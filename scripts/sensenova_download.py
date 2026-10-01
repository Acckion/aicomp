"""Download the pinned official checkpoint with resumable, checked HTTP ranges.

Large artifacts stay in volatile RAM storage; only progress is written to AIC.
Whole-file SHA256 must match official Hugging Face LFS metadata before use.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import requests

REPO = 'sensenova/SenseNova-Vision-7B-MoT'
REVISION = '79548fcc5b954598799b9317f8d3ec5e347d5c0e'
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--directory', default='/dev/shm/aicomp_sensenova/model')
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    folder = Path(args.directory)
    folder.mkdir(parents=True, exist_ok=True)
    output = ROOT / 'experiments/sensenova_probe'
    output.mkdir(parents=True, exist_ok=True)
    response = requests.get(f'https://huggingface.co/api/models/{REPO}/tree/{REVISION}', timeout=60)
    response.raise_for_status()
    entries = [f for f in response.json() if f['path'].endswith(('.json', '.txt', '.safetensors'))]
    record = {'repo': REPO, 'revision': REVISION, 'files': [], 'volatile_storage': True}
    for item in entries:
        name, size = item['path'], item['size']
        target = folder / name
        digest = item.get('lfs', {}).get('oid')
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
        if target.exists() and target.stat().st_size == size:
            actual = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
            if not digest or actual == digest:
                record['files'].append({'name': name, 'bytes': size, 'sha256': actual})
                continue
        if size < 1024 * 1024:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            assert len(r.content) == size, name
            target.write_bytes(r.content)
        else:
            partial = target.with_suffix(target.suffix + '.part')
            state_path = partial.with_suffix(partial.suffix + '.json')
            chunk = 32 * 1024 * 1024
            done = set()
            if partial.exists() and state_path.exists():
                saved = json.loads(state_path.read_text())
                if saved['revision'] == REVISION and saved['size'] == size and saved['chunk'] == chunk:
                    done = set(saved['done'])
            fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o600)
            os.ftruncate(fd, size)

            def fetch(index):
                left = index * chunk
                right = min(size, left + chunk) - 1
                for attempt in range(6):
                    try:
                        with requests.get(url, headers={'Range': f'bytes={left}-{right}'},
                                          stream=True, timeout=(30, 120)) as r:
                            r.raise_for_status()
                            assert r.status_code == 206, (name, r.status_code)
                            assert r.headers.get('Content-Range') == f'bytes {left}-{right}/{size}'
                            offset = left
                            for data in r.iter_content(1024 * 1024):
                                assert offset + len(data) <= right + 1
                                view = memoryview(data)
                                while view:
                                    written = os.pwrite(fd, view, offset)
                                    assert written > 0
                                    offset += written
                                    view = view[written:]
                            assert offset == right + 1, (name, offset, right)
                        return index
                    except Exception:
                        if attempt == 5:
                            raise
                        time.sleep(min(2 ** attempt, 20))

            total = (size + chunk - 1) // chunk
            start = time.monotonic()
            try:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    tasks = [pool.submit(fetch, i) for i in range(total) if i not in done]
                    for future in as_completed(tasks):
                        done.add(future.result())
                        state_path.write_text(json.dumps({'revision': REVISION, 'size': size,
                                                          'chunk': chunk, 'done': sorted(done)}))
                        if len(done) % 16 == 0 or len(done) == total:
                            print(f'{name}: {len(done)}/{total} chunks; {time.monotonic()-start:.0f}s', flush=True)
                            (output / 'download_status.json').write_text(json.dumps({
                                'stage': 'downloading', 'file': name, 'chunks': len(done),
                                'total_chunks': total, 'time': time.time()}))
                os.fsync(fd)
            finally:
                os.close(fd)
            actual = hashlib.file_digest(partial.open('rb'), 'sha256').hexdigest()
            assert digest and actual == digest, ('SHA256 mismatch', name, actual, digest)
            partial.replace(target)
            state_path.unlink(missing_ok=True)
        assert target.stat().st_size == size
        actual = hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()
        assert not digest or actual == digest
        record['files'].append({'name': name, 'bytes': size, 'sha256': actual})
    (output / 'model_identity.json').write_text(json.dumps(record, indent=2))
    (folder.parent / 'download.complete').write_text(str(folder))
    print('MODEL_DOWNLOAD_COMPLETE; official LFS checksums verified', flush=True)


if __name__ == '__main__':
    main()
