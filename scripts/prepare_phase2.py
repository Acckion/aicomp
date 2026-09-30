"""Extract official RGB inputs for inference only, with CRC and image validation."""
import json,zipfile
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'data/phase2';out.mkdir(exist_ok=True)
with zipfile.ZipFile(ROOT/'AIC2026_PHASE_2_1000.zip') as z:
    entries=[i for i in z.infolist() if i.filename.startswith('visible/') and not i.is_dir()]
    if not entries:
        raise ValueError('No visible/ RGB folder in official archive')
    assert len(entries)==1000,len(entries)
    images=[]
    for i,entry in enumerate(sorted(entries,key=lambda e:e.filename)):
        path=Path(entry.filename)
        assert len(path.parts)==2 and '..' not in path.parts
        dst=out/path;dst.parent.mkdir(exist_ok=True)
        # zipfile.read checks CRC before returning data.
        raw=z.read(entry)
        if not dst.exists():dst.write_bytes(raw)
        else:assert dst.read_bytes()==raw
        with Image.open(dst) as image:
            image.load();w,h=image.size
        images.append({'id':i+1,'file_name':entry.filename,'width':w,'height':h})
    (out/'images.json').write_text(json.dumps({'images':images,'purpose':'official phase2 inference only; no annotations; excluded from training'},indent=2))
print('Prepared 1000 RGB phase2 inputs for inference only')
