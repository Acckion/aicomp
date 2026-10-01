"""BF16 understanding-only runtime, with dormant image experts on CPU."""
import os
import sys
import json
from pathlib import Path

import torch

def load_model(model_path, source_path, output):
    sys.path.insert(0, str(source_path))
    from inference.sensenova_vision import SenseNovaVisionModel
    torch.set_num_threads(2)
    assert torch.cuda.device_count() == 4
    for i in range(4):
        torch.cuda.set_per_process_memory_fraction(8.5*1024**3/torch.cuda.get_device_properties(i).total_memory, i)

    class SharedMemoryModel(SenseNovaVisionModel):
        def _infer_device_map(self, model):
            mapping = {'vit_model': 0, 'connector': 0, 'vit_pos_embed': 0,
                       'time_embedder': 0, 'latent_pos_embed': 0, 'vae2llm': 0, 'llm2vae': 0,
                       'language_model.model.embed_tokens': 0, 'language_model.lm_head': 0,
                       'language_model.model.norm': 0, 'language_model.model.norm_moe_gen': 'cpu',
                       'language_model.model.rotary_emb': 0}
            for i, layer in enumerate(model.language_model.model.layers):
                device = 0 if i < 4 else 1 if i < 12 else 2 if i < 20 else 3
                prefix = f'language_model.model.layers.{i}'
                mapping[prefix] = device
                for name, _ in layer.named_modules():
                    if name and 'moe_gen' in name and not any('moe_gen' in p for p in name.split('.')[:-1]):
                        mapping[f'{prefix}.{name}'] = 'cpu'
            (Path(output)/'device_map.json').write_text(json.dumps(mapping, indent=2))
            return mapping

    return SharedMemoryModel(model_path=str(model_path), device='cuda', dtype='bf16',
                            offload_folder='/dev/shm/aicomp_sensenova/offload')
