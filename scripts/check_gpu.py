"""Check CUDA operations and collective communication without training any model."""
from datetime import timedelta
import os
import torch
import torch.distributed as dist

rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=timedelta(seconds=90))
value = torch.tensor([float(dist.get_rank() + 1)], device='cuda')
dist.all_reduce(value)
world = dist.get_world_size()
assert value.item() == world * (world + 1) / 2
a = torch.ones((128,128), device='cuda')
assert (a @ a)[0,0].item() == 128
print(f'GPU {rank}: {torch.cuda.get_device_name(rank)} CUDA and NCCL passed', flush=True)
dist.barrier()
dist.destroy_process_group()
