import torch
import torch.distributed as dist
from torch.multiprocessing import Process

def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        dist.send(tensor=tensor, dst=1)
        print(f'Rank {rank} sent data {tensor[0]}')
    elif rank == 1:
        dist.recv(tensor=tensor, src=0)
        print(f'Rank {rank} received data {tensor[0]}')

def init_processes(rank, size, fn, master_ip, backend='gloo'):
    dist.init_process_group(backend, init_method=f'tcp://{master_ip}:29500', rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 2 
    master_ip = "192.168.0.97" 
    init_processes(1, size, run, master_ip)
