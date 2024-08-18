import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        dist.send(tensor=tensor, dst=1)
        print(f'Rank {rank} enviou dados {tensor[0]}')
    elif rank == 1:
        dist.recv(tensor=tensor, src=0)
        print(f'Rank {rank} recebeu dados {tensor[0]}')

def init_processes(rank, size, fn, master_ip, backend='gloo'):
    dist.init_process_group(backend, init_method=f'tcp://{master_ip}:29500', rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    size = 2
    master_ip = input("Digite o IP do nó mestre: ")
    rank = int(input("Digite o rank deste nó (0 para mestre, 1 para trabalhador): "))
    init_processes(rank, size, run, master_ip)