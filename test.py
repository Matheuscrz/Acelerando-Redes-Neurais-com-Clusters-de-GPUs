import cupy as cp
import numpy as np
import time

# Função para executar no GPU
def gpu_func():
    x_gpu = cp.arange(10000000)  # 10 milhões de elementos
    start = time.time()
    y_gpu = cp.exp(x_gpu)
    cp.cuda.Stream.null.synchronize()  # Sincroniza o fluxo para garantir que a computação esteja concluída
    end = time.time()
    print(f"Tempo de execução no GPU: {end - start} segundos")

# Função para executar no CPU
def cpu_func():
    x_cpu = np.arange(10000000)  # 10 milhões de elementos
    start = time.time()
    y_cpu = np.exp(x_cpu)
    end = time.time()
    print(f"Tempo de execução no CPU: {end - start} segundos")

# Executando as funções
gpu_func()
cpu_func()
