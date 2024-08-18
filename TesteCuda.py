import torch
import torch.nn as nn
import torch.optim as optim
import time

# Definindo um modelo de rede neural simples
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Função para medir o tempo de execução
def measure_time(device):
    model = SimpleModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Dados fictícios
    inputs = torch.randn(100000, 10).to(device)
    targets = torch.randn(100000, 1).to(device)
    
    # Treinamento
    start_time = time.time()
    for epoch in range(100):  # 100 épocas de treinamento
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    end_time = time.time()
    
    return end_time - start_time

# Executando na CPU
device = torch.device('cpu')
cpu_time = measure_time(device)
print(f"Tempo de execução na CPU: {cpu_time:.4f} segundos")

# Verificando se a GPU está disponível e executando na GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_time = measure_time(device)
    print(f"Tempo de execução na GPU: {gpu_time:.4f} segundos")
else:
    print("GPU não disponível.")
