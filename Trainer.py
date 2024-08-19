import torch
from torch.utils.data import Dataset, DataLoader
from utils import MyTrainDataset # MyTrainDataset é uma classe de conjunto de dados personalizada

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup():
        # Inicializa o grupo de processos
    init_process_group(backend='nccl') #gloo para CPU


class Trainer:
    def _init_(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            save_every: int,
            snapshot_path: str,
    ) -> None:
        """
        Inicializa a classe Trainer.

        Args:
            model: O modelo de rede neural a ser treinado
            train_data: DataLoader contendo os dados de treinamento
            optimizer: Otimizador para atualizar os pesos do modelo
            gpu_id: ID da GPU a ser usada para treinamento
            save_every: Intervalo de salvamento do modelo em checkpoints
        """
        self.gpu_id = int(os.environ.get("LOCAL_RANK")) 
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        if os.path.exists(snapshot_path):
            print(f"Carregando snapshot de {snapshot_path}")
            self._load_snapshot(snapshot_path)
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
            snapshot = torch.load(snapshot_path)
            self.model.load_state_dict(snapshot["MODEL_STATE"])
            self.epochs_run = snapshot["EPOCHS_RUN"]
            print(f"Snapshot carregado de {snapshot_path} com {self.epochs_run} épocas treinadas")
    
    def _run_batch(self, source, targets):
        """
        Executa uma etapa de treinamento para um lote de dados.

        Args:
            source: Dados de entrada para o modelo
            targets: Rótulos/targets correspondentes aos dados de entrada
        """
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = torch.nn.CrossEntropyLoss()(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        """
        Executa uma época de treinamento.

        Args:
            epoch: O número da época atual
        """
        b_sz = len(next(iter(self.train_data)) [0]) # Tamanho do lote
        print(f"[GPU {self.gpu_id}] Epoch {epoch} | Batchsize {b_sz} | Steps: {len(self.train_data)}") 
        for i, (source, targets) in enumerate(self.train_data): # Itera sobre os lotes
            source = source.to(self.gpu_id) # Move os dados para a GPU
            targets = targets.to(self.gpu_id) # Move os rótulos para a GPU
            self._run_batch(source, targets) # Executa uma etapa de treinamento

    def _save_snapshot(self, epoch):
        """
        Salva um checkpoint do modelo treinado.
        
        Args:
            epoch: O número da época atual
        """
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, f"snapshot.pt") # Salva o checkpoint
        print(f"Epoch {epoch} | Ponto de verificação de treinamento salvo em snapshot.pt")

    def train(self, max_epochs: int):
        """
        Treina o modelo por um número específico de épocas.
        
        Args:
            max_epochs: Número total de épocas para treinar o modelo
        """
        for epoch in range(self.epochs_run, max_epochs): # Itera sobre as épocas
            self._run_epoch(epoch) # Executa uma época de treinamento
            if self.gpu_id == 0 and epoch % self.save_every == 0: # Salva o modelo a cada 'save_every' épocas
                self._save_snapshot(epoch)
    
    def load_train_objs():
        """
        Carrega os objetos necessários para treinamento.
        
        Returns:
            train_set: Conjunto de dados de treinamento
            model: Modelo de rede neural
            optimizer: Otimizador para atualizar os pesos do modelo
        """
        train_set = MyTrainDataset(2048) #Carrega seu conjunto de dados
        model = torch.nn.Linear(20,1) #Carrega seu modelo
        optmizer = torch.optim.SGD(model.parameters(), lr=1e-3) #carrega seu otimizador
        return train_set, model, optimizer
    
    def prepare_dataloader(dataset: Dataset, batch_size: int):
        """
        Prepara um DataLoader para um conjunto de dados.
        
        Args:
            dataset: Conjunto de dados para carregar
            batch_size: Tamanho do lote para carregar
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True, # Otimiza a transferência de dados para a GPU
            shuffle=False, # Não embaralha os dados
            sampler=DistributedSampler(dataset), # Usa um sampler distribuído
        )
    
    def main(total_epochs: int, save_every: int, snapshot_path: str = "snapshot.pt"):
        """
        Função principal para treinamento distribuído.
        
        Args:
            rank: Identificador do processo atual
            world_size: Número total de processos
            total_epochs: Número total de épocas para treinar o modelo
            save_every: Intervalo de salvamento do modelo em checkpoints
        """
        ddp_setup() # Configura o treinamento distribuído
        train_set, model, optimizer = load_train_objs() # Carrega os objetos necessários para treinamento
        train_data = prepare_dataloader(train_set, batch_size=32) # Prepara o DataLoader para o conjunto de dados
        trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path) # Inicializa o treinador
        trainer.train(total_epochs) # Treina o modelo
        destroy_process_group() # Finaliza o grupo de processos

    if _name_=="_main_":
        """
        Inicia o treinamento distribuído.
        
        Args:
            sys.argv[1]: Número total de épocas para treinar o modelo
            sys.argv[2]: Intervalo de salvamento do modelo em checkpoints
        """
        import sys
        total_epochs = int(sys.argv[1]) # Número total de épocas para treinar o modelo
        save_every = int(sys.argv[2]) # Intervalo de salvamento do modelo em checkpoints
        main (total_epochs, save_every) # Inicia o treinamento distribuído