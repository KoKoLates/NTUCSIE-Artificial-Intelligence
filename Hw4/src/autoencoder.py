import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

class Autoencoder(nn.Module):
    """" implementation of autoencoder """
    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        super(Autoencoder, self).__init__()
        self.encoder: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.ReLU()
        )
        self.decoder: nn.Sequential = nn.Sequential(
            nn.Linear(encoding_dim // 2, encoding_dim),
            nn.Linear(encoding_dim, input_dim),
        )
    
    def forward(self, x: np.ndarray) -> nn.Sequential:
        #TODO: 5%
        return self.decoder(self.encoder(x))
    
    def fit(
        self, 
        X: np.ndarray, 
        epochs: int = 10, 
        batch_size: int = 32
    ) -> None:
        """ training autoencoder """
        #TODO: 5%
        optimizer: optim.Adam = optim.Adam(self.parameters(), lr=0.001)
        loss_func: nn.MSELoss = nn.MSELoss()
        data_loader = DataLoader(
            dataset=TensorDataset(torch.tensor(X, dtype=torch.float)),
            batch_size=batch_size,
            shuffle=False
        )

        history: list = []
        for _ in tqdm(range(epochs)):
            epoch_loss: torch.Tensor = 0
            for batch in data_loader:
                batch_tensor: torch.Tensor = torch.cat(batch)
                optimizer.zero_grad()
                loss: torch.Tensor = loss_func(batch_tensor, self(batch_tensor))
                loss.backward()
                optimizer.step()
                epoch_loss += loss

            epoch_loss /= len(data_loader)
            history.append(epoch_loss.item())

        plt.plot(history)
        plt.title('autoencoder training loss')
        plt.savefig('./output/autoencoder')
        plt.clf()        
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        #TODO: 2%
        output_tensor: torch.Tensor = self.encoder(torch.tensor(X, dtype=torch.float))
        return output_tensor.detach().numpy()
    
    def reconstruct(self, X: torch.Tensor) -> np.ndarray:
        #TODO: 2%
        tensor_output: torch.Tensor = self.decoder(torch.tensor(\
            self.transform(X.detach().numpy()), dtype=torch.float))
        return tensor_output.detach().numpy()


class DenoisingAutoencoder(Autoencoder):
    """ implementation of denoise autoencoder """
    def __init__(
        self, 
        input_dim: int, 
        encoding_dim: int, 
        noise_factor: torch.float = 0.2
    ) -> None:
        super(DenoisingAutoencoder, self).__init__(input_dim, encoding_dim)
        self.noise_factor: torch.float = noise_factor
    
    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        #TODO: 3%
        mean: torch.Tensor = torch.zeros(x.size())
        _std: torch.Tensor = torch.zeros(x.size()) + self.noise_factor
        return x + torch.normal(mean=mean, std=_std)
    
    def fit(self, 
        X: np.ndarray, 
        epochs: int = 10, 
        batch_size: int = 32
    ) -> None:
        #TODO: 4%
        optimizer: optim.Adam = optim.Adam(self.parameters(), lr=0.001)
        loss_func: nn.MSELoss = nn.MSELoss()
        data_loader = DataLoader(
            dataset=TensorDataset(torch.tensor(X, dtype=torch.float)),
            batch_size=batch_size,
            shuffle=False
        )

        history: list = []
        for _ in tqdm(range(epochs)):
            epoch_loss: torch.Tensor = 0
            for batches in data_loader:
                batch_tensor: torch.Tensor = torch.cat([self.add_noise(batch) for batch in batches])
                optimizer.zero_grad()
                loss: torch.Tensor = loss_func(batch_tensor, self(batch_tensor))
                loss.backward()
                optimizer.step()
                epoch_loss += loss

            epoch_loss /= len(data_loader)
            history.append(epoch_loss.item())
        
        plt.plot(history)
        plt.title('denoise autoencoder training loss')
        plt.savefig('./output/denoise_autoencoder')
        plt.clf()


