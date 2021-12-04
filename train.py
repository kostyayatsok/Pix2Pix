import torch
import torch.nn as nn
from model import Generator
from data import get_dataloaders
import numpy as np

class Trainer:
    def __init__(self, batch_size) -> None:
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.generator = Generator(n_blocks=7, in_ch=1, hid_ch=64, out_ch=3)
        self.optimizer = torch.optim.Adam(self.generator.parameters())
        self.train_loader, self.val_loader = get_dataloaders(batch_size)
        self.criterion = nn.L1Loss()
        self.n_epoch = 100
    def __call__(self):
        for epoch in range(self.n_epoch):
            print(f"Started epoch {epoch:03d}/{self.n_epoch:03d}", end=' ')
            self.generator.train()
            losses = []
            for input, target in self.train_loader:
                self.optimizer.zero_grad()
                
                output, loss = self.process_batch(input, target)
                losses.append(loss.item())
                
                loss.backward()
                self.optimizer.step()
            val_losses = []
            with torch.no_grad():
                self.generator.eval()
                for input, target in self.val_loader:
                    output, loss = self.process_batch(input, target)
                    val_losses.append(loss.item())
            loss = np.mean(losses)
            val_loss = np.mean(val_losses)
            print(f"loss: {loss:.4f}, val_loss: {val_loss:.4f}")
    def process_batch(self, input, target):
        input = input.to(self.device)
        target = target.to(self.device)
        
        output = self.generator(input)
        loss = self.criterion(output)

        return output, loss
