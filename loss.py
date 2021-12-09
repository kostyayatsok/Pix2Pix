import torch
import torch.nn as nn

class Pix2PixLoss(nn.Module):
    def __init__(self, _lambda=100.) -> None:
        super().__init__()
        self.l1_criterion = nn.L1Loss()
        self.ce_criterion = nn.CrossEntropyLoss()
        self._lambda = _lambda
    def forward(self, batch: dict()) -> None:
        batch['g_l1_loss'] =  self.l1_criterion(
            batch['fake_image'], batch['real_image'])
        
        batch['d_fake_loss'] = self.ce_criterion(
            batch['d_fake'], torch.zeros_like(batch['d_fake'])
        )
        batch['d_real_loss'] = self.ce_criterion(
            batch['d_real'], torch.ones_like(batch['d_real'])
        )
        batch['d_loss'] = (batch['d_real_loss'] + batch['d_fake_loss']) / 2
        batch['g_loss'] = batch['d_loss'] + self._lambda * batch['g_l1_loss']
    