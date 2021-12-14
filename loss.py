import torch
import torch.nn as nn

class Pix2PixLoss(nn.Module):
    def __init__(self, _lambda=100.) -> None:
        super().__init__()
        self.l1_criterion = nn.L1Loss()
        self.ce_criterion = nn.BCEWithLogitsLoss()
        self._lambda = _lambda
    def forward(self, batch: dict()) -> None:
        batch['g_l1_loss'] =  self.l1_criterion(
            batch['fake_image'], batch['real_image'])
        
        fake_labels = torch.zeros(
            batch['d_fake'].size(),
            dtype=torch.float,
            device=batch['d_fake'].device,
        ) #/ 10 #using label smoothing
        batch['d_fake_loss'] = self.ce_criterion(batch['d_fake'], fake_labels)
        batch['d_real_loss'] = self.ce_criterion(batch['d_real'], 1 - fake_labels)
        batch['g_fake_loss'] = self.ce_criterion(batch['d_fake'], 1 - fake_labels)
        
        batch['d_loss'] = (batch['d_real_loss'] + batch['d_fake_loss']) / 2
        batch['g_loss'] = batch['g_fake_loss'] + self._lambda * batch['g_l1_loss'] 
    