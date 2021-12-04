import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        activation: nn.Module=nn.ReLU(),
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            activation,
            nn.Conv2d(in_ch, out_ch, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(out_ch),
        )
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.net(x)

class UpBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        activation: nn.Module=nn.ReLU(),
        dropout: int=0
    ) -> None:
        
        super().__init__()
        self.net = nn.Sequential(
            activation,
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.tensor, residual: torch.tensor) -> torch.tensor:
        x = self.net(x)
        if residual is not None:
            x = torch.cat([x, residual], dim=1)
        return x

class Generator(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        in_ch: int,
        hid_ch: int,
        out_ch: int
    ) -> None:
        
        super().__init__()
        encoder = []
        encoder.append(
            nn.Conv2d(in_ch, hid_ch, kernel_size=4, padding=1, stride=2)
        )
        cur_ch = hid_ch
        for i in range(1, n_blocks-1):
            encoder.append(DownBlock(
                cur_ch, min(2**i, 8) * hid_ch, nn.LeakyReLU(0.2)))
            cur_ch = min(2**i, 8) * hid_ch
        encoder.append(
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(cur_ch, min(2**i, 8) * hid_ch, kernel_size=4, padding=1, stride=2),
            )
        )
        self.encoder = nn.ModuleList(encoder)
        
        decoder = []
        for i in range(n_blocks-2, -1, -1):
            dropout = 0.5 if i > 3 else 0.
            decoder.append(UpBlock(cur_ch, min(2**i, 8) * hid_ch, dropout=dropout))
            cur_ch = 2 * min(2**i, 8) * hid_ch
        decoder.append(UpBlock(cur_ch, out_ch, nn.Tanh()))
        self.decoder = nn.ModuleList(decoder)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        residuals = []
        for layer in self.encoder:
            residuals.append(x)
            x = layer(x)
        residuals[0] = None
        for i, layer in enumerate(self.decoder):
            x = layer(x, residuals[-(i+1)])
        return x