import torch
import torch.nn as nn

def init_weights(model):
    def init_fn(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    model.apply(init_fn)

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
            nn.Conv2d(in_ch, out_ch, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch, track_running_stats=False),
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
                in_ch, out_ch, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch, track_running_stats=False),
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
            nn.Conv2d(in_ch, hid_ch, kernel_size=4, padding=1, stride=2, bias=False)
        )
        cur_ch = hid_ch
        for i in range(1, n_blocks-1):
            encoder.append(DownBlock(
                cur_ch, min(2**i, 8) * hid_ch, nn.LeakyReLU(0.2)))
            cur_ch = min(2**i, 8) * hid_ch
        encoder.append(
            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(
                    cur_ch, min(2**i, 8) * hid_ch,
                    kernel_size=4, padding=1, stride=2, bias=False
                ),
            )
        )
        self.encoder = nn.ModuleList(encoder)
        
        decoder = []
        for i in range(n_blocks-1, 0, -1):
            dropout = 0.5 if i > 3 else 0.
            decoder.append(
                UpBlock(cur_ch, min(2**(i-1), 8) * hid_ch, dropout=dropout))
            cur_ch = 2 * min(2**(i-1), 8) * hid_ch
        decoder.append(UpBlock(cur_ch, out_ch))
        self.decoder = nn.ModuleList(decoder)
        
        init_weights(self)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        residuals = []
        for layer in self.encoder:
            residuals.append(x)
            x = layer(x)
        residuals[0] = None
        for i, layer in enumerate(self.decoder):
            x = layer(x, residuals[-(i+1)])
        return nn.Tanh()(x)

class DiscBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int=2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch, track_running_stats=False),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, n_blocks: int, in_ch: int, hid_ch: int) -> None:
        super().__init__()
        layers = []
        layers.append(
            nn.Conv2d(in_ch, hid_ch, kernel_size=4, padding=1, stride=2, bias=False)
        )
        cur_ch = hid_ch
        for i in range(1, n_blocks):
            layers.append(DiscBlock(cur_ch, min(2**i, 8) * hid_ch, stride=2))
            cur_ch = min(2**i, 8) * hid_ch
        layers.append(DiscBlock(cur_ch, min(2**i, 8) * hid_ch, stride=1))
        layers.append(
            nn.Conv2d(cur_ch, 1, kernel_size=4, padding=1, stride=1, bias=False)
        )
        self.net = nn.Sequential(*layers)
        
        init_weights(self)    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
