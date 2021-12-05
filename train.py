import torch
import torch.nn as nn
from model import Generator
from data import get_facade_dataloaders
import numpy as np
import torchvision.transforms.functional as F

class Trainer:
    def __init__(self, batch_size) -> None:
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.generator = Generator(n_blocks=7, in_ch=3, hid_ch=64, out_ch=3).to(self.device)
        print(f"Create generator with {sum([p.numel() for p in self.generator.parameters()])} parameters")
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.train_loader, self.val_loader = get_facade_dataloaders(batch_size)
        self.criterion = nn.L1Loss()
        self.n_epoch = 250
    def __call__(self):
        for epoch in range(1, self.n_epoch+1):
            print(f"Started epoch {epoch:04d}/{self.n_epoch:04d}:", end=' ', flush=True)
            self.generator.train()
            losses = []
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                
                output, loss = self.process_batch(**batch)
                
                loss.backward()
                self.optimizer.step()
#                 self.scheduler.step(loss)
                losses.append(loss.item())
            
            pil_image = F.to_pil_image(output[0]+0.5)
            pil_image.save("images/output_train.jpg")
            pil_image = F.to_pil_image(batch['target'][0]+0.5)
            pil_image.save("images/target_train.jpg")
            pil_image = F.to_pil_image(batch['input'][0])
            pil_image.save("images/input_train.jpg")
            
            val_losses = []
            with torch.no_grad():
                # self.generator.eval()
                for batch in self.val_loader:
                    output, loss = self.process_batch(**batch)
                    val_losses.append(loss.item())
                pil_image = F.to_pil_image(output[0]+0.5)
                pil_image.save("images/output_val.jpg")
                pil_image = F.to_pil_image(batch['target'][0]+0.5)
                pil_image.save("images/target_val.jpg")
                pil_image = F.to_pil_image(batch['input'][0])
                pil_image.save("images/input_val.jpg")
            loss = np.mean(losses)
            val_loss = np.mean(val_losses)
            print(f"loss: {loss:.6f}, val_loss: {val_loss:.6f}, lr: {self.optimizer.param_groups[0]['lr']:.7f}", flush=True)
            if epoch % 10 == 0:
                torch.save(self.generator.state_dict(), "generator.pt")
    def process_batch(self, input, target, *args, **kwargs):
        input = input.to(self.device)
        target = target.to(self.device)
        
        output = self.generator(input)
        loss = self.criterion(output, target)

        return output, loss
