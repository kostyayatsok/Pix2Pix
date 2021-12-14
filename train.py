import torch
from model import Generator, Discriminator
from data import get_facade_dataloaders
from loss import Pix2PixLoss
from test import calc_fid
from utils import Timer, MetricTracker 
import numpy as np


WANDB = False
if WANDB:
    import wandb
    import matplotlib.pyplot as plt
    wandb.init(project='Pix2Pix')

class Trainer:
    def __init__(
        self, batch_size: int, log_freq: int=5,
        save_freq: int=10, fid_freq: int=50,
        start_epoch: int=0
    ) -> None:
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.generator = Generator(
            n_blocks=8, in_ch=3, hid_ch=64, out_ch=3).to(self.device)
        if start_epoch > 0:
            self.generator.load_state_dict(torch.load('generator.pt'))
        print(f"Created generator with "
              f"{sum([p.numel() for p in self.generator.parameters()])} "
              f"parameters")

        self.discriminator = Discriminator(
            n_blocks=3, in_ch=6, hid_ch=64).to(self.device)
        if start_epoch > 0:
            self.discriminator.load_state_dict(torch.load('discriminator.pt'))
        print(f"Created discriminator with"
              f" {sum([p.numel() for p in self.discriminator.parameters()])}"
              f" parameters")

        self.g_opt = torch.optim.AdamW(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_opt = torch.optim.AdamW(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

#         self.g_opt = torch.optim.SGD(self.generator.parameters(), lr=0.0002)
#         self.d_opt = torch.optim.SGD(self.discriminator.parameters(), lr=0.0002)

        
        self.train_loader, self.val_loader = get_facade_dataloaders(batch_size, num_workers=3)
        
        self.criterion = Pix2PixLoss(_lambda=100.)
        self.n_epoch = 400
        self.start_epoch = start_epoch
        self.step = start_epoch * len(self.train_loader)
        self.g_warmup_steps = 0
        
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.fid_freq = fid_freq
        
        self.timer = Timer()
        self.tracker = MetricTracker(eternals=['time'])
        
    def __call__(self):
        self.generator.train()
        self.discriminator.train()        
        for self.epoch in range(self.start_epoch+1, self.n_epoch+1):
            self.timer.start('epoch_duration')
            print(
                f"Epoch {self.epoch:04d}/{self.n_epoch:04d}:", flush=True)
            
            self.timer.start("train")
            for i, batch in enumerate(self.train_loader):                
                self.d_opt.zero_grad()
                self.g_opt.zero_grad()
                
                self.process_batch(batch)
                
                if self.step % 2 and batch['d_loss'] > 0.1:#
                    batch['d_loss'].backward()
#                     torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 50)
                    self.d_opt.step()
                else:
                    batch['g_loss'].backward()
#                     torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 50)
                    self.g_opt.step()

                self.tracker(batch, suffix='train')
                
                self.step += 1

                if WANDB and self.step % self.log_freq == 0:
                    self.timer.start("log_train")
                    self.log('train', batch)
                    self.timer.end("log_train")
                    self.tracker({"log_train": self.timer.get("log_train")}, suffix='time')
            self.timer.end("train")
            self.tracker({"train": self.timer.get("train")}, suffix='time')

            with torch.no_grad():
                self.timer.start("val") 
                for batch in self.val_loader:
                    self.process_batch(batch)
                    self.tracker(batch, suffix='val')
                self.timer.end("val")
                self.tracker({"val": self.timer.get("val")}, suffix='time')
                
            if self.epoch % self.save_freq == 0:
                self.timer.start("save")
                torch.save(self.generator.state_dict(), "generator.pt")
                torch.save(self.discriminator.state_dict(), "discriminator.pt")
                self.timer.end("save")
                self.tracker({"save": self.timer.get("save")}, suffix='time', count=self.save_freq)
            if self.epoch % self.fid_freq == 0:
#                 self.timer.start("fid_train")
#                 self.tracker(
#                     {
#                         'fid': calc_fid(
#                             self.generator, self.val_loader,
#                             self.device, './facades/train/', './predictions/'
#                         ) 
#                     }
#                     , suffix='train'
#                 )
#                 self.timer.end("fid_train")
#                 self.tracker({"fid_train": self.timer.get("fid_train")}, suffix='time', count=self.fid_freq)

                self.timer.start("fid_val")
                self.tracker(
                    {
                        'fid': calc_fid(
                            self.generator, self.val_loader,
                            self.device, './facades/val/', './predictions_val/'
                        )
                    }, suffix='val'
                )
                self.timer.end("fid_val")
                self.tracker({"fid_val": self.timer.get("fid_val")}, suffix='time', count=self.fid_freq)

            if WANDB:
                self.timer.start("log_val")
                self.log('val', batch)
                self.timer.end("log_val") 
                self.tracker({"log_val": self.timer.get("log_val")}, suffix='time')
            self.timer.end("epoch_duration")
            
            self.tracker({"epoch_duration": self.timer.get("epoch_duration")})
            
    def process_batch(self, batch: dict()) -> None:
        batch['mask'] = batch['mask'].to(self.device)
        batch['real_image'] = batch['real_image'].to(self.device)
        
        batch['fake_image'] = self.generator(batch['mask'])
        batch['d_fake'] = self.discriminator(torch.cat((batch['mask'], batch['fake_image']), dim=1))
        batch['d_real'] = self.discriminator(torch.cat((batch['mask'], batch['real_image']), dim=1))
        
        self.criterion(batch)
        
    def log(self, mode, batch=None):
        if not WANDB:
            return
        log = self.tracker.get_group(mode)
        log['step'] = self.step
        log['epoch'] = self.epoch
        log['epoch_duration'] = self.tracker['epoch_duration']

        if mode == 'val':
            time = self.tracker.get_group('time')
            data = [[label, val] for (label, val) in time.items()]
            table = wandb.Table(data=data, columns = ["label", "value"])
            log["time"] = wandb.plot.bar(table, "label", "value", title="Time")
        
            
        if batch is not None:
            idx = np.random.randint(batch['real_image'].size(0))
            
            real_image = ((batch['real_image'][idx].detach().cpu().numpy().transpose([1, 2, 0]) + 1) * 127.5).astype(np.uint8)
            fake_image = ((batch['fake_image'][idx].detach().cpu().numpy().transpose([1, 2, 0]) + 1) * 127.5).astype(np.uint8)
            mask = ((batch['mask'][idx].detach().cpu().numpy().transpose([1, 2, 0]) + 1) * 127.5).astype(np.uint8)
            d_fake = (torch.sigmoid(batch['d_fake'][idx]).detach().cpu().numpy() * 255).astype(np.uint8)
            d_real = (torch.sigmoid(batch['d_real'][idx]).detach().cpu().numpy() * 255).astype(np.uint8)

            log[f'real_{mode}'] = wandb.Image(real_image)
            log[f'fake_{mode}'] = wandb.Image(fake_image)
            log[f'mask_{mode}'] = wandb.Image(mask)
            log[f'd_fake_{mode}'] = wandb.Image(d_fake)
            log[f'd_real_{mode}'] = wandb.Image(d_real)
         
        wandb.log(log)
