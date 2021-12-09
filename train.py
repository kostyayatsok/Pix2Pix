import torch
from model import Generator, Discriminator
from data import get_facade_dataloaders
from loss import Pix2PixLoss
from test import calc_fid
from utils import Timer, MetricTracker 

WANDB = False
if WANDB:
    import wandb
    import matplotlib.pyplot as plt
    wandb.init(project='Pix2Pix')

class Trainer:
    def __init__(
        self, batch_size: int, log_freq: int=10,
        save_freq: int=10, fid_freq: int=10,
    ) -> None:
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.generator = Generator(
            n_blocks=7, in_ch=3, hid_ch=64, out_ch=3).to(self.device)
        print(f"Create generator with "
              f"{sum([p.numel() for p in self.generator.parameters()])} "
              f"parameters")

        self.discriminator = Discriminator(
            n_blocks=3, in_ch=3, hid_ch=64).to(self.device)
        
        print(f"Create discriminator with"
              f" {sum([p.numel() for p in self.discriminator.parameters()])}"
              f" parameters")

        self.g_opt = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.train_loader, self.val_loader = get_facade_dataloaders(batch_size)
        
        self.criterion = Pix2PixLoss()
        self.n_epoch = 250
        self.step = 0
        
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.fid_freq = fid_freq
        
        self.timer = Timer()
        self.tracker = MetricTracker(eternals=['time'])
        
    def __call__(self):
        self.generator.train()
        self.discriminator.train()
        for epoch in range(1, self.n_epoch+1):
            self.timer.start('epoch_time')
            print(
                f"Epoch {epoch:04d}/{self.n_epoch:04d}:", end='\r', flush=True)
            
            self.timer.start("train")
            for i, batch in enumerate(self.train_loader):                
                self.d_opt.zero_grad()
                self.g_opt.zero_grad()
                
                self.process_batch(batch)
                
                self.discriminator.requires_grad_=True
                self.generator.requires_grad_=False
                batch['d_loss'].backward()
                self.d_opt.step()

                self.discriminator.requires_grad_=False                
                self.generator.requires_grad_=True
                batch['g_loss'].backward()
                self.g_opt.step()

                self.tracker(batch, suffix='train')
                
                self.step += 1

                if WANDB and self.step % self.log_freq == 0:
                    self.timer.start("log_train")
                    self.log('train')
                    self.timer.end("log_train")
                    self.tracker(self.timer.get("log_train"), suffix='time')
            self.timer.end("train")

            with torch.no_grad():
                self.timer.start("val") 
                for batch in self.val_loader:
                    self.process_batch(batch)
                    self.tracker(batch, suffix='val')
                self.timer.end("val")
            
            if epoch % self.save_freq == 0:
                self.timer.start("save")
                torch.save(self.generator.state_dict(), "generator.pt")
                torch.save(self.discriminator.state_dict(), "discriminator.pt")
                self.timer.end("save")
            
            if epoch % self.fid_freq == 0:
                self.timer.start("fid_train")
                self.tracker(
                    calc_fid(
                        self.generator, self.val_loader,
                        self.device, './facades/train/', './predictions/'
                    ), suffix='train'
                )
                self.timer.end("fid_train")
                self.timer.start("fid_val")
                self.tracker(
                    calc_fid(
                        self.generator, self.val_loader,
                        self.device, './facades/val/', './predictions/'
                    ), suffix='val'
                )
                self.timer.end("fid_val")
            
            if WANDB:
                self.timer.start("log_val")
                self.log('val')
                self.timer.end("log_val") 

            self.timer.end("epoch")
            
            self.tracker(self.timer.all(), suffix='time', exclude=['train_log'])
            
    def process_batch(self, batch: dict()) -> None:
        batch['mask'] = batch['mask'].to(self.device)
        batch['real_image'] = batch['real_image'].to(self.device)
        
        batch['fake_image'] = self.generator(batch['mask'])
        batch['d_fake'] = self.discriminator(batch['fake_image'])
        batch['d_real'] = self.discriminator(batch['mask'])
        
        self.criterion(batch)
        
    def log(self):
        if not WANDB:
            return
        log = self.tracker.all()
        
        time = {}
        for k, v in log.items():
            if k.endswith('time') and k != 'epoch_time':
                log.popitem((k, v))
                time[k] = v
        plt.pie(time.values(), labels=time.keys())
        plt.axis('equal')
        log["time"] = plt
        
        #TODO images log
        
        wandb.log(log)
