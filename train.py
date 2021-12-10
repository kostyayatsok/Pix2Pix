import torch
from model import Generator, Discriminator
from data import get_facade_dataloaders
from loss import Pix2PixLoss
from test import calc_fid
from utils import Timer, MetricTracker 

WANDB = True
if WANDB:
    import wandb
    import matplotlib.pyplot as plt
    wandb.init(project='Pix2Pix')

class Trainer:
    def __init__(
        self, batch_size: int, log_freq: int=5,
        save_freq: int=10, fid_freq: int=100,
    ) -> None:
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.generator = Generator(
            n_blocks=8, in_ch=3, hid_ch=64, out_ch=3).to(self.device)
        print(f"Created generator with "
              f"{sum([p.numel() for p in self.generator.parameters()])} "
              f"parameters")

        self.discriminator = Discriminator(
            n_blocks=3, in_ch=6, hid_ch=64).to(self.device)
        
        print(f"Created discriminator with"
              f" {sum([p.numel() for p in self.discriminator.parameters()])}"
              f" parameters")

        self.g_opt = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        self.train_loader, self.val_loader = get_facade_dataloaders(batch_size)
        
        self.criterion = Pix2PixLoss()
        self.n_epoch = 10000
        self.step = 0
        
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.fid_freq = fid_freq
        
        self.timer = Timer()
        self.tracker = MetricTracker(eternals=['time'])
        
    def __call__(self):
        self.generator.train()
        self.discriminator.train()
        for self.epoch in range(1, self.n_epoch+1):
            self.timer.start('epoch_duration')
            print(
                f"Epoch {self.epoch:04d}/{self.n_epoch:04d}:", flush=True)
            
            self.timer.start("train")
            for i, batch in enumerate(self.train_loader):                
                self.d_opt.zero_grad()
                self.g_opt.zero_grad()
                
                self.process_batch(batch)
                
                if self.step % 50:
                    self.discriminator.requires_grad=False                
                    self.generator.requires_grad=True
                    batch['g_loss'].backward()
                    self.g_opt.step()
                else:
                    self.discriminator.requires_grad=True
                    self.generator.requires_grad=False
                    batch['d_loss'].backward()
                    self.d_opt.step()

                self.tracker(batch, suffix='train')
                
                self.step += 1

                if WANDB and self.step % self.log_freq == 0:
                    self.timer.start("log_train")
                    self.log()
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
                self.timer.start("fid_train")
                self.tracker(
                    {
                        'fid': calc_fid(
                            self.generator, self.val_loader,
                            self.device, './facades/train/', './predictions/'
                        ) 
                    }
                    , suffix='train'
                )
                self.timer.end("fid_train")
                self.tracker({"fid_train": self.timer.get("fid_train")}, suffix='time', count=self.fid_freq)

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
                self.log('val')
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
        
    def log(self, mode='train'):
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
        
        #TODO images log
        
        wandb.log(log)
