import torch
import numpy as np

SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', default='train',
                    help='train/test mode')
args = parser.parse_args()

from train import Trainer
from test import calc_fid

trainer = Trainer(
    batch_size=1,
    log_freq=200,
    save_freq=20,
    fid_freq=20,
    start_epoch=0 if args.mode == 'train' else 1
)


if args.mode == 'train':
    trainer()
elif args.mode == 'test':
    trainer.generator.eval()
    fid = calc_fid(trainer.generator, trainer.val_loader, trainer.device, './facades/val/', './predictions_val/')
    print(f"FID: {fid}")
else:
    raise "Incorrect mode"