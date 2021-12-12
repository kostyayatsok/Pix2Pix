import torch
import numpy as np

SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

from train import Trainer

trainer = Trainer(1, start_epoch=61, fid_freq=10)
trainer()
