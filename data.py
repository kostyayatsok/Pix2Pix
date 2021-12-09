from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset, random_split
import os
import glob
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image
import torch
import numpy as np

class FacadeDataset(Dataset):
    def __init__(self, root, part) -> None:
        super().__init__()
        data_path = os.path.join(*[root, 'facades', part])
        if not os.path.exists(data_path):
            os.system("wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz")
            os.system("tar -qxf facades.tar.gz")
        self.paths = glob.glob(f'{data_path}/*.jpg')
        self.len = len(self.paths)
    def __len__(self) -> int:
        return self.len
    def __getitem__(self, index: int) -> dict():
        input = torch.from_numpy(
            np.array(Image.open(self.paths[index]))
        ).transpose(1, 2).transpose(0, 1)
        
        img_size = 256
        target = input[:,:,:img_size] / 255. - 0.5
        input = input[:,:,img_size:] / 255.
#         input_t = self.transforms(input)
#         target_t = self.transforms(target)-0.5
#        input_t = input_t / torch.max(input_t)
        return {
            'mask': input,
            'real_image': target,
            'name': self.paths[index].split('/')[-1][:-4]
        }

def get_facade_dataloaders(
    batch_size, root='.', num_workers=1
):
    train_loader = DataLoader(
        FacadeDataset(root, 'train'), batch_size=batch_size,
        shuffle=True, pin_memory=True#, num_workers=num_workers
    )
    val_loader = DataLoader(
        FacadeDataset(root, 'val'), batch_size=batch_size,
        shuffle=False, pin_memory=True#, num_workers=num_workers
    )
    return train_loader, val_loader