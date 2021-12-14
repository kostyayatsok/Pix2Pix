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
    def __init__(self, root, part, transforms=lambda x : x) -> None:
        super().__init__()
        data_path = os.path.join(*[root, 'facades', part])
        if not os.path.exists(data_path):
            os.system("wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz")
            os.system("tar -qxf facades.tar.gz")
        self.paths = glob.glob(f'{data_path}/*.jpg')
        self.len = len(self.paths)
        self.transforms = transforms
    def __len__(self) -> int:
        return self.len
    def __getitem__(self, index: int) -> dict():
        input = torch.from_numpy(
             np.array(Image.open(self.paths[index])).transpose((2, 0, 1))
        )

        img_size = 256
        
        target = input[:,:,:img_size]
        input = input[:,:,img_size:]

        target = target / 127.5 - 1
        input = input / 127.5 - 1
        
        img = torch.cat((input, target), dim=0)
        img = self.transforms(img)
        input, target = img[:3], img[3:]

        return {
            'mask': input,
            'real_image': target,
            'name': self.paths[index].split('/')[-1][:-4]
        }

def get_facade_dataloaders(
    batch_size, root='.', num_workers=1
):
    transforms = nn.Sequential(
        T.Resize(286),
        T.RandomCrop(256),
        T.RandomHorizontalFlip()
    )
    train_loader = DataLoader(
        FacadeDataset(root, 'train', transforms), batch_size=batch_size,
        shuffle=True, pin_memory=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        FacadeDataset(root, 'val'), batch_size=batch_size,
        shuffle=False, pin_memory=True, num_workers=num_workers
    )
    return train_loader, val_loader