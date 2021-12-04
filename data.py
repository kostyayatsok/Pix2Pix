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
    def __init__(self, root, transforms) -> None:
        super().__init__()
        data_path = os.path.join(*[root, 'base'])
        if not os.path.exists(data_path):
            os.system("wget https://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip")
            os.system("unzip -q CMP_facade_DB_base.zip")
        self.masks = glob.glob(f'{data_path}/*.png')
        self.real = glob.glob(f'{data_path}/*.jpg')
        self.len = len(self.masks)
        
        self.transforms = transforms
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        input = torch.from_numpy(
            np.array(Image.open(self.masks[index]))
        ).unsqueeze(0)
        target = T.ToTensor()(Image.open(self.real[index]))
        mask = torch.ones_like(target)
        
        input_t = self.transforms(input)
        target_t = self.transforms(target)-0.5
        mask_t =  self.transforms(target).bool()
        
        input_t = input_t / torch.max(input_t)
        return input_t, target_t, mask_t

def get_dataloaders(
    batch_size, root='.', image_size=1024, val_split=0.05, num_workers=1
):
    transforms = T.Compose([
        T.CenterCrop(image_size),
    ])
    dataset = FacadeDataset(root, transforms)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    print(f"Using {train_size} images for training and {val_size} for validation")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, pin_memory=True)#, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, pin_memory=True)#, num_workers=num_workers)
    return train_loader, val_loader