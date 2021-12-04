from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset, random_split
import os
import glob
import torchvision.transforms as T
import torch.nn as nn
class FacadeDataset(Dataset):
    def __init__(self, root, transforms) -> None:
        super().__init__()
        data_path = os.path.join(*[root, 'CMP_facade_DB_base','base'])
        self.masks = glob.glob(f'{data_path}/*.png')
        self.real = glob.glob(f'{data_path}/*.jpg')
        self.len = len(self.masks)
        
        self.transforms = transforms
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        input = read_image(self.masks[index])
        target = read_image(self.real[index])
        
        input_t = self.transforms(input/255.)
        target_t = self.transforms(target/255.)
        return input_t, target_t

def get_dataloaders(
    batch_size, root='.', image_size=1024, val_split=0.1, num_workers=1
):
    transforms = nn.Sequential(
        T.CenterCrop(image_size),
    )
    dataset = FacadeDataset(root, transforms)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, pin_memory=True)#, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, pin_memory=True)#, num_workers=num_workers)
    return train_loader, val_loader