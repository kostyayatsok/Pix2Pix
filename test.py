import os
from model import Generator
from data import get_facade_dataloaders
import torch
import torchvision.transforms.functional as F
out_dir = 'predictions/'
weights = 'generator.pt'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

os.makedirs(out_dir, exist_ok=True)

generator = Generator(n_blocks=7, in_ch=3, hid_ch=64, out_ch=3).to(device)
generator.load_state_dict(torch.load(weights))
train_loader, _ = get_facade_dataloaders(32)
with torch.no_grad():
    for batch in train_loader:
        input, name = batch['input'], batch['name']
        input = input.to(device)
        output = generator(input)
        for image, n in zip(output, name):
            pil_image = F.to_pil_image(image+0.5)
            pil_image.save(f"{out_dir}/{n}.jpg")