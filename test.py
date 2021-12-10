import os
from model import Generator
from data import get_facade_dataloaders
import torch
import torchvision.transforms.functional as F
from subprocess import PIPE, run

@torch.no_grad()
def calc_fid(generator, loader, device, in_dir, out_dir='tmp/'):
    os.makedirs(out_dir, exist_ok=True)
    for batch in loader:
        input, name = batch['mask'], batch['name']
        input = input.to(device)
        fake_images = generator(input)
        for image, n in zip(fake_images, name):
            pil_image = F.to_pil_image(image+0.5)
            pil_image.save(f"{out_dir}/{n}.jpg")
    
    result = run(
        f'python3 -m pytorch_fid {out_dir} {in_dir} --device {device}',
        stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True
    )
    fid = float(result.stdout.split()[-1])
    return fid
if __name__ == "__main__":
    out_dir = 'predictions/'
    weights = 'generator.pt'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    generator = Generator(n_blocks=7, in_ch=3, hid_ch=64, out_ch=3).to(device)
    generator.load_state_dict(torch.load(weights))
    _, loader = get_facade_dataloaders(32)
    calc_fid(generator, loader, device, out_dir)
