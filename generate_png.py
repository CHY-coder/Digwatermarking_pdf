import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
from torch.nn import functional as F
import utils

image_size = 64
data = "../digwm_data"
checkpoint = './model/encoder_epoch_end.pth'
generate_0 = './generate/0'
generate_1 = './generate/1'
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  # 数值在[0,1]
    # transforms.Lambda(lambda x: x.mul(255))
])
eval_dataset = datasets.ImageFolder(data, transform)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=True)


with torch.no_grad():
    encoder = utils.load_model(checkpoint, 'cpu', 'encoder')
    for batch_id, (x, message) in enumerate(eval_loader):
        y = encoder(x, message)
        y = torch.clamp(y, min=0, max=1)
        if message[0] == 0:
            utils.save_images(y, generate_0, batch_id)
        else:
            utils.save_images(y, generate_1, batch_id)

