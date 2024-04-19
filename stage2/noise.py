import torch
import utils
from torch.nn import functional as F
from torchvision import transforms
import numpy as np


class GaussianNoise(torch.nn.Module):
    """Adds Gaussian noise to an image."""

    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, tensor):
        return tensor + torch.randn(tensor.size()).to(tensor.device) * self.std + self.mean


class ColorJitter(torch.nn.Module):
    """Randomly changes the brightness, contrast, saturation, and hue of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.transform = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation,
                                                hue=hue)

    def forward(self, img):
        return self.transform(img)


def generate_random_number():
    mean = 1
    std_dev = 0.2  # 较小的标准差意味着大部分数值将更加集中在均值附近
    number = np.random.normal(mean, std_dev)

    # 确保生成的数在0到2之间
    number = max(min(number, 2), 0)
    return number


def add_noise(batch):
    b, c, h, w = batch.shape

    # Define the sequence of transformations
    transforms_list = transforms.Compose([
        transforms.Resize((round(h * generate_random_number()), round(w * generate_random_number()))),  # Resize
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Rotate, translate, scale
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Perspective transformation
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Blur
        GaussianNoise(mean=0., std=0.1),  # Gaussian noise
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color manipulation
        transforms.Resize((64,64))
    ])

    # Apply the transformations
    batch_transformed = torch.stack([transforms_list(img) for img in batch])
    return batch_transformed
def add_noise2(batch, args, global_step):
    N, C, H, W = batch.shape

    global_step = torch.tensor(global_step).float()
    ramp_fn = lambda ramp: torch.min(global_step / ramp, torch.tensor(1.0))

    # resizing, translation, scaling, rotation
    img = utils.apply_transformations(batch, args, ramp_fn, H, W)

    # perspective
    pers_rate = torch.rand([]) * ramp_fn(args.rnd_perspec_ramp) * args.rnd_perspec
    img = utils.perspective_img(img, W, pers_rate)

    # blur
    probs = [0.25, 0.25]  # Probability for gauss and line blur types
    N_blur = 7
    sigrange_gauss = [1.0, 3.0]
    sigrange_line = [0.25, 1.0]
    wmin_line = 3
    kernel = utils.random_blur_kernel(probs, N_blur, sigrange_gauss, sigrange_line, wmin_line)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # 增加一个维度，用于通道数，现在形状是 [1, N_blur, N_blur]
    img = F.conv2d(img, kernel.to(img.device), padding=N_blur // 2)
    img = torch.clamp(img, 0, 1)

    # noise
    rnd_noise = torch.rand([]) * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise
    img = utils.noise(img, rnd_noise)

    # color manipulation
    img = utils.color_manipulation(img, args, ramp_fn)

    return img