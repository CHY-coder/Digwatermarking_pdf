import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.res1 = ResidualBlock(32)
        self.conv2 = ConvLayer(33, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X, M):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.res1(y)

        # concatenate M(0/1)
        B, C, H, W = y.size()
        message_channel = torch.zeros((B, 1, H, W))
        for i, m in enumerate(M):
            message_channel[i, :, :, :] = m
        y = torch.cat((y, message_channel), dim=1)

        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y

class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.conv1 = ConvLayer(3, 32, kernel_size=3, stride=2)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 32, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv4 = ConvLayer(64, 64, kernel_size=3, stride=1)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv5 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in5 = torch.nn.InstanceNorm2d(128, affine=True)
        self.conv6 = ConvLayer(128, 128, kernel_size=3, stride=1)
        self.in6 = torch.nn.InstanceNorm2d(128, affine=True)
        
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(128 * 64 * 64, 512)  # 假设输入图像大小为 64 * 64
        self.fc2 = torch.nn.Linear(512, 2)


        self.relu = torch.nn.ReLU()
    
    def forward(self, y):
        out = self.relu(self.in1(self.conv1(y)))
        out = self.relu(self.in2(self.conv2(out)))
        out = self.relu(self.in3(self.conv3(out)))
        out = self.relu(self.in4(self.conv4(out)))
        out = self.relu(self.in5(self.conv5(out)))
        out = self.relu(self.in6(self.conv6(out)))
        out = self.flatten(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)

        return out



class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

class Discriminator(torch.nn.Module):
    def __init__(self, H, W):
        super(Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            # input layer (3, H, W)
            torch.nn.Linear(3*H*W, 1024),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            # hidden layer
            torch.nn.Linear(1024, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(0.3),
            # output layer
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.model(x)
        return output


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
        transforms.Resize((h * generate_random_number(), w * generate_random_number())),  # Resize
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Rotate, translate, scale
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Perspective transformation
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Blur
        GaussianNoise(mean=0., std=0.1),  # Gaussian noise
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color manipulation
    ])

    # Apply the transformations
    batch_transformed = torch.stack([transforms_list(img) for img in batch])
    return batch_transformed