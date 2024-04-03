import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import sys
sys.path.append(r"/app/deepvecfont")
from models.imgsr.modules import TrainOptions, create_model


def create_sr_model():
    imgsr_opt = TrainOptions().parse()
    imgsr_opt.isTrain = False
    imgsr_opt.batch_size = 1
    imgsr_opt.phase = 'test'
    imgsr_model = create_model(imgsr_opt)
    imgsr_model.setup(imgsr_opt)

    return imgsr_model


class Encoder(torch.nn.Module):
    def __init__(self, num_residual_blocks=4):
        super(Encoder, self).__init__()

        # Initial convolution layers
        self.conv1 = ResidualBlock(1,16)
        self.conv2 = ResidualBlock(17,32)

        # Message embedding layer
        self.message_embedding = MessageEmbeddingLayer(input_dim=1, output_channels=1, H=64, W=64)

        # Residual layers
        self.residual_blocks = torch.nn.ModuleList([ResidualBlock(32,32) for _ in range(num_residual_blocks)])

        # Upsampling Layers
        self.conv3 = torch.nn.Sequential(ConvLayer(32, 16, kernel_size=3, stride=1),
                                         torch.nn.BatchNorm2d(16, affine=True),
                                         torch.nn.ReLU(),
                                         ConvLayer(16, 8, kernel_size=3, stride=1),
                                         torch.nn.BatchNorm2d(8, affine=True),
                                         torch.nn.ReLU(),
                                         ConvLayer(8, 1, kernel_size=9, stride=1),
                                         )

    def forward(self, X, M):

        X = X - 0.5
        y = self.conv1(X)

        # concatenate M(0/1)
        M = M.unsqueeze(-1).to(y.device)
        M_embedded = self.message_embedding(M)
        y = torch.cat((y, M_embedded), dim=1)

        y = self.conv2(y)

        for block in self.residual_blocks:
            y = block(y)

        y = self.conv3(y)

        return y


class MessageEmbeddingLayer(nn.Module):
    def __init__(self, input_dim, output_channels, H, W):
        super(MessageEmbeddingLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_channels * H * W)
        self.bn = nn.BatchNorm1d(output_channels * H * W)
        self.output_channels = output_channels
        self.H = H
        self.W = W

        self.relu = torch.nn.LeakyReLU(0.2)

    def forward(self, message):
        message = message.float()
        embedded_message = self.fc(message)
        embedded_message = self.bn(embedded_message)
        embedded_message = self.relu(embedded_message)
        embedded_message = embedded_message.view(-1, self.output_channels, self.H,
                                                 self.W)
        return embedded_message



# class Decoder(torch.nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#
#         self.conv1 = ConvLayer(1, 32, kernel_size=3, stride=2)
#         self.bn1 = torch.nn.BatchNorm2d(32, affine=True)
#         self.conv2 = ConvLayer(32, 32, kernel_size=3, stride=1)
#         self.bn2 = torch.nn.BatchNorm2d(32, affine=True)
#         self.conv3 = ConvLayer(32, 64, kernel_size=3, stride=2)
#         self.bn3 = torch.nn.BatchNorm2d(64, affine=True)
#         self.conv4 = ConvLayer(64, 64, kernel_size=3, stride=1)
#         self.bn4 = torch.nn.BatchNorm2d(64, affine=True)
#         self.conv5 = ConvLayer(64, 128, kernel_size=3, stride=2)
#         self.bn5 = torch.nn.BatchNorm2d(128, affine=True)
#         self.conv6 = ConvLayer(128, 128, kernel_size=3, stride=1)
#         self.bn6 = torch.nn.BatchNorm2d(128, affine=True)
#
#         self.flatten = torch.nn.Flatten()
#         self.fc1 = torch.nn.Linear(128 * 8 * 8, 512)  # 假设输入图像大小为 64 * 64
#         self.fc2 = torch.nn.Linear(512, 2)
#
#         self.relu = torch.nn.ReLU()
#
#     def forward(self, y):
#         y = y - 0.5
#         out = self.relu(self.bn1(self.conv1(y)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.relu(self.bn3(self.conv3(out)))
#         out = self.relu(self.bn4(self.conv4(out)))
#         out = self.relu(self.bn5(self.conv5(out)))
#         out = self.relu(self.bn6(self.conv6(out)))
#         out = self.flatten(out)
#         out = self.relu(self.fc1(out))
#         out = self.fc2(out)
#
#         return out


class Decoder(torch.nn.Module):
    def __init__(self, num_residual_blocks=4):
        super(Decoder, self).__init__()

        # 初始卷积层
        self.conv1 = ResidualBlock(1,16)
        self.conv2 = ResidualBlock(16,32)

        # 残差块
        self.residual_blocks = torch.nn.ModuleList([ResidualBlock(32,32) for _ in range(num_residual_blocks)])

        # 信息提取层
        self.info_extract = torch.nn.Sequential(torch.nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
                                                torch.nn.BatchNorm2d(16, affine=True),
                                                torch.nn.ReLU(),
                                                torch.nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),
                                                torch.nn.BatchNorm2d(8, affine=True),
                                                torch.nn.ReLU(),
                                                torch.nn.Flatten(),
                                                torch.nn.Linear(8 * 16 * 16, 2),
                                                )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.info_extract(x)
        return x


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding_mode='reflect'):
        super(ConvLayer, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                      padding=kernel_size // 2, padding_mode=padding_mode)

    def forward(self, x):
        return self.conv2d(x)


class ChannelAttention(torch.nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(channels // reduction, channels, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(torch.nn.Module):
    """A residual block."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = torch.nn.Sequential(ConvLayer(in_channels, out_channels, kernel_size=3, stride=1),
                                         torch.nn.BatchNorm2d(out_channels, affine=True),
                                         torch.nn.ReLU()
                                         )

        self.conv2 = torch.nn.Sequential(ConvLayer(out_channels, out_channels, kernel_size=3, stride=1),
                                         torch.nn.BatchNorm2d(out_channels, affine=True),
                                         # 这里移除了ReLU，以便在累加残差后应用
                                         )

        self.ca = ChannelAttention(out_channels)

        self.match_dimensions = in_channels != out_channels
        if self.match_dimensions:
            # 调整残差路径上的维度以匹配主路径的输出
            self.residual_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            self.residual_bn = torch.nn.BatchNorm2d(out_channels)

        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        if self.match_dimensions:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.ca(out)

        out += residual
        out = self.relu(out)

        return out


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = ConvLayer(1, 8, kernel_size=3, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(8, affine=True)
        self.conv2 = ConvLayer(8, 16, kernel_size=3, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(16, affine=True)
        self.conv3 = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.bn3 = torch.nn.BatchNorm2d(32, affine=True)
        self.conv4 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.bn4 = torch.nn.BatchNorm2d(64, affine=True)
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(64, 1)
        self.relu = torch.nn.LeakyReLU(0.2)

    def forward(self, x):
        x = x - 0.5
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.relu(self.bn3(self.conv3(output)))
        output = self.relu(self.bn4(self.conv4(output)))
        output = self.global_avg_pool(output)
        output = output.view(output.size(0), -1)
        output = torch.sigmoid(self.fc(output))
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
