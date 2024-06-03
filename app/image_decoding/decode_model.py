import torch
import torch.nn as nn


class TrinaryDecoder(nn.Module):
    def __init__(self, binary_decoder):
        super(TrinaryDecoder, self).__init__()
        self.binary_decoder = binary_decoder
        self.new_fc = nn.Linear(2, 3)

    def forward(self, x):
        x = self.binary_decoder(x)
        x = self.new_fc(x)
        return x


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
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = torch.nn.Sequential(ConvLayer(in_channels, out_channels, kernel_size=3, stride=1),
                                         torch.nn.BatchNorm2d(out_channels, affine=True),
                                         torch.nn.ReLU()
                                         )

        self.conv2 = torch.nn.Sequential(ConvLayer(out_channels, out_channels, kernel_size=3, stride=1),
                                         torch.nn.BatchNorm2d(out_channels, affine=True),
                                         )

        self.ca = ChannelAttention(out_channels)

        self.match_dimensions = in_channels != out_channels
        if self.match_dimensions:
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
