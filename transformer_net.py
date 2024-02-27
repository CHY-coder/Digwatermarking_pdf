import torch


class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.res1 = ResidualBlock(32)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
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
        massage = torch.full((B, 1, H, W), M)
        y = torch.cat((y, massage), dim=1)

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
            # input layer (1, H, W)
            torch.nn.Linear(H*W, 1024),
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


def transform_net(encoded_image, args, global_step):
    ramp_fn = lambda ramp: min(float(global_step) / ramp, 1.0)

    rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri
    rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
    # 假设utils.get_rnd_brightness_tf()返回一个形状为[batch_size, 1, 1, 1]的张量，包含随机亮度值
    # 我们将在这里模拟一个简化版本
    rnd_brightness = torch.randn((args.batch_size, 1, 1, 1)) * rnd_bri

    rnd_noise = torch.rand([]) * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise

    contrast_low = 1.0 - (1.0 - args.contrast_low) * ramp_fn(args.contrast_ramp)
    contrast_high = 1.0 + (args.contrast_high - 1.0) * ramp_fn(args.contrast_ramp)

    rnd_sat = torch.rand([]) * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat

    # 模糊模拟省略
    
    noise = torch.randn_like(encoded_image) * rnd_noise
    encoded_image = encoded_image + noise
    encoded_image = torch.clamp(encoded_image, 0, 1)

    contrast_scale = torch.rand((encoded_image.size(0), 1, 1, 1)) * (contrast_high - contrast_low) + contrast_low
    encoded_image = encoded_image * contrast_scale
    encoded_image = encoded_image + rnd_brightness
    encoded_image = torch.clamp(encoded_image, 0, 1)

    encoded_image_lum = torch.sum(encoded_image * torch.tensor([0.3, 0.6, 0.1]).view(1, 1, 1, 3), dim=3, keepdim=True)
    encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    encoded_image = encoded_image.reshape(-1, 400, 400, 3)

    return encoded_image
