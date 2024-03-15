import torch
from PIL import Image
import os
import glob
from model import Encoder, Decoder
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.nn import functional as F
import random

def load_images(directory, size=None, scale=None):
    images = []
    for filename in glob.glob(os.path.join(directory, '*.[pP][nN][gG]'), recursive=True):
        img = Image.open(filename).convert('RGB')
        if size is not None:
            img = img.resize((size, size), Image.ANTIALIAS)
        elif scale is not None:
            img_width, img_height = img.size
            new_width = int(img_width / scale)
            new_height = int(img_height / scale)
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
        images.append(img)
    return images

def load_model(path, device, name):
    if name == 'encoder':
        model = Encoder()
    elif name == 'decoder':
        model = Decoder()
    model_dict = torch.load(path)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()
    return model

def save_images(tensor, directory, filename_prefix="image_", file_format="png"):
    """
    将归一化的 (b, c, h, w) 形状的 PyTorch Tensor 存储为图像文件。

    参数：
    tensor (torch.Tensor) : 归一化后的图像数据，形状为 (b, c, h, w)。
    directory (str)       : 图像存储的目标目录。
    filename_prefix (str) : 图像文件名的前缀，默认为 "image_"。
    file_format (str)     : 图像文件格式，默认为 "png"。

    返回值：
    None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(tensor.shape[0]):
        img = tensor[i].detach().cpu().numpy()
        img *= 255  # 将归一化数据转换回 0-255 范围
        img = img.astype(np.uint8)  # 转换为 uint8 类型
        img = img.transpose(1, 2, 0)  # 将 (c, h, w) 转换为 (h, w, c) 方便 PIL 处理

        pil_image = Image.fromarray(img)
        file_path = os.path.join(directory, f"{filename_prefix}{i}.{file_format}")
        pil_image.save(file_path)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def save_model(encoder, decoder, discri, args, e='end'):
    encoder.eval().cpu()
    encoder_model_filename = "encoder_epoch_" + e + ".pth"
    encoder_model_path = os.path.join(args.save_model_dir, encoder_model_filename)
    torch.save(encoder.state_dict(), encoder_model_path)

    decoder.eval().cpu()
    decoder_model_filename = "decoder_epoch_" + e + ".pth"
    decoder_model_path = os.path.join(args.save_model_dir, decoder_model_filename)
    torch.save(decoder.state_dict(), decoder_model_path)

    discri.eval().cpu()
    discri_model_filename = "discri_epoch_" + e + ".pth"
    discri_model_path = os.path.join(args.save_model_dir, discri_model_filename)
    torch.save(encoder.state_dict(), discri_model_path)


def eval_model(encoder, decoder, args, device):
    if args.eval_data is None:
        return
    encoder.eval()
    decoder.eval()
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # 数值在[0,1]
        # transforms.Lambda(lambda x: x.mul(255))
    ])
    eval_dataset = datasets.ImageFolder(args.eval_data, transform)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
    with torch.no_grad():
        img_total = 0
        img_correct = 0
        for batch_id, (x, message) in enumerate(eval_loader):
            img_total = img_total + len(x)
            x = x.to(device)
            message = message.to(device)
            y = encoder(x, message)
            y = torch.clamp(y, min=0, max=1)
            m = decoder(y)
            probabilities = F.softmax(m, dim=1)
            _, predicted_classes = probabilities.max(dim=1)
            img_correct = img_correct + sum(message == predicted_classes)
        return img_total, img_correct

def perspective_img(imgs, d):
    b, c, h, w = imgs.shape
    tl_x = random.uniform(-d, d)  # Top left corner, top
    tl_y = random.uniform(-d, d)  # Top left corner, left
    bl_x = random.uniform(-d, d)  # Bot left corner, bot
    bl_y = random.uniform(-d, d)  # Bot left corner, left
    tr_x = random.uniform(-d, d)  # Top right corner, top
    tr_y = random.uniform(-d, d)  # Top right corner, right
    br_x = random.uniform(-d, d)  # Bot right corner, bot
    br_y = random.uniform(-d, d)  # Bot right corner, right

    rect = np.array([
        [0, 0],
        [w, 0],
        [w, w],
        [0, w]], dtype="float32")

    dst = np.array([
        [tl_x, tl_y],
        [tr_x + w, tr_y],
        [br_x + w, br_y + w],
        [bl_x, bl_y + w]], dtype="float32")

    out = transforms.functional.perspective(imgs, rect, dst, interpolation=Image.BILINEAR, fill=1)
    return out

def random_blur_kernel(probs, N_blur, sigrange_gauss, sigrange_line, wmin_line):
    N = N_blur
    coords = torch.stack(torch.meshgrid(torch.arange(N_blur), torch.arange(N_blur), indexing='ij'), -1) - (.5 * (N-1))
    coords = coords.float()
    manhat = torch.sum(torch.abs(coords), -1)

    # nothing, default
    vals_nothing = (manhat < .5).float()

    # gauss
    sig_gauss = torch.rand([]) * (sigrange_gauss[1] - sigrange_gauss[0]) + sigrange_gauss[0]
    vals_gauss = torch.exp(-torch.sum(coords**2, -1) / 2.0 / sig_gauss**2)

    # line
    theta = torch.rand([]) * 2.0 * np.pi
    v = torch.tensor([torch.cos(theta), torch.sin(theta)])
    dists = torch.sum(coords * v, -1)

    sig_line = torch.rand([]) * (sigrange_line[1] - sigrange_line[0]) + sigrange_line[0]
    w_line = torch.rand([]) * (.5 * (N-1) + .1 - wmin_line) + wmin_line

    vals_line = torch.exp(-dists**2 / 2.0 / sig_line**2) * (manhat < w_line).float()

    # Select blur type based on probs
    t = torch.rand([])
    if t < probs[0]:
        vals = vals_gauss
    elif t < probs[0] + probs[1]:
        vals = vals_line
    else:
        vals = vals_nothing

    vals = vals / torch.sum(vals)

    return vals

def get_rnd_brightness_torch(rnd_bri, rnd_hue, batch_size):
    # Generate random hue adjustments
    rnd_hue = torch.rand((batch_size, 3, 1, 1), dtype=torch.float32) * (2 * rnd_hue) - rnd_hue
    # Generate random brightness adjustments
    rnd_brightness = torch.rand((batch_size, 1, 1, 1), dtype=torch.float32) * (2 * rnd_bri) - rnd_bri
    # Return the combined adjustments
    return rnd_hue + rnd_brightness
def color_manipulation(encoded_image, args, global_step):
    global_step = torch.tensor(global_step).float()
    ramp_fn = lambda ramp: torch.min(global_step / ramp, torch.tensor(1.0))

    rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri
    rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
    rnd_brightness = get_rnd_brightness_torch(rnd_bri, rnd_hue, args.batch_size)

    contrast_low = 1. - (1. - args.contrast_low) * ramp_fn(args.contrast_ramp)
    contrast_high = 1. + (args.contrast_high - 1.) * ramp_fn(args.contrast_ramp)
    contrast_params = [contrast_low, contrast_high]

    rnd_sat = torch.rand(1) * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat

    contrast_scale = torch.rand(encoded_image.size(0), device=encoded_image.device) * (
                contrast_params[1] - contrast_params[0]) + contrast_params[0]
    contrast_scale = contrast_scale.view(-1, 1, 1, 1)

    encoded_image = encoded_image * contrast_scale
    encoded_image = encoded_image + rnd_brightness
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # 计算亮度图
    lum_weights = torch.tensor([0.3, 0.6, 0.1], device=encoded_image.device).view(1, 3, 1, 1)
    encoded_image_lum = torch.sum(encoded_image * lum_weights, dim=1, keepdim=True)

    # 调整饱和度
    encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    # 如果需要改变图像大小
    # encoded_image = encoded_image.view(-1, 3, 400, 400)

    return encoded_image

def noise(encoded_image, args, global_step):
    global_step = torch.tensor(global_step).float()
    ramp_fn = lambda ramp: torch.min(global_step / ramp, torch.tensor(1.0))

    rnd_noise = torch.rand([]) * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise
    noise = torch.normal(mean=0.0, std=rnd_noise, size=encoded_image.size())
    encoded_image = encoded_image + noise
    encoded_image = torch.clamp(encoded_image, 0, 1)

    return encoded_image

