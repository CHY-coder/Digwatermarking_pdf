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
        transforms.ToTensor(),  # 数值在[0,1]
        # transforms.Lambda(lambda x: x.mul(255))
    ])
    eval_dataset = datasets.ImageFolder(args.eval_data, transform)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)
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



