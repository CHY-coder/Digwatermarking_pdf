import glob
from torchvision import transforms
import torch.onnx
from stage1 import utils
import argparse
import os
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):

        self.index_list = [13985, 25861, 10068, 10396]
        self.transform = transform
        self.file_path = sorted(glob.glob(os.path.join(img_dir, '*.png')))

        self.file_path_test = []
        for i in self.index_list:
            self.file_path_test.append(self.file_path[i])
        self.file_path_test = sorted(self.file_path_test)

        self.file_name = [os.path.basename(file_path) for file_path in self.file_path_test]

    def __len__(self):
        return len(self.file_path_test)

    def __getitem__(self, idx):

        img_path = self.file_path_test[idx]
        img_name = self.file_name[idx]
        img = Image.open(img_path)
        img_gray = self.transform(img)
        return img_gray, img_name


parser = argparse.ArgumentParser(description="parser for generate png.")
parser.add_argument("--data", type=str, default="../../data/ori_png64/0", help="The original img path.")
parser.add_argument("--checkpoint", type=str, default='./model/20240329_015045/encoder_epoch_5.pth', help="The checkpoint path.")
parser.add_argument("--output", type=str, default='../../data/en_png_3.29_20', help="The output path.")
parser.add_argument("--img_size", type=int, default=64, help="The img size.")

args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])
dataset = CustomImageDataset(args.data, transform)

if not os.path.exists(args.output):
    os.makedirs(args.output)
generate_0 = os.path.join(args.output, '0')
generate_1 = os.path.join(args.output, '1')


with torch.no_grad():
    encoder = utils.load_model(args.checkpoint, 'cpu', 'encoder')
    for data, file_name in dataset:
        data = data.unsqueeze(0)
        y1 = encoder(data, torch.tensor([1]))
        y0 = encoder(data, torch.tensor([0]))
        y1 = torch.clamp(y1, min=0, max=1)
        y0 = torch.clamp(y0, min=0, max=1)
        utils.save_images(y1, generate_1, file_name)
        utils.save_images(y0, generate_0, file_name)


