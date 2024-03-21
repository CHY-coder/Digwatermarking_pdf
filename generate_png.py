from torchvision import transforms
import torch.onnx
import utils
import argparse
import os
from torch.utils.data import Dataset
from PIL import Image
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.file_path = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.file_name = [f for f in os.listdir(img_dir) if
                          f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        img_path = self.file_path[idx]
        img_name = self.file_name[idx]
        img = Image.open(img_path)
        img_gray = self.transform(img)
        return img_gray, img_name

parser = argparse.ArgumentParser(description="parser for generate png.")
parser.add_argument("--data", type=str, default="../digwm_data/png64", help="The original img path.")
parser.add_argument("--checkpoint", type=str, default='./model/encoder_epoch_end.pth', help="The checkpoint path.")
parser.add_argument("--output", type=str, default='./generate', help="The output path.")
parser.add_argument("--img_size", type=int, default=64, help="The img size.")

args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  # 数值在[0,1]
])
dataset = CustomImageDataset(args.data, transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

if not os.path.exists(args.output):
    os.makedirs(args.output)
generate_0 = os.path.join(args.output, '0')
generate_1 = os.path.join(args.output, '1')

with torch.no_grad():
    encoder = utils.load_model(args.checkpoint, 'cpu', 'encoder')
    for batch_data, batch_file_names in dataloader:
        for data, file_name in zip(batch_data, batch_file_names):
            data = data.unsqueeze(0)
            y1 = encoder(data, [1])
            y0 = encoder(data, [0])
            y1 = torch.clamp(y1, min=0, max=1)
            y0 = torch.clamp(y0, min=0, max=1)
            utils.save_images(y1, generate_1, file_name)
            utils.save_images(y0, generate_0, file_name)


