import os
import glob

import torch
from PIL import Image
from model import Decoder, add_noise
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda")


class PngDataset(Dataset):
    def __init__(self, png_path_list, message):

        self.png_path_list = png_path_list
        self.message = message

        self.transform = transforms.Compose([
                        transforms.Resize([64,64]),
                        transforms.Grayscale(num_output_channels=1),
                        transforms.ToTensor(),
                        ])

    def __len__(self):
        return len(self.svg_path_list)

    def __getitem__(self, idx):

        png_tensor = Image.open(self.png_path_list[idx])
        png_tensor = self.transform(png_tensor)

        png_message = torch.tensor(self.message)

        return png_tensor, png_message


png_path = "../stage1/en_png/image_1"
png_path_list = glob.glob(os.path.join(png_path, '*.png'))
png_path_list = sorted(png_path_list)


decoder = Decoder().to(device).eval()
decoder.load_state_dict(torch.load("model_checkpoints/decoder_epoch_end.pth", map_location=device))

png_folder = os.path.basename(png_path)
if png_folder == "image_0":
    message = 0
elif png_folder == "image_1":
    message = 1
else:
    message = None

dataset = PngDataset(png_path_list,message=message)

count = 0
for idx, data in enumerate(dataset):
    png_tensor, png_message = data

    png_tensor = png_tensor.unsqueeze(0).to(device)
    png_tensor = add_noise(png_tensor)
    message = decoder(png_tensor)
    if int(message.argmax(dim=-1)) == png_message:
        count += 1

    print("准确率为：",round(count/(idx+1),4))

