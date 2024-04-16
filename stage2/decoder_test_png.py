import os
import glob

import torch
import pydiffvg
from PIL import Image
from model import Decoder
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


png_path = "./test_png"
png_path_list = glob.glob(os.path.join(png_path, '*.png'))
png_path_list = sorted(png_path_list)
print(len(png_path_list))


decoder = Decoder().to(device).eval()
decoder.load_state_dict(torch.load("../stage1/model/20240329_092223/decoder_epoch_12.pth", map_location=device))

message = 0

dataset = PngDataset(png_path_list,message=message)

count = 0
for idx, data in enumerate(dataset):
    png_tensor, png_message = data

    png_tensor = png_tensor.unsqueeze(0).to(device)

    message = decoder(png_tensor)

    if int(message.argmax(dim=-1)) == png_message:
        count += 1
    else:
        pydiffvg.imwrite(png_tensor.squeeze(0).permute(1, 2, 0).repeat(1, 1, 3).cpu(), f'test_png/{idx}.png',
                         gamma=1.0)

    print("准确率为：",round(count/(idx+1),4))



