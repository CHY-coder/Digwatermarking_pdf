import os
import glob
import sys

import torch
import pydiffvg
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from model import Decoder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np


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


png_path = "./test_png/1-pt"
png_path_list = glob.glob(os.path.join(png_path, '*.png'))
row = 10
colum = 22

def extract_number(filename):
    number = int(''.join(filter(str.isdigit, filename)))
    return number

png_path_list = sorted(png_path_list, key=extract_number)

print(len(png_path_list))
print(png_path_list)

path_a = "../stage1/model/20240416_162316/decoder_epoch_end.pth"
# path_a = "../stage1/model/20240426_170623/decoder_epoch_end.pth"

decoder = Decoder().to(device).eval()
decoder.load_state_dict(torch.load(path_a , map_location=device))


me = 1
s = '有限公司-有限公司2024年3月合并申报明细'

dataset = PngDataset(png_path_list,message=me)


# count = 0
# for idx, data in enumerate(dataset):
#     png_tensor, png_message = data
#
#     png_tensor = png_tensor.unsqueeze(0).to(device)
#
#     message = decoder(png_tensor)
#
#     if int(message.argmax(dim=-1)) == png_message:
#         count += 1
#     # else:
#     #     pydiffvg.imwrite(png_tensor.squeeze(0).permute(1, 2, 0).repeat(1, 1, 3).cpu(), f'test_png/{idx}.png',
#     #                      gamma=1.0)
#
#     print("准确率为：",round(count/(idx+1),4))

# list_code = []
# list_codes = []
# code = []
p = []
# m = torch.tensor(me)
# m = int(m)
matrix = np.zeros((row, colum))
for idx, data in enumerate(dataset):
    png_tensor, png_message = data

    png_tensor = png_tensor.unsqueeze(0).to(device)

    message = decoder(png_tensor)
    
    i = int(idx / colum)
    j = int(idx % colum)
    matrix[i][j] = int(message.argmax(dim=-1))

    # list_code.append(int(message.argmax(dim=-1)))

    # if (idx+1)%10==0:
    #     list_codes.append(list_code)
    #     code.append(round(sum(list_code)/len(list_code)))
    #     list_code = torch.tensor(list_code)
    #     p.append(sum(list_code==m)/len(list_code))
    #     list_code = []

# print(list_codes)
# print(code)
p = [ sum(matrix[:,i] == int(me)) / row for i in range(colum)]
for c,i in zip(s,p):
    print(str(c)+'('+str(me)+')'+':'+str(i))
