import os
import glob

import numpy as np
import torch
from PIL import Image
from model import Decoder, TrinaryDecoder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class PngDataset(Dataset):
    def __init__(self, png_path_list):

        self.png_path_list = png_path_list

        self.transform = transforms.Compose([
                        transforms.Resize([64,64]),
                        transforms.Grayscale(num_output_channels=1),
                        transforms.ToTensor(),
                        ])

    def __len__(self):
        return len(self.png_path_list)

    def __getitem__(self, idx):
        png_tensor = Image.open(self.png_path_list[idx])
        return self.transform(png_tensor)


def sorted_png_paths(png_path):
    png_files = glob.glob(os.path.join(png_path, '*.png'))
    extract_number = lambda filename: int(''.join(filter(str.isdigit, filename)))
    return sorted(png_files, key=extract_number)


def decoder_load(path, device):
    decoder = Decoder().to(device).eval()
    tri_decoder = TrinaryDecoder(decoder).to(device).eval()
    tri_decoder.load_state_dict(torch.load(path, map_location=device))
    return tri_decoder


def code_pre(decoder_list, dataset, num_code):

    current_code = []
    predicted_codes = []

    for idx, data in enumerate(dataset):

        png_tensor = data
        png_tensor = png_tensor.unsqueeze(0).to(device)

        message = 0
        for decoder in decoder_list:
            message += decoder(png_tensor)

        result = int(message.argmax(dim=-1))

        current_code.append(result)

        if (idx+1)%42==0:
            predicted_codes.append(current_code)
            current_code=[]

    return predicted_codes


def decode_acc_print(predicted_codes, true_codes):
    count = 0
    for idx, (code1, code2) in enumerate(zip(predicted_codes, true_codes)):
        matches = 0
        total = len(code1)
        error_positions = []

        for pos, (char1, char2) in enumerate(zip(code1, code2)):
            if char1 == char2:
                matches += 1
            else:
                error_positions.append(pos + 1)

        code1_str = ''.join(str(char) for char in code1)
        code2_str = ''.join(str(char) for char in code2)

        accuracy = (matches / total) * 100

        c = code1_str.count('1')

        print("预测编码：", code1_str)
        print("真实编码：", code2_str)
        print(f"编码 {idx + 1} 的准确率: {accuracy:.2f}%")
        print("校验结果：", c % 2 == 0)

        if error_positions:
            print(f"错误位：{', '.join(map(str, error_positions))}")
        else:
            count += 1
            print("无错误位")

        print("*" * 100)

    print("总体解码准确率:", count / len(true_codes))


if __name__ == "__main__":

    code_test1 = """111111222000
111111222000
000000000111
000000000111
000222222000
000222222000
111222111111
111222111111
000000222000
000000222000
111000000111
111000000111
222111222222
222111222222
111222111222
111222111222
000000111222
000000111222
222111000222
222111000222
222222222222
222222222222"""

    codes_test1 = code_test1.split("\n")
    true_codes_test1 = [[int(char) for char in row] for row in codes_test1]

    code_test2 = """111111222000111111111122222222221111111111
111111222000111111111122222222221111111111
000000000111111111111122222000001111100000
000000000111111111111122222000001111100000
000222222000111110000000000111112222211111
000222222000111110000000000111112222211111
111222111111000000000011111000000000000000
111222111111000000000011111000000000000000
000000222000111111111122222222220000000000
000000222000111111111122222222220000000000
111000000111222222222222222222222222222222
111000000111222222222222222222222222222222
222111222222000001111100000111111111100000
222111222222000001111100000111111111100000
111222111222000002222200000000000000000000
111222111222000002222200000000000000000000
000000111222111112222211111222221111100000
000000111222111112222211111222221111100000
222111000222111111111111111222222222200000
222111000222111111111111111222222222200000
222222222222111112222200000000002222200000
222222222222111112222200000000002222200000"""

    codes_test2 = code_test2.split("\n")
    true_codes_test2 = [[int(char) for char in row.strip()] for row in codes_test2]

    device = torch.device("cuda")

    png_path = "/home/chenzhe/a_py_project/Digwatermarking/stage3/CRAFT-pytorch-master/gen/te1/3"
    path_a = "/home/chenzhe/a_py_project/Digwatermarking/test/model/a3/20240528_032218/decoder_pro_epoch_79.pth"
    path_b = "/home/chenzhe/a_py_project/Digwatermarking/test/model/a3/20240528_025713/decoder_pro_epoch_79.pth"
    num_code = 42

    png_path_list = sorted_png_paths(png_path)
    dataset = PngDataset(png_path_list)
    decoder = decoder_load(path_a, device)
    decoder2 = decoder_load(path_b,device)
    predicted_codes = code_pre([decoder, decoder2], dataset, num_code=num_code)

    decode_acc_print(predicted_codes, true_codes_test2)
