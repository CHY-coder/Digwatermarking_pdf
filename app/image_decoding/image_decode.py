import os
import glob

import torch
from PIL import Image
from image_decoding.decode_model import Decoder, TrinaryDecoder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class PngDataset(Dataset):
    def __init__(self, png_path_list):
        self.png_path_list = png_path_list

        self.transform = transforms.Compose([
            transforms.Resize([64, 64]),
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


def code_pre(decoder_list, dataset):
    current_code = []

    for idx, data in enumerate(dataset):
        png_tensor = data
        png_tensor = png_tensor.unsqueeze(0).to(device)

        message = 0
        for decoder in decoder_list:
            message += decoder(png_tensor)

        result = int(message.argmax(dim=-1))
        current_code.append(result)

    return current_code


device = torch.device("cpu")

path_a = "image_decoding/decoder_weights/20240528_032218/decoder_pro_epoch_79.pth"
path_b = "image_decoding/decoder_weights/20240528_025713/decoder_pro_epoch_79.pth"

decoder = decoder_load(path_a, device)
decoder2 = decoder_load(path_b, device)


def decode_id(png_path):

    png_path_list = sorted_png_paths(png_path)
    dataset = PngDataset(png_path_list)

    predict_code = code_pre([decoder, decoder2], dataset)

    result = ''.join(map(str, predict_code))
    final_result = '\n'.join(result[i:i + 14] for i in range(0, len(result), 14))

    return final_result


