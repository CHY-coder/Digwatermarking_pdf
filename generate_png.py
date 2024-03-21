from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
import utils
import argparse
import os

parser = argparse.ArgumentParser(description="parser for generate png.")
parser.add_argument("--data", type=str, default="../digwm_data", help="The original img path.")
parser.add_argument("--checkpoint", type=str, default='./model/encoder_epoch_end.pth', help="The checkpoint path.")
parser.add_argument("--output", type=str, default='./generate', help="The output path.")
parser.add_argument("--img_size", type=int, default=64, help="The img size.")

args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  # 数值在[0,1]
])
eval_dataset = datasets.ImageFolder(args.data, transform)
eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=True)

if not os.path.exists(args.output):
    os.makedirs(args.output)
generate_0 = os.path.join(args.output, '0')
generate_1 = os.path.join(args.output, '1')

with torch.no_grad():
    encoder = utils.load_model(args.checkpoint, 'cpu', 'encoder')
    for batch_id, (x, message) in enumerate(eval_loader):
        y = encoder(x, message)
        y = torch.clamp(y, min=0, max=1)
        if message[0] == 0:
            utils.save_images(y, generate_0, batch_id)
        else:
            utils.save_images(y, generate_1, batch_id)

