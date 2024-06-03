import argparse
import os
import sys
import time
import datetime

import torch
import torch.onnx
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.nn import functional as F
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from model import Decoder, TrinaryDecoder
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn



def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def save_model(decoder, args, e='end'):

    decoder.eval().cpu()
    decoder_model_filename = "decoder_pro_epoch_" + e + ".pth"
    decoder_model_path = os.path.join(args.save_model_dir, decoder_model_filename)
    torch.save(decoder.state_dict(), decoder_model_path)


def train(args):
    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.ImageFolder("../data/traindata", transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    path_b = "/home/chenzhe/a_py_project/Digwatermarking/test/model/a3/20240527_085007/decoder_pro_epoch_69.pth"

    decoder = Decoder().to(device)
    trinary_decoder = TrinaryDecoder(decoder).to(device)
    trinary_decoder.load_state_dict(torch.load(path_b, map_location=device))

    optimizer_de = torch.optim.Adam(trinary_decoder.parameters(), lr=1e-4)

    img0_total = 2573+858+1701+1144
    img1_total = 2580+858+1699+1144
    img2_total = 2880+858+1701+1143

    criterion_de = torch.nn.CrossEntropyLoss()

    lr_decay_rate = 0.9
    scheduler_de = StepLR(optimizer_de, step_size=1, gamma=lr_decay_rate)

    global_step = 0

    for e in range(100):
        trinary_decoder.train()

        m_loss = 0
        count = 0

        for batch_id, (x, message) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch

            optimizer_de.zero_grad()

            x = torch.clamp(x, min=0, max=1)
            x = x.to(device)
            m = trinary_decoder(x)
            m_l = criterion_de(m, message.to(device))
            m_loss = m_loss + m_l

            m_l.backward()
            optimizer_de.step()
            global_step = global_step + 1

            if (batch_id + 1) % 50 == 0:
                mesg = f"{time.ctime()}\tEpoch {e + 1}:\t[{count}/{len(train_dataset)}]\tmessage loss: {m_loss / (batch_id + 1)}"
                print(mesg)

        scheduler_de.step()

        trinary_decoder.eval()

        with torch.no_grad():
            img_total = 0
            img0_correct = 0
            img1_correct = 0
            img2_correct = 0
            for batch_id, (x, message) in enumerate(train_loader):
                img_total = img_total + len(x)
                x = x.to(device)
                message = message.to(device)
                x = torch.clamp(x, min=0, max=1)

                m = trinary_decoder(x)
                probabilities = F.softmax(m, dim=1)
                _, predicted_classes = probabilities.max(dim=1)

                img0_correct = img0_correct + ((message == 0) & (message == predicted_classes)).float().sum().item()
                img1_correct = img1_correct + ((message == 1) & (message == predicted_classes)).float().sum().item()
                img2_correct = img2_correct + ((message == 2) & (message == predicted_classes)).float().sum().item()

        result = (f"message 0 [{img0_correct}/{img0_total}]\taccuracy: {img0_correct / img0_total}\t"
                  f"message 1 [{img1_correct}/{img1_total}]\taccuracy: {img1_correct / img1_total}\t"
                  f"message 2 [{img2_correct}/{img2_total}]\taccuracy: {img2_correct / img2_total}\t")
        print(result)

        if args.save_model_dir is not None and (e+1)%10==0:
            save_model(trinary_decoder, args, e=str(e))
            trinary_decoder.to(device).train()
            print('save model.')

    save_model(trinary_decoder, args)


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for Digital watermarking")

    main_arg_parser.add_argument("--save-model-dir", type=str, required=True, default="model",
                                  help="path to folder where trained model will be saved.")
    main_arg_parser.add_argument("--device", type=str, default="cuda",
                                  help="set device.")

    args = main_arg_parser.parse_args()

    args.save_model_dir = os.path.join(args.save_model_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    check_paths(args)
    train(args)


if __name__ == "__main__":
    main()

