import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from model import Encoder, Decoder, Discriminator
from vgg import Vgg16


def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    if args.cuda:
        device = torch.device("cuda")
    # elif args.mps:
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    encoder = Encoder().to(device)
    optimizer_en = Adam(encoder.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)

    discri = Discriminator(args.image_size, args.image_size).to(device)
    optimizer_discri = Adam(discri.parameters(), args.lr)
    criterion = torch.nn.BCELoss()


    decoder = Decoder().to(device)
    optimizer_de = Adam(decoder.parameters(), args.lr)

    for e in range(args.epochs):
        
        encoder.train()
        discri.train()
        decoder.train()

        agg_content_loss = 0.
        count = 0
        for batch_id, (x, message) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch

            optimizer_en.zero_grad()
            optimizer_discri.zero_grad()
            optimizer_de.zero_grad()

            x = x.to(device)
            y = encoder(x, message)
            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            vq_loss = args.vq_weight * mse_loss(y, x)

            features_y = vgg(y)
            features_x = vgg(x)
            percep_loss = 0
            for ft_y, ft_x in zip(features_y, features_x):
                gm_y = utils.gram_matrix(ft_y)
                gm_x = utils.gram_matrix(ft_x)
                percep_loss += mse_loss(gm_y, gm_x)
            percep_loss = args.percep_weight * percep_loss

            discri_x = discri(x)
            discri_y = discri(y)
            discri_loss = criterion(discri_x, torch.ones_like(discri_x)) + criterion(discri_y, torch.zeros_like(discri_y))
            discri_loss = args.A_weight * discri_loss


            total_loss.backward()
            optimizer_en.step()

            agg_content_loss += content_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  (agg_content_loss) / (batch_id + 1)
                )
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                encoder.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(encoder.state_dict(), ckpt_model_path)
                encoder.to(device).train()

    # save model
    encoder.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(encoder.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for Digital watermarking")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=2,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=2,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset(glyph image), the path should point to a folder ")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--vq_weight", type=float, default=5,
                                  help="weight for vq_loss, default is 5")
    train_arg_parser.add_argument("--percep_weight", type=float, default=0.01,
                                  help="weight for percep_loss, default is 0.01")
    train_arg_parser.add_argument("--A_weight", type=float, default=1,
                                  help="weight for A_loss, default is 1")
    train_arg_parser.add_argument("--m_weight", type=float, default=1,
                                  help="weight for m_loss, default is 1")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation arguments")
    eval_arg_parser.add_argument("--image", type=str, required=True,
                                 help="path to glyph image you want to add digital watermark")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output glyph image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for adding a digital watermark to glyph image. "
                                      "If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    eval_arg_parser.add_argument("--cuda", type=int, default=False,
                                 help="set it to 1 for running on cuda, 0 for CPU")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")
    # eval_arg_parser.add_argument('--mps', action='store_true', default=False, help='enable macOS GPU training')

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    # if not args.mps and torch.backends.mps.is_available():
    #     print("WARNING: mps is available, run with --mps to enable macOS GPU")

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    # else:
    #     stylize(args)


if __name__ == "__main__":
    main()
