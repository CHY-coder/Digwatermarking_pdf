import argparse
import os
import sys
import time
import re
import logging
import datetime

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
import matplotlib.pyplot as plt
from torch.nn import functional as F


import utils
from model import Encoder, Decoder, Discriminator, add_noise
from vgg import Vgg16

# 设置日志格式
def setup_logger(prefix='model_training', log_dir='logs', console_level=logging.ERROR):
    """
    初始化并配置日志器，返回一个已经配置好的logger实例。

    参数:
        prefix (str): 日志文件名的前缀。
        log_dir (str): 存放日志文件的目录，默认为'logs'。
        console_level (logging.LEVEL): 控制台日志级别，默认只显示错误及以上级别的信息。

    返回:
        logger: 配置好的logging.Logger实例。
    """

    def create_log_filename():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f'{prefix}_{timestamp}.log'

    # 创建日志目录（如果不存在）
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建自定义日志文件名
    log_file_name = create_log_filename()
    log_path = os.path.join(log_dir, log_file_name)

    # 创建一个logger
    logger = logging.getLogger(prefix)
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于写入自定义日志文件
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # 创建一个formatter，用于设置日志格式
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)

    # 添加FileHandler到logger
    logger.addHandler(file_handler)

    # 创建一个StreamHandler，用于将错误级别及以上的日志输出到控制台
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_level)
    stream_handler.setFormatter(formatter)

    # 添加StreamHandler到logger
    logger.addHandler(stream_handler)

    return logger
def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        # if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
        #     os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)
def evaluate(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.img_scale != 1:
        ori_image = utils.load_images(args.img_dir, scale=args.img_scale)
    else:
        ori_image = utils.load_images(args.img_dir, size=args.img_size)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    img_tensors = [transform(img) for img in ori_image]
    img_tensors_batch = torch.stack(img_tensors, dim=0)

    with torch.no_grad():
        encoder = utils.load_model(args.encoder, device, 'encoder')
        decoder = utils.load_model(args.decoder, device, 'decoder')
        output = encoder(img_tensors_batch, args.message)
        output = torch.clamp(output, min=0, max=1)
        utils.save_images(output, args.output_image)
        m = decoder(output)
        probabilities = F.softmax(m, dim=1)
        _, predicted_classes = probabilities.max(dim=1)
        print(predicted_classes)

def train(args):
    logger = setup_logger()
    logger.info("Training process started.")

    if args.cuda:
        device = torch.device("cuda")
    # elif args.mps:
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Grayscale(num_output_channels=1), # 单通道
        transforms.ToTensor(), # 数值在[0,1]
        # transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    encoder = Encoder().to(device)
    optimizer_en = Adam(encoder.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)

    discri = Discriminator().to(device)
    optimizer_discri = Adam(discri.parameters(), args.lr)
    criterion = torch.nn.BCELoss()


    decoder = Decoder().to(device)
    optimizer_de = Adam(decoder.parameters(), args.lr)
    criterion_de = torch.nn.CrossEntropyLoss()

    try:
        for e in range(args.epochs):

            encoder.train()
            discri.train()
            decoder.train()

            vq_loss = 0
            percep_loss = 0
            discri_loss = 0
            m_loss = 0
            loss = 0
            count = 0
            for batch_id, (x, message) in enumerate(train_loader):
                n_batch = len(x)
                count += n_batch

                optimizer_en.zero_grad()
                optimizer_discri.zero_grad()
                optimizer_de.zero_grad()

                x = x.to(device)
                y = encoder(x, message)
                y = torch.clamp(y, min=0, max=1)
                # y = utils.normalize_batch(y)
                # x = utils.normalize_batch(x)

                vq_l = args.vq_weight * mse_loss(y, x)
                vq_loss = vq_loss + vq_l

                features_y = vgg(y)
                features_x = vgg(x)
                p_loss = 0
                for ft_y, ft_x in zip(features_y, features_x):
                    gm_y = utils.gram_matrix(ft_y)
                    gm_x = utils.gram_matrix(ft_x)
                    p_loss += mse_loss(gm_y, gm_x)
                percep_l = args.percep_weight * p_loss
                percep_loss = percep_loss + percep_l

                discri_x = discri(x)
                discri_y = discri(y)
                d_loss = criterion(discri_x, torch.ones_like(discri_x)) + criterion(discri_y,
                                                                                    torch.zeros_like(discri_y))
                discri_l = args.A_weight * d_loss
                discri_loss = discri_loss + discri_l

                y = add_noise(y)
                y = torch.clamp(y, min=0, max=1)
                m = decoder(y)
                m_l = args.m_weight * criterion_de(m, message.to(device))
                m_loss = m_loss + m_l

                l = vq_l + percep_l + discri_l + m_l
                loss = loss + l
                l.backward()

                optimizer_en.step()
                optimizer_discri.step()
                optimizer_de.step()

                if (batch_id + 1) % args.log_interval == 0:
                    mesg = "{}\tEpoch {}:\t[{}/{}]\ttotal loss: {:.6f}\tvisual quality: {:.6f}\t" \
                           "perceptual: {:.6f}\tdiscriminator: {:.6f}\tmessage: {:.6f}".format(
                        time.ctime(), e + 1, count, len(train_dataset), loss / (batch_id + 1), vq_loss / (batch_id + 1),
                                      percep_loss / (batch_id + 1), discri_loss / (batch_id + 1),
                                      m_loss / (batch_id + 1)
                    )
                    logger.info(mesg)
                    print(mesg)

            img_total, img1_correct, img0_correct = utils.eval_model(encoder, decoder, args, device)
            result = "{}\tEpoch {}:\tmessage 1 [{}/{}]\taccuracy: {:.6f}\tmessage 0 [{}/{}]\taccuracy: {:.6f}\t".format(
                time.ctime(), e + 1, img1_correct, img_total, img1_correct / img_total, img0_correct, img_total, img0_correct / img_total
            )
            logger.info(result)
            print(result)

            if args.save_model_dir is not None:
                utils.save_model(encoder, decoder, discri, args, str(e))
                encoder.to(device).train()
                decoder.to(device).train()
                discri.to(device).train()
                logger.info('save model.')
                print('save model.')

    except:
        logger.exception("An error occurred:", exc_info=True)
    finally:
        logger.info("Training process completed (or possibly interrupted).")
        # save model
        utils.save_model(encoder, decoder, discri, args)
        logger.info('save model at ' + args.save_model_dir)



def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for Digital watermarking")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=10,
                                  help="number of training epochs, default is 10")
    train_arg_parser.add_argument("--batch-size", type=int, default=16,
                                  help="batch size for training, default is 16")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset(glyph image), the path should point to a folder ")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint", type=str, default=None,
                                  help="Load checkpoint to initialize the model.")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--vq_weight", type=float, default=5,
                                  help="weight for vq_loss, default is 5")
    train_arg_parser.add_argument("--percep_weight", type=float, default=1,
                                  help="weight for percep_loss, default is 1")
    train_arg_parser.add_argument("--A_weight", type=float, default=1,
                                  help="weight for A_loss, default is 1")
    train_arg_parser.add_argument("--m_weight", type=float, default=1,
                                  help="weight for m_loss, default is 1")
    train_arg_parser.add_argument("--lr", type=float, default=1e-4,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=1000,
                                  help="number of batches after which a checkpoint of the trained model will be created")
    train_arg_parser.add_argument("--eval_data", type=str, default=None,
                                  help="Evaluate dataset, default is None.")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation arguments")
    eval_arg_parser.add_argument("--img_dir", type=str, required=True,
                                 help="path to glyph image you want to add digital watermark")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output glyph image")
    eval_arg_parser.add_argument("--encoder", type=str, required=True,
                                 help="saved encoder to be used for adding a digital watermark to glyph image. ")
    eval_arg_parser.add_argument("--decoder", type=str, required=True,
                                 help="saved decoder to be used for getting 0/1 message from glyph image. ")
    eval_arg_parser.add_argument("--cuda", type=int, default=False,
                                 help="set it to 1 for running on cuda, 0 for CPU")
    eval_arg_parser.add_argument("--img_size", type=int, default=64,
                                 help="set image size, default is 64")
    eval_arg_parser.add_argument("--img_scale", type=int, default=1,
                                 help="set image scale, default is 1")
    eval_arg_parser.add_argument("--message", type=int, nargs='+',
                                 help="set image message.")


    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    elif args.subcommand == "eval":
        evaluate(args)

if __name__ == "__main__":
    main()
