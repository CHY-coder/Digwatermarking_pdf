import argparse
import os
import sys
import time
import datetime

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx
from torch.nn import functional as F

from stage1 import utils
from model import Encoder, Decoder, Discriminator
from noise import add_noise2
from vgg import Vgg16

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
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
    logger = utils.setup_logger()
    logger.info("Training process started." + ' '.join(sys.argv[1:]))

    device = torch.device(args.device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Grayscale(num_output_channels=1), # 单通道
        transforms.ToTensor(), # 数值在[0,1]
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 建立模型或加载已有的模型继续训练，并设置优化器
    if args.en_checkpoint is not None and os.path.exists(args.en_checkpoint):
        encoder = utils.load_model(args.en_checkpoint, device, "encoder")
    else:
        encoder = Encoder().to(device)
    optimizer_en = Adam(encoder.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)

    if args.disc_checkpoint is not None and os.path.exists(args.disc_checkpoint):
        discri = utils.load_model(args.disc_checkpoint, device, "discriminator")
    else:
        discri = Discriminator().to(device)
    optimizer_discri = Adam(discri.parameters(), args.lr)
    criterion = torch.nn.BCELoss()

    if args.de_checkpoint is not None and os.path.exists(args.de_checkpoint):
        decoder = utils.load_model(args.de_checkpoint, device, "decoder")
    else:
        decoder = Decoder().to(device)
    optimizer_de = Adam(decoder.parameters(), args.lr)
    criterion_de = torch.nn.CrossEntropyLoss()

    # 设置更新次数计数器，add_noise2有用
    global_step = 0

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
                y = encoder(x, message) #编码器编码信息
                y = torch.clamp(y, min=0, max=1) #将数据限制在[0,1]

                vq_l = args.vq_weight * mse_loss(y, x)
                vq_loss = vq_loss + vq_l

                # 用vgg16计算percep_loss
                features_y = vgg(y)
                features_x = vgg(x)
                p_loss = 0
                for ft_y, ft_x in zip(features_y, features_x):
                    gm_y = utils.gram_matrix(ft_y)
                    gm_x = utils.gram_matrix(ft_x)
                    p_loss += mse_loss(gm_y, gm_x)
                percep_l = args.percep_weight * p_loss
                percep_loss = percep_loss + percep_l

                # 判别器
                discri_x = discri(x)
                discri_y = discri(y)
                d_loss = criterion(discri_x, torch.ones_like(discri_x)) + criterion(discri_y,
                                                                                    torch.zeros_like(discri_y))
                discri_l = args.A_weight * d_loss
                discri_loss = discri_loss + discri_l

                # y = add_noise(y)
                y = add_noise2(y, args, global_step)
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
                global_step = global_step + 1

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
    train_arg_parser.add_argument("--batch-size", type=int, default=8,
                                  help="batch size for training, default is 8")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset(glyph image), the path should point to a folder ")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint", type=str, default=None,
                                  help="Load checkpoint to initialize the model.")
    train_arg_parser.add_argument("--image-size", type=int, default=64,
                                  help="size of training images, default is 64 X 64")
    train_arg_parser.add_argument("--device", type=str, required='cpu',
                                  help="set device, e.g cup/cuda/cuda:1")
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
                                  help="learning rate, default is 1e-4")
    train_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    train_arg_parser.add_argument("--en_checkpoint", type=str, default=None,
                                  help="provide encoder checkpoint path to continue training.")
    train_arg_parser.add_argument("--de_checkpoint", type=str, default=None,
                                  help="provide decoder checkpoint path to continue training.")
    train_arg_parser.add_argument("--disc_checkpoint", type=str, default=None,
                                  help="provide discriminator checkpoint path to continue training.")
    train_arg_parser.add_argument("--eval_data", type=str, default=None,
                                  help="Evaluate dataset, default is None.")
    # add_noise2的噪声参数
    train_arg_parser.add_argument('--rnd_bri_ramp', type=int, default=1000,
                                  help="默认在模型参数更新1000次之前，实际噪声扰动系数=（当前更新次数/1000）* 设置扰动系数")
    train_arg_parser.add_argument('--rnd_sat_ramp', type=int, default=1000)
    train_arg_parser.add_argument('--rnd_hue_ramp', type=int, default=1000)
    train_arg_parser.add_argument('--rnd_noise_ramp', type=int, default=1000)
    train_arg_parser.add_argument('--contrast_ramp', type=int, default=1000)
    train_arg_parser.add_argument('--rnd_resize_ramp', type=int, default=1000)
    train_arg_parser.add_argument('--rnd_trans_ramp', type=int, default=1000)
    train_arg_parser.add_argument('--rnd_scal_ramp', type=int, default=1000)
    train_arg_parser.add_argument('--rnd_rot_ramp', type=int, default=1000)
    train_arg_parser.add_argument('--rnd_perspec_ramp', type=int, default=1000)

    train_arg_parser.add_argument('--rnd_bri', type=float, default=.3, help='随机亮度变化的幅度')
    train_arg_parser.add_argument('--rnd_noise', type=float, default=.02, help='随机添加噪声的程度')
    train_arg_parser.add_argument('--rnd_sat', type=float, default=1.0, help='随机饱和度变化的幅度')
    train_arg_parser.add_argument('--rnd_hue', type=float, default=.1, help='随机色调变化的幅度')
    train_arg_parser.add_argument('--contrast_low', type=float, default=.5, help='随机对比度最低值')
    train_arg_parser.add_argument('--contrast_high', type=float, default=1.5, help='随机对比度最高值')
    train_arg_parser.add_argument('--rnd_resize', type=float, default=.1, help='图像大小随机系数')
    train_arg_parser.add_argument('--rnd_trans', type=float, default=0.2, help='随机平移的程度')
    train_arg_parser.add_argument('--rnd_scal', type=float, default=.1, help='随机尺度变化的比例')
    train_arg_parser.add_argument('--rnd_rot', type=float, default=30, help='随机旋转的角度')
    train_arg_parser.add_argument('--rnd_perspec', type=float, default=0.1, help='随机透视变换的程度')

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
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        args.save_model_dir = os.path.join(args.save_model_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        check_paths(args)
        train(args)
    elif args.subcommand == "eval":
        evaluate(args)

if __name__ == "__main__":
    main()
