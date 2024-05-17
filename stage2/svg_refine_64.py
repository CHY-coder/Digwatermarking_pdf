import os
import glob
import sys
import time

import pydiffvg
import torch
import argparse
from torchvision import transforms
from save_svg import save_svg_paths_only
from torch.utils.data import Dataset
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from model import Decoder
from noise import add_noise, add_noise2
from utils import setup_logger

gamma = 1.0

device = torch.device("cuda:0")
torch.cuda.set_device(device)


class SvgAndPngDataset(Dataset):
    def __init__(self, svg_path_list, png_path_list, message):

        assert len(svg_path_list) == len(png_path_list)
        self.len = len(svg_path_list)

        self.svg_path_list = svg_path_list
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

        canvas_width, canvas_height, shapes, shape_groups = (
            pydiffvg.svg_to_scene(self.svg_path_list[idx]))

        png_tensor = Image.open(self.png_path_list[idx])
        png_tensor = self.transform(png_tensor)

        png_message = torch.tensor(self.message)
        name = os.path.basename(self.png_path_list[idx]).split('.')[0]

        return canvas_width, canvas_height, shapes, shape_groups, png_tensor, png_message, name


def main(args):
    logger = setup_logger()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    svg_path_list = glob.glob(os.path.join(args.svg_path, '*.svg'))
    svg_path_list = sorted(svg_path_list)

    png_path_list = glob.glob(os.path.join(args.en_png_path, '*.png'))
    png_path_list = sorted(png_path_list)

    png_folder = os.path.basename(args.en_png_path)
    if png_folder == "0":
        message = 0
    elif png_folder == "1":
        message = 1
    else:
        message = None

    render = pydiffvg.RenderFunction.apply
    # imgsr_model = create_sr_model()
    message_criterion = torch.nn.CrossEntropyLoss()

    decoder = Decoder().to(device).eval()
    decoder.load_state_dict(torch.load(args.decoder_checkpoint_path, map_location=device))

    dataset = SvgAndPngDataset(svg_path_list, png_path_list, message=message)
    end = min(args.start + args.delta, dataset.len)

    idx = 0
    acc = 0

    other = []

    for i in range(args.start, end):
        canvas_width, canvas_height, shapes, shape_groups, png_tensor, png_message, name = dataset[i]
        idx += 1
        png_tensor = png_tensor.unsqueeze(0).to(device)
        png_message = png_message.unsqueeze(0).to(device)

        # # 将分辨率从64*64提升到256*256
        # imgsr_model.set_test_input(png_tensor)
        # with torch.no_grad():
        #     imgsr_model.forward()
        # png_tensor = imgsr_model.fake_B

        points_vars = []
        for path in shapes:
            path.points.requires_grad = True
            points_vars.append(path.points)
        color_vars = {}
        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars[group.fill_color.data_ptr()] = group.fill_color
        color_vars = list(color_vars.values())

        # Optimize
        if not bool(points_vars) or not bool(shapes) or not bool(color_vars):
            other.append(name)
            # save_svg_paths_only(f'{args.out_path}/{name}.svg',
            #                     canvas_width, canvas_height, shapes, shape_groups)
            continue
        points_optim = torch.optim.Adam(points_vars, lr=1.0)
        color_optim = torch.optim.Adam(color_vars, lr=0.01)

        # Adam iterations.
        for t in range(args.num_iter):
            points_optim.zero_grad()
            color_optim.zero_grad()
            # Forward pass: render the image.
            scene_args = pydiffvg.RenderFunction.serialize_scene( \
                canvas_width, canvas_height, shapes, shape_groups)
            img = render(args.out_width,  # width
                         args.out_height,  # height
                         2,  # num_samples_x
                         2,  # num_samples_y
                         0,  # seed
                         None,  # bg
                         *scene_args)
            # Compose img with white background
            img = img.to(device)
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                              device=device) * (1 - img[:, :, 3:4])
            img = img[:, :, :1]
            # Convert img from HWC to NCHW
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

            image_loss = (img - png_tensor).pow(2).mean()

            # 重新映射到64*64进行解码
            img = render(64,  # width
                         64,  # height
                         2,  # num_samples_x
                         2,  # num_samples_y
                         0,  # seed
                         None,  # bg
                         *scene_args)
            img = img.to(device)
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                              device=device) * (1 - img[:, :, 3:4])
            img = img[:, :, :1]
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

            img = add_noise2(img, args, args.globle)
            img_message = decoder(img)

            message_loss = message_criterion(img_message, png_message)

            loss = image_loss + 1e-5 * message_loss
            loss.backward()

            # Take a gradient descent step.
            points_optim.step()
            color_optim.step()
            for group in shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)

            if t == args.num_iter - 1:
                save_svg_paths_only(f'{args.out_path}/{name}.svg',
                                    canvas_width, canvas_height, shapes, shape_groups)

        scene_args = pydiffvg.RenderFunction.serialize_scene( \
            canvas_width, canvas_height, shapes, shape_groups)

        # 重新映射到64*64进行解码
        img = render(64,  # width
                     64,  # height
                     2,  # num_samples_x
                     2,  # num_samples_y
                     0,  # seed
                     None,  # bg
                     *scene_args)
        img = img.to(device)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=device) * (1 - img[:, :, 3:4])
        img = img[:, :, :1]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)

        img = add_noise2(img, args, args.globle)
        img_message = decoder(img)

        png_tensor = add_noise(png_tensor)
        png_message = decoder(png_tensor)

        if int(img_message.argmax(dim=-1)) == int(png_message.argmax(dim=-1)):
            acc += 1
            pre = '正确'
        else:
            pre = '错误'
        result = "{}\t第{}张:\t文件名 {}\t预测 {}\t目前为止准确率 {:.2f}\t".format(
            time.ctime(), idx, name, pre, (acc / idx))
        logger.info(result)
        print(result)

    print(other)
    logger.info((' ').join(other))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iter", type=int, default=200)
    parser.add_argument("--out_width",type=int, default=64)
    parser.add_argument("--out_height", type=int, default=64)

    parser.add_argument("--decoder_checkpoint_path", type=str,
                        default="../stage1/model/20240419_145013/decoder_epoch_2.pth", help="decoder模型参数文件")

    parser.add_argument("--svg_path", type=str,
                        default="../../data/svg35", help="svg所在文件夹")

    parser.add_argument("--ori_png_path", type=str,
                        default="../../data/ori_png64_35/0", help="原始png所在文件夹")

    parser.add_argument("--en_png_path", type=str,
                        default="../stage1/output/0", help="需要对齐的编码png所在文件夹")

    parser.add_argument("--out_path", type=str,
                        default="./results/refine_svg_64/0", help="结果保存路径")

    parser.add_argument("--start", type=int,
                        default=0, help="开始的索引")
    parser.add_argument("--delta", type=int,
                        default=2000, help="处理的数据量")
    parser.add_argument("--globle", type=int,
                        default=5000, help="")

    parser.add_argument('--rnd_bri_ramp', type=int, default=1000)
    parser.add_argument('--rnd_sat_ramp', type=int, default=1000)
    parser.add_argument('--rnd_hue_ramp', type=int, default=1000)
    parser.add_argument('--rnd_noise_ramp', type=int, default=1000)
    parser.add_argument('--contrast_ramp', type=int, default=1000)
    parser.add_argument('--rnd_resize_ramp', type=int, default=1000)
    parser.add_argument('--rnd_trans_ramp', type=int, default=1000)
    parser.add_argument('--rnd_scal_ramp', type=int, default=1000)
    parser.add_argument('--rnd_rot_ramp', type=int, default=1000)
    parser.add_argument('--rnd_perspec_ramp', type=int, default=1000)

    parser.add_argument('--rnd_bri', type=float, default=.3)
    parser.add_argument('--rnd_noise', type=float, default=.02)
    parser.add_argument('--rnd_sat', type=float, default=1.0)
    parser.add_argument('--rnd_hue', type=float, default=.1)
    parser.add_argument('--contrast_low', type=float, default=.5)
    parser.add_argument('--contrast_high', type=float, default=1.5)
    parser.add_argument('--rnd_resize', type=float, default=.1)
    parser.add_argument('--rnd_trans', type=float, default=0.2)
    parser.add_argument('--rnd_scal', type=float, default=.1)
    parser.add_argument('--rnd_rot', type=float, default=30)
    parser.add_argument('--rnd_perspec', type=float, default=0.1)

    args = parser.parse_args()

    main(args)
