import os
import glob
import sys

import datetime
import pydiffvg
import torch
import argparse
from torchvision import transforms
from save_svg import save_svg_paths_only
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from model import Decoder, create_sr_model
from noise import add_noise, add_noise2

gamma = 1.0

device = torch.device("cuda")


class SvgAndPngDataset(Dataset):
    def __init__(self, svg_path_list, png_path_list, message):

        assert len(svg_path_list) == len(png_path_list)

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

        return canvas_width, canvas_height, shapes, shape_groups, png_tensor, png_message


def main(args):

    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    ori_path_list = glob.glob(os.path.join(args.ori_png_path, '*.png'))
    ori_path_list = sorted(ori_path_list)
    ori_tensor = Image.open(ori_path_list[args.num_svg])
    ori_tensor = transform(ori_tensor)

    pydiffvg.imwrite(ori_tensor.permute(1, 2, 0).repeat(1, 1, 3).cpu(), f'{args.out_path}/a_ori_png.png',gamma=gamma)

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
    imgsr_model = create_sr_model()
    message_criterion = torch.nn.CrossEntropyLoss()

    decoder = Decoder().to(device).eval()
    decoder.load_state_dict(torch.load(args.decoder_checkpoint_path, map_location=device))

    dataset = SvgAndPngDataset(svg_path_list, png_path_list, message=message)

    canvas_width, canvas_height, shapes, shape_groups, png_tensor, png_message = dataset[args.num_svg]

    pydiffvg.imwrite(png_tensor.permute(1,2,0).repeat(1,1,3).cpu(), f'{args.out_path}/a_target_64.png', gamma=gamma)

    png_tensor = png_tensor.unsqueeze(0).to(device)
    png_message = png_message.unsqueeze(0).to(device)

    # 将分辨率从64*64提升到256*256
    imgsr_model.set_test_input(png_tensor)
    with torch.no_grad():
        imgsr_model.forward()
    png_tensor = imgsr_model.fake_B
    pydiffvg.imwrite(png_tensor.squeeze(0).permute(1, 2, 0).repeat(1, 1, 3).cpu(), f'{args.out_path}/a_target_256.png', gamma=gamma)

    scene_args = pydiffvg.RenderFunction.serialize_scene( \
        canvas_width, canvas_height, shapes, shape_groups)

    img = render(args.out_width,  # width
                 args.out_height,  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 0,  # seed
                 None,  # bg
                 *scene_args)
    # The output image is in linear RGB space. Do Gamma correction before saving the image.
    pydiffvg.imwrite(img.cpu(), f'{args.out_path}/init.png', gamma=gamma)

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
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    # Adam iterations.
    for t in range(args.num_iter):
        print('iteration:', t)
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
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=device) * (1 - img[:, :, 3:4])
        # Save the intermediate render.
        pydiffvg.imwrite(img.cpu(), f'{args.out_path}/iter_{t}.png', gamma=gamma)
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
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                          device=device) * (1 - img[:, :, 3:4])
        img = img[:, :, :1]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW

        img = add_noise(img)
        img_message = decoder(img)

        message_loss = message_criterion(img_message, png_message)

        loss = image_loss + 1e-5 * message_loss
        print('render loss:', loss.item())
        loss.backward()

        # Take a gradient descent step.
        points_optim.step()
        color_optim.step()
        for group in shape_groups:
            group.fill_color.data.clamp_(0.0, 1.0)

        if t == args.num_iter - 1:
            save_svg_paths_only(f'{args.out_path}/final.svg',
                                canvas_width, canvas_height, shapes, shape_groups)

    scene_args = pydiffvg.RenderFunction.serialize_scene( \
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(args.out_width,  # width
                 args.out_height,  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 0,  # seed
                 None,  # bg
                 *scene_args)
    pydiffvg.imwrite(img.cpu(), f'{args.out_path}/final.png', gamma=gamma)

    # 重新映射到64*64进行解码
    img = render(64,  # width
                 64,  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 0,  # seed
                 None,  # bg
                 *scene_args)

    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3,
                                                      device=device) * (1 - img[:, :, 3:4])
    img = img[:, :, :1]
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)

    img = add_noise(img)
    img_message = decoder(img)

    png_tensor = add_noise(png_tensor)
    png_message = decoder(png_tensor)

    print("*"*200)
    print("编码数字为：", int(img_message.argmax(dim=-1)))
    print("*" * 200)
    print("正确编码数字为：", int(png_message.argmax(dim=-1)))
    print("*" * 200)

    # from subprocess import call
    #
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # call(["ffmpeg", "-framerate", "24", "-i",
    #     f"{args.out_path}/iter_%d.png", "-vb", "20M",
    #     f"{args.out_path}/total_{args.num_iter}_{timestamp}.mp4"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iter", type=int, default=200)
    parser.add_argument("--num_svg", type=int, default=25861)
    parser.add_argument("--out_width",type=int, default=256)
    parser.add_argument("--out_height", type=int, default=256)

    parser.add_argument("--decoder_checkpoint_path", type=str,
                        default="../stage1/model/20240403/decoder_epoch_17.pth", help="decoder模型参数文件")

    parser.add_argument("--svg_path", type=str,
                        default="../../data/svg256", help="svg所在文件夹")

    parser.add_argument("--ori_png_path", type=str,
                        default="../../data/png64/0", help="原始png所在文件夹")

    parser.add_argument("--en_png_path", type=str,
                        default="../stage1/output/0", help="需要对齐的编码png所在文件夹")

    parser.add_argument("--out_path", type=str,
                        default="./results/refine_svg", help="结果保存路径")

    args = parser.parse_args()

    main(args)
