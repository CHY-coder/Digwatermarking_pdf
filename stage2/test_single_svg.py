import os
import glob
import time
import sys

import pydiffvg
import torch
import argparse
from torchvision import transforms
from save_svg import save_svg_paths_only
from torch.utils.data import Dataset, DataLoader
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from stage1.model import Decoder, add_noise
from model import create_sr_model
from PIL import Image

gamma = 1.0

device = torch.device("cuda")


class Args:
    def __init__(self, num_iter, out_width, out_height, decoder_checkpoint_path, svg_path, en_png_path, out_path):
        self.num_iter = num_iter
        self.out_width = out_width
        self.out_height = out_height
        self.decoder_checkpoint_path = decoder_checkpoint_path
        self.svg_path = svg_path
        self.en_png_path = en_png_path
        self.out_path = out_path


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

    index_list = [13985, 25861, 10068, 10396]

    svg_path_list = glob.glob(os.path.join(args.svg_path, '*.svg'))
    svg_path_list = sorted(svg_path_list)

    svg_paths = []
    for i in index_list:
        svg_paths.append(svg_path_list[i])
    svg_paths = sorted(svg_paths)

    svg_name_list = [os.path.basename(file_path) for file_path in svg_paths]

    png_path_list = glob.glob(os.path.join(args.en_png_path, '*.png'))
    png_path_list = sorted(png_path_list)

    decoder = Decoder().to(device).eval()
    decoder.load_state_dict(torch.load(args.decoder_checkpoint_path, map_location=device))

    png_folder = os.path.basename(args.en_png_path)

    if png_folder == "0":
        message = 0
    elif png_folder == "1":
        message = 1
    else:
        message = None

    imgsr_model = create_sr_model()
    render = pydiffvg.RenderFunction.apply
    message_criterion = torch.nn.CrossEntropyLoss()
    dataset = SvgAndPngDataset(svg_paths, png_path_list, message=message)

    output_path = f'{args.out_path}/svg_{message}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for idx, data in enumerate(dataset):

        print('iteration:', idx)

        output_filename = f'{args.out_path}/svg_{message}/{svg_name_list[idx]}'
        # if os.path.exists(output_filename):
        #     continue

        canvas_width, canvas_height, shapes, shape_groups, png_tensor, png_message = data

        png_tensor = png_tensor.unsqueeze(0).to(device)
        png_message = png_message.unsqueeze(0).to(device)

        # 将分辨率从64*64提升到256*256
        imgsr_model.set_test_input(png_tensor)
        with torch.no_grad():
            imgsr_model.forward()
        png_tensor = imgsr_model.fake_B

        points_vars = []
        color_vars = {}
        for path in shapes:
            path.points.requires_grad = True
            points_vars.append(path.points)
        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars[group.fill_color.data_ptr()] = group.fill_color
        color_vars = list(color_vars.values())

        points_optim = torch.optim.Adam(points_vars, lr=1.0)
        color_optim = torch.optim.Adam(color_vars, lr=0.01)

        for t in range(args.num_iter):

            points_optim.zero_grad()
            color_optim.zero_grad()

            scene_args = pydiffvg.RenderFunction.serialize_scene( \
                canvas_width, canvas_height, shapes, shape_groups)
            img = render(args.out_width,  # width
                         args.out_height,  # height
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
            loss.backward()

            points_optim.step()
            color_optim.step()

            for group in shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)

            if t == args.num_iter - 1:
                save_svg_paths_only(output_filename,
                                    canvas_width, canvas_height, shapes, shape_groups)

        scene_args = pydiffvg.RenderFunction.serialize_scene( \
            canvas_width, canvas_height, shapes, shape_groups)

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

        # img = add_noise(img)
        img_message = decoder(img)

        print("编码数字为：", int(img_message.argmax(dim=-1)))
        print("正确编码数字为：", message)


if __name__ == "__main__":

    decoder_checkpoint_path = "../stage1/model/20240403_095840/decoder_epoch_end.pth"

    args1 = Args(
        num_iter=200,
        out_width=256,
        out_height=256,
        decoder_checkpoint_path=decoder_checkpoint_path,
        svg_path="../../data/svg256",
        en_png_path="../stage1/output/0",
        out_path="./results/refine_svg_test_4"
    )

    start_time = time.time()
    main(args1)
    end_time = time.time()
    print(f"第一次运行时间: {end_time - start_time} 秒")

    args2 = Args(
        num_iter=200,
        out_width=256,
        out_height=256,
        decoder_checkpoint_path=decoder_checkpoint_path,
        svg_path="../../data/svg256",
        en_png_path="../stage1/output/1",
        out_path="./results/refine_svg_test_4"
    )

    start_time = time.time()
    main(args2)
    end_time = time.time()
    print(f"第二次运行时间: {end_time - start_time} 秒")



