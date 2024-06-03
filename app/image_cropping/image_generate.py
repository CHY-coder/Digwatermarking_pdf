import os
from PIL import Image
import image_cropping.file_utils as file_utils
from natsort import natsorted


def gen_image(input_folder, seg_folder, result_folder):

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    image_list, _, _ = file_utils.get_files(input_folder)
    count = 0
    image_list = natsorted(image_list)
    print(image_list)
    for k, image_path in enumerate(image_list):
        original_image = Image.open(image_path)
        img = os.path.basename(image_path).split('.')[0]
        seg_txt = seg_folder + '/res_' + img + '.txt'
        coordinates = []
        with open(seg_txt) as f:
            lines = f.readlines()
            for line in lines:
                if line != '\n':
                    line = line.strip()
                    line = list(map(int, line.split(',')))
                    coordinates.append(line)

        for i, coord in enumerate(coordinates):
            # 假设coord是一个8个数字的列表[x1, y1, x2, y2, x3, y3, x4, y4]
            # 我们需要找到边界框的最小矩形
            count += 1
            xs = coord[::2]
            ys = coord[1::2]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # 使用Pillow裁剪图像
            cropped_image = original_image.crop((min_x, min_y, max_x, max_y))
            # 保存裁剪后的图像
            cropped_image.save(f'{result_folder}/{img}_cropped_{count}.png')



