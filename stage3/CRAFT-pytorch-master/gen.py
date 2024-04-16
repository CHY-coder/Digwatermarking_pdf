from PIL import Image
import numpy as np

# 假设你已经有了CRAFT的输出坐标，下面是如何从图像中裁剪文字的示例

# 加载原始图像
original_image = Image.open('./data/1.jpg')

# 这应该是CRAFT输出的坐标，每个坐标表示一个文字的区域
# 每四个一组，表示一个文本的边界框
seg_txt = './result/res_1.txt'
coordinates = []
with open(seg_txt) as f:
    lines = f.readlines()
    for line in lines:
        if line != '\n':
            line = line.strip()
            line = list(map(int, line.split(',')))
            coordinates.append(line)

# 对于CRAFT输出的每组坐标
for i, coord in enumerate(coordinates):
    # 假设coord是一个8个数字的列表[x1, y1, x2, y2, x3, y3, x4, y4]
    # 我们需要找到边界框的最小矩形
    xs = coord[::2]
    ys = coord[1::2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # 使用Pillow裁剪图像
    cropped_image = original_image.crop((min_x, min_y, max_x, max_y))

    # 保存裁剪后的图像
    cropped_image.save(f'./gen/cropped_text_{i}.png')

