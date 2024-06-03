import os
import time

import gradio as gr

from pathlib import Path
import shutil

from pdf_to_image.pdf_to_image import pdf_to_images

from image_cropping.image_crop import split_char
from image_cropping.image_generate import gen_image
from image_decoding.image_decode import decode_id


def decimal_to_ternary(decimal_number):
    # 将十进制数转换为三进制数
    ternary_number = []
    while decimal_number > 0:
        ternary_number.append(decimal_number % 3)
        decimal_number //= 3
    ternary_number.reverse()

    # 如果转换后的三进制数不足10位，用0填充
    while len(ternary_number) < 13:
        ternary_number.insert(0, 0)

    return ternary_number


def add_and_mod(ternary_number):
    # 对10位三进制数进行求和并取3的模
    sum_mod = sum(ternary_number) % 3

    # 将模添加到10位数的开头
    result = [sum_mod] + ternary_number
    # 再复制一位方便切割
    result = [sum_mod] + result
    str_list = [str(num) for num in result]
    ID = ''.join(str_list)
    return ID


def process_decimal(decimal_number):
    ternary_number = decimal_to_ternary(decimal_number)
    result = add_and_mod(ternary_number)
    return result


def clear_folder(folder_list):
    for folder in folder_list:
        for item in folder.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def check(code):
    codes = code.split('\n')
    result = []
    for id in codes:
        id_int = [int(n) for n in id]
        if sum(id_int[1:]) % 3 == id_int[0]:
            result.append('True')
        else:
            result.append('False')
    final_result = '\n'.join(result)
    return final_result


def encode(num):
    number = int(num)
    if number>1594323:
        result = {"id":"null (大于最大可编码数字)"}
    else:
        result = {"id":process_decimal(number)}

    return result["id"]


def decode(file_path):

    file_path = file_path.name
    file_name = os.path.basename(file_path)

    input_folder = "pdf"
    image_folder = "image"
    seg_folder = "seg_result"
    result_folder = "cropped_img"

    for folder in [input_folder, image_folder, seg_folder, result_folder]:
        os.makedirs(folder, exist_ok=True)

    clear_folder([Path(input_folder), Path(image_folder), Path(seg_folder), Path(result_folder)])

    input_dir = Path(input_folder)
    dest_path = input_dir / file_name
    shutil.copy(file_path, dest_path)

    start_time = time.time()

    # PDF 转图片
    start = time.time()
    pdf_to_images(input_folder, image_folder)
    end = time.time()
    print(f"pdf转图片运行时间: {end - start:.2f} 秒")

    # 图片分割
    start = time.time()
    split_char(image_folder, seg_folder)
    end = time.time()
    print(f"图片分割运行时间: {end - start:.2f} 秒")

    # 分割图片生成
    start = time.time()
    gen_image(image_folder, seg_folder, result_folder)
    end = time.time()
    print(f"分割图片生成运行时间: {end - start:.2f} 秒")

    # 分割图片解码
    start = time.time()
    code = decode_id(result_folder)
    end = time.time()
    print(f"分割图片解码运行时间: {end - start:.2f} 秒")

    # 解码ID验证
    start = time.time()
    decode_result = check(code)
    end = time.time()
    print(f"解码ID验证运行时间: {end - start:.2f} 秒")

    result = {"id": code, "result": decode_result}

    total_time = time.time() - start_time
    print(f"总运行时间: {total_time:.2f} 秒")

    return result["id"], result["result"]


with gr.Blocks() as demo:

    with gr.Tab("encode"):
        with gr.Row():
            user_input = gr.Textbox(
                label="公司id: ",
                placeholder="请输入数字",
                elem_id="input_box",
            )
        with gr.Row():
            encode_out = gr.Textbox(label="编码id: ")

        with gr.Row():
            encodeBtn = gr.Button(value="编码", variant="primary", scale=0)

    with gr.Tab("decode"):
        file_download = gr.File(label="文档扫描pdf上传: ")

        with gr.Row():
            with gr.Column(scale=1):
                output = gr.Textbox(label="解码id: ")
            with gr.Column(scale=1):
                output2 = gr.Textbox(label="校验位验证结果: ")

        with gr.Row():
            decodeBtn = gr.Button(value="解码", variant="primary", scale=0)

    encodeBtn.click(encode,[user_input],[encode_out])
    decodeBtn.click(decode,[file_download], [output,output2])

demo.queue()
demo.launch(server_name="127.0.0.1", server_port=8891, share=True)