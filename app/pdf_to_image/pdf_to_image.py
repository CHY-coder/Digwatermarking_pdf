import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
import os


def pdf_to_images(pdf_folder, output_folder, target_width=7014, target_height=5077, bit_depth=32):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    pdf_path = os.path.join(pdf_folder, pdf_files[0])

    # 打开 PDF 文件
    pdf_document = fitz.open(pdf_path)

    # 计算缩放比例
    page = pdf_document.load_page(0)
    original_width, original_height = page.rect.width, page.rect.height

    zoom_x = target_width / original_width
    zoom_y = target_height / original_height
    mat = fitz.Matrix(zoom_x, zoom_y)

    # 遍历每一页
    for page_num in range(len(pdf_document)):

        if page_num>3:
            break

        # 获取页对象
        page = pdf_document.load_page(page_num)

        # 使用缩放矩阵将页转换为图片
        pix = page.get_pixmap(matrix=mat, alpha=bit_depth == 32)  # 根据位深设置透明度

        # 检查位深并转换
        if bit_depth == 32:
            mode = "RGBA"
        elif bit_depth == 24:
            mode = "RGB"
        else:
            raise ValueError("Unsupported bit depth. Only 24 and 32 are supported.")

        # 创建图片路径
        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")

        # 转换图片并调整对比度和亮度
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

        # 调整对比度
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)  # 增加对比度
        # 调整亮度
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.2)  # 增加亮度

        # 保存图片
        img.save(image_path)

        print(f"Page {page_num + 1} saved as {image_path}")

