import fontforge
import os
import re

def adjust_svg_viewbox(svg_content, canvas_size=256):
    # 构建新的 viewBox 字符串
    new_viewbox = f'viewBox="0 0 {canvas_size} {canvas_size}"'
    # 使用正则表达式查找并替换 viewBox 属性
    svg_content = re.sub(r'viewBox="[^"]*"', new_viewbox, svg_content, count=1)
    return svg_content

def export_glyphs_to_svg_with_viewbox(font_path, output_dir, canvas_size=256):
    font = fontforge.open(font_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for glyph in font.glyphs():
        # bbox = glyph.boundingBox()
        # if bbox[2] - bbox[0] == 0 or bbox[3] - bbox[1] == 0:
        #     continue
        file_name = f"{glyph.glyphname}.svg"
        file_path = os.path.join(output_dir, file_name)
        glyph.export(file_path)

        with open(file_path, 'r') as file:
            svg_data = file.read()

        svg_data = adjust_svg_viewbox(svg_data, canvas_size)

        with open(file_path, 'w') as file:
            file.write(svg_data)

font_path = "./simsun.ttc"  # 替换为你的字体文件路径
output_dir = "./svg256"  # 替换为你的输出目录
canvas_size = 256
export_glyphs_to_svg_with_viewbox(font_path, output_dir, canvas_size)
