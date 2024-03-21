import fontforge
import os
import re
import argparse

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

parser = argparse.ArgumentParser(description="parser for glyphs to svg.")
parser.add_argument("--font_path", type=str, default="./simsun.ttc", help="glyphs path.")
parser.add_argument("--output_dir", type=str, default="./svg256", help="output svg path.")
parser.add_argument("--canvas_size", type=int, default=256, help="canvas size.")
args = parser.parse_args()
export_glyphs_to_svg_with_viewbox(args.font_path, args.output_dir, args.canvas_size)
