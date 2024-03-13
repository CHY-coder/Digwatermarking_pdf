import fontforge
import os
import psMat

def export_glyphs_to_svg(font_path, output_dir, glyph_size=64):
    # 加载字体文件
    font = fontforge.open(font_path)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历字体中的所有字形
    for glyph in font.glyphs():
        # 设置导出的 SVG 的大小
        glyph.transform(psMat.scale(glyph_size / glyph.width))
        # 设置文件名：原字形的名称 + ".svg"
        file_name = "{}.svg".format(glyph.glyphname)
        file_path = os.path.join(output_dir, file_name)
        # 导出字形为 SVG 文件
        glyph.export(file_path)

# 字体文件路径
font_path = "./simsun.ttc"
# 输出目录
output_dir = "./svg64"
# 导出字形为 SVG
export_glyphs_to_svg(font_path, output_dir)

