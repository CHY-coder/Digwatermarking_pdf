import fontforge

def export_glyphs_to_png(font_path, output_dir, pixel_height=64):
    font = fontforge.open(font_path)
    for glyph in font.glyphs():
        output_path = f"{output_dir}/{glyph.glyphname}.png"
        glyph.export(output_path, pixel_height)
    font.close()

# 调用函数
font_path = "./simsun.ttc"  # 字体文件路径
output_dir = "./output_png"  # 输出目录路径
export_glyphs_to_png(font_path, output_dir)
