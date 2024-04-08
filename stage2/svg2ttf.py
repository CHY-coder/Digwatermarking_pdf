import fontforge
import os
import sys

def svg_to_ttf(svg_dir, output_ttf_path):
    # 创建一个新的字体
    font = fontforge.font()

    # 为每个SVG文件创建一个字形
    for svg_filename in os.listdir(svg_dir):
        if svg_filename.endswith(".svg"):
           # 去除文件名开头的"uni"和文件扩展名，剩下的是Unicode码点的十六进制表示
            unicode_str = os.path.splitext(svg_filename)[0][3:]
            codepoint = int(unicode_str, 16)  # 将字符串从十六进制转换为整数

            # 创建字形
            glyph = font.createChar(codepoint)
            
            # 导入SVG到字形
            glyph.importOutlines(os.path.join(svg_dir, svg_filename))
            
            # 自动设置字形的宽度
            glyph.left_side_bearing = glyph.right_side_bearing = 0
            # glyph.autoWidth()

    # 生成TTF文件
    font.generate(output_ttf_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python svg_to_ttf.py <svg_directory> <output_ttf_path>")
    else:
        svg_dir = sys.argv[1]
        output_ttf_path = sys.argv[2]
        svg_to_ttf(svg_dir, output_ttf_path)



