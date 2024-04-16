import sys
sys.path.append('/usr/lib/python3/dist-packages')
import fontforge
import os


def svg_to_ttf(svg_dir, output_ttf_path):
    # 创建一个新的字体
    font = fontforge.font()

    # 为每个SVG文件创建一个字形
    for svg_filename in os.listdir(svg_dir):
        if svg_filename.endswith(".svg"):
            s = os.path.splitext(svg_filename)[0]

            if s.startswith('uni') and s != 'union':
                # 去除文件名开头的"uni"和文件扩展名，剩下的是Unicode码点的十六进制表示
                unicode_str = s[3:]
                # print(unicode_str)
                codepoint = int(unicode_str, 16)  # 将字符串从十六进制转换为整数

                # 创建字形
                glyph = font.createChar(codepoint)
                
                # 导入SVG到字形
                file_path = os.path.join(svg_dir, svg_filename)

                glyph.importOutlines(file_path)
                glyph.left_side_bearing = glyph.right_side_bearing = 15

    # 生成TTF文件
    font.generate(output_ttf_path)

if __name__ == "__main__":

    svg_to_ttf('/home/chenzhe/a_py_project/Digwatermark/Digwatermarking_pdf_run/stage2/results/refine_svg_64/0',
               './0.ttf')






