import fontforge
import argparse
import os

def export_glyphs_to_png(font_path, output_dir, pixel_height=64):
    font = fontforge.open(font_path)

    img0_path = os.path.join(output_dir, '0')
    img1_path = os.path.join(output_dir, '1')
    if not os.path.exists(img0_path):
        os.makedirs(img0_path)
    if not os.path.exists(img1_path):
        os.makedirs(img1_path)

    for glyph in font.glyphs():
        output_path0 = f"{img0_path}/{glyph.glyphname}.png"
        output_path1 = f"{img1_path}/{glyph.glyphname}.png"
        glyph.export(output_path0, pixel_height)
        glyph.export(output_path1, pixel_height)
    font.close()

parser = argparse.ArgumentParser(description="parser for glyphs to png.")
parser.add_argument("--font_path", type=str, default="./simsun.ttc", help="glyphs path.")
parser.add_argument("--output_dir", type=str, default="./png64", help="output png path.")
parser.add_argument("--pixel", type=int, default=64, help="set pixel height.")
args = parser.parse_args()
export_glyphs_to_png(args.font_path, args.output_dir, args.pixel)
