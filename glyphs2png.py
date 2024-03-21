import fontforge
import argparse

def export_glyphs_to_png(font_path, output_dir, pixel_height=64):
    font = fontforge.open(font_path)
    for glyph in font.glyphs():
        output_path = f"{output_dir}/{glyph.glyphname}.png"
        glyph.export(output_path, pixel_height)
    font.close()

parser = argparse.ArgumentParser(description="parser for glyphs to png.")
parser.add_argument("--font_path", type=str, default="./simsun.ttc", help="glyphs path.")
parser.add_argument("--output_dir", type=str, default="./png64", help="output png path.")
parser.add_argument("--pixel", type=int, default=64, help="set pixel height.")
args = parser.parse_args()
export_glyphs_to_png(args.font_path, args.output_dir, args.pixel)
