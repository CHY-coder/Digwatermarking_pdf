# Reproduction of AutoStegaFont: Synthesizing Vector Fonts for Hiding Information in Documents

# Usage
To train the encoder and decoder in the first stage, run
```
cd stage1
python train.py train --dataset ../../data/ori_png64 --save-model-dir ./model --image-size 64 --device cuda --lr 1e-4 --eval_data ../../data/ori_png64 --vq_weight 3 --percep_weight 0.1 --batch-size 16
```
Then, generate the encoded glyph images.
```
cd stage1
python generate_png.py
```
To train the SVG glyphs in the second stage, run
```
cd stage2
python svg_refine_64.py
```
Then generate fonts from SVG glyphs.
```
cd stage3
node icon2font.js
node svg2ttf.js
```
Demo
```
cd app
python gradio_app.py
```

# Attention
## You must install fontforge when use glyphs2svg and glyphs2png.
```
sudo apt-get install fontforge
sudo apt-get install python3-fontforge

way 1
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages
python3 glyphs2svg.py

way 2
fontforge -script glyphs2png.py

way 3 (This way do not set size)
fontforge -lang=ff -c 'Open($1); SelectWorthOutputting(); foreach Export("svg"); endloop;' font.ttf 
```
## we put deepvecfont into /app/deepvecfont and modify modify the checkpoints_dir in /app/deepvecfont/models/imgsr/modules.py line 50 to '/app/deepvecfont/experiments'

# Reference
```
init code: https://github.com/pytorch/examples/blob/main/fast_neural_style/README.md
noise layer: https://github.com/tancik/StegaStamp
```
