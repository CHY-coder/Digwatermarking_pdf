# Reproduction of AutoStegaFont: Synthesizing Vector Fonts for Hiding Information in Documents

# Attention
## You must install fontforge when use glyphs2svg and glyphs2png.
```
sudo apt-get install fontforge

way 1
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages
python3 glyphs2svg.py

way 2
fontforge -script glyphs2png.py

way 3 (This way do not set size)
fontforge -lang=ff -c 'Open($1); SelectWorthOutputting(); foreach Export("svg"); endloop;' font.ttf 
```
# Reference
```
init code: https://github.com/pytorch/examples/blob/main/fast_neural_style/README.md
noise layer: https://github.com/tancik/StegaStamp
```