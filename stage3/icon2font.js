const SVGIcons2SVGFontStream = require('svgicons2svgfont');
const fs = require('fs');
const path = require('path');

const fontStream = new SVGIcons2SVGFontStream({
  fontName: 'test',
});

fontStream
  .pipe(fs.createWriteStream('fonts/test.svg'))
  .on('finish', function () {
    console.log('字体成功创建！');
  })
  .on('error', function (err) {
    console.log(err);
  });

// 设定图标文件夹路径
const iconsDirectory = '/home/chenzhe/a_py_project/Digwatermark/Digwatermarking_pdf_run/stage2/results/test/1';
// 读取图标文件夹中的所有文件
const iconFiles = fs.readdirSync(iconsDirectory);

for (let i = 0; i < iconFiles.length; i++) {
  const fileName = iconFiles[i];
  // 确保只处理SVG文件，且文件名以"uni"开头但不是"union"
  if (path.extname(fileName) === '.svg' && fileName.startsWith('uni') && fileName.slice(0, 5).toLowerCase() !== 'union') {
    const glyph = fs.createReadStream(path.join(iconsDirectory, fileName));
    // 提取文件名的后四位作为Unicode值
    const unicodeStr = fileName.slice(-8, -4);
    const unicode = String.fromCharCode(parseInt(unicodeStr, 16));
    glyph.metadata = {
      unicode: [unicode],
      name: fileName.slice(0, -4), // 移除文件扩展名，使用整个文件名（不包括扩展名）作为名称
    };
    fontStream.write(glyph);
  }
}

// 不要忘记结束流
fontStream.end();
