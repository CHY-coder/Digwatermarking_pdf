var fs = require('fs');
var svg2ttf = require('svg2ttf');

var ttf = svg2ttf(fs.readFileSync('fonts/test.svg', 'utf8'), {});
fs.writeFileSync('test.ttf', Buffer.from(ttf.buffer));
