const path = require('path');

module.exports = {
  entry: './src/index.js',
  node: {
    fs: 'empty'
  },
  output: {
    filename: 'main.js',
    path: path.resolve(__dirname, 'dist')
  }
};