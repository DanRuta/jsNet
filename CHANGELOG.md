# 3.4.0 - Bug fixes and improvements
---
#### Global
- Removed implicit softmax from last layer, to allow multi-variate regression (#42)

#### WebAssembly
- Added fix for Webpack loading of NetWASM.js
- Added net.delete() for clean-up

#### OutputLayer
- Added OutputLayer class

#### Examples
- Added example project for loading jsNet through Webpack
- Added example for using multiple WASM network instances

#### Bug fixes
- WASM misassigned learning rate defaults
- WASM momentum not training

# 3.3.0 - Misc Improvements
---
#### Network
- Added confusion matrix
- Made it possible to pass volume (3D array) input data
- Added callback interval config to .train()
- Added collectErrors config to .train() and .test()

#### InputLayer
- Added InputLayer class

#### Examples
- Added MNIST dev enviromnment example

# 3.2.0 - IMG data, validation, early stopping
---
#### Network
- Added weight+bias importing and exporting via images, using IMGArrays
- Added validation config to .train(), with interval config
- Added early stopping to validation, with threshold stopping condition
- Added early stopping patience condition
- Added early stopping divergence condition
- Breaking change: "error" key in training callbacks have been changed to "trainingError"
- Breaking change: Removed ability to use either data keys 'expected' and 'output'. Now just 'expected'.

#### NetUtil
- Added splitData function
- Added normalize function

#### NetMath
- Added root mean squared error cost function
- Added momentum weight update function
- Breaking change: Renamed "vanilla update fn" to "vanilla sgd"

# 3.1.0 - Optimizations
---
#### ConvLayer
- Optimized errors structure
- Optimized bias structure
- Optimized activations structure
- Optimized weights structure
- Optimized deltaWeights structure
- Optimized deltaBiase structure

#### NetUtil
- Optimized convolve

#### FCLayer
- Optimized weights structure
- Optimized bias structure
- Optimized deltaWeights structure
- Optimized sums structure
- Optimized errors structure and net errors propagation
- Optimized activations structure
- Optimized forward()
- Optimized backward()
- Optimized deltaBias structure

#### Global
- Changed framework loading to allow choosing between versions at runtime
- Added basic server and browser + nodejs demos for how to load jsNet
- Bug fixes
- Changed the way classes were bundled, to fix some bundler compatibility issues (see #33)

# 3.0.0 - WebAssembly
---
#### WebAssembly
- Complete, rewritten, WebAssembly version of jsNet

#### Global
- Many bug fixes
- Removed default configuration values: l1, l2
- Added layer specific activation function config for FC layers, and ability to turn it off
- Reworked regularization algorithm
- Reworked layer specific activation function assignments
- Reworked net error propagation, using softmax
- net.forward() function now returns softmax activations by default

#### JavaScript
- Removed babel transpilation (it's 2018)

# 2.1.0 - Optimizations
---
#### NetUtil
- Optimized addZeroPadding() - ~68% faster
- Optimized uniform() - ~588% faster
- Optimized gaussian() - ~450% faster

#### FCLayer
- Optimized resetDeltaWeights() and applyDeltaWeights() - ~18% faster (overall)

#### NetMath
- Optimized softmax() - ~924% faster

#### ConvLayer
- Restricted filters' dropout maps only to when dropout is configured - less memory usage

#### Bug Fixes
- Fixed bug caused by minification and disabled name mangling

# 2.0.0 - Convolutional Networks
---
#### Network
- New name: jsNet
- Restructured to allow multiple layer types
- Added conv config for configuring filterSize, zeroPadding, stride ConvLayer default values
- Added pool config for configuring size and stride PoolLayer default values
- Added (input) channels config, used by ConvLayers
- Re-wrote the JSON import/export. Check README for details on backward compatibility
- Removed ability to create a network by just giving layer types in the list
- Can check the version number via Network.version
- Renamed adaptiveLR to updateFn

#### ConvLayer
- Added ConvLayer.js 🎉 with activation, filterCount, filterSize, zeroPadding and stride configs

#### Filter
- Added Filter.js

#### PoolLayer
- Added PoolLayer, with stride and activation configs

#### NetUtil
- Added NetUtil.js
- Added addZeroPadding
- Added arrayToMap
- Added arrayToVolume
- Added 4 other helper functions

#### NetMath
- Renamed noadaptivelr to vanillaupdatefn

#### FCLayer
- Renamed Layer.js to FCLayer.js. Layer still exists as an alias to what is now FCLayer

#### Bug Fixes
- Fixed training callback giving iteration index, not count (-1)

# 1.5.0 - Training, Misc
---
#### Network
- Made string configs (activation, adaptiveLR, cost, distribution) case/underscore/space insensitive.
- Allow custom activation functions to be configured
- Allow custom cost functions to be configured
- Allow custom weight distribution functions to be configured
- Added time elapsed to training/testing logs and elapsed milliseconds to training callback
- Added log option to training/testing to disable logging for each one
- Added mini batch SGD training, and miniBatchSize config for .train() as its config
- Breaking change: classes are no longer required straight into global context. See readme.
- Added shuffle .train() option
- Added callback .test() option
- Breaking change: Updated default values

#### NetMath
- Breaking change (if you were using the NetMath functions directly): Made config functions' names lower case

#### Bug Fixes
- Fixed iteration error logged when testing being the averaged total at that point

# 1.4.0 - Weights Initialization
---
#### Network
- Reworked current weights initialization to be configurable
- Set old weights init to uniform distribution, with additional limit config
- Added mean as weightsConfig option, for gaussian distribution
- Added stdDeviation as weightsConfig option, for gaussian distribution

#### NetMath
- Added standardDeviation to NetMath
- Added gaussian weights distribution
- Added xavierNormal weights distribution
- Added lecunUniform weights distribution
- Added lecunNormal weights distribution
- Added xavierUniform weights distribution

# v1.3.0 - Regularization
---
#### Network
- Added dropout, with dropout configuration
- Added L2 regularization, via l2 strength config
- Added L1 regularization, via l1 strength config
- Added max norm regularization, via the maxNorm threshold value config

#### Bug Fixes
- Fixed error value logged accumulating across epochs, instead of resetting to 0
- Fixed epoch counter logged resetting instead of accumulating across training sessions

# v1.2.0 - Activation functions (Part 1)
---
#### Network
- Added lreluSlope, for lrelu activation
- Added eluAlpha, for elu activation

#### NetMath
- Added tanh activation function
- Added relu activation function
- Added lrelu activation function
- Added rrelu activation function
- Added lecuntanh activation function
- Added sech to NetMath
- Added elu activation function

# v1.1.0 - Update Functions
---
#### Network
- Added rho as a network configuration
- Added rmsDecay as a network configuration
- Added adaptiveLR as a network configuration

#### NetMath
- Added adam as adaptiveLR configuration
- Added RMSProp as adaptiveLR configuration
- Added adagrad as adaptiveLR configuration
- Added gain as adaptiveLR configuration
- Added Mean Squared Error cost function

# v1.0.0
----
Initial release