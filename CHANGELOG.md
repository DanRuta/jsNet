# Upcoming
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

# v1.1.0 - Adaptive learning rates
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