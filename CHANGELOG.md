# Upcoming
---
#### Network
- Made string configs (activation, adaptiveLR, cost, distribution) case/underscore/space insensitive.
- Allow custom activation functions to be configured

#### NetMath
- Breaking change (if you were using the NetMath functions directly): Made config functions' names lower case

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