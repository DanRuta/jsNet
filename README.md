jsNet
=
[![Build Status](https://travis-ci.org/DanRuta/jsNet.svg?branch=master)](https://travis-ci.org/DanRuta/jsNet)&nbsp;&nbsp;&nbsp;&nbsp;JS: [![Coverage Status](https://coveralls.io/repos/github/DanRuta/jsNet/badge.svg?branch=master)](https://coveralls.io/github/DanRuta/jsNet?branch=master) C++: [![codecov](https://codecov.io/gh/DanRuta/jsNet/branch/master/graph/badge.svg)](https://codecov.io/gh/DanRuta/jsNet)
---

[![NPM](https://nodei.co/npm/jsnet.png?downloads=true&downloadRank=true&stars=true)](https://nodei.co/npm/jsnet/)

jsNet is a javascript based deep learning framework for basic and convolutional neural networks. It is functional in both nodejs and in the browser.

*Note: The npm package defaults to the JavaScript only version, for now.*

## Current WebAssembly version: 3.0.
The WebAssembly version is now finished, and fully matches the JavaScript feature set. This notice will be removed in general release 3.1.

*Disclaimer: I am the sole developer on this, and I'm learning things as I go along. There may be things I've misunderstood, not done quite right, or done outright wrong. If you notice something wrong, please let me know, and I'll fix it (or submit a PR).*

## Demos
https://ai.danruta.co.uk/mnist - Interactive MNIST Digit classifier, using FCLayers only.

https://ai.danruta.co.uk/webassembly - Performance comparison between JS and WebAssembly (v2.0) versions.

##  Usage
There are two different versions in the dist folder. The original ```jsNet.min.js``` is the JavaScript only version.
To use the WebAssembly version, use the ```jsNetWebAssembly.min.js``` file. You will also need the ```NetWASM.js``` file. This is the 'glue' code created by emscripten, and it will be importing the ```NetWASM.wasm``` file, so you need to have that available, also.

### JavaScript
When using in the browser, you just include the ```jsNet.min.js``` file. In nodejs, you just ```npm install jsnet``` and  require it like so:
```javascript
const {Network, Layer, FCLayer, ConvLayer, PoolLayer, Filter, Neuron, NetMath, NetUtil} = require("jsNet")
// Get just what you need.
```
Layer is an alias for FCLayer, for people not using the library for convolutional networks.

### WebAssembly

For the WebAssembly version, you will need the ```jsNetWebAssembly.min.js```, ```NetWASM.js``` and ```NetWASM.wasm``` files, found in the dist folder.

I've kept the API almost identical to the JavaScript only version. The biggest difference is that with WebAssembly, currently, the file must be served by a server.

The second difference is that the wasm gets lazy loaded, meaning you have to wait for it to be loaded before you can create the network. You can listen for the ```jsNetWASMLoaded``` event to know when it's ready, like so:

```javascript
window.addEventListener("jsNetWASMLoaded", () => { /* ready */ })
```

Finally, you need to assign the module when creating the network, like so:
```javascript
const net = new Network({Module: Module})
```
This makes it easier to use in nodejs.


### Constructing
---

I will use [the MNIST dataset](https://github.com/cazala/mnist) in the examples below.

A network can be built in three different ways:

##### 1
With absolutely no parameters, and it will build a 3 FCLayer net. It will figure out some appropriate sizes for them once you pass it some data.
```javascript
const net = new Network()

// WebAssembly
const net = new Network({Module: Module})
// This config must always be included when using the  WebAssembly version
```
##### 2
By giving a list of numbers, and the network will configure some FCLayers with that many neurons.
```javascript
const net = new Network({
    layers: [784, 100, 10]
})
```

##### 3
Or you can fully configure the layers by constructing them. Check below what configurations are available for each layer.
```javascript
// Example 1 - fully connected network
const net = new Network({
    layers: [new Layer(784), new Layer(100), new Layer(10)]
})
// Example 2 - convolutional network
const net = new Network({
    layers: [new FCLayer(784), new ConvLayer(8, {filterSize: 3}), new PoolLayer(2), new FCLayer(196), new FCLayer(10)]
})
```

The usual arrangement of layers would folow something like this:

```FCLayer -> [ConvLayer]* -> [[ConvLayer]* || PoolLayer]+ -> FCLayer+```

In words, an FCLayer, maybe followed by pairs of Conv and (optional) Pool layers (starting with Conv), and at the end, at least one FCLayer.
The first FCLayer needs to have as many neurons in it as there are data points per iteration, and the last FCLayer needs to have as many neurons as there are classes for your data set.

When building a convolutional network, make sure that the number of neurons in the FC layer following a Conv or Pool layer matches the number of outgoing activations in the layer preceding it. See below for a tip on easily calculating that number.

### Training
----

The data structure must be an object with key ```input``` having an array of numbers, and key ```expected``` or ```output``` holding the expected output of the network. For example, the following are both valid inputs for both training and testing.
```javascript
{input: [1,0,0.2], expected: [1, 2]}
{input: [1,0,0.2], output: [1, 2]}
```
You train the network by passing a set of data. The network will log to the console the error and epoch number, after each epoch, as well as time elapsed and average epoch duration.
```javascript
const {training} = mnist.set(800, 200) // Get the training data from the mnist library, linked above

const net = new Network()
net.train(training) // This on its own is enough
.then(() => console.log("done")) // Training resolves a promise, meaning you can add further code here (eg testing)
```

Until more options are implemented, softmax is used by default during training, on the last layer. As such, activation configurations are not used there.

##### Options
###### Epochs
By default, this is ```1``` and represents how many times the data passed will be used.
```javascript
net.train(training, {epochs: 5}) // This will run through the training data 5 times
```
###### Callback
You can also provide a callback in the options parameter, which will get called after each iteration (Maybe updating a graph?). The callback is passed how many iterations have passed, the error, the milliseconds elapsed and the input data for that iteration.
```javascript
const doSomeStuff = ({iterations, error, elapsed, input}) => ....
net.train(training, {callback: doSomeStuff})
```
###### Log
You can turn off the logging by passing log: false in the options parameter.
```javascript
net.train(training, {log: false})
```
###### Mini Batch Size
You can use mini batch SGD training by specifying a mini batch size to use (changing it from the default, 1). You can set it to true, and it will default to how many classifications there are in the training data.

```javascript
net.train(training, {miniBatchSize: 10})
```

###### Shuffle
You can randomly shuffle the training data before it is used by setting the shuffle option to true
```javascript
net.train(training, {shuffle: true})
```

### Testing
---
Once the network is trained, you can test it like so:
```javascript
const {training, test} = mnist.set(800, 200)
net.train(training).then(() => net.test(test))
```
This resolves a promise, with the average test error percentage.

##### Options
###### Log
You can turn off the logging by passing log: false in the options parameter.
```javascript
const {training, test} = mnist.set(800, 200)
net.train(training).then(() => net.test(test, {log: false}))
```
###### Callback
Like with training, you can provide a callback for testing, which will get called after each iteration. The callback is passed how many iterations have passed, the error, the milliseconds elapsed and the input data for that iteration.
```javascript
const doSomeStuff = ({iterations, error, elapsed, input}) => ....
net.train(training).then(() => net.test(test, {callback: doSomeStuff}))
```

### Exporting
---
Weights data is exported as a JSON object.
```javascript
const data = trainedNet.toJSON()
```

### Importing
---
Only the weights are exported. You still need to build the net with the same structure and configs, eg activation function.
```javascript
const freshNetwork = new Network(...)
freshNetwork.fromJSON(data)
```
If using exported data from before version 2.0.0, just do a find-replace of "neurons" -> "weights" on the exported data and it will work with the new version.

### Trained usage
---
Once the network has been trained, tested and imported into your page, you can use it via the ```forward``` function.
```javascript
const userInput = [1,0,1,0,0.5] // Example input
const netResult = net.forward(userInput)
```
This will return an array of the **softmax** activations in the output layer.


## Configurations
---
**String configs are case/space/underscore insensitive.**

Without setting any configs, the default values are equivalent to the following configuration:
```javascript
const net = new Network()
// is equivalent to
const net = new Network({
    activation: "sigmoid",
    learningRate: 0.2,
    cost: "meansquarederror",
    dropout: 1,
    l2: undefined,
    l1: undefined,
    layers: [ /* 3 FCLayers */ ]
    updateFn: "vanillaupdatefn",
    weightsConfig: {
        distribution: "xavieruniform"
    }
})
```

### Network

You can check the framework version via Network.version (static).

|  Attribute | What it does | Available Configurations | Default value |
|:-------------:| :-----:| :-----:| :---: |
| learningRate | The speed at which the net will learn. | Any number | 0.2 (see below for exceptions) |
| cost | Cost function to use when printing out the net error | crossEntropy, meanSquaredError | meansquarederror |
| channels | Specifies the number of channels in the input data. EG, 3 for RGB images. Used by convolutional networks. | Any number | undefined |
| conv | (See ConvLayer) An object where the optional keys filterSize, zeroPadding and stride set values for all Conv layers to default to  | Object | {} |
| pool | (See PoolLayer) An object where the optional keys size and stride set values for all Pool layers to default to  | Object | {} |

##### Examples
```javascript
net = new Network({learningRate: 0.2})
net = new Network({cost: "crossEntropy"})
net = new Network({cost: (target, output) => ...})
convnet = new Network({
    layers: [...some fc, conv and pool layers ...],
    conv: {
        filterSize: 3,
        zeroPadding: 1,
        stride: 1
    },
    pool: {
        size: 2
    }
})
```

You can set custom cost functions. They are given the iteration's expected output as the first parameter and the actual output as the second parameter, and they need to return a single number.

Learning rate is 0.2 by default, except when using the following configurations:

| Modifier| Type | Default value|
|:-------------:| :-----: | :-----: |
| RMSProp | updateFn | 0.001 |
| adam | updateFn | 0.01 |
| adadelta | updateFn | undefined |
| tanh, lecuntanh | activation | 0.001 |
| relu, lrelu, rrelu, elu | activation | 0.01 |

### Weight update functions
|  Attribute | What it does | Available Configurations | Default value |
|:-------------:| :-----:| :-----:| :---: |
| updateFn | The function used for updating the weights/bias. The vanillaupdatefn option just sets the network to update the weights without any changes to learning rate. | vanillaupdatefn, gain, adagrad, RMSProp, adam , adadelta| vanillaupdatefn |
| rmsDecay | The decay rate for RMSProp, when used | Any number | 0.99 |
| rho | Momentum for Adadelta, when used | Any number | 0.95 |

##### Examples
```javascript
net = new Network({updateFn: "adagrad"})
net = new Network({updateFn: "RMS_Prop", rmsDecay: 0.99})
net = new Network({updateFn: "adadelta", rho: 0.95})
```

### Activation Function
|  Attribute | What it does | Available Configurations | Default value |
|:-------------:| :-----:| :-----:| :---: |
| activation | Activation function used by neurons | sigmoid, tanh, relu, lrelu, rrelu, lecuntanh, elu | sigmoid |
| lreluSlope | Slope for lrelu, when used | Any number | -0.0005 |
| eluAlpha | Alpha value for elu, when used | Any number | 1 |

\* When constructing convolutional networks, one of the rectified linear unit activation functions may be more suitable.

##### Examples
```javascript
net = new Network({activation: "sigmoid"})
net = new Network({activation: "lrelu", lreluSlope: -0.0005})
net = new Network({activation: "elu", eluAlpha: 1})
net = new Network({activation: x => x})
```
You can set your own activation functions in the JavaScript version (but not the WebAssebly version). The functions are given as parameters:
- The sum of the previous layer's activations and the neuron's bias
- If the function should calculate the prime (during backprop) - boolean
- A reference to the neuron/filter being activated (in pool layers, the reference is to the net).

The network is bound as the function's scope, meaning you can access its data through ```this```.
The function needs to return a single number.

### Regularization
|  Attribute | What it does | Available Configurations | Default value |
|:-------------:| :-----:| :-----:| :---: |
| dropout | Probability a neuron will **not** be dropped | Any number, or false to disable (equivalent to 1) | 1 (disabled) |
| l2 | L2 regularization strength | any number, or true (which sets it to 0.001) | undefined |
| l1 | L1 regularization strength | any number, or true (which sets it to 0.005) | undefined |
| maxNorm | Max norm threshold | any number, or true (which sets it to 1000) | undefined |

##### Examples
```javascript
net = new Network({dropout: 0.5})
net = new Network({l1: 0.005})
net = new Network({l2: 0.001})
net = new Network({maxNorm: 1000})
```

You can do elastic net regularization by including both l1 and l2 regularization configs.

### Weights Initialization
You can specify configuration options for weights initialization via the weightsConfig object. The values below go in the weightsConfig object.

|  Attribute | What it does | Available Configurations | Default value |
|:-------------:| :-----:| :-----:| :---: |
| distribution | The distribution of the weights values in a neuron | uniform, gaussian, xavierNormal, lecunUniform, lecunNormal, xavierUniform | xavierUniform |
| limit | Used with uniform to dictate the maximum absolute value of the weight. 0 centered. | Any number | 0.1 |
| mean| Used with gaussian to dictate the center value of the middle of the bell curve distribution | Any number | 0 |
| stdDeviation | Used with gaussian to dictate the spread of the data. | Any number | 0.05 |

###### Xavier Normal
This samples weights from a gaussian distribution with variance: ``` 2 / (fanIn + fanOut)```

###### Xavier Uniform
This samples weights from a uniform distribution with limit: ``` sqrt(6 / (fanIn + fanOut)) ```

###### Lecun Normal
This samples weights from a gaussian distribution with variance: ``` 1 / fanIn```

###### Lecun Uniform
This samples weights from a uniform distribution with limit: ``` sqrt(3 / fanIn) ```

Xavier Normal/Uniform falls back to Lecun Normal/Uniform on the last layer, where there is no fanOut to use.

You can set custom weights distribution functions. They are given as parameters the number of weights needed and the weightsConfig object, additionally containing a layer's fanIn and/or fanOut. It must return an array of weights.

##### Examples
```javascript
net = new Network({weightsConfig: {
    distribution: "uniform",
    limit: 0.1
}})
net = new Network({weightsConfig: {
    distribution: "gaussian",
    mean: 0,
    stdDeviation: 1
}})
net = new Network({weightsConfig: {distribution: "xavierNormal"}})
net = new Network({weightsConfig: {distribution: "lecunUniform"}})
net = new Network({weightsConfig: {distribution: n => [...new Array(n)]}})
```

### FCLayer (Fully connected layer)
```FCLayer(int num_neurons[, object configs])```

The first parameter, an integer, is for how many neurons the layer will have. The second, is an object where the configurations below go.

|  Attribute | What it does | Available Configurations | Default value |
|:-------------:| :-----:| :-----:| :---: |
| activation | Activation function to use (see below notes) | false, sigmoid, tanh, relu, lrelu, rrelu, lecuntanh, elu, _function_ | The net's activation function |

##### Examples
```javascript
// 20 neurons, sigmoid activation function
net = new Network({
    activation: "sigmoid",
    layers: [..., new FCLayer(20), ...]
})
// 100 neurons, no activation function
net = new Network({
    activation: "sigmoid",
    layers: [..., new FCLayer(20, {activation: false}), ...]
})
// 15 neurons, tanh activation function
net = new Network({
    activation: "sigmoid",
    layers: [..., new FCLayer(20, {activation: "tanh"}), ...]
})
```

Softmax is used by default on the last layer. Activation configurations are therefore not used there.

### ConvLayer (Convolutional layer)
```ConvLayer(int num_filters[, object configs])```

The first parameter, an integer, is for how many filters to use in the layer. The second, is an object where the configurations below go.

|  Attribute | What it does | Available Configurations | Default value |
|:-------------:| :-----:| :-----:| :---: |
| filterSize | The spacial dimensions of each filter's weights. Giving 3 creates a 3x3 map in each channel | Any odd number | 3 |
| zeroPadding | How much to pad the input map with zero values. Default value keeps output map dimension the same as the input | Any number | Rounded down filterSize/2, keeping dimensions the same (equivalent to 'SAME' in TensorFlow) |
| stride | How many values to move between convolutions | Any number | 1 |
| activation | Activation function to use (see below notes) | false, sigmoid, tanh, relu, lrelu, rrelu, lecuntanh, elu, _function_ | false |

You need to make sure you configure the hyperparameters correctly (you'll be told if something's wrong), to have the filter convolve across all input values and avoiding otherwise decimal outgoing spacial dimensions.

### Tip
You can calculate the spacial dimensions of a convolution layer's outgoing activation volume with the following formula:
```
size out = (size in - filter size + 2 * zero padding) / stride + 1
```

#### About the activation function
Sometimes, you may read about ReLU layers being used, and such. However, it made much more sense in the implementation to just do the activation in the ConvLayer, as it would be more computationally efficient than using a dedicated layer. Therefore there are no such 'activation' layers, as you just specify the activation in the network configs.

By default, the Conv layer activation is turned off (similar to configuring with ```false```). However, you can configure it via the activation key. You can provide a custom function (in the JavaScript only version), or use the string name of an existing activation function, similar to configuring the network activation. (See above)

##### Examples
```javascript
net = new Network({
    activation: "relu",
    layers: [..., new ConvLayer(8, {filterSize: 3, activation: false}, ...)] // Conv layer will use no activation
})
net = new Network({
    activation: "relu",
    layers: [..., new ConvLayer(8, {zeroPadding: 0, stride: 2}), ....] // Conv layer will use ReLU activation
})
net = new Network({
    activation: "relu",
    layers: [..., new ConvLayer(8, {filterSize: 5, activation: "elu"}), ....] // Conv layer will use eLU
})
```

### PoolLayer
```PoolLayer(int span[, object configs])```

The first parameter, an integer, is for the size of area to pool across (Eg, 2, for a 2x2 area). The default value is 2.
The second is an object where the configurations below go.

|  Attribute | What it does | Available Configurations | Default value |
|:-------------:| :-----:| :-----:| :---: |
| stride | How many values to move between pooling | Any number | layer.size |
| activation | Activation function to use (see below notes) | false, sigmoid, tanh, relu, lrelu, rrelu, lecuntanh, elu, function | false |

The pooling operation used is max pool.

##### Examples

```javascript
net = new Network({
    layers: [
        new FCLayer(784), // 28 x 28
        new ConvLayer(8, {filterSize: 3, zeroPadding: 1, stride: 1}), // 28 x 28
        new PoolLayer(2, {stride: 2}), // 14 x 14
        new FCLayer(196),
        new FCLayer(50),
        new FCLayer(10)
    ],
    activation: "lrelu",
    updateFn: "adagrad",
    learningRate: 0.05
})
```

### Tip
When using Pool layers following a convolutional layer, it is more computationally efficient to perform the activation function in the pool layer instead of doing it in the conv layer. This is only true for increasing functions (the included activation functions are ok). The logic behind it is that max pooling will pick the highest value out of a set of values. It makes sense to only compute the activation of a single value instead of a group of them, as the pooling choice would not be affected by an increasing function.

For example, using the following set-up compared to the one above, the training was about 18% (average of 4) faster (with nearly identical results). This optimization may be even more dramatic for Pool layers with bigger sizes.
```javascript
net = new Network({
    layers: [
        new FCLayer(784),
        new ConvLayer(8, {filterSize: 3, zeroPadding: 1, stride: 1, activation: false}), // !
        new PoolLayer(2, {stride: 2, activation: "lrelu"}), // !
        new FCLayer(196),
        new FCLayer(50),
        new FCLayer(10)
    ],
    activation: "lrelu",
    updateFn: "adagrad",
    learningRate: 0.05
})
```

## Future plans
---
More and more features will be added, as time goes by, and I learn more. General improvements and optimizations will be added throughout. Breaking changes will be documented. Check the changelog to see the history of added features.

##### Short term
The WebAssembly version, a complete re-write, has just shipped in general release version 3.0. Next, in 3.1, I'll be going through a list of optimizations ideas I've racked up. You can keep track of progress on the dev branch.

##### Long term
Once that is done, I will be experimenting with a WebGL version, using shaders to run computations on the GPU.

## Contributing
---
I am always looking for feedback, suggestions and ideas, especially if something's not right, or it can be improved/optimized.

Pull requests are always welcome. Just make sure the tests all pass and coverage is at (or nearly) at 100%.
To develop, first ```npm install``` the dev dependencies. You can then run ```grunt``` to listen for file changes and transpile, and you can run the mocha tests via ```npm test```, where you can also see the coverage.

To build the WebAssembly version, you will need to be able to use emscripten to compile locally. Check out [this article](https://medium.com/statuscode/setting-up-the-ultimate-webassembly-c-workflow-6484efa3e162) I wrote if you need any help setting it up. Once set up, run ```npm run build``` to set up the environment. Grunt will do the compilation during development.
