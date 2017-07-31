# Network.js
[![Build Status](https://travis-ci.org/DanRuta/Network.js.svg?branch=master)](https://travis-ci.org/DanRuta/Network.js)&nbsp;&nbsp;&nbsp;&nbsp;[![Coverage Status](https://coveralls.io/repos/github/DanRuta/Network.js/badge.svg?branch=master)](https://coveralls.io/github/DanRuta/Network.js?branch=master)

Network.js is promise based implementation of a (currently) basic neural network, functional in nodejs as well as the browser. The focus was end user ease of use. 

This project is in its infancy, and more features and optimisations will periodically be added.

##  Usage

I will use [the MNIST dataset](https://github.com/cazala/mnist) in the examples below.

## Demo
https://ai.danruta.co.uk - Interactive MNIST Digit classifier

### Constructing
---
A network can be built in a few different ways: 

##### 1 
With absolutely no parameters, and it will figure out an appropriate structure once you pass it some data.
```javascript
const net = new Network()
```
##### 2
By giving a list of numbers, and the network will configure some layers with that many neurons.
```javascript
const net = new Network({
    layers: [784, 100, 10]
})
```
##### 3
By specifying just a list of the type of layer you'd like to use. The number of neurons are calculated from input/output sizes and the hidden layers are then given some appropriate sizes.
(This will be more useful when more layer types are added. Currently only a fully connected layer is implemented, named "Layer", for now)
```javascript
const net = new Network({
    layers: [Layer, Layer, Layer]
})
```
##### 4
Or you can fully configure the layers by constructing them. The layers currently can only have its number of neurons configured.
```javascript
const net = new Network({
    layers: [new Layer(784), new Layer(100), new Layer(10)]
})
```
The default values are values I have found produced good results in the datasets I used for testing. You will, of course, get best results by hand picking configurations appropriate to your own data set.

### Training
----


The data structure must (currently) be an object with key ```input``` having an array of numbers, and key ```expected``` or ```output``` holding the expected output of the network. For example, the following are both valid inputs for both training and testing.
```javascript
{input: [1,0,0.2], expected: [1, 2]}
{input: [1,0,0.2], output: [1, 2]}
```
You train the network by passing a set of data. The network will log to the console the error and epoch number, after each epoch.
```javascript
const {training} = mnist.set(800, 200) // Get the training data from the mnist library, linked above

const net = new Network()
net.train(training) // This on its own is enough
.then(() => console.log("done")) // This resolves a promise, meaning you can add further code here (eg testing)
```
You can also provide a callback, which will get called after each iteration (Maybe updating a graph?). The callback is passed how many iterations have passed, the error, and the input data for that iteration. 

### Testing
---
Once the network is trained, you can test it like so:
```javascript
const {training, test} = mnist.set(800, 200)
net.train(training).then(() => net.test(test))
```
The network will log the testing iteration and the error. This also resolves a promise, with the average test error percentage.

### Exporting
---
Layer and weights data is exported as a JSON object.
```javascript
const data = trainedNet.toJSON()
```

### Importing
---
```javascript
const freshNetwork = new Network()
freshNetwork.fromJSON(data)
```

### Trained usage
---
Once the network has been trained, tested and imported into your page, you can use it via the ```forward``` function.
```javascript
const userInput = [1,0,1,0,0.5] // Example input
const netResult = net.forward(userInput)
```
This will return an array of the activations in the output layer.
You can run them through a softmax function by using NetMath.
```javascript
const normalizedResults = NetMath.softmax(netResult)
```

## Configurations
---
### Network
|  Attribute | What it does | Available Configurations | Default value |
|:-------------:| :-----:| :-----:| :---: |
| learningRate | The speed at which the net will learn. | Any number | 0.2 (see below for exceptions) |
| cost | Cost function to use when printing out the net error | crossEntropy, meanSquaredError | crossEntropy |

##### Examples
```javascript
net = new Network({learningRate: 0.2})
net = new Network({cost: "crossEntropy"})
```
Learning rate is 0.2 by default, except when using the following configurations:

| Modifier| Type | Default value| 
|:-------------:| :-----: | :-----: |
| RMSProp | adaptiveLR | 0.001 |
| adam | adaptiveLR | 0.01 |
| adadelta | adaptiveLR | undefined |
| tanh, lecuntanh | activation | 0.001 |
| relu, lrelu, rrelu, elu | activation | 0.01 |

### Adaptive Learning Rate
|  Attribute | What it does | Available Configurations | Default value |
|:-------------:| :-----:| :-----:| :---: |
| adaptiveLR | The function used for updating the weights/bias. Null just sets the network to update the weights without any changes to learning rate. | null, gain, adagrad, RMSProp, adam , adadelta| null |
| rmsDecay | The decay rate for RMSProp | Any number | 0.99 |
| rho | Momentum for Adadelta | Any number | 0.95 |

##### Examples
```javascript
net = new Network({adaptiveLR: "adagrad"})
net = new Network({adaptiveLR: "RMSProp", rmsDecay: 0.99})
net = new Network({adaptiveLR: "adadelta", rho: 0.95})
```
### Activation Function
|  Attribute | What it does | Available Configurations | Default value |
|:-------------:| :-----:| :-----:| :---: |
| activation | Activation function used by neurons | sigmoid, tanh, relu, lrelu, rrelu, lecuntanh, elu | sigmoid |
| lreluSlope | Slope for lrelu | Any number | 0.99 |
| eluAlpha | Alpha value for ELU | Any number | 1 |

##### Examples
```javascript
net = new Network({activation: "sigmoid"})
net = new Network({activation: "lrelu", lreluSlope: 0.99})
net = new Network({activation: "elu", eluAlpha: 1})
```
### Regularization
|  Attribute | What it does | Available Configurations | Default value |
|:-------------:| :-----:| :-----:| :---: |
| dropout | Probability a neuron will be dropped | Any number, or false to disable (equivalent to 1) | 0.5 |
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
| distribution | The distribution of the weights values in a neuron | uniform, gaussian, xavierNormal, lecunUniform, lecunNormal, xavierUniform | uniform |
| limit | Used with uniform to dictate the maximum absolute value of the weight. 0 centered. | Any number | 0.1 |
| mean| Used with gaussian to dictate the center value of the middle of the bell curve distribution | Any number | 0 |
| stdDeviation | Used with gaussian to dictate the spread of the data. | Any number | 0.05 |

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
```

Xavier Normal/Uniform falls back to Lecun Normal/Uniform on the last layer, where there is no fanOut to use.

## Future plans
---
More and more features will be added to this little library, as time goes by, and I learn more. General library improvements and optimisations will be added throughout. Breaking changes will be documented.

##### Short term
 The first few changes have been adding more configuration options, such as activation functions, cost functions, regularization, adaptive learning, weights init, etc. Check the changelog for details. Next up is mini batch SGD and some general library improvement ideas I've logged along the way.

##### Mid term
Conv, Pool and BatchNorm layers.

##### Long term
Once that is done, and there is a decent selection of configurations, and features, I will be focusing all my attention to some novel, hardcore optimisations, as part of my final year university project. Afterwards, I plan to incorporate other network types, eg LSTM networks.

## Contributing
---
Always looking for feedback, suggestions and ideas, especially if something's not right, or it can be improved/optimized.
Pull requests are always welcome. Just make sure the tests all pass and coverage is at (or nearly) at 100%.
To develop, first ```npm install``` the dev dependencies. You can then run ```grunt``` to listen for file changes and transpile, and you can run the mocha tests via ```npm test```, where you can also see the coverage.
