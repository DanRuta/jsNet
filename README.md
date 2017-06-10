# Network.js
[![Build Status](https://travis-ci.org/DanRuta/Network.js.svg?branch=master)](https://travis-ci.org/DanRuta/Network.js)&nbsp;&nbsp;&nbsp;&nbsp;[![Coverage Status](https://coveralls.io/repos/github/DanRuta/Network.js/badge.svg?branch=master)](https://coveralls.io/github/DanRuta/Network.js?branch=master)

Network.js is promise based implementation of a (currently) really basic neural network, functional in node as well as the browser. The focus was end user simplicity. 

This project is in its infancy, and more features and optimisations will periodically be added.

##  Usage

I will use [the MNIST dataset](https://github.com/cazala/mnist) in the examples below.

### Constructing
---
A network can be built in a few different ways, 

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
table

### Testing
---
Once the network is trained, you can test it like so:
```javascript
const {training, test} = mnist.set(800, 200)
net.train(training).then(() => net.test(test))
```
The network will log the testing iteration and the error. This also resolves a promise.

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
#### Network
|  attribute | What it does | Available Configurations | Default value |
|:-------------:| :-----:| :-----:| :---: |
| Learning Rate | The speed at which the net will learn. | Any number | 0.2 |
| Activation | Activation function used by neurons | "sigmoid" | "sigmoid" |
| Cost | Cost function to use when printing out the net error | "crossEntropy", "meanSquaredError" | "crossEntropy" |
## Future plans
---
##### Short term
More and more features will be added to this little library, as time goes by, and I learn more. The first few changes will be adding more configuration options, such as activation functions, cost functions, etc. General library improvements and optimisations will be added throughout. Breaking changes will be documented.
##### Long term
Keeping the same level of ease of use in mind, I will add Conv and Pool layers.
Once that is done, and there is a decent selection of configurations, and features, I will be focusing all my attention to some harcore optimisations.

## Contributing
---
Pull requests are always welcome, as long as the tests all pass and coverage is at (or nearly) at 100%.
To develop, first ```npm install``` the dev dependencies. You can then run ```grunt``` to listen for file changes and transpile, and you can run the mocha tests via ```npm test```, where you can also see the coverage.


