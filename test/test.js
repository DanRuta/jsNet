"use strict"

const chaiAsPromised = require("chai-as-promised")
const chai = require("chai")
const assert = chai.assert
const expect = chai.expect
const sinonChai = require("sinon-chai")
const sinon = require("sinon")
chai.use(sinonChai)
chai.use(chaiAsPromised);

require("../dist/Network.concat.js")

describe("Tests", () => {
    it("Network is loaded", () => expect(Network).to.not.be.undefined)
    it("Layer is loaded", () => expect(Layer).to.not.be.undefined)
    it("Neuron is loaded", () => expect(Neuron).to.not.be.undefined)
    it("NetMath is loaded", () => expect(NetMath).to.not.be.undefined)
})

describe("Network", () => {

    describe("Constructor", () => {

        describe("Config defaults", () => {

            let net
            beforeEach(() => net = new Network())

            it("Defaults the activation to sigmoid and sets the function from NetMath to net.activation", () => {
                expect(net.activation).to.equal(NetMath.sigmoid)
            })

            it("Defaults the learning rate to 0.2", () => {
                expect(net.learningRate).to.equal(0.2)
            })

            it("Sets the net.epochs to 0", () => {
                expect(net.epochs).to.equal(0)
            })

            it("Sets the net iterations to 0", () => {
                expect(net.iterations).to.equal(0)
            })

            it("Defaults the cost function to 'crossEntropy'", () => {
                expect(net.cost).to.equal(NetMath.crossEntropy)
            })
        })

        it("Can create a new Network with no parameters", () => expect(new Network()).instanceof(Network))

        it("Sets the network state to not-defined when defining with no layers parameter", () => {
            const net = new Network()
            expect(net.state).to.equal("not-defined")
        })

        it("Assigns a list of constructed layers as the net's layers", () => {
            const layers = [new Layer(), new Layer()]
            const net = new Network({layers: layers})
            expect(net.layers).to.eq(layers)
        })

        it("Sets the network state to initialised (after the initLayers function) when layers are given and they are all constructed", () => {
            const layers = [new Layer(), new Layer()]
            const net = new Network({layers: layers})
            expect(net.state).to.equal("initialised")
        })

        it("Constructs a list of Layers with sizes respective to numbers when giving a list of numbers as layers parameter", () => {
            const net = new Network({layers: [2,4,1]})
            expect(net.layers.every(layer => layer instanceof Layer)).to.be.true
        })

        it("Sets the network state to initialised (after the initLayers function) when building layers with integers", () => {
            const net = new Network({layers: [2,4,1]})
            expect(net.state).to.equal("initialised")
        })

        it("Throws an error when mixing constructed and non constructed Layers", () => {
            const layers = [Layer, new Layer()]
            assert.throw(() => new Network({layers}), "There was an error constructing from the layers given.")
        })

        it("Saves the non-constructed layers in a property definedLayers, when using just layer classes", () => {
            const layers = [Layer, Layer]
            const net = new Network({layers})
            expect(net.definedLayers).to.eq(layers)
        })
    })

    describe("initLayers", () => {

        const netThis = {} 
        const net = new Network()

        beforeEach(() => sinon.spy(net, "joinLayer"))
        afterEach(() => net.joinLayer.restore())

        it("Does nothing when the net's state is already initialised", () => {
            const netThis = {state: "initialised"}
            net.initLayers.bind(netThis)()

            expect(netThis.state).to.equal("initialised")
            expect(net.joinLayer).to.not.be.called
        })

        it("Calls the joinLayer function with each layer when state is constructed", () => {
            const layer1 = new Layer(1)
            const layer2 = new Layer(2)
            const netThis = {state: "constructed", layers: [layer1, layer2], joinLayer: net.joinLayer}
            net.initLayers.bind(netThis)()

            expect(netThis.state).to.equal("initialised")
            expect(net.joinLayer).to.have.been.calledTwice
            expect(net.joinLayer).to.have.been.calledWith(layer1)
            expect(net.joinLayer).to.have.been.calledWith(layer2)
        })

        it("Calculates reasonable sizes for layers, when state is defined (with a small net)", () => {
            const netThis = {state: "defined", definedLayers: [Layer, Layer, Layer, Layer], joinLayer: net.joinLayer}
            net.initLayers.bind(netThis, 3, 2)()
            expect(netThis.layers.map(layer => layer.size)).to.deep.equal([3, 5, 3, 2])
            expect(net.joinLayer.callCount).to.equal(4)
        })

        it("Calculates reasonable sizes for layers, when state is defined (with a big net)", () => {
            const netThis = {state: "defined", definedLayers: [Layer, Layer, Layer, Layer, Layer, Layer], joinLayer: net.joinLayer}
            net.initLayers.bind(netThis, 784, 10)()

            expect(netThis.state).to.equal("initialised")
            expect(net.joinLayer.callCount).to.equal(6)
            expect(netThis.layers.map(layer => layer.size)).to.deep.equal([784, 417, 315, 214, 112, 10])
        })

        it("Creates three Layers when state is not-defined. First and last layer sizes respective to input/output, middle is in-between", () => {
            const netThis = {state: "not-defined", joinLayer: net.joinLayer, layers: []}
            net.initLayers.bind(netThis, 3, 2)()

            expect(netThis.state).to.equal("initialised")
            expect(netThis.layers.map(layer => layer.size)).to.deep.equal([3, 5, 2])
            expect(net.joinLayer).to.have.been.calledThrice
        })

        it("Creates three Layers when state is not-defined. (the same, but with big net)", () => {
            const netThis = {state: "not-defined", joinLayer: net.joinLayer, layers: []}
            net.initLayers.bind(netThis, 784, 10)()
            expect(netThis.state).to.equal("initialised")
            expect(net.joinLayer).to.have.been.calledThrice
            expect(netThis.layers.map(layer => layer.size)).to.deep.equal([784, 204, 10])
        })

        it("Sets the network's activation function to the layers", () => {
            const layer1 = new Layer(2)
            const layer2 = new Layer(3)
            const net = new Network({layers: [layer1, layer2]})

            expect(layer1.activation).to.equal(NetMath.sigmoid)
            expect(layer2.activation).to.equal(NetMath.sigmoid)
        })
    })

    describe("joinLayer", () => {

        let net, layer1, layer2

        beforeEach(() => {
            net = new Network()
            layer1 = new Layer(2)
            layer2 = new Layer(3)
        }) 


        it("Does nothing to a single layer network", () => {
            net.layers = [layer1]
            net.joinLayer(layer1)
            expect(layer1.nextLayer).to.be.undefined
            expect(layer1.prevLayer).to.be.undefined
        })


        it("Assigns layer2 to layer1's next layer", () => {
            net.layers = [layer1, layer2]
            net.joinLayer(layer1, 0)
            net.joinLayer(layer2, 1)
            expect(layer1.nextLayer).to.equal(layer2)
        })

        it("Assigns layer1 to layer2's prev layer", () => {
            net.layers = [layer1, layer2]
            net.joinLayer(layer1, 0)
            net.joinLayer(layer2, 1)
            expect(layer2.prevLayer).to.equal(layer1)
        })
    })

    describe("forward", () => {

        it("Throws an error if called when the network is not in an 'initialised' state", () => {
            const net = new Network()
            expect(net.forward.bind(net)).to.throw("The network layers have not been initialised.")
        })

        it("Throws an error if called with no data", () => {
            const net = new Network({layers: [new Layer(3)]})
            expect(net.forward.bind(net)).to.throw("No data passed to Network.forward()")
        })

        it("Logs a warning if the given input array length does not match input length", () => {
            sinon.spy(console, "warn")
            const net = new Network({layers: [new Layer(3)]})
            net.forward([1,2,3,4])
            expect(console.warn).to.have.been.calledWith("Input data length did not match input layer neurons count.")
            console.warn.restore()
        })


        it("Sets the activation of all input neurons to the value of the input", () => {
            const net = new Network({layers: [new Layer(3)]})
            net.forward([1,2,3])
            expect(net.layers[0].neurons[0].activation).to.equal(1)
            expect(net.layers[0].neurons[1].activation).to.equal(2)
            expect(net.layers[0].neurons[2].activation).to.equal(3)
        })

        it("Calls every layer's (except the first's) forward function", () => {
            const layer1 = new Layer(3)
            const layer2 = new Layer(4)
            const layer3 = new Layer(7)
            const net = new Network({layers: [layer1, layer2, layer3]})

            sinon.spy(layer1, "forward")
            sinon.spy(layer2, "forward")
            sinon.spy(layer3, "forward")

            net.forward([1,2,3])
            expect(layer1.forward).to.have.not.been.called
            expect(layer2.forward).to.have.been.called
            expect(layer3.forward).to.have.been.called
        })

        it("Returns the activations of the neurons in the last layer", () => {
            const layer1 = new Layer(3)
            const layer2 = new Layer(4)
            const layer3 = new Layer(7)
            const net = new Network({layers: [layer1, layer2, layer3]})

            const result = net.forward([1,2,3])
            const activations = net.layers[2].neurons.map(n => n.activation)

            expect(result).to.deep.equal(activations)
        })
    })

    describe("backward", () => {
        it("Throws an error if called with no data", () => {
            const net = new Network({layers: [new Layer(3)]})
            expect(net.backward.bind(net)).to.throw("No data passed to Network.backward()")
        })

        it("Logs a warning if the given 'expected' array length does not match output neurons count", () => {
            sinon.spy(console, "warn")
            const net = new Network({layers: [new Layer(3), new Layer(2), new Layer(4)]})
            net.backward([1,2,3])
            expect(console.warn).to.have.been.calledWith("Expected data length did not match output layer neurons count.")
            console.warn.restore()
        })

        it("Calls the backward() function of every layer, in reverse order", () => {

            const layer2 = new Layer(2)
            const layer3 = new Layer(3)
            const net = new Network({layers: [new Layer(1), layer2, layer3]})

            const l2Spy = sinon.spy(layer2, "backward")
            const l3Spy = sinon.spy(layer3, "backward")

            net.backward([1,2,3])

            expect(l2Spy).to.have.been.called
            expect(l3Spy).to.have.been.called
            sinon.assert.callOrder(l3Spy, l2Spy)
        })
    })

    describe("resetDeltaWeights", () => {
        it("Sets the delta weights of all neurons to 0", () => {
            const layer1 = new Layer(2)
            const layer2 = new Layer(2)
            const net = new Network({layers: [layer1, layer2]})
            layer2.neurons.forEach(neuron => neuron.deltaWeights = [1,1])

            net.resetDeltaWeights()
            expect(layer2.neurons[0].deltaWeights).to.deep.equal([0,0])
            expect(layer2.neurons[1].deltaWeights).to.deep.equal([0,0])
        })
    })

    describe("applyDeltaWeights", () => {
        it("Increments the weights of all neurons with their respective deltas (when learning rate is 1)", () => {
            const layer1 = new Layer(2)
            const layer2 = new Layer(3)
            const net = new Network({learningRate: 1, layers: [layer1, layer2]})

            layer2.neurons.forEach(neuron => neuron.weights = [0.25, 0.25])
            layer2.neurons.forEach(neuron => neuron.deltaWeights = [0.5, 0.5])

            net.applyDeltaWeights()
            
            expect(layer1.weights).to.be.undefined
            expect(layer2.neurons[0].weights).to.deep.equal([0.75, 0.75])
            expect(layer2.neurons[1].weights).to.deep.equal([0.75, 0.75])
            expect(layer2.neurons[2].weights).to.deep.equal([0.75, 0.75])
        })

        it("Increments the bias of all neurons with their deltaBias", () => {
            const layer1 = new Layer(2)
            const layer2 = new Layer(3)
            const net = new Network({learningRate: 1, layers: [layer1, layer2]})

            layer2.neurons.forEach(neuron => neuron.bias = 0.25)
            layer2.neurons.forEach(neuron => neuron.deltaBias = 0.5)

            net.applyDeltaWeights()

            expect(layer1.bias).to.be.undefined
            expect(layer2.neurons[0].bias).to.equal(0.75)
            expect(layer2.neurons[1].bias).to.equal(0.75)
            expect(layer2.neurons[2].bias).to.equal(0.75)
        })
    })

    describe("toJSON", () => {
        const layer1 = new Layer(2)
        const layer2 = new Layer(3)
        const net = new Network({layers: [layer1, layer2], activation: "sigmoid"})

        it("Exports the correct number of layers", () => {
            const json = net.toJSON()
            expect(json.layers).to.not.be.undefined
            expect(json.layers).to.have.lengthOf(2)
        })

        it("Exports weights correctly", () => {

            layer2.neurons.forEach(neuron => neuron.weights = neuron.weights.map(w => 1))

            const json = net.toJSON()
            expect(json.layers[1].neurons).to.not.be.undefined
            expect(json.layers[1].neurons).to.have.lengthOf(3)
            expect(json.layers[1].neurons[0].weights).to.deep.equal([1,1])
            expect(json.layers[1].neurons[1].weights).to.deep.equal([1,1])
            expect(json.layers[1].neurons[2].weights).to.deep.equal([1,1])
        })

        it("Exports bias correctly", () => {
            layer2.neurons.forEach(neuron => neuron.bias = 1)

            const json = net.toJSON()
            expect(json.layers[1].neurons[0].bias).to.equal(1)
            expect(json.layers[1].neurons[1].bias).to.equal(1)
            expect(json.layers[1].neurons[2].bias).to.equal(1)  
        })
    })

    describe("fromJSON", () => {
        const testData = {
            layers: [
                {
                    neurons: [{},{}]
                },
                {
                    neurons: [
                        {bias: 1, weights: [1,1]},
                        {bias: 1, weights: [1,1]},
                        {bias: 1, weights: [1,1]}
                    ]
                }
            ]
        }

        it("Throws an error if no data is given", () => {
            const net = new Network()
            expect(net.fromJSON.bind(net)).to.throw("No JSON data given to import.")
            expect(net.fromJSON.bind(net, null)).to.throw("No JSON data given to import.")
        })

        it("Clears out the old layers", () => {
            const net = new Network({layers: [2,5,3,6]})

            net.fromJSON(testData)
            expect(net.layers).to.have.lengthOf(2)
        })

        it("Calls the iinitLayers function to join the layers together", () => {
            const net = new Network()
            sinon.spy(net, "initLayers")

            net.fromJSON(testData)

            expect(net.initLayers).to.be.called
        })

        it("Sets the net state to 'constructed'", () => {
            const net = new Network()
            const stub = sinon.stub(net, "initLayers")
            net.state = "initialised" // just to make sure
            net.fromJSON(testData)
            expect(net.state).to.equal("constructed")
        })
    })

    describe("train", () => {

        let net

        beforeEach(() => {
            net = new Network({layers: [2, 3, 2]})
            sinon.stub(net, "forward").callsFake(() => [1,1])
            sinon.stub(net, "backward")
            sinon.stub(net, "resetDeltaWeights")
            sinon.stub(net, "applyDeltaWeights")
            sinon.stub(net, "initLayers")
            sinon.stub(NetMath, "crossEntropy")
            sinon.stub(console, "log") // Get rid of output spam
        }) 

        afterEach(() => {
            net.forward.restore()
            net.backward.restore()
            net.resetDeltaWeights.restore()
            net.applyDeltaWeights.restore()
            net.initLayers.restore()
            NetMath.crossEntropy.restore()
            console.log.restore()
        }) 

        const testData = [
            {input: [0,0], expected: [0, 0]},
            {input: [0,1], expected: [0, 1]},
            {input: [1,0], expected: [1, 0]},
            {input: [1,1], expected: [1, 1]}
        ]
        const badTestData = [
            {input: [0,0], expected: [0, 0]},
            {input: [0,1], expected: [0, 1]},
            {input: [1,0], NOPE: [1, 0]},
            {input: [1,1], expected: [1, 1]}
        ]
        const testDataWithOutput = [
            {input: [0,0], output: [0, 0]},
            {input: [0,1], output: [0, 1]},
            {input: [1,0], output: [1, 0]},
            {input: [1,1], output: [1, 1]}
        ]
        const testDataWithMixedExpectedOutput = [
            {input: [0,0], output: [0, 0, 1, 0, 1]},
            {input: [0,1], expected: [0, 1, 0, 1, 0]},
            {input: [1,0], output: [1, 0, 1, 0, 1]},
            {input: [1,1], expected: [1, 1, 0, 0, 1]}
        ]

        it("Returns a promise", () => {
            expect(net.train(testData)).instanceof(Promise)
        })

        it("Rejects the promise when no data is given", () => {
            return expect(net.train()).to.be.rejectedWith("No data provided")
        })

        it("Rejects the promise if some data does not have the key 'input' and 'expected'/'output'", () => {
            return expect(net.train(badTestData)).to.be.rejectedWith("Data set must be a list of objects with keys: 'input' and 'expected' (or 'output')")
        })

        it("Resolves the promise when you give it data", () => {
            return expect(net.train(testData)).to.be.fulfilled
        })

        it("Accepts 'output' as an alternative name for expected values", () => {
            return expect(net.train(testDataWithOutput)).to.be.fulfilled  
        })

        it("Does one iteration when not passing any config data", () => {
            net.epochs = 0
            return net.train(testData).then(() => {
                expect(net.epochs).to.equal(1)
            })
        })

        it("Increments the net.iterations by 4 when doing 1 epoch on a data set of 4 items", () => {
            return net.train(testData).then(() => {
                expect(net.iterations).to.equal(4)
            })
        })

        it("Does five epochs when passing epochs = 5 as config", () => {
            return net.train(testData, {epochs:5}).then(() => {
                expect(net.epochs).to.equal(5)
            })
        })

        it("Calls the resetDeltaWeights function for each iteration", () => {
            return net.train(testData).then(() => {
                expect(net.resetDeltaWeights.callCount).to.equal(4)
            })
        })

        it("Calls the forward function for each iteration", () => {
            return net.train(testData).then(() => {
                expect(net.forward.callCount).to.equal(4)
            })
        })

        it("Calls the backward function for each iteration", () => {
            return net.train(testData).then(() => {
                expect(net.backward.callCount).to.equal(4)
            })
        })

        it("Calls the applyDeltaWeights function for each iteration", () => {
            return net.train(testData, {}, true).then(() => {
                expect(net.applyDeltaWeights.callCount).to.equal(4)
            })
        })

        it("Calls the cost function for each iteration", () => {
            sinon.spy(net, "cost")
            return net.train(testData).then(() => {
                expect(net.cost.callCount).to.equal(4)
                net.cost.restore()
            })
        })

        it("Calls a given callback with an object containing keys: 'iterations', 'error' and 'input', for each iteration", () => {
            sinon.stub(console, "warn")

            return net.train(testData, {callback: console.warn}).then(() => {
                expect(console.warn).to.have.been.called
                expect(console.warn.callCount).to.equal(4)
                expect(console.warn).to.have.been.calledWith(sinon.match.has("iterations"))
                expect(console.warn).to.have.been.calledWith(sinon.match.has("error"))
                expect(console.warn).to.have.been.calledWith(sinon.match.has("input"))
                console.warn.restore()
            })
        })

        it("Calls the initLayers function when the net state is not 'initialised'", () => {
            const network = new Network()
            sinon.stub(network, "forward")
            sinon.spy(network, "initLayers")

            return network.train(testData).then(() => {
                expect(network.initLayers).to.have.been.called
            })
        })

        it("Calls the initLayers function with the length of the first input and length of first expected", () => {
            const network = new Network()
            sinon.stub(network, "forward")
            sinon.spy(network, "initLayers")

            return network.train(testData).then(() => {
                expect(network.initLayers).to.have.been.calledWith(2, 2)
                network.initLayers.restore()
            })

        })

        it("Also calls the initLayers function correctly when the first item in the dataSet is named as output", () => {
            const network = new Network({layers: [Layer, Layer, Layer]})
            sinon.stub(network, "forward")
            sinon.spy(network, "initLayers")

            return network.train(testDataWithMixedExpectedOutput).then(() => {
                expect(network.initLayers).to.have.been.calledWith(2,5)
                network.initLayers.restore()
            })
        })

        it("Logs to the console once for each epoch", () => {
            return net.train(testData, {epochs: 3}).then(() => {
                expect(console.log.callCount).to.equal(3)
            })
        })
    })

    describe("test", () => {

        let net
        beforeEach(() => {
            net = new Network({layers: [2,4,3]})
            net.cost = (x,y) => x+y
            sinon.spy(net, "cost")
            sinon.stub(net, "forward")
        }) 

        afterEach(() => {
            net.cost.restore()
        })
        after(() => {
            net.forward.restore()
        }) 

        const testData = [
            {input: [0,0], expected: [0, 0]},
            {input: [0,1], expected: [0, 1]},
            {input: [1,0], expected: [1, 0]},
            {input: [1,1], expected: [1, 1]}
        ]
        const testDataOutput = [
            {input: [0,0], output: [0, 0]},
            {input: [0,1], expected: [0, 1]},
            {input: [1,0], expected: [1, 0]},
            {input: [1,1], expected: [1, 1]}
        ]


        it("Returns a promise", () => {
            expect(net.test(testData)).instanceof(Promise)
        })  

        it("Rejects the promise when no data is given", () => {
            return expect(net.test()).to.be.rejectedWith("No data provided")
        })

        it("Calls the network forward function for each item in the test set", () => {
            return net.test(testData).then(() => {
                expect(net.forward.callCount).to.equal(4)
            })
        })

        it("Resolves with a number, indicating error", () => {
            return net.test(testData).then((result) => {
                expect(typeof result).to.equal("number")
            })
        })

        it("Calls the cost function for each iteration", () => {
            return net.test(testData).then(() => {
                expect(net.cost.callCount).to.equal(4)
            })
        })

        it("Accepts test date with output key instead of expected", () => {
            return net.test(testDataOutput).then(() => {
                expect(net.cost.callCount).to.equal(4)
            })
        })

        it("Logs to the console once for each iteration", () => {
            sinon.spy(console, "log")
            return net.test(testData).then(() => {
                expect(console.log.callCount).to.equal(4)
                console.log.restore()
            })
        })
    })
})

describe("Layer", () => {
    describe("Constructor", () => {
        it("Can create a new Layer with no parameters", () => expect(new Layer()).instanceof(Layer))

        it("Creates a list of neurons with the size given as parameter", () => {
            const layer = new Layer(10)
            expect(layer.neurons).to.not.be.undefined
            expect(layer.neurons.length).to.equal(10)
        })
    })

    describe("assignNext", () => {

        const layer = new Layer()

        it("Adds reference of a layer to its nextLayer property", () => {
            const layer2 = new Layer()
            layer2.assignNext(layer)
            expect(layer2.nextLayer).to.equal(layer)
        })
    })

    describe("assignPrev", () => {

        const layer = new Layer()

        it("Adds a reference to a layer to its prevLayer property", () => {
            const layer2 = new Layer()
            layer2.assignPrev(layer)
            expect(layer2.prevLayer).to.equal(layer)
        })

        it("Inits all the neurons in the layer's neurons array", () => {
            const layer1 = new Layer(2)
            const layer2 = new Layer(2)

            sinon.spy(layer2.neurons[0], "init") 
            sinon.spy(layer2.neurons[1], "init") 

            layer2.assignPrev(layer)
            expect(layer2.neurons[0].init).to.have.been.called
            expect(layer2.neurons[1].init).to.have.been.called

            layer2.neurons[0].init.restore()
            layer2.neurons[1].init.restore()
        })
    })

    describe("forward", () => {

        let layer1, layer2

        beforeEach(() => {
            layer1 = new Layer(2)
            layer2 = new Layer(3)
        })

        it("Sets the sum of each neuron to the bias, when all the weights are 0", () => {
            const net = new Network({layers: [layer1, layer2]})

            layer1.neurons.forEach(neuron => neuron.activation = Math.random())
            layer2.neurons.forEach(neuron => neuron.weights = [0,0,0])
            layer2.forward()
            expect(layer2.neurons[0].sum).to.equal(layer2.neurons[0].bias)
            expect(layer2.neurons[1].sum).to.equal(layer2.neurons[1].bias)
            expect(layer2.neurons[2].sum).to.equal(layer2.neurons[2].bias)
        })

        it("Sets the sum of each neuron to the bias + previous layer's activations when weights are 1", () => {
            const net = new Network({layers: [layer1, layer2]})

            layer1.neurons.forEach(neuron => neuron.activation = 2)
            layer2.neurons.forEach(neuron => neuron.weights = [1,1,1])
            layer2.neurons.forEach(neuron => neuron.bias = 1)

            layer2.forward()
            expect(layer2.neurons[0].sum).to.equal(5)
        })

        it("Sets the neurons' activation to the sigmoid of their sums when the config activation function is sigmoid", () => {
            const net = new Network({
                activation: "sigmoid",
                layers: [layer1, layer2]
            })

            net.forward([1,2])
            expect(layer2.neurons[0].activation).to.equal(NetMath.sigmoid(layer2.neurons[0].sum))
            expect(layer2.neurons[1].activation).to.equal(NetMath.sigmoid(layer2.neurons[1].sum))
            expect(layer2.neurons[2].activation).to.equal(NetMath.sigmoid(layer2.neurons[2].sum))
        })
    })

    describe("backward", () => {

        let layer1, layer2, layer3, net

        beforeEach(() => {
            layer1 = new Layer(2)
            layer2 = new Layer(3)
            layer3 = new Layer(4)
            net = new Network({layers: [layer1, layer2, layer3], activation: "sigmoid"})
        })

        it("Sets each neurons' error to expected - their activation, in output layer", () => {
            const net = new Network({layers: [layer1, layer2]})

            layer2.neurons.forEach(neuron => neuron.activation = 0.5)
            net.backward([1,2,3])
            expect(layer2.neurons.map(n => n.error)).to.deep.equal([0.5, 1.5, 2.5])
        })

        it("Sets each neurons' derivative to the layer's activation function prime of the neuron's sum", () => {
            layer2.neurons.forEach(neuron => neuron.sum = 0.5)
            net.backward([1,2,3,4])
            const expectedDerivatives = [...new Array(3)].map(v => NetMath.sigmoid(0.5, true))

            expect(layer2.neurons.map(n => n.derivative)).to.deep.equal(expectedDerivatives)
        })

        it("Sets each neuron's error to the derivative * next layer's neurons' weighted errors (in hidden layers)", () => {
            layer3.neurons.forEach(neuron => {
                neuron.error = 0.5
                neuron.weights = neuron.weights.map(w => 1)
            })
            layer2.neurons.forEach(neuron => neuron.sum = 0.5)
            layer2.backward()

            const expectedErrors = [...new Array(3)].map(v => NetMath.sigmoid(0.5, true) * 2)

            expect(layer2.neurons.map(n => n.error)).to.deep.equal(expectedErrors)
        })

        it("For each neuron, it increments each of its delta weights with the respective weight's neuron's activation * this neuron's error", () => {
            layer2.neurons.forEach(neuron =>neuron.activation = 0.5)
            layer3.neurons.forEach(neuron => neuron.activation = 0.5)
            layer3.backward([1,2,3,4])
            expect(layer3.neurons[0].deltaWeights).to.deep.equal([0.25, 0.25, 0.25])
        })

        it("Increments each neuron's bias with the its error", () => {
            layer3.neurons.forEach(neuron => neuron.activation = 0.5) 
            layer3.backward([1,2,3,4])
            expect(layer3.neurons.map(n => n.deltaBias)).to.deep.equal([0.5, 1.5, 2.5, 3.5])
        })

    })
})

describe("Neuron", () => {

    describe("Constructor", () => {

        let neuron

        beforeEach(() => neuron = new Neuron({bias: 1, weights: [1,2]}))

        it("Can construct a neuron", () => expect(new Neuron()).instanceof(Neuron))

        it("Sets the imported attribute to true when given import data", () => {
            expect(neuron.imported).to.be.true
        })

        it("Sets the neuron weights to the imported weights, when given import data", () => {
            expect(neuron.weights).to.deep.equal([1,2])
        })

        it("Sets the neuron bias to the imported bias, when given imported data", () => {
            expect(neuron.bias).to.equal(1)
        })

        it("Does nothing when there is no imported data given", () => {
            const neuron2 = new Neuron()
            expect(neuron2.imported).to.be.undefined
            expect(neuron2.weights).to.be.undefined
            expect(neuron2.bias).to.be.undefined
        })
    })

    describe("init", () => {
  
        it("Creates a weights array, of length the same as given parameter", () => {
            const neuron = new Neuron()
            neuron.init(5)
            expect(neuron.weights.length).to.equal(5)
        })

        it("Weights are all between -0.1 and +0.1", () => {
            const neuron = new Neuron()
            neuron.init(5)
            expect(neuron.weights.every(w => w>=-0.1 && w<=0.1)).to.be.true
        })

        it("Creates a bias value between -0.1 and +0.1", () => {
            const neuron = new Neuron()
            neuron.init(5)
            expect(neuron.bias).to.not.be.undefined
            expect(neuron.bias).to.be.at.most(0.1)
            expect(neuron.bias).to.be.at.least(-0.1)
        })

        it("Creates an array of delta weights with the same length as the weights array", () => {
            const neuron = new Neuron()
            neuron.init(5)
            expect(neuron.deltaWeights).to.not.be.undefined
            expect(neuron.deltaWeights.length).to.equal(neuron.weights.length)
        })

        it("Sets all the delta weights to 0", () => {
            const neuron = new Neuron()
            neuron.init(5)
            expect(neuron.deltaWeights).to.deep.equal([0,0,0,0,0])
        })

        it("Does not change the weights when the neuron is marked as imported", () => {
            const neuron = new Neuron()
            neuron.imported = true
            neuron.weights = ["test"]
            neuron.init()
            expect(neuron.weights).to.deep.equal(["test"])
        })

        it("Does not change the bias when the neuron is marked as imported", () => {
            const neuron = new Neuron()
            neuron.imported = true
            neuron.weights = ["test"]
            neuron.bias = "test"
            neuron.init()
            expect(neuron.bias).to.equal("test")
        })
    })
})


describe("Netmath", () => {

    describe("Sigmoid", () => {
        it("sigmoid(1.681241237) == 0.8430688214048092", () => {
            expect(NetMath.sigmoid(1.681241237)).to.equal(0.8430688214048092)
        })
        it("sigmoid(0.8430688214048092, true) == 0.21035474941074114", () => {
            expect(NetMath.sigmoid(0.8430688214048092, true)).to.equal(0.21035474941074114)
        })        
    })

    describe("Cross Entropy", () => {
        it("crossEntropy([1,0,0.3], [0,1, 0.8]) == 70.16654147569186", () => {
            expect(NetMath.crossEntropy([1,0,0.3], [0,1, 0.8])).to.equal(70.16654147569186)
        })
    })

    describe("Softmax", () => {
        it("softmax([23, 54, 167, 3]) == [0.0931174089068826, 0.21862348178137653, 0.6761133603238867, 0.012145748987854251]", () => {
            expect(NetMath.softmax([23, 54, 167, 3])).to.deep.equal([0.0931174089068826, 0.21862348178137653, 0.6761133603238867, 0.012145748987854251])
        })
    })
})