"use strict"

const chaiAsPromised = require("chai-as-promised")
const chai = require("chai")
const assert = chai.assert
const expect = chai.expect
const sinonChai = require("sinon-chai")
const sinon = require("sinon")
chai.use(sinonChai)
chai.use(chaiAsPromised);

const {Network, Layer, Neuron, NetMath} = require("../dist/Network.concat.js")

describe("Tests", () => {
    it("Network is loaded", () => expect(Network).to.not.be.undefined)
    it("Layer is loaded", () => expect(Layer).to.not.be.undefined)
    it("Neuron is loaded", () => expect(Neuron).to.not.be.undefined)
    it("NetMath is loaded", () => expect(NetMath).to.not.be.undefined)
})

describe("Network", () => {

    describe("Constructor", () => {

        describe("Config and defaults", () => {

            let net
            beforeEach(() => net = new Network())

            it("Defaults the activation to sigmoid and sets the function from NetMath to net.activation", () => {
                expect(net.activation.name).to.equal("bound sigmoid")
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

            it("Defaults the cost function to 'crossentropy'", () => {
                expect(net.cost).to.equal(NetMath.crossentropy)
            })

            it("Defaults the adaptiveLR to noadaptivelr", () => {
                expect(net.adaptiveLR).to.equal("noadaptivelr")
            })

            it("Sets the net.weightUpdateFn to NetMath.noadaptivelr when setting it to false, null, or 'noadaptivelr'", () => {
                const net2 = new Network({adaptiveLR: null})
                expect(net2.weightUpdateFn).to.equal(NetMath.noadaptivelr)
                const net3 = new Network({adaptiveLR: false})
                expect(net3.weightUpdateFn).to.equal(NetMath.noadaptivelr)
                const net4 = new Network({adaptiveLR: "noadaptivelr"})
                expect(net4.weightUpdateFn).to.equal(NetMath.noadaptivelr)
            })

            it("Sets the net.weightUpdateFn to NetMath.adagrad when setting it to 'adagrad'", () => {
                const net2 = new Network({adaptiveLR: "adagrad"})
                expect(net2.weightUpdateFn).to.equal(NetMath.adagrad)
            })

            it("Defaults the net.rmsDecay to 0.99 if the adaptiveLR is rmsprop", () => {
                const net2 = new Network({adaptiveLR: "rmsprop"})
                expect(net2.rmsDecay).to.equal(0.99)
            })

            it("Sets the net.rmsDecay to use input, if supplied", () => {
                const net2 = new Network({adaptiveLR: "rmsprop", rmsDecay: 0.9})
                expect(net2.rmsDecay).to.equal(0.9)
            })

            it("Does not set an rmsDecay, if adaptiveLR is not rmsprop, even if supplied", () => {
                const net2 = new Network({adaptiveLR: "adagrad", rmsDecay: 0.9})
                expect(net2.rmsDecay).to.be.undefined
            })

            it("Defaults the learning rate to 0.01 if the adaptiveLR is rmsprop", () => {
                const net2 = new Network({adaptiveLR: "rmsprop"})
                expect(net2.learningRate).to.equal(0.001)
            })

            it("Still allows user learning rates to be set, even if adaptiveLR is rmsprop", () => {
                const net2 = new Network({adaptiveLR: "rmsprop", learningRate: 0.5})
                expect(net2.learningRate).to.equal(0.5)
            })

            it("Defaults the learning rate to 0.01 if the adaptiveLR is adam", () => {
                const net2 = new Network({adaptiveLR: "adam"})
                expect(net2.learningRate).to.equal(0.01)
            })

            it("Still allows user learning rates to be set, even if adaptiveLR is adam", () => {
                const net2 = new Network({adaptiveLR: "adam", learningRate: 0.5})
                expect(net2.learningRate).to.equal(0.5)
            })

            it("Defaults the net.rho to 0.95 if the adaptiveLR is adadelta", () => {
                const net2 = new Network({adaptiveLR: "adadelta"})
                expect(net2.rho).to.equal(0.95)
            })

            it("Still allows user rho values to be set", () => {
                const net2 = new Network({adaptiveLR: "adadelta", rho: 0.5})
                expect(net2.rho).to.equal(0.5)
            })

            it("Still sets a rho value, even if a learning rate is given", () => {
                const net2 = new Network({adaptiveLR: "adadelta", rho: 0.9, learningRate: 0.01})
                expect(net2.rho).to.equal(0.9)
                expect(net2.learningRate).to.equal(0.01)
            })

            it("Defaults the learningRate to 0.001 if the activation is tanh", () => {
                const net = new Network({activation: "tanh"})
                expect(net.learningRate).to.equal(0.001)
            })

            it("Still allows a user to set a learningRate when activation is tanh", () => {
                const net = new Network({activation: "tanh", learningRate: 0.01})
                expect(net.learningRate).to.equal(0.01)
            })

            it("Defaults the learningRate to 0.01 when activation is relu", () => {
                const net = new Network({activation: "relu"})
                expect(net.learningRate).to.equal(0.01)
            })

            it("Still allows a user to set a learningRate when activation is relu", () => {
                const net = new Network({activation: "relu", learningRate: 0.001})
                expect(net.learningRate).to.equal(0.001)
            })

            it("Defaults lreluSlope to -0.0005 when using lrelu activation", () => {
                const net = new Network({activation: "lrelu"})
                expect(net.lreluSlope).to.equal(-0.0005)
            })

            it("Still allows a user to set a lreluSlope when activation is lrelu", () => {
                const net = new Network({activation: "lrelu", lreluSlope: -0.001})
                expect(net.lreluSlope).to.equal(-0.001)
            })

            it("Defaults the learningRate to 0.01 when activation is lrelu", () => {
                const net = new Network({activation: "lrelu"})
                expect(net.learningRate).to.equal(0.01)
            })

            it("Still allows a user to set a learningRate when activation is lrelu", () => {
                const net = new Network({activation: "lrelu", learningRate: 0.001})
                expect(net.learningRate).to.equal(0.001)
            })

            it("Sets the net.activationConfig to the given activation config value.", () => {
                const net = new Network({activation: "relu"})
                expect(net.activationConfig).to.equal("relu")
            })

            it("Defaults the learningRate to 0.01 when activation is rrelu", () => {
                const net = new Network({activation: "rrelu"})
                expect(net.learningRate).to.equal(0.01)
            })

            it("Still allows a user to set a learningRate when activation is rrelu", () => {
                const net = new Network({activation: "rrelu", learningRate: 0.001})
                expect(net.learningRate).to.equal(0.001)
            })

            it("Defaults the learningRate to 0.001 when activation is lecuntanh", () => {
                const net = new Network({activation: "lecuntanh"})
                expect(net.learningRate).to.equal(0.001)
            })

            it("Still allows a user to set a learningRate when activation is lecuntanh", () => {
                const net = new Network({activation: "lecuntanh", learningRate: 0.01})
                expect(net.learningRate).to.equal(0.01)
            })

            it("Defaults the learningRate to 0.01 when activation is elu", () => {
                const net = new Network({activation: "elu"})
                expect(net.learningRate).to.equal(0.01)
            })

            it("Still allows a user to set a learningRate when activation is elu", () => {
                const net = new Network({activation: "elu", learningRate: 0.001})
                expect(net.learningRate).to.equal(0.001)
            })

            it("Defaults the eluAlpha to 1 when activation is elu", () => {
                const net = new Network({activation: "elu"})
                expect(net.eluAlpha).to.equal(1)
            })

            it("Still allows a user to set a eluAlpha when activation is elu", () => {
                const net = new Network({activation: "elu", eluAlpha: 2})
                expect(net.eluAlpha).to.equal(2)
            })

            it("Defaults the dropout to 0.5", () => {
                expect(net.dropout).to.equal(0.5)
            })

            it("Allows custom dropout value", () => {
                const net = new Network({dropout: 0.9})
                expect(net.dropout).to.equal(0.9)
            })

            it("Allows disabling dropout by setting the config to false (setting the value to 1)", () => {
                const net = new Network({dropout: false})
                expect(net.dropout).to.equal(1)
            })

            it("Sets the net.l2 value to the l2 value given as parameter", () => {
                const net = new Network({l2: 0.0005})
                expect(net.l2).to.equal(0.0005)
            })

            it("Doesn't set the net.l2 to anything if the l2 parameter is missing", () => {
                expect(net.l2).to.be.undefined
            })

            it("Sets the l2 value to 0.001 if the configuration given is boolean true", () => {
                const net = new Network({l2: true})
                expect(net.l2).to.equal(0.001)
            })

            it("Sets the net.l1 value to the l1 value given as parameter", () => {
                const net = new Network({l1: 0.0005})
                expect(net.l1).to.equal(0.0005)
            })

            it("Doesn't set the net.l1 to anything if the l1 parameter is missing", () => {
                expect(net.l1).to.be.undefined
            })

            it("Sets the l1 value to 0.005 if the configuration given is boolean true", () => {
                const net = new Network({l1: true})
                expect(net.l1).to.equal(0.005)
            })

            it("Sets the net.maxNorm value to the value given as configuration, if given", () => {
                const net = new Network({maxNorm: 1000})
                expect(net.maxNorm).to.equal(1000)
            })

            it("Sets the net.maxNorm value to 1000 if maxNorm is configured as 'true'", () => {
                const net = new Network({maxNorm: true})
                expect(net.maxNorm).to.equal(1000)
            })

            it("Sets the net.maxNormTotal to 0 when maxNorm configuration is given", () => {
                const net = new Network({maxNorm: true})
                expect(net.maxNormTotal).to.equal(0)
            })

            it("Defaults the net.weightsConfig.distribution to uniform", () => {
                const net = new Network({weightsConfig: {limit: 1}})
                expect(net.weightsConfig).to.not.be.undefined
                expect(net.weightsConfig.distribution).to.equal("uniform")
            })

            it("Allows setting the net.weightsConfig.distribution to different config", () => {
                const net = new Network({weightsConfig: {distribution: "gaussian"}})
                expect(net.weightsConfig.distribution).to.equal("gaussian")
            })

            it("Defaults the net.weightsConfig.limit to 0.1 if distribution is uniform", () => {
                const net = new Network({weightsConfig: {distribution: "uniform"}})
                expect(net.weightsConfig.limit).to.not.be.undefined
                expect(net.weightsConfig.limit).to.equal(0.1)
            })

            it("Defaults the net.weightsConfig.limit to 0.1 if no weightsConfig is given and distribution is defaulted to uniform", () => {
                const net = new Network()
                expect(net.weightsConfig.distribution).to.equal("uniform")
                expect(net.weightsConfig.limit).to.not.be.undefined
                expect(net.weightsConfig.limit).to.equal(0.1)
            })

            it("Does not set the net.weightsConfig.limit to anything if distribution is not of uniform type", () => {
                const net = new Network({weightsConfig: {distribution: "gaussian"}})
                expect(net.weightsConfig.limit).to.be.undefined
            })

            it("Allows setting net.weightsConfig.limit to own config", () => {
                const net = new Network({weightsConfig: {distribution: "uniform", limit: 2}})
                expect(net.weightsConfig.limit).to.equal(2)
            })

            it("Defaults the net.weightsConfig.mean to 0 when distribution is gaussian", () => {
                const net = new Network({weightsConfig: {distribution: "gaussian"}})
                expect(net.weightsConfig.mean).to.equal(0)
            })

            it("Sets the net.weightsConfig.mean to the given weightsConfig.mean, if provided", () => {
                const net = new Network({weightsConfig: {distribution: "gaussian", mean: 1}})
                expect(net.weightsConfig.mean).to.equal(1)
            })

            it("Does not set the net.weightsConfig.mean if the distribution is not gaussian", () => {
                const net = new Network({weightsConfig: {distribution: "not gaussian", mean: 1}})
                expect(net.weightsConfig.mean).to.be.undefined
            })

            it("Defaults the net.weightsConfig.stdDeviation to 0.05 when distribution is gaussian", () => {
                const net = new Network({weightsConfig: {distribution: "gaussian"}})
                expect(net.weightsConfig.stdDeviation).to.equal(0.05)
            })

            it("Sets the net.weightsConfig.stdDeviation to the given weightsConfig.stdDeviation, if provided", () => {
                const net = new Network({weightsConfig: {distribution: "gaussian", stdDeviation: 2}})
                expect(net.weightsConfig.stdDeviation).to.equal(2)
            })

            it("Does not set the net.weightsConfig.stdDeviation if the distribution is not gaussian", () => {
                const net = new Network({weightsConfig: {distribution: "not gaussian", stdDeviation: 2}})
                expect(net.weightsConfig.stdDeviation).to.be.undefined
            })
        })

        it("Can create a new Network with no parameters", () => expect(new Network()).instanceof(Network))

        it("Sets the network state to not-defined when defining with no layers parameter", () => {
            const net = new Network()
            expect(net.state).to.equal("not-defined")
        })

        it("Sets the initial net.error to 0", () => {
            const net = new Network()
            expect(net.error).to.equal(0)
        })

        it("Sets the initial net.l2Error to 0 if l2 is configured", () => {
            const net = new Network({l2: true})
            expect(net.l2Error).to.equal(0)
        })

        it("Doesn't set the net.l2Error if l2 is not configured", () => {
            const net = new Network()
            expect(net.l2Error).to.be.undefined
        })

        it("Sets the initial net.l1Error to 0 if l1 is configured", () => {
            const net = new Network({l1: true})
            expect(net.l1Error).to.equal(0)
        })

        it("Doesn't set the net.l1Error if l1 is not configured", () => {
            const net = new Network()
            expect(net.l1Error).to.be.undefined
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

        it("Allows uppercase activation function configs (lecuntanh when configuring as LecunTanh)", () => {
            const net = new Network({activation: "LecunTanh"})
            expect(net.activation.name).to.equal("bound lecuntanh")
        })

        it("Allows snake_case activation function configs (lecuntanh when configuring as lecun_tanh)", () => {
            const net = new Network({activation: "lecun_tanh"})
            expect(net.activation.name).to.equal("bound lecuntanh")
        })

        it("Allows white space activation function configs (lecuntanh when configuring as lecun tanh)", () => {
            const net = new Network({activation: "lecun tanh"})
            expect(net.activation.name).to.equal("bound lecuntanh")
        })

        it("Allows uppercase adaptiveLR function configs (rmsprop when configuring as RMSProp)", () => {
            const net = new Network({adaptiveLR: "RMSProp"})
            expect(net.adaptiveLR).to.equal("rmsprop")
        })

        it("Allows snake_case adaptiveLR function configs (rmsprop when configuring as rms_prop)", () => {
            const net = new Network({adaptiveLR: "rms_prop"})
            expect(net.adaptiveLR).to.equal("rmsprop")
        })

        it("Allows white space adaptiveLR function configs (rmsprop when configuring as rms prop)", () => {
            const net = new Network({adaptiveLR: "rms prop"})
            expect(net.adaptiveLR).to.equal("rmsprop")
        })

        it("Allows uppercase cost function configs (crossentropy when configuring as crossEntropy)", () => {
            const net = new Network({cost: "crossEntropy"})
            expect(net.cost.name).to.equal("crossentropy")
        })

        it("Allows snake_case cost function configs (meansquarederror when configuring as mean_squared_error)", () => {
            const net = new Network({cost: "mean_squared_error"})
            expect(net.cost.name).to.equal("meansquarederror")
        })

        it("Allows white space cost function configs (meansquarederror when configuring as mean squared error)", () => {
            const net = new Network({cost: "mean squared error"})
            expect(net.cost.name).to.equal("meansquarederror")
        })

        it("Allows uppercase weights distribution config (xaviernormal when configuring as xavierNormal)", () => {
            const net = new Network({weightsConfig: {distribution: "xavierNormal"}})
            expect(net.weightsConfig.distribution).to.equal("xaviernormal")
        })

        it("Allows snake_case weights distribution config (xaviernormal when configuring as Xavier_Normal)", () => {
            const net = new Network({weightsConfig: {distribution: "Xavier_Normal"}})
            expect(net.weightsConfig.distribution).to.equal("xaviernormal")
        })

        it("Allows white space weights distribution config (lecununiform when configuring as Lecun Uniform)", () => {
            const net = new Network({weightsConfig: {distribution: "Lecun Uniform"}})
            expect(net.weightsConfig.distribution).to.equal("lecununiform")
        })

        it("Allows setting a custom function as the activation function", () => {
            const customActivation = x => x
            const net = new Network({activation: customActivation})
            expect(net.activation).to.equal(customActivation)
            expect(net.activation("test")).to.equal("test")
        })

        it("Allows setting a custom cost function", () => {
            const customCost = x => x
            const net = new Network({cost: customCost})
            expect(net.cost).to.equal(customCost)
            expect(net.cost("test")).to.equal("test")
        })

        it("Allows setting a custom weights distribution function", () => {
            const customWD = x => [...new Array(x)]
            const layer1 = new Layer(2)
            const layer2 = new Layer(3)
            const net = new Network({weightsConfig: {distribution: customWD}, layers: [layer1, layer2]})
            expect(layer2.weightsConfig.distribution).to.equal(customWD)
            expect(layer2.weightsConfig.distribution(10)).to.have.lengthOf(10)
        })
    })

    describe("initLayers", () => {

        const netThis = {} 
        const net = new Network()

        beforeEach(() => sinon.stub(net, "joinLayer"))
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
            const netThis = {state: "constructed", layers: [layer1, layer2], joinLayer: net.joinLayer, activation: NetMath.sigmoid}
            net.initLayers.bind(netThis)()

            expect(netThis.state).to.equal("initialised")
            expect(net.joinLayer).to.have.been.calledTwice
            expect(net.joinLayer).to.have.been.calledWith(layer1)
            expect(net.joinLayer).to.have.been.calledWith(layer2)
        })

        it("Calculates reasonable sizes for layers, when state is defined (with a small net)", () => {
            const netThis = {state: "defined", definedLayers: [Layer, Layer, Layer, Layer], joinLayer: net.joinLayer, activation: NetMath.sigmoid}
            net.initLayers.bind(netThis, 3, 2)()
            expect(netThis.layers.map(layer => layer.size)).to.deep.equal([3, 5, 3, 2])
            expect(net.joinLayer.callCount).to.equal(4)
        })

        it("Calculates reasonable sizes for layers, when state is defined (with a big net)", () => {
            const netThis = {state: "defined", definedLayers: [Layer, Layer, Layer, Layer, Layer, Layer], joinLayer: net.joinLayer, activation: NetMath.sigmoid}
            net.initLayers.bind(netThis, 784, 10)()

            expect(netThis.state).to.equal("initialised")
            expect(net.joinLayer.callCount).to.equal(6)
            expect(netThis.layers.map(layer => layer.size)).to.deep.equal([784, 417, 315, 214, 112, 10])
        })

        it("Creates three Layers when state is not-defined. First and last layer sizes respective to input/output, middle is in-between", () => {
            const netThis = {state: "not-defined", joinLayer: net.joinLayer, layers: [], activation: NetMath.sigmoid}
            net.initLayers.bind(netThis, 3, 2)()

            expect(netThis.state).to.equal("initialised")
            expect(netThis.layers.map(layer => layer.size)).to.deep.equal([3, 5, 2])
            expect(net.joinLayer).to.have.been.calledThrice
        })

        it("Creates three Layers when state is not-defined. (the same, but with big net)", () => {
            const netThis = {state: "not-defined", joinLayer: net.joinLayer, layers: [], activation: NetMath.sigmoid}
            net.initLayers.bind(netThis, 784, 10)()
            expect(netThis.state).to.equal("initialised")
            expect(net.joinLayer).to.have.been.calledThrice
            expect(netThis.layers.map(layer => layer.size)).to.deep.equal([784, 204, 10])
        })

        it("Sets the network's activation function to the layers", () => {
            const layer1 = new Layer(2)
            const layer2 = new Layer(3)
            const net = new Network({layers: [layer1, layer2], activation: "sigmoid"})

            expect(layer1.activation.name).to.equal("bound sigmoid")
            expect(layer2.activation.name).to.equal("bound sigmoid")
        })
    })

    describe("joinLayer", () => {

        let net, layer1, layer2, layer3

        beforeEach(() => {
            net = new Network({weightsConfig: {distribution: "uniform"}})
            layer1 = new Layer(2)
            layer2 = new Layer(3)
            layer3 = new Layer(4)
        }) 

        it("Does nothing to a single layer network", () => {
            net.layers = [layer1]
            net.joinLayer(layer1)
            expect(layer1.nextLayer).to.be.undefined
            expect(layer1.prevLayer).to.be.undefined
        })

        it("Assigns the network's activation function to the layer", () => {
            layer1.activation = undefined
            net.layers = [layer1]
            net.activation = "test"
            net.joinLayer(layer1)
            expect(layer1.activation).to.equal("test")
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

        it("Does not set the network's rho value to the layer if it does not exist", () => {
            net.layers = [layer1]
            net.rho = undefined
            net.joinLayer(layer1)
            expect(layer1.rho).to.be.undefined
        })

        it("Sets the layer.weightsConfig to the net.weightsConfig", () => {
            net.layers = [layer1]
            net.weightsConfig = {test: "stuff"}
            net.joinLayer(layer1)
            expect(layer1.weightsConfig).to.have.key("test")
        })

        it("Assigns the layer2 weightsConfig.fanIn to the number of neurons in layer1", () => {
            net.layers = [layer1]
            layer1.weightsConfig = {}
            net.joinLayer(layer2, 1)
            expect(layer2.weightsConfig.fanIn).to.equal(2)
        })

        it("Assigns the layer2 weightsConfig.fanOut to the number of neurons in layer3", () => {
            net.layers = [layer1, layer2]
            layer2.weightsConfig = {}
            layer3.weightsConfig = {}
            net.joinLayer(layer3, 2)
            expect(layer2.weightsConfig.fanOut).to.equal(4)
        })

        it("Assigns a reference to the Network class to each layer", () => {
            net.layers = [layer1]
            layer1.weightsConfig = {}
            net.joinLayer(layer2, 1)
            expect(layer2.net).to.equal(net)
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
            const net = new Network({learningRate: 1, layers: [layer1, layer2], adaptiveLR: "noadaptivelr"})

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
            const net = new Network({learningRate: 1, layers: [layer1, layer2], adaptiveLR: "noadaptivelr"})

            layer2.neurons.forEach(neuron => neuron.bias = 0.25)
            layer2.neurons.forEach(neuron => neuron.deltaBias = 0.5)

            net.applyDeltaWeights()

            expect(layer1.bias).to.be.undefined
            expect(layer2.neurons[0].bias).to.equal(0.75)
            expect(layer2.neurons[1].bias).to.equal(0.75)
            expect(layer2.neurons[2].bias).to.equal(0.75)
        })

        it("Increments the net.l2Error by each weight, applied to the L2 formula", () => {
            const layer1 = new Layer(2)
            const layer2 = new Layer(1)
            const net = new Network({activation: "sigmoid", learningRate: 0.2, l2: 0.001, layers: [layer1, layer2]})

            layer2.neurons.forEach(neuron => neuron.weights = [0.25, 0.25])
            layer2.neurons.forEach(neuron => neuron.deltaWeights = [0.5, 0.5])

            sinon.stub(net, "weightUpdateFn")
            net.applyDeltaWeights()

            expect(net.l2Error).to.equal(0.0000625)

            net.weightUpdateFn.restore()
        })

        it("Increments the net.l1Error by each weight, applied to the L1 formula", () => {
            const layer1 = new Layer(2)
            const layer2 = new Layer(1)
            const net = new Network({activation: "sigmoid", learningRate: 0.2, l1: 0.005, layers: [layer1, layer2]})

            layer2.neurons.forEach(neuron => neuron.weights = [0.25, 0.25])
            layer2.neurons.forEach(neuron => neuron.deltaWeights = [0.5, 0.5])

            sinon.stub(net, "weightUpdateFn")
            net.applyDeltaWeights()

            expect(net.l1Error).to.equal(0.0025)

            net.weightUpdateFn.restore()
        })

        it("Increments the net.maxNormTotal if the net.maxNorm is configured", () => {
            const layer2 = new Layer(1)
            const net = new Network({maxNorm: 3, layers: [new Layer(2), layer2]})

            layer2.neurons.forEach(neuron => neuron.weights = [0.25, 0.25])

            sinon.stub(net, "weightUpdateFn").callsFake(x => x)
            sinon.stub(NetMath, "maxNorm").callsFake(x => x)
            net.applyDeltaWeights()

            expect(NetMath.maxNorm).to.be.called
            expect(net.maxNormTotal).to.equal(0.3535533905932738) // sqrt ( 2 * 0.25**2 )
 
            NetMath.maxNorm.restore()
            net.weightUpdateFn.restore()
        })

        it("Does not increment net.maxNormTotal if the net.maxNorm is not configured", () => {
            const layer2 = new Layer(1)
            const net = new Network({layers: [new Layer(2), layer2]})

            layer2.neurons.forEach(neuron => neuron.weights = [0.25, 0.25])

            sinon.stub(net, "weightUpdateFn").callsFake(x => x)
            net.applyDeltaWeights()

            expect(net.maxNormTotal).to.be.undefined

            net.weightUpdateFn.restore()
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
            net = new Network({layers: [2, 3, 2], adaptiveLR: null})
            sinon.stub(net, "forward").callsFake(() => [1,1])
            sinon.stub(net, "backward")
            sinon.stub(net, "resetDeltaWeights")
            sinon.stub(net, "applyDeltaWeights")
            sinon.stub(net, "initLayers")
            sinon.stub(NetMath, "crossentropy")
            sinon.stub(console, "log") // Get rid of output spam
        }) 

        afterEach(() => {
            net.forward.restore()
            net.backward.restore()
            net.resetDeltaWeights.restore()
            net.applyDeltaWeights.restore()
            net.initLayers.restore()
            NetMath.crossentropy.restore()
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
        const testDataX10 = [
            {input: [0,0], expected: [0, 0]},
            {input: [0,1], expected: [0, 1]},
            {input: [1,0], expected: [1, 0]},
            {input: [0,0], expected: [0, 0]},
            {input: [0,1], expected: [0, 1]},
            {input: [1,0], expected: [1, 0]},
            {input: [0,0], expected: [0, 0]},
            {input: [0,1], expected: [0, 1]},
            {input: [1,0], expected: [1, 0]},
            {input: [1,1], expected: [1, 1]}
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

        it("Calls the resetDeltaWeights function for each iteration, +1", () => {
            return net.train(testData).then(() => {
                expect(net.resetDeltaWeights.callCount).to.equal(5)
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

        it("Calls a given callback with an object containing keys: 'elapsed', 'iterations', 'error' and 'input', for each iteration", () => {
            sinon.stub(console, "warn")

            return net.train(testData, {callback: console.warn}).then(() => {
                expect(console.warn).to.have.been.called
                expect(console.warn.callCount).to.equal(4)
                expect(console.warn).to.have.been.calledWith(sinon.match.has("iterations"))
                expect(console.warn).to.have.been.calledWith(sinon.match.has("error"))
                expect(console.warn).to.have.been.calledWith(sinon.match.has("input"))
                expect(console.warn).to.have.been.calledWith(sinon.match.has("elapsed"))
                console.warn.restore()
            })
        })

        it("Calls the initLayers function when the net state is not 'initialised'", () => {
            const network = new Network({adaptiveLR: null})
            sinon.stub(network, "forward")
            sinon.spy(network, "initLayers")

            return network.train(testData).then(() => {
                expect(network.initLayers).to.have.been.called
            })
        })

        it("Calls the initLayers function with the length of the first input and length of first expected", () => {
            const network = new Network({adaptiveLR: null})
            sinon.stub(network, "forward")
            sinon.spy(network, "initLayers")

            return network.train(testData).then(() => {
                expect(network.initLayers).to.have.been.calledWith(2, 2)
                network.initLayers.restore()
            })

        })

        it("Also calls the initLayers function correctly when the first item in the dataSet is named as output", () => {
            const network = new Network({layers: [Layer, Layer, Layer], adaptiveLR: null})
            sinon.stub(network, "forward")
            sinon.spy(network, "initLayers")

            return network.train(testDataWithMixedExpectedOutput).then(() => {
                expect(network.initLayers).to.have.been.calledWith(2,5)
                network.initLayers.restore()
            })
        })

        it("Logs to the console once for each epoch, +2 (for start/stop logs)", () => {
            return net.train(testData, {epochs: 3}).then(() => {
                expect(console.log.callCount).to.equal(5)
            })
        })

        it("Does not log anything to the console if the log option is set to false", () => {
            return net.train(testData, {log: false}).then(() => {
                expect(console.log).to.not.be.called
            })
        })

        it("Sets all layer states to training during training", () => {
            return net.train(testData, {epochs: 1, callback: () => {
                expect(net.layers[0].state).to.equal("training")
                expect(net.layers[1].state).to.equal("training")
                expect(net.layers[2].state).to.equal("training")
            }})
        })

        it("Returns all layer states to initialised after training is stopped", () => {
            return net.train(testData).then(() => {
                expect(net.layers[0].state).to.equal("initialised")
                expect(net.layers[1].state).to.equal("initialised")
                expect(net.layers[2].state).to.equal("initialised")
            })
        })

        it("Resets the net.l2Error if it's configured", () => {
            const net = new Network({l2: true})
            sinon.stub(net, "applyDeltaWeights")
            net.l2Error = 999
            return net.train(testData).then(() => {
                expect(net.l2Error).to.equal(0)
            })
        })

        it("Does not set net.l2Error to anything if it wasn't configured", () => {
            net.l2Error = undefined
            return net.train(testData).then(() => {
                expect(net.l2Error).to.be.undefined
            })
        })

        it("Resets the net.l1Error if it's configured", () => {
            const net = new Network({l1: true})
            sinon.stub(net, "applyDeltaWeights")
            net.l1Error = 999
            return net.train(testData).then(() => {
                expect(net.l1Error).to.equal(0)
            })
        })

        it("Does not set net.l1Error to anything if it wasn't configured", () => {
            net.l1Error = undefined
            return net.train(testData).then(() => {
                expect(net.l1Error).to.be.undefined
            })
        })

        it("Only applies the weight deltas for half the iterations when miniBatchSize is set to 2", () => {
            return net.train(testData, {miniBatchSize: 2}).then(() => {
                expect(net.applyDeltaWeights.callCount).to.equal(2)
            })
        })

        it("Only resets weight deltas for half the iterations +1 when miniBatchSize is set to 2", () => {
            return net.train(testData, {miniBatchSize: 2}).then(() => {
                expect(net.resetDeltaWeights.callCount).to.equal(3)
            })
        })

        it("Applies any left over calculated weights if finishing training part way to a mini batch (10 items, batch size 3)", () => {
            return net.train(testDataX10, {miniBatchSize: 3}).then(() => {
                expect(net.applyDeltaWeights.callCount).to.equal(4)
                expect(net.resetDeltaWeights.callCount).to.equal(4)
            })
        })

        it("If miniBatchSize is configured as true, it is defaulted to the number of classifications", () => {
            return net.train(testData, {miniBatchSize: true}).then(() => {
                expect(net.miniBatchSize).to.equal(2)
                expect(console.log).to.be.calledWith(`Training started. Epochs: 1 Batch Size: 2`)
            })
        })

        it("Shuffles the input data if the shuffle option is set to true", () => {
            sinon.stub(net, "shuffle")
            return net.train(testData, {shuffle: true}).then(() => {
                expect(net.shuffle).to.be.calledWith(testData)
                net.shuffle.restore()
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

        it("Logs to the console once for each iteration, +1 at the end", () => {
            sinon.spy(console, "log")
            return net.test(testData).then(() => {
                expect(console.log.callCount).to.equal(5)
                console.log.restore()
            })
        })

        it("Does not log anything to the console if the log option is set to false", () => {
            sinon.spy(console, "log")
            return net.test(testData, {log: false}).then(() => {
                expect(console.log).to.not.be.called
                console.log.restore()
            })
        })
    })

    describe("format", () => {

        let net

        before(() => net = new Network())

        it("Returns undefined if passed undefined", () => {
            expect(net.format(undefined)).to.be.undefined
        })

        it("Turns a string to lower case", () => {
            const testString = "aAbB"
            const result = net.format(testString) 
            expect(result).to.equal("aabb")
        })

        it("Removes white spaces", () => {
            const testString = " aA bB "
            const result = net.format(testString) 
            expect(result).to.equal("aabb")
        })

        it("Removes underscores", () => {
            const testString = "_aA_bB_"
            const result = net.format(testString) 
            expect(result).to.equal("aabb")
        })

        it("Formats given milliseconds to milliseconds only when under a second", () => {
            const testMils = 100
            expect(net.format(testMils, "time")).to.equal("100ms")
        })

        it("Formats given milliseconds to seconds only when under a minute", () => {
            const testMils = 10000
            expect(net.format(testMils, "time")).to.equal("10s")
        })

        it("Formats given milliseconds to minutes and seconds only when under an hour", () => {
            const testMils = 100000
            expect(net.format(testMils, "time")).to.equal("1m 40s")
        })

        it("Formats given milliseconds to hours, minutes and seconds when over an hour", () => {
            const testMils = 10000000
            expect(net.format(testMils, "time")).to.equal("2h 46m 40s")
        })
    })

    describe("shuffle", () => {

        const testArr = [1,2,3,4,5, "a", "b", "c"]
        const original = testArr.slice(0) 
        const net = new Network()
        net.shuffle(testArr)

        it("Keeps the same number of elements", () => {
            expect(testArr).to.have.lengthOf(8)
        })

        it("Changes the order of the elements", () => {
            expect(testArr).to.not.deep.equal(original)
        })

        it("Does not include any new elements", () => {
            expect(testArr.every(elem => original.includes(elem))).to.be.true
        })

        it("Still includes all original elements", () => {
            expect(original.every(elem => testArr.includes(elem))).to.be.true
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

        it("Sets the layer state to not-initialised", () => {
            const layer = new Layer(10)
            expect(layer.state).to.equal("not-initialised")
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
        let layer1
        let layer2

        beforeEach(() => {
            layer1 = new Layer(2)
            layer2 = new Layer(2)
            layer2.weightsConfig = {limit: 0.1}
            layer2.net = {weightsInitFn: NetMath.uniform}
            sinon.stub(layer2.neurons[0], "init") 
            sinon.stub(layer2.neurons[1], "init")
        })

        afterEach(() => {
            layer2.neurons[0].init.restore()
            layer2.neurons[1].init.restore()
        })

        it("Adds a reference to a layer to its prevLayer property", () => {
            layer2.assignPrev(layer)
            expect(layer2.prevLayer).to.equal(layer)
        })

        it("Creates the neuron its weights array, with .length the same as given parameter", () => {
            layer2.assignPrev(layer1)
            expect(layer2.neurons[0].weights.length).to.equal(2)
        })

        it("Does not change the weights when the neuron is marked as imported", () => {
            layer2.neurons[0].imported = true
            layer2.neurons[0].weights = ["test"]
            layer2.assignPrev(layer1)
            expect(layer2.neurons[0].weights).to.deep.equal(["test"])
        })

        it("Gives the neuron a bias value between -0.1 and +0.1", () => {
            layer2.assignPrev(layer1)
            expect(layer2.neurons[0].bias).to.not.be.undefined
            expect(layer2.neurons[0].bias).to.be.at.most(0.1)
            expect(layer2.neurons[0].bias).to.be.at.least(-0.1)
        })

        it("Does not change the bias when the neuron is marked as imported", () => {
            layer2.neurons[0].imported = true
            layer2.neurons[0].bias = "test"
            layer2.assignPrev(layer1)
            expect(layer2.neurons[0].bias).to.equal("test")
        })

        it("Inits all the neurons in the layer's neurons array", () => {
            layer2.assignPrev(layer)
            expect(layer2.neurons[0].init).to.have.been.called
            expect(layer2.neurons[1].init).to.have.been.called
        })

        it("Calls the neuron's init function with the prev layer's neurons count", () => {
            layer2.assignPrev(layer1)
            expect(layer2.neurons[0].init).to.have.been.calledWith(2)
            expect(layer2.neurons[1].init).to.have.been.calledWith(2)
        })

        it("Calls the neuron's init function with adaptiveLR and activationConfig", () => {
            layer2.net.adaptiveLR = "test"
            layer2.net.activationConfig = "stuff"
            layer2.assignPrev(layer1)
            expect(layer2.neurons[0].init).to.have.been.calledWith(2, sinon.match({"adaptiveLR": "test"}))
            expect(layer2.neurons[0].init).to.have.been.calledWith(2, sinon.match({"activationConfig": "stuff"}))
            expect(layer2.neurons[1].init).to.have.been.calledWith(2, sinon.match({"adaptiveLR": "test"}))
            expect(layer2.neurons[1].init).to.have.been.calledWith(2, sinon.match({"activationConfig": "stuff"}))
        })

        it("Calls the neuron's init function with eluAlpha", () => {
            layer2.net.eluAlpha = 1
            layer2.assignPrev(layer1)
            expect(layer2.neurons[0].init).to.have.been.calledWith(2, sinon.match({"eluAlpha": 1}))
            expect(layer2.neurons[1].init).to.have.been.calledWith(2, sinon.match({"eluAlpha": 1}))
        })

        it("Sets the layer state to initialised", () => {
            layer2.assignPrev(layer1)
            expect(layer2.state).to.equal("initialised")
        })

        it("Calls the NetMath.uniform function when the weightsInitFn is uniform", () => {
            sinon.stub(layer2.net, "weightsInitFn")
            layer2.assignPrev(layer1)
            expect(layer2.net.weightsInitFn).to.be.called
            layer2.net.weightsInitFn.restore()
        })
    })

    describe("forward", () => {

        let layer1, layer2

        beforeEach(() => {
            layer1 = new Layer(2)
            layer2 = new Layer(3)
            layer1.weightsInitFn = NetMath.uniform
            layer2.weightsInitFn = NetMath.uniform
            layer2.weightsConfig = {limit: 0.1}
        })

        it("Sets the sum of each neuron to the bias, when all the weights are 0", () => {
            const net = new Network({layers: [layer1, layer2], dropout: 1})

            layer1.neurons.forEach(neuron => neuron.activation = Math.random())
            layer2.neurons.forEach(neuron => neuron.weights = [0,0,0])
            layer2.forward()
            expect(layer2.neurons[0].sum).to.equal(layer2.neurons[0].bias)
            expect(layer2.neurons[1].sum).to.equal(layer2.neurons[1].bias)
            expect(layer2.neurons[2].sum).to.equal(layer2.neurons[2].bias)
        })

        it("Sets the sum of each neuron to the bias + previous layer's activations when weights are 1", () => {
            const net = new Network({layers: [layer1, layer2], dropout: 1})

            layer1.neurons.forEach(neuron => neuron.activation = 2)
            layer2.neurons.forEach(neuron => neuron.weights = [1,1,1])
            layer2.neurons.forEach(neuron => neuron.bias = 1)

            layer2.forward()
            expect(layer2.neurons[0].sum).to.equal(5)
        })

        it("Sets the neurons' activation to the sigmoid of their sums when the config activation function is sigmoid", () => {
            const net = new Network({
                activation: "sigmoid",
                layers: [layer1, layer2],
                weightsConfig: {limit: 0.1},
                dropout: 1
            })

            net.forward([1,2])
            expect(layer2.neurons[0].activation).to.equal(NetMath.sigmoid(layer2.neurons[0].sum))
            expect(layer2.neurons[1].activation).to.equal(NetMath.sigmoid(layer2.neurons[1].sum))
            expect(layer2.neurons[2].activation).to.equal(NetMath.sigmoid(layer2.neurons[2].sum))
        })

        it("Sets some neurons's .dropped value to true", () => {
            const net = new Network({layers: [new Layer(5), new Layer(15)], dropout: 0.5})
            net.layers[0].neurons.forEach(neuron => neuron.activation = Math.random())
            net.layers[1].state = "training"
            net.layers[1].neurons.forEach(neuron => neuron.weights = [0,0,0,0,0])
            net.layers[1].forward([1,2,3,4,5])

            expect(net.layers[1].neurons.filter(n => n.dropped).length).to.be.at.least(1)
        })

        it("Doesn't apply the dropout if the layer state is not training", () => {
            const net = new Network({layers: [new Layer(5), new Layer(15)], dropout: 0.5})
            net.layers[0].neurons.forEach(neuron => neuron.activation = Math.random())
            net.layers[1].state = "initialised"
            net.layers[1].neurons.forEach(neuron => neuron.weights = [0,0,0,0,0])
            net.layers[1].forward([1,2,3,4,5])

            expect(net.layers[1].neurons.filter(n => n.dropped).length).to.equal(0)
        })

        it("Sets dropped neurons' activations to 0", () => {
            const net = new Network({layers: [new Layer(5), new Layer(15)], dropout: 0.5})
            net.layers[0].neurons.forEach(neuron => neuron.activation = Math.random())
            net.layers[1].neurons.forEach(neuron => neuron.weights = [0,0,0,0,0])
            net.layers[1].state = "training"
            net.layers[1].forward([1,2,3,4,5])
            expect(net.layers[1].neurons.find(n => n.dropped).activation).to.equal(0)
        })
    })

    describe("backward", () => {

        let layer1, layer2, layer3, net

        beforeEach(() => {
            layer1 = new Layer(2)
            layer2 = new Layer(3)
            layer3 = new Layer(4)
            layer1.net = {}
            layer2.net = {}
            layer3.net = {}
            net = new Network({layers: [layer1, layer2, layer3], activation: "sigmoid", dropout: 0})
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
            layer2.neurons.forEach(neuron => neuron.activation = 0.5)
            layer3.neurons.forEach(neuron => neuron.activation = 0.5)
            layer3.net = {miniBatchSize: 1}
            layer3.backward([1,2,3,4])
            expect(layer3.neurons[0].deltaWeights).to.deep.equal([0.25, 0.25, 0.25])
        })

        it("Increments each neuron's bias with the its error", () => {
            layer3.neurons.forEach(neuron => neuron.activation = 0.5) 
            layer3.backward([1,2,3,4])
            expect(layer3.neurons.map(n => n.deltaBias)).to.deep.equal([0.5, 1.5, 2.5, 3.5])
        })

        it("Sets the neuron error to 0 if dropped", () => {
            layer3.neurons.forEach(neuron => {
                neuron.error = 0.5
                neuron.weights = neuron.weights.map(w => 1)
            })
            layer2.neurons.forEach(neuron => neuron.sum = 0.5)
            layer2.neurons[0].dropped = true
            layer2.neurons[0].error = 1
            layer2.backward()

            expect(layer2.neurons[0].error).to.equal(0)
            expect(layer2.neurons[1].error).to.not.equal(0)
        })

        it("Does not increment weight deltas for neurons in prev layer that are dropped", () => {
            layer3.neurons.forEach(neuron => {
                neuron.error = 0.5
                neuron.weights = neuron.weights.map(w => 1)
            })
            layer2.neurons.forEach(neuron => neuron.sum = 0.5)
            layer2.neurons[0].dropped = true
            layer3.neurons[0].deltaWeights[0] = 0

            layer2.backward()

            expect(layer3.neurons[0].deltaWeights[0]).to.equal(0)
        })

        it("Skips doing backward pass for dropped neurons", () => {
            layer3.neurons.forEach(neuron => {
                neuron.error = 0.5
                neuron.weights = neuron.weights.map(w => 1)
            })
            layer2.neurons.forEach(neuron => neuron.sum = 0.5)
            layer2.neurons[0].dropped = true
            layer3.neurons[0].deltaWeights[0] = 0

            layer2.neurons[0].deltaBias = "test"
            layer2.neurons[0].deltaWeights = ["test", "test"]
            layer2.neurons[0].derivative = "test"

            layer2.backward()

            expect(layer2.neurons[0].deltaBias).to.equal(0)
            expect(layer2.neurons[0].deltaWeights).to.deep.equal(["test", "test"])
            expect(layer2.neurons[0].derivative).to.equal("test")
        })

        it("Sets weight deltas to the normal delta + the l2 value", () => {
            layer2.neurons.forEach(neuron => neuron.activation = 0.5)
            layer3.neurons.forEach(neuron => {
                neuron.deltaWeights = [0.25,0.25,0.25,0.25]
                neuron.activation = 0.25
            })
            layer3.net = {l2: 0.001, miniBatchSize: 1}

            layer3.backward([0.3, 0.3, 0.3, 0.3])
            expect(layer3.neurons[0].deltaWeights[0].toFixed(6)).to.equal("0.275006")
        })

        it("Sets weight deltas to the normal delta + the l1 value", () => {
            layer2.neurons.forEach(neuron => neuron.activation = 0.5)
            layer3.neurons.forEach(neuron => {
                neuron.deltaWeights = [0.25,0.25,0.25,0.25]
                neuron.activation = 0.25
            })
            layer3.net = {l1: 0.005, miniBatchSize: 1}

            layer3.backward([0.3, 0.3, 0.3, 0.3])
            expect(layer3.neurons[0].deltaWeights[0].toFixed(6)).to.equal("0.275031")
        })

        it("Regularizes by a tenth as much when the miniBatchSize is configured as 10", () => {
            layer2.neurons.forEach(neuron => neuron.activation = 0.5)
            layer3.neurons.forEach(neuron => {
                neuron.deltaWeights = [0.25,0.25,0.25,0.25]
                neuron.activation = 0.25
            })
            layer3.net = {l1: 0.005, miniBatchSize: 10}

            layer3.backward([0.3, 0.3, 0.3, 0.3])
            expect(layer3.neurons[0].deltaWeights[0].toFixed(6)).to.equal("0.275003")

            layer3.neurons.forEach(neuron => {
                neuron.deltaWeights = [0.25,0.25,0.25,0.25]
                neuron.activation = 0.25
            })
            layer3.net.l2 = 0.001
            layer3.net.l1 = 0
            layer3.backward([0.3, 0.3, 0.3, 0.3])
            expect(layer3.neurons[0].deltaWeights[0].toFixed(6)).to.equal("0.275001")
        })
    })
})

describe("Neuron", () => {

    describe("constructor", () => {

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
  
        let neuron, neuron2

        beforeEach(() => {
            neuron = new Neuron()
            neuron.weights = [...new Array(5)].map(v => Math.random()*0.2-0.1)

            neuron2 = new Neuron()
            neuron2.weights = [...new Array(3)].map(v => Math.random()*0.2-0.1)
        })

        it("Creates an array of delta weights with the same length as the weights array", () => {
            neuron.init(5)
            expect(neuron.deltaWeights).to.not.be.undefined
            expect(neuron.deltaWeights.length).to.equal(neuron.weights.length)
        })

        it("Sets all the delta weights to 0", () => {
            neuron.init(5)
            expect(neuron.deltaWeights).to.deep.equal([0,0,0,0,0])
        })

        it("Creates a weightGains array if the adaptiveLR parameter is gain, with same size as weights, with 1 values", () => {
            neuron2.init(3, {adaptiveLR: "gain"})
            expect(neuron2.weightGains).to.not.be.undefined
            expect(neuron2.weightGains).to.have.lengthOf(3)
            expect(neuron2.weightGains).to.deep.equal([1,1,1])
        })

        it("Creates a biasGain value of 1 if the adaptiveLR parameter is gain", () => {
            neuron2.init(3, {adaptiveLR: "gain"})
            expect(neuron2.biasGain).to.equal(1)
        })

        it("Does not create the weightGains and biasGain when the adaptiveLR is not gain", () => {
            neuron2.init(3, {adaptiveLR: "not gain"})
            expect(neuron2.weightGains).to.be.undefined
            expect(neuron2.biasGain).to.be.undefined
        })

        it("Creates a weightsCache array, with same dimension as weights, if the adaptiveLR is adagrad, with 0 values", () => {
            neuron2.init(3, {adaptiveLR: "adagrad"})
            expect(neuron2.weightsCache).to.not.be.undefined
            expect(neuron2.weightsCache).to.have.lengthOf(3)
            expect(neuron2.weightsCache).to.deep.equal([0,0,0])
        })

        it("Creates a weightsCache array, with same dimension as weights, if the adaptiveLR is rmsprop, with 0 values", () => {
            neuron2.init(3, {adaptiveLR: "rmsprop"})
            expect(neuron2.weightsCache).to.not.be.undefined
            expect(neuron2.weightsCache).to.have.lengthOf(3)
            expect(neuron2.weightsCache).to.deep.equal([0,0,0])
        })

        it("Creates a biasCache value of 0 if the adaptiveLR parameter is adagrad", () => {
            neuron2.init(3, {adaptiveLR: "adagrad"})
            expect(neuron2.biasCache).to.equal(0)
        })

        it("Creates a biasCache value of 0 if the adaptiveLR parameter is rmsprop", () => {
            neuron2.init(3, {adaptiveLR: "adagrad"})
            expect(neuron2.biasCache).to.equal(0)
        })

        it("Does not create the weightsCache or biasCache if the adaptiveLR is not adagrad", () => {
            neuron2.init(3, {adaptiveLR: "not adagrad"})
            expect(neuron2.weightsCache).to.be.undefined
            expect(neuron2.biasCache).to.be.undefined
        })

        it("Does not create the weightsCache or biasCache if the adaptiveLR is not rmsprop", () => {
            neuron2.init(3, {adaptiveLR: "not rmsprop"})
            expect(neuron2.weightsCache).to.be.undefined
            expect(neuron2.biasCache).to.be.undefined
        })

        it("Creates and sets neuron.m to 0 if the adaptiveLR parameter is adam", () => {
            neuron2.init(3, {adaptiveLR: "adam"})
            expect(neuron2.m).to.not.be.undefined
            expect(neuron2.m).to.equal(0)
        })

        it("Creates and sets neuron.v to 0 if the adaptiveLR parameter is adam", () => {
            neuron2.init(3, {adaptiveLR: "adam"})
            expect(neuron2.v).to.not.be.undefined
            expect(neuron2.v).to.equal(0)
        })

        it("Does not create neuron.m or neuron.v when the adaptiveLR parameter is not adam", () => {
            neuron2.init(3, {adaptiveLR: "not adam"})
            expect(neuron2.m).to.be.undefined
            expect(neuron2.v).to.be.undefined
        })

        it("Creates a weightsCache array, with same dimension as weights, if the adaptiveLR is adadelta, with 0 values", () => {
            neuron2.init(3, {adaptiveLR: "adadelta"})
            expect(neuron2.weightsCache).to.not.be.undefined
            expect(neuron2.weightsCache).to.have.lengthOf(3)
            expect(neuron2.weightsCache).to.deep.equal([0,0,0])
        })

        it("Creates a adadeltaBiasCache value of 0 if the adaptiveLR parameter is adadelta", () => {
            neuron2.init(3, {adaptiveLR: "adadelta"})
            expect(neuron2.adadeltaBiasCache).to.equal(0)
        })

        it("Creates a adadeltaCache array, with same dimension as weights, if the adaptiveLR is adadelta, with 0 values", () => {
            neuron2.init(3, {adaptiveLR: "adadelta"})
            expect(neuron2.adadeltaCache).to.not.be.undefined
            expect(neuron2.adadeltaCache).to.have.lengthOf(3)
            expect(neuron2.adadeltaCache).to.deep.equal([0,0,0])
        })

        it("Does not create adadeltaBiasCache or adadeltaCache when the adaptiveLR is adagrad or rmsprop", () => {
            neuron2.init(3, {adaptiveLR: "adagrad"})
            expect(neuron2.adadeltaCache).to.be.undefined
            expect(neuron2.adadeltaBiasCache).to.be.undefined
            const neuron3 = new Neuron()
            neuron3.weights = [...new Array(3)].map(v => Math.random()*0.2-0.1)
            neuron3.init(3, {adaptiveLR: "rmsprop"})
            expect(neuron3.adadeltaCache).to.be.undefined
            expect(neuron3.adadeltaBiasCache).to.be.undefined
        })

        it("Creates a random neuron.rreluSlope number if the activationConfig value is rrelu", () => {
            neuron2.init(3, {activationConfig: "rrelu"})
            expect(neuron2.rreluSlope).to.not.be.undefined
            expect(neuron2.rreluSlope).to.be.a.number
            expect(neuron2.rreluSlope).to.be.at.most(0.0011)
        })

        it("Sets the neuron.eluAlpha to the given value, if given a value", () => {
            neuron2.init(3, {activationConfig: "elu", eluAlpha: 0.5})
            expect(neuron2.eluAlpha).to.equal(0.5)
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

    describe("Tanh", () => {

        it("tanh(1)==0.7615941559557649", () => {
            expect(NetMath.tanh(1)).to.equal(0.7615941559557649)
        })

        it("tanh(0.5)==0.46211715726000974", () => {
            expect(NetMath.tanh(0.5)).to.equal(0.46211715726000974)
        })

        it("tanh(0.5, true)==0.7864477329659275", () => {
            expect(NetMath.tanh(0.5, true)).to.equal(0.7864477329659275)
        })

        it("tanh(1.5, true)==0.18070663892364855", () => {
            expect(NetMath.tanh(1.5, true)).to.equal(0.18070663892364855)
        })

        it("Doesn't return NaN if the input value is too high", () => {
            expect(NetMath.tanh(900)).to.not.be.NaN
            expect(NetMath.tanh(900, true)).to.not.be.NaN
        })
    })

    describe("relu", () => {

        it("relu(2)==2", () => {
            expect(NetMath.relu(2)).to.equal(2)
        })

        it("relu(-2)==0", () => {
            expect(NetMath.relu(-2)).to.equal(0)
        })

        it("relu(2, true)==1", () => {
            expect(NetMath.relu(2, true)).to.equal(1)
        })

        it("relu(-2, true)==0", () => {
            expect(NetMath.relu(-2, true)).to.equal(0)
        })
    })

    describe("rrelu", () => {

        it("rrelu(2, false, {rreluSlope: 0.0005})==2", () => {
            expect(NetMath.rrelu(2, false, {rreluSlope: 0.0005})).to.equal(2)
        })

        it("rrelu(-2, false, {rreluSlope: 0.0005})==0", () => {
            expect(NetMath.rrelu(-2, false, {rreluSlope: 0.0005})).to.equal(0.0005)
        })

        it("rrelu(2, true, {rreluSlope: 0.0005})==1", () => {
            expect(NetMath.rrelu(2, true, {rreluSlope: 0.0005})).to.equal(1)
        })

        it("rrelu(-2, true, {rreluSlope: 0.0005})==0", () => {
            expect(NetMath.rrelu(-2, true, {rreluSlope: 0.0005})).to.equal(0.0005)
        })
    })

    describe("lrelu", () => {

        it("lrelu(2)==2", () => {
            expect(NetMath.lrelu.bind({lreluSlope:-0.0005}, 2)()).to.equal(2)
        })

        it("lrelu(-2)==-0.001", () => {
            expect(NetMath.lrelu.bind({lreluSlope:-0.0005}, -2)()).to.equal(-0.001)
        })

        it("lrelu(2, true)==1", () => {
            expect(NetMath.lrelu.bind({lreluSlope:-0.0005}, 2, true)()).to.equal(1)
        })

        it("lrelu(-2, true)==0", () => {
            expect(NetMath.lrelu.bind({lreluSlope:-0.0005}, -2, true)()).to.equal(-0.0005)
        })
    })

    describe("sech", () => {

        it("sech(1)==0.6480542736638853", () => {
            expect(NetMath.sech(1)).to.equal(0.6480542736638853)
        })

        it("sech(-0.5)==0.886818883970074", () => {
            expect(NetMath.sech(-0.5)).to.equal(0.886818883970074)
        })
    })

    describe("lecuntanh", () => {

        it("lecuntanh(2)==1.4929388053842507", () => {
            expect(NetMath.lecuntanh(2)).to.equal(1.4929388053842507)
        })

        it("lecuntanh(-2)==-1.4929388053842507", () => {
            expect(NetMath.lecuntanh(-2)).to.equal(-1.4929388053842507)
        })

        it("lecuntanh(2, true)==0.2802507761872869", () => {
            expect(NetMath.lecuntanh(2, true)).to.equal(0.2802507761872869)
        })

        it("lecuntanh(-2, true)==0.2802507761872869", () => {
            expect(NetMath.lecuntanh(-2, true)).to.equal(0.2802507761872869)
        })
    })

    describe("elu", () => {

        it("elu(2)==2", () => {
            expect(NetMath.elu.bind(null, 2, false, {eluAlpha: 1})()).to.equal(2)
        })

        it("elu(-0.25)==-0.22119921692859512", () => {
            expect(NetMath.elu.bind(null, -0.25, false, {eluAlpha: 1})()).to.equal(-0.22119921692859512)
        })

        it("elu(2, true)==1", () => {
            expect(NetMath.elu.bind(null, 2, true, {eluAlpha: 1})()).to.equal(1)
        })

        it("elu(-0.5, true)==0.6065306597126334", () => {
            expect(NetMath.elu.bind(null, -0.5, true, {eluAlpha: 1})()).to.equal(0.6065306597126334)
        })
    })

    describe("Cross Entropy", () => {

        it("crossentropy([1,0,0.3], [0,1, 0.8]) == 70.16654147569186", () => {
            expect(NetMath.crossentropy([1,0,0.3], [0,1, 0.8])).to.equal(70.16654147569186)
        })
    })

    describe("Softmax", () => {

        it("softmax([23, 54, 167, 3]) == [0.0931174089068826, 0.21862348178137653, 0.6761133603238867, 0.012145748987854251]", () => {
            expect(NetMath.softmax([23, 54, 167, 3])).to.deep.equal([0.0931174089068826, 0.21862348178137653, 0.6761133603238867, 0.012145748987854251])
        })
    })

    describe("Mean Squared Error", () => {

        it("meansquarederror([13,17,18,20,24], [12,15,20,22,24]) == 2.6", () => {
            expect(NetMath.meansquarederror([13,17,18,20,24], [12,15,20,22,24])).to.equal(2.6)
        })
    })

    describe("noadaptivelr", () => {

        const fn = NetMath.noadaptivelr.bind({learningRate: 0.5})

        it("Increments a weight with half of its delta weight when the learning rate is 0.5", () => {
            expect(fn(10, 10)).to.equal(15)
            expect(fn(10, 20)).to.equal(20)
            expect(fn(10, -30)).to.equal(-5)
        })
    })

    describe("gain", () => {

        let neuron

        beforeEach(() => neuron = new Neuron())

        it("Doubles a value when the gain is 2 and learningRate 1", () => {
            neuron.biasGain = 2
            const result = NetMath.gain.bind({learningRate: 1}, 10, 5, neuron)()
            expect(result).to.equal(20)
        })

        it("Halves a value when the learning rate is 0.1 and gain is -5", () => {
            neuron.biasGain = -5
            const result = NetMath.gain.bind({learningRate: 0.1}, 5, 5, neuron)()
            expect(result).to.equal(2.5)
        })

        it("Increments a neuron's bias gain by 0.05 when the bias value doesn't change sign", () => {
            const fakeThis = {
                learningRate: 1
            }
            neuron.bias = 0.1
            neuron.biasGain =  1
            NetMath.gain.bind(fakeThis, 0.1, 1, neuron)()
            expect(neuron.biasGain).to.equal(1.05)
        })

        it("Does not increase the gain to more than 5", () => {
            const fakeThis = {
                learningRate: 1
            }
            neuron.bias = 0.1
            neuron.biasGain =  4.99
            NetMath.gain.bind(fakeThis, 0.1, 1, neuron)()
            expect(neuron.biasGain).to.equal(5)
        })

        it("Multiplies a neuron's bias gain by 0.95 when the value changes sign", () => {
            const fakeThis = {
                learningRate: -10
            }
            neuron.bias = 0.1
            neuron.biasGain =  1
            NetMath.gain.bind(fakeThis, 0.1, 1, neuron)()
            expect(neuron.biasGain).to.equal(0.95)
        })

        it("Does not reduce the bias gain to less than 0.5", () => {
            const fakeThis = {
                learningRate: -10
            }
            neuron.bias = 0.1
            neuron.biasGain =  0.51
            NetMath.gain.bind(fakeThis, 0.1, 1, neuron)()
            expect(neuron.biasGain).to.equal(0.5)
        })

        it("Increases weight gain the same way as the bias gain", () => {
            const fakeThis = {
                learningRate: 1
            }
            neuron.weights = [0.1, 0.1]
            neuron.weightGains =  [1, 4.99]
            NetMath.gain.bind(fakeThis, 0.1, 1, neuron, 0)()
            NetMath.gain.bind(fakeThis, 0.1, 1, neuron, 1)()
            expect(neuron.weightGains[0]).to.equal(1.05)
            expect(neuron.weightGains[1]).to.equal(5)
        })

        it("Decreases weight gain the same way as the bias gain", () => {
            const fakeThis = {
                learningRate: -10
            }
            neuron.weights = [0.1, 0.1]
            neuron.weightGains =  [1, 0.51]
            NetMath.gain.bind(fakeThis, 0.1, 1, neuron, 0)()
            NetMath.gain.bind(fakeThis, 0.1, 1, neuron, 1)()
            expect(neuron.weightGains[0]).to.equal(0.95)
            expect(neuron.weightGains[1]).to.equal(0.5)
        })
    })

    describe("adagrad", () => {

        let neuron

        beforeEach(() => neuron = new Neuron())

        it("Increments the neuron's biasCache by the square of its deltaBias", () => {
            neuron.biasCache = 0
            NetMath.adagrad.bind({learningRate: 2}, 1, 3, neuron)()
            expect(neuron.biasCache).to.equal(9)
        })

        it("Returns a new value matching the formula for adagrad", () => {
            neuron.biasCache = 0
            const result = NetMath.adagrad.bind({learningRate: 0.5}, 1, 3, neuron)() 
            expect(result.toFixed(1)).to.equal("1.5")
        })

        it("Increments the neuron's weightsCache the same was as the biasCache", () => {
            neuron.weightsCache = [0, 1, 2]
            const result1 = NetMath.adagrad.bind({learningRate: 2}, 1, 3, neuron, 0)()
            const result2 = NetMath.adagrad.bind({learningRate: 2}, 1, 4, neuron, 1)()
            const result3 = NetMath.adagrad.bind({learningRate: 2}, 1, 2, neuron, 2)()
            expect(neuron.weightsCache[0]).to.equal(9)
            expect(neuron.weightsCache[1]).to.equal(17)
            expect(neuron.weightsCache[2]).to.equal(6)

            expect(result1.toFixed(1)).to.equal("3.0")
            expect(result2.toFixed(1)).to.equal("2.9")
            expect(result3.toFixed(1)).to.equal("2.6")
        })
    })

    describe("rmsprop", () => {
        let neuron

        beforeEach(() => neuron = new Neuron())

        it("Sets the cache value to the correct formula", () => {
            neuron.biasCache = 10
            NetMath.rmsprop.bind({learningRate: 2, rmsDecay: 0.99}, 1, 3, neuron)()
            expect(neuron.biasCache).to.equal(9.99) // 9.9 + 0.01 * 9
        })

        it("Returns a new value matching the formula for rmsprop, using this new cache value", () => {
            neuron.biasCache = 10
            const result = NetMath.rmsprop.bind({learningRate: 0.5, rmsDecay: 0.99}, 1, 3, neuron)()
            expect(result.toFixed(2)).to.equal("1.47") // 1 + 0.5 * 3 / 3.1607
        })

        it("Updates the weightsCache the same way as the biasCache", () => {
            neuron.weightsCache = [0, 1, 2]
            const result1 = NetMath.rmsprop.bind({learningRate: 0.5, rmsDecay: 0.99}, 1, 3, neuron, 0)()
            const result2 = NetMath.rmsprop.bind({learningRate: 0.5, rmsDecay: 0.99}, 1, 4, neuron, 1)()
            const result3 = NetMath.rmsprop.bind({learningRate: 0.5, rmsDecay: 0.99}, 1, 2, neuron, 2)()
            expect(neuron.weightsCache[0].toFixed(2)).to.equal("0.09")
            expect(neuron.weightsCache[1].toFixed(2)).to.equal("1.15")
            expect(neuron.weightsCache[2].toFixed(2)).to.equal("2.02")

            expect(result1.toFixed(1)).to.equal("6.0")
            expect(result2.toFixed(1)).to.equal("2.9")
            expect(result3.toFixed(1)).to.equal("1.7")
        })
    })

    describe("adam", () => {

        let neuron

        beforeEach(() => neuron = new Neuron())

        it("Sets the neuron.m to the correct value, following the algorithm", () => {
            neuron.m = 0.1
            NetMath.adam.bind({learningRate: 0.01}, 1, 0.2, neuron)()
            expect(neuron.m.toFixed(2)).to.equal("0.11") // 0.9 * 0.1 + (1-0.9) * 0.2
        })

        it("Sets the neuron.v to the correct value, following the algorithm", () => {
            neuron.v = 0.1
            NetMath.adam.bind({learningRate: 0.01}, 1, 0.2, neuron)()
            expect(neuron.v.toFixed(5)).to.equal("0.09994") // 0.999 * 0.1 + (1-0.999) * 0.2*0.2
        })

        it("Calculates a value correctly, following the algorithm", () => {
            neuron.m = 0.121
            neuron.v = 0.045
            const result = NetMath.adam.bind({learningRate: 0.01, iterations: 0.2}, -0.3, 0.02, neuron)()
            expect(result.toFixed(6)).to.equal("-0.298474")            
        })
    })

    describe("adadelta", () => {

        let neuron

        beforeEach(() => neuron = new Neuron())

        it("Sets the neuron.biasCache to the correct value, following the adadelta formula", () => {
            neuron.biasCache = 0.5
            NetMath.adadelta.bind({rho: 0.95}, 0.5, 0.2, neuron)()
            expect(neuron.biasCache).to.equal(0.477) // 0.95 * 0.5 + (1-0.95) * 0.2**2
        })

        it("Sets the weightsCache to the correct value, following the adadelta formula, same as biasCache", () => {
            neuron.weightsCache = [0.5, 0.75]
            neuron.adadeltaCache = [0, 0]
            NetMath.adadelta.bind({rho: 0.95}, 0.5, 0.2, neuron, 0)()
            NetMath.adadelta.bind({rho: 0.95}, 0.5, 0.2, neuron, 1)()
            expect(neuron.weightsCache[0]).to.equal(0.477)
            expect(neuron.weightsCache[1].toFixed(4)).to.equal("0.7145")
        })

        it("Creates a value for the bias correctly, following the algorithm", () => {
            neuron.biasCache = 0.5
            neuron.adadeltaBiasCache = 0.25
            const newValue = NetMath.adadelta.bind({rho: 0.95}, 0.5, 0.2, neuron)()
            expect(newValue.toFixed(5)).to.equal("0.64479") // 0.5 + sqrt(~0.25/~0.477) * 0.2 (~ because of eps)
        })

        it("Creates a value for the weight correctly, the same was as the bias", () => {
            neuron.weightsCache = [0.5, 0.75]
            neuron.adadeltaCache = [0.1, 0.2]
            const newValue1 = NetMath.adadelta.bind({rho: 0.95}, 0.5, 0.2, neuron, 0)()
            const newValue2 = NetMath.adadelta.bind({rho: 0.95}, 0.5, 0.2, neuron, 1)()
            expect(newValue1.toFixed(5)).to.equal("0.59157") // 0.5 + sqrt(~0.1/~0.5) * 0.2
            expect(newValue2.toFixed(5)).to.equal("0.60581")
        })

        it("Updates the neuron.adadeltaBiasCache with the correct value, following the formula", () => {
            neuron.biasCache = 0.5
            neuron.adadeltaBiasCache = 0.25
            NetMath.adadelta.bind({rho: 0.95}, 0.5, 0.2, neuron)()
            expect(neuron.adadeltaBiasCache).to.equal(0.2395) // 0.95 * 0.25 + (1-0.95) * 0.2*0.2 
        })

        it("Updates the neuron.adadeltaCache with the correct value, following the formula, same as adadeltaBiasCache", () => {
            neuron.weightsCache = [0.5, 0.75]
            neuron.adadeltaCache = [0.1, 0.2]
            NetMath.adadelta.bind({rho: 0.95}, 0.5, 0.2, neuron, 0)()
            NetMath.adadelta.bind({rho: 0.95}, 0.5, 0.2, neuron, 1)()
            expect(neuron.adadeltaCache[0]).to.equal(0.097) // 0.95 * 0.1 + (1-0.95) * 0.2*0.2
            expect(neuron.adadeltaCache[1]).to.equal(0.192)
        })
    })

    describe("maxNorm", () => {

        let net

        beforeEach(() => net = new Network())

        it("Sets the net.maxNormTotal to 0", () => {
            net.maxNormTotal = 1
            NetMath.maxNorm.bind(net)()
            expect(net.maxNormTotal).to.equal(0)
        })

        it("Scales weights if their L2 exceeds the configured max norm threshold", () => {
            const layer2 = new Layer(2)
            const net = new Network({layers: [new Layer(1), layer2], maxNorm: 1})

            layer2.neurons[0].weights = [2, 2] 
            net.maxNormTotal = 2.8284271247461903 // maxNormTotal = sqrt (2**2 * 2) = 2.8284271247461903

            NetMath.maxNorm.bind(net)()
            expect(layer2.neurons[0].weights).to.deep.equal([0.7071067811865475, 0.7071067811865475])
        })

        it("Does not scale weights if their L2 does not exceed the configured max norm threshold", () => {
            const layer2 = new Layer(2)
            const net = new Network({layers: [new Layer(1), layer2], maxNorm: 1000})

            layer2.neurons[0].weights = [2, 2] 
            net.maxNormTotal = 2.8284271247461903 // maxNormTotal = sqrt (2**2 * 2) = 2.8284271247461903

            NetMath.maxNorm.bind(net)()
            expect(layer2.neurons[0].weights).to.deep.equal([2, 2])
        })
    })

    describe("uniform", () => {

        it("Returns the same number of values as the size value given", () => {
            const result = NetMath.uniform(10, {limit: 0.1})
            expect(result.length).to.equal(10)
        })

        it("Weights are all between -0.1 and +0.1 when weightsConfig.limit is 0.1", () => {
            const result = NetMath.uniform(10, {limit: 0.1})
            expect(result.every(w => w>=-0.1 && w<=0.1)).to.be.true
        })

        it("Inits some weights at values bigger |0.1| when weightsConfig.limit is 1000", () => {
            const result = NetMath.uniform(10, {limit: 1000})
            expect(result.some(w => w<=-0.1 || w>=0.1)).to.be.true
        })

        it("Creates weights that are more or less uniform", () => {
            const result = NetMath.uniform(10000, {limit: 1})

            const decData = {}
            result.forEach(w => decData[Math.abs(w*10).toString()[0]] = (decData[Math.abs(w*10).toString()[0]]|0) + 1)
            const sorted = Object.keys(decData).map(k=>decData[k]).sort((a,b) => a<b)
            expect(sorted[0] - sorted[sorted.length-1]).to.be.at.most(200)
        })
    })

    describe("standardDeviation", () => {

        it("Returns 6.603739470936145 for [3,5,7,8,5,25,8,4]", () => {
            expect(NetMath.standardDeviation([3,5,7,8,5,25,8,4])).to.equal(6.603739470936145)
        })
    })

    describe("gaussian", () => {

        it("Returns the same number of values as the size value given", () => {
            const result = NetMath.gaussian(10, {mean: 0, stdDeviation: 1})
            expect(result.length).to.equal(10)
        })

        it("The standard deviation of the weights is roughly 1 when set to 1", () => {
            const result = NetMath.gaussian(1000, {mean: 0, stdDeviation: 1})
            const std = NetMath.standardDeviation(result)
            expect(Math.round(std*10)/10).to.be.at.most(1.15)
            expect(Math.round(std*10)/10).to.be.at.least(0.85)
        })

        it("The standard deviation of the weights is roughly 5 when set to 5", () => {
            const result = NetMath.gaussian(1000, {mean: 0, stdDeviation: 5})
            const std = NetMath.standardDeviation(result)
            expect(Math.round(std*10)/10).to.be.at.most(1.15 * 5)
            expect(Math.round(std*10)/10).to.be.at.least(0.885 * 5)
        })

        it("The mean of the weights is roughly 0 when set to 0", () => {
            const result = NetMath.gaussian(1000, {mean: 0, stdDeviation: 1})
            const mean = result.reduce((p,c) => p+c) / 1000
            expect(mean).to.be.at.most(0.1)
            expect(mean).to.be.at.least(-0.1)
        })

        it("The mean of the weights is roughly 10 when set to 10", () => {
            const result = NetMath.gaussian(1000, {mean: 10, stdDeviation: 1})
            const mean = result.reduce((p,c) => p+c) / 1000
            expect(mean).to.be.at.most(10.15)
            expect(mean).to.be.at.least(9.85)
        })
    })

    describe("lecunnormal", () => {

        it("Returns the same number of values as the size value given", () => {
            const result = NetMath.lecunnormal(10, {fanIn: 5})
            expect(result.length).to.equal(10)
        })

        it("The standard deviation of the weights is roughly 0.05 when the fanIn is 5", () => {
            const result = NetMath.lecunnormal(1000, {fanIn: 5})
            const std = NetMath.standardDeviation(result)
            expect(Math.round(std*1000)/1000).to.be.at.most(0.60)
            expect(Math.round(std*1000)/1000).to.be.at.least(0.40)
        })

        it("The mean of the weights is roughly 0", () => {
            const result = NetMath.lecunnormal(1000, {fanIn: 5})
            const mean = result.reduce((p,c) => p+c) / 1000
            expect(mean).to.be.at.most(0.1)
            expect(mean).to.be.at.least(-0.1)
        })
    })

    describe("lecununiform", () => {

        it("Returns the same number of values as the size value given", () => {
            const result = NetMath.lecununiform(10, {fanIn: 5})
            expect(result.length).to.equal(10)
        })

        it("Weights are all between -0.5 and +0.5 when fanIn is 12", () => {
            const result = NetMath.lecununiform(1000, {fanIn: 12})
            expect(result.every(w => w>=-0.5 && w<=0.5)).to.be.true
        })

        it("Inits some weights at values bigger |0.5| when fanIn is smaller (8)", () => {
            const result = NetMath.lecununiform(1000, {fanIn: 8})
            expect(result.some(w => w<=-0.05 || w>=0.05)).to.be.true
        })

        it("Creates weights that are more or less uniform", () => {
            const result = NetMath.lecununiform(10000, {fanIn: 12})

            const decData = {}
            result.forEach(w => decData[Math.abs(w*10).toString()[0]] = (decData[Math.abs(w*10).toString()[0]]|0) + 1)
            const sorted = Object.keys(decData).map(k=>decData[k]).sort((a,b) => a<b)
            expect(sorted[0] - sorted[sorted.length-1]).to.be.at.most(210)
        })
    })

    describe("xaviernormal", () => {

        it("Returns the same number of values as the size value given", () => {
            const result = NetMath.xaviernormal(10, {fanIn: 5, fanOut: 10})
            expect(result.length).to.equal(10)
        })

        it("The standard deviation of the weights is roughly 0.25 when the fanIn is 5 and fanOut is 25", () => {
            const result = NetMath.xaviernormal(1000, {fanIn: 5, fanOut: 25})
            const std = NetMath.standardDeviation(result)
            expect(Math.round(std*1000)/1000).to.be.at.most(0.3)
            expect(Math.round(std*1000)/1000).to.be.at.least(0.2)
        })

        it("The mean of the weights is roughly 0", () => {
            const result = NetMath.xaviernormal(1000, {fanIn: 5, fanOut: 25})
            const mean = result.reduce((p,c) => p+c) / 1000
            expect(mean).to.be.at.most(0.1)
            expect(mean).to.be.at.least(-0.1)
        })

        it("Falls back to using lecunnormal if there is no fanOut available", () => {
            sinon.spy(NetMath, "lecunnormal")
            const result = NetMath.xaviernormal(10, {fanIn: 5})
            expect(NetMath.lecunnormal).to.have.been.calledWith(10, {fanIn: 5})
            expect(result.length).to.equal(10)
            NetMath.lecunnormal.restore()
        })
    })

    describe("xavieruniform", () => {

        it("Returns the same number of values as the size value given", () => {
            const result = NetMath.xavieruniform(10, {fanIn: 10})
            expect(result.length).to.equal(10)
        })

        it("Weights are all between -0.5 and +0.5 when fanIn is 10 and fanOut is 15", () => {
            const result = NetMath.xavieruniform(1000, {fanIn: 10, fanOut: 15})
            expect(result.every(w => w>=-0.5 && w<=0.5)).to.be.true
        })

        it("Inits some weights at values bigger |0.5| when fanIn+fanOut is smaller (5+5=10)", () => {
            const result = NetMath.xavieruniform(1000, {fanIn: 5, fanOut: 5})
            expect(result.some(w => w<=-0.05 || w>=0.05)).to.be.true
        })

        it("Creates weights that are more or less uniform", () => {
            const result = NetMath.xavieruniform(1000, {fanIn: 5, fanOut: 5})

            const decData = {}
            result.forEach(w => decData[Math.abs(w*10).toString()[0]] = (decData[Math.abs(w*10).toString()[0]]|0) + 1)
            const sorted = Object.keys(decData).map(k=>decData[k]).sort((a,b) => a<b)
            expect(sorted[0] - sorted[sorted.length-1]).to.be.at.most(200)
        })

        it("Falls back to using lecunnormal if there is no fanOut available", () => {
            sinon.spy(NetMath, "lecununiform")
            const result = NetMath.xavieruniform(10, {fanIn: 5})
            expect(NetMath.lecununiform).to.have.been.calledWith(10, {fanIn: 5})
            expect(result.length).to.equal(10)
            NetMath.lecununiform.restore()
        })
    })
})