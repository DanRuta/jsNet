"use strict"

const chaiAsPromised = require("chai-as-promised")
const chai = require("chai")
const assert = chai.assert
const expect = chai.expect
const sinonChai = require("sinon-chai")
const sinon = require("sinon")
chai.use(sinonChai)
chai.use(chaiAsPromised)

const {Network, Layer, FCLayer, ConvLayer, PoolLayer, Neuron, Filter, NetMath, NetUtil} = require("../dist/jsNetJS.concat.js")

describe("Loading", () => {

    it("Network is loaded", () => expect(Network).to.not.be.undefined)
    it("Layer is loaded", () => expect(Layer).to.not.be.undefined)
    it("Neuron is loaded", () => expect(Neuron).to.not.be.undefined)
    it("Filter is loaded", () => expect(Filter).to.not.be.undefined)
    it("NetMath is loaded", () => expect(NetMath).to.not.be.undefined)
    it("NetUtil is loaded", () => expect(NetUtil).to.not.be.undefined)
    it("FCLayer is loaded", () => expect(FCLayer).to.not.be.undefined)
    it("ConvLayer is loaded", () => expect(ConvLayer).to.not.be.undefined)
    it("PoolLayer is loaded", () => expect(PoolLayer).to.not.be.undefined)

    it("Loads Layer as an alias of FCLayer", () => {

        const layer = new Layer()
        const fclayer = new FCLayer()

        expect(Layer).to.equal(FCLayer)
        expect(layer).to.deep.equal(fclayer)
    })

    it("Statically returns the Network version when accessing via .version", () => {
        expect(Network.version).to.equal("3.2.0")
    })
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

            it("Defaults the cost function to 'meansquarederror'", () => {
                expect(net.cost).to.equal(NetMath.meansquarederror)
            })

            it("Defaults the updateFn to vanillasgd", () => {
                expect(net.updateFn).to.equal("vanillasgd")
            })

            it("Sets the net.weightUpdateFn to NetMath.vanillasgd when setting it to false, null, or 'vanillasgd'", () => {
                const net2 = new Network({updateFn: null})
                expect(net2.weightUpdateFn).to.equal(NetMath.vanillasgd)
                const net3 = new Network({updateFn: false})
                expect(net3.weightUpdateFn).to.equal(NetMath.vanillasgd)
                const net4 = new Network({updateFn: "vanillasgd"})
                expect(net4.weightUpdateFn).to.equal(NetMath.vanillasgd)
            })

            it("Sets the net.weightUpdateFn to NetMath.adagrad when setting it to 'adagrad'", () => {
                const net2 = new Network({updateFn: "adagrad"})
                expect(net2.weightUpdateFn).to.equal(NetMath.adagrad)
            })

            it("Defaults the net.rmsDecay to 0.99 if the updateFn is rmsprop", () => {
                const net2 = new Network({updateFn: "rmsprop"})
                expect(net2.rmsDecay).to.equal(0.99)
            })

            it("Sets the net.rmsDecay to use input, if supplied", () => {
                const net2 = new Network({updateFn: "rmsprop", rmsDecay: 0.9})
                expect(net2.rmsDecay).to.equal(0.9)
            })

            it("Does not set an rmsDecay, if updateFn is not rmsprop, even if supplied", () => {
                const net2 = new Network({updateFn: "adagrad", rmsDecay: 0.9})
                expect(net2.rmsDecay).to.be.undefined
            })

            it("Defaults the learning rate to 0.001 if the updateFn is rmsprop", () => {
                const net2 = new Network({updateFn: "rmsprop"})
                expect(net2.learningRate).to.equal(0.001)
            })

            it("Still allows user learning rates to be set, even if updateFn is rmsprop", () => {
                const net2 = new Network({updateFn: "rmsprop", learningRate: 0.5})
                expect(net2.learningRate).to.equal(0.5)
            })

            it("Defaults the learning rate to 0.01 if the updateFn is adam", () => {
                const net2 = new Network({updateFn: "adam"})
                expect(net2.learningRate).to.equal(0.01)
            })

            it("Still allows user learning rates to be set, even if updateFn is adam", () => {
                const net2 = new Network({updateFn: "adam", learningRate: 0.5})
                expect(net2.learningRate).to.equal(0.5)
            })

            it("Defaults the learning rate to 0.2 if the updateFn is momentum", () => {
                const net2 = new Network({updateFn: "momentum"})
                expect(net2.learningRate).to.equal(0.2)
            })

            it("Still allows user configurable learning rate, if the updateFn is momentum", () => {
                const net2 = new Network({updateFn: "momentum", learningRate: 0.1})
                expect(net2.learningRate).to.equal(0.1)
            })

            it("Defaults the net.rho to 0.95 if the updateFn is adadelta", () => {
                const net2 = new Network({updateFn: "adadelta"})
                expect(net2.rho).to.equal(0.95)
            })

            it("Still allows user rho values to be set", () => {
                const net2 = new Network({updateFn: "adadelta", rho: 0.5})
                expect(net2.rho).to.equal(0.5)
            })

            it("Still sets a rho value, even if a learning rate is given", () => {
                const net2 = new Network({updateFn: "adadelta", rho: 0.9, learningRate: 0.01})
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

            it("Sets the net.rreluSlope to a random number between 0 and 0.001", () => {
                expect(net.rreluSlope).to.be.at.least(0)
                expect(net.rreluSlope).to.be.at.most(0.001)
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

            it("Defaults the dropout to 1", () => {
                expect(net.dropout).to.equal(1)
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

            it("Leaves the net.l2 at 0 if the l2 parameter is set to false", () => {
                const net = new Network({l2: false})
                expect(net.l2).to.be.equal(0)
            })

            it("Leaves the net.l2 at 0 if the l2 parameter is not given", () => {
                expect(net.l2).to.be.equal(0)
            })

            it("Sets the l2 value to 0.001 if the configuration given is boolean true", () => {
                const net = new Network({l2: true})
                expect(net.l2).to.equal(0.001)
            })

            it("Sets the net.l1 value to the l1 value given as parameter", () => {
                const net = new Network({l1: 0.0005})
                expect(net.l1).to.equal(0.0005)
            })

            it("Leaves the net.l1 value a 0 if the l1 parameter is set to false", () => {
                const net = new Network({l1: false})
                expect(net.l1).to.be.equal(0)
            })

            it("Sets the l1 value to 0.005 if the configuration given is boolean true", () => {
                const net = new Network({l1: true})
                expect(net.l1).to.equal(0.005)
            })

            it("Leaves the net.l1 value a 0 if the l1 parameter is not given", () => {
                expect(net.l1).to.be.equal(0)
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

            it("Defaults the net.weightsConfig.distribution to xavieruniform", () => {
                const net = new Network({weightsConfig: {limit: 1}})
                expect(net.weightsConfig).to.not.be.undefined
                expect(net.weightsConfig.distribution).to.equal("xavieruniform")
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

            it("Sets the net.conv.filterSize to whatever value is given", () => {
                const net = new Network({conv: {filterSize: 3}})
                expect(net.conv.filterSize).to.equal(3)
            })

            it("Does not otherwise set net.conv.filterSize to anything", () => {
                expect(net.conv.filterSize).to.be.undefined
            })

            it("Sets the net.conv.zeroPadding to whatever value is given", () => {
                const net = new Network({conv: {zeroPadding: 1}})
                expect(net.conv.zeroPadding).to.equal(1)
            })

            it("Does not otherwise set net.conv.zeroPadding to anything", () => {
                expect(net.conv.zeroPadding).to.be.undefined
            })

            it("Sets the net.conv.stride to whatever value is given", () => {
                const net = new Network({conv: {stride: 1}})
                expect(net.conv.stride).to.equal(1)
            })

            it("Does not otherwise set net.conv.stride to anything", () => {
                expect(net.conv.stride).to.be.undefined
            })

            it("Sets the net.channels to whatever value is given", () => {
                const net = new Network({channels: 3})
                expect(net.channels).to.equal(3)
            })

            it("Does not otherwise set net.channels to anything", () => {
                expect(net.channels).to.be.undefined
            })

            it("Sets the net.pool.size to the value given", () => {
                const net = new Network({pool: {size: 3}})
                expect(net.pool.size).to.equal(3)
            })

            it("Does not otherwise set net.pool.size to anything", () => {
                expect(net.pool.size).to.be.undefined
            })

            it("Sets the net.pool.stride to the value given", () => {
                const net = new Network({pool: {stride: 1}})
                expect(net.pool.stride).to.equal(1)
            })

            it("Does not otherwise set net.pool.stride to anything", () => {
                expect(net.pool.stride).to.be.undefined
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
            const net = new Network({l2: false})
            expect(net.l2Error).to.be.undefined
        })

        it("Sets the initial net.l1Error to 0 if l1 is configured", () => {
            const net = new Network({l1: true})
            expect(net.l1Error).to.equal(0)
        })

        it("Doesn't set the net.l1Error if l1 is not configured", () => {
            const net = new Network({l1: false})
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

        it("Allows uppercase updateFn function configs (rmsprop when configuring as RMSProp)", () => {
            const net = new Network({updateFn: "RMSProp"})
            expect(net.updateFn).to.equal("rmsprop")
        })

        it("Allows snake_case updateFn function configs (rmsprop when configuring as rms_prop)", () => {
            const net = new Network({updateFn: "rms_prop"})
            expect(net.updateFn).to.equal("rmsprop")
        })

        it("Allows white space updateFn function configs (rmsprop when configuring as rms prop)", () => {
            const net = new Network({updateFn: "rms prop"})
            expect(net.updateFn).to.equal("rmsprop")
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

        it("Does not change a layer's activation function if it already has one", () => {
            layer1.activation = "something"
            net.layers = [layer1]
            net.activation = "something else"
            net.joinLayer(layer1)
            expect(layer1.activation).to.equal("something")
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
            net.layers = [layer1, layer2, layer3]
            layer2.weightsConfig = {}
            layer3.weightsConfig = {}
            net.joinLayer(layer2, 1)
            expect(layer2.weightsConfig.fanOut).to.equal(4)
        })

        it("Assigns a reference to the Network class to each layer", () => {
            net.layers = [layer1]
            layer1.weightsConfig = {}
            net.joinLayer(layer2, 1)
            expect(layer2.net).to.equal(net)
        })

        it("Sets the layer2 state to initialised", () => {
            net.layers = [layer1]
            layer1.weightsConfig = {}
            net.joinLayer(layer2, 1)
            expect(layer2.state).to.equal("initialised")
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

        it("Returns the softmax of the neurons in the last layer", () => {
            const layer1 = new Layer(3)
            const layer2 = new Layer(4)
            const layer3 = new Layer(7)
            const net = new Network({layers: [layer1, layer2, layer3]})

            const result = net.forward([1,2,3])
            const activations = net.layers[2].neurons.map(n => n.sum)

            expect(result).to.deep.equal(NetMath.softmax(activations))
        })

        it("Returns just the output without softmax if there is only one value in the output layer", () => {
            const layer1 = new Layer(3)
            const layer2 = new Layer(4)
            const layer3 = new Layer(1)
            const net = new Network({layers: [layer1, layer2, layer3]})

            const result = net.forward([1,2,3])
            const activations = net.layers[2].neurons.map(n => n.sum)

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

        it("Calls the every layer except the first's resetDeltaWeights() function", () => {
            const layer1 = new FCLayer(2)
            const layer2 = new FCLayer(2)
            const layer3 = new FCLayer(2)
            const net = new Network({layers: [layer1, layer2, layer3]})

            sinon.spy(layer1, "resetDeltaWeights")
            sinon.spy(layer2, "resetDeltaWeights")
            sinon.spy(layer3, "resetDeltaWeights")

            net.resetDeltaWeights()

            expect(layer1.resetDeltaWeights).to.not.be.called
            expect(layer2.resetDeltaWeights).to.be.called
            expect(layer3.resetDeltaWeights).to.be.called
        })
    })

    describe("applyDeltaWeights", () => {

        it("Calls ever layer except the first's applyDeltaWeights() function", () => {
            const layer1 = new Layer(2)
            const layer2 = new Layer(3)
            const layer3 = new Layer(4)

            const net = new Network({layers: [layer1, layer2, layer3]})

            sinon.spy(layer1, "applyDeltaWeights")
            sinon.spy(layer2, "applyDeltaWeights")
            sinon.spy(layer3, "applyDeltaWeights")

            net.applyDeltaWeights()
            expect(layer1.applyDeltaWeights).to.not.be.called
            expect(layer2.applyDeltaWeights).to.be.called
            expect(layer3.applyDeltaWeights).to.be.called
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
        const convNet = new Network({layers: [new FCLayer(1024), new ConvLayer(2, {filterSize: 3})]})

        it("Exports the correct number of layers", () => {
            const json = net.toJSON()
            expect(json.layers).to.not.be.undefined
            expect(json.layers).to.have.lengthOf(2)
        })

        it("Exports weights correctly", () => {

            layer2.neurons.forEach(neuron => neuron.weights = neuron.weights.map(w => 1))

            const json = net.toJSON()
            expect(json.layers[1].weights).to.not.be.undefined
            expect(json.layers[1].weights).to.have.lengthOf(3)
            expect(json.layers[1].weights[0].weights).to.deep.equal([1,1])
            expect(json.layers[1].weights[1].weights).to.deep.equal([1,1])
            expect(json.layers[1].weights[2].weights).to.deep.equal([1,1])
        })

        it("Exports bias correctly", () => {
            layer2.neurons.forEach(neuron => neuron.bias = 1)

            const json = net.toJSON()
            expect(json.layers[1].weights[0].bias).to.equal(1)
            expect(json.layers[1].weights[1].bias).to.equal(1)
            expect(json.layers[1].weights[2].bias).to.equal(1)
        })

        it("Exports a conv layer's weights correctly", () => {

            convNet.layers[1].filters[0].weights = [[[1,2,3],[4,5,6],[7,8,9]]]
            convNet.layers[1].filters[1].weights = [[[4,5,6],[7,8,9],[1,2,3]]]

            const convJson = convNet.toJSON()
            expect(convJson.layers[1].weights).to.not.be.undefined
            expect(convJson.layers[1].weights).to.have.lengthOf(2)
            expect(convJson.layers[1].weights[0].weights[0]).to.have.lengthOf(3)
            expect(convJson.layers[1].weights[0].weights[0][0]).to.have.lengthOf(3)
            expect(convJson.layers[1].weights[0].weights).to.deep.equal([[[1,2,3],[4,5,6],[7,8,9]]])
            expect(convJson.layers[1].weights[1].weights).to.deep.equal([[[4,5,6],[7,8,9],[1,2,3]]])
        })

        it("Exports a conv layer's bias correctly", () => {

            convNet.layers[1].filters[0].bias = 1
            convNet.layers[1].filters[1].bias = 2

            const convJson = convNet.toJSON()
            expect(convJson.layers[1].weights[0].bias).to.equal(1)
            expect(convJson.layers[1].weights[1].bias).to.equal(2)
        })
    })

    describe("fromJSON", () => {

        const testData = {
            layers: [
                {
                    weights: [{},{}]
                },
                {
                    weights: [
                        {bias: 1, weights: [1,1]},
                        {bias: 2, weights: [2,2]},
                        {bias: 3, weights: [3,3]}
                    ]
                }
            ]
        }

        const testDataConv = {
            layers: [
                {
                    weights: [{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}]
                },
                {
                    weights: [
                        {bias: 1, weights: [[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]],[[13,14],[15,16]]]},
                        {bias: 2, weights: [[[5,6],[7,8]],[[9,10],[11,12]],[[13,14],[15,16]],[[1,2],[3,4]]]}
                    ]
                }
            ]
        }

        it("Throws an error if no data is given", () => {
            const net = new Network()
            expect(net.fromJSON.bind(net)).to.throw("No JSON data given to import.")
            expect(net.fromJSON.bind(net, null)).to.throw("No JSON data given to import.")
        })

        it("Throws an error if the number of layers in the import data does not match the net's", () => {
            const net = new Network({layers: [2,3,4,5]})
            expect(net.fromJSON.bind(net, testData)).to.throw("Mismatched layers (2 layers in import data, but 4 configured)")
        })

        it("Throws an error if the FCLayer weights container shape is mismatched", () => {
            const net = new Network({layers: [3,3]})
            expect(net.fromJSON.bind(net, testData)).to.throw("Mismatched weights count. Given: 2 Existing: 3. At layers[1], neurons[0]")
        })

        it("Calls the resetDeltaWeights() function", () => {
            const net = new Network({layers: [new FCLayer(2), new FCLayer(3)]})
            sinon.spy(net, "resetDeltaWeights")
            net.fromJSON(testData)
            expect(net.resetDeltaWeights).to.be.called
        })

        it("Sets the weights and biases to the import data values", () => {
            const net = new Network({layers: [new FCLayer(2), new FCLayer(3)]})
            net.fromJSON(testData)
            expect(net.layers[1].neurons[0].bias).to.equal(1)
            expect(net.layers[1].neurons[1].bias).to.equal(2)
            expect(net.layers[1].neurons[2].bias).to.equal(3)
            expect(net.layers[1].neurons[0].weights).to.deep.equal([1,1])
            expect(net.layers[1].neurons[1].weights).to.deep.equal([2,2])
            expect(net.layers[1].neurons[2].weights).to.deep.equal([3,3])
        })

        it("Sets the weights and biases in a conv layer to the import data values", () => {
            const net = new Network({channels: 4, layers: [new FCLayer(16), new ConvLayer(2, {filterSize: 2})]})
            net.fromJSON(testDataConv)
            expect(net.layers[1].filters[0].bias).to.equal(1)
            expect(net.layers[1].filters[1].bias).to.equal(2)
            expect(net.layers[1].filters[0].weights).to.deep.equal([[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]],[[13,14],[15,16]]])
            expect(net.layers[1].filters[1].weights).to.deep.equal([[[5,6],[7,8]],[[9,10],[11,12]],[[13,14],[15,16]],[[1,2],[3,4]]])
        })

        it("Throws an error if the ConvLayer weights depth is mismatched", () => {
            const net = new Network({channels: 1, layers: [new FCLayer(16), new ConvLayer(2, {filterSize: 2})]})
            expect(net.fromJSON.bind(net, testDataConv)).to.throw("Mismatched weights depth. Given: 4 Existing: 1. At: layers[1], filters[0]")
        })

        it("Throws an error if the ConvLayer weights spacial dimension is mismatched", () => {
            const net = new Network({channels: 4, layers: [new FCLayer(16), new ConvLayer(2, {filterSize: 3})]})
            expect(net.fromJSON.bind(net, testDataConv)).to.throw("Mismatched weights size. Given: 2 Existing: 3. At: layers[1], filters[0]")
        })
    })

    describe("train", () => {

        let net

        beforeEach(() => {
            net = new Network({layers: [2, 3, 2], updateFn: null, l2: false})
            sinon.stub(net, "forward").callsFake(() => [1,1])
            sinon.stub(net, "backward")
            sinon.stub(net, "resetDeltaWeights")
            sinon.stub(net, "applyDeltaWeights")
            sinon.stub(net, "initLayers")
            sinon.stub(NetMath, "meansquarederror")
            sinon.stub(console, "log") // Get rid of output spam
        })

        afterEach(() => {
            net.forward.restore()
            net.backward.restore()
            net.resetDeltaWeights.restore()
            net.applyDeltaWeights.restore()
            net.initLayers.restore()
            NetMath.meansquarederror.restore()
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

        it("Rejects the promise if some data does not have the key 'input' and 'expected'", () => {
            return expect(net.train(badTestData)).to.be.rejectedWith("Data set must be a list of objects with keys: 'input' and 'expected")
        })

        it("Resolves the promise when you give it data", () => {
            return expect(net.train(testData)).to.be.fulfilled
        })

        it("Does not accept 'output' as an alternative name for expected values", () => {
            return expect(net.train(testDataWithOutput)).to.not.be.fulfilled
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

        it("Calls the backward function with the errors, calculated as target-output, inverted for the correct classes", () => {
            return net.train([testData[1]]).then(() => {
                expect(net.backward).to.be.calledWith([-1, 0])
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

        it("Calls a given callback with an object containing keys: 'elapsed', 'iterations', 'validations', 'trainingError', 'validationError', and 'input', for each iteration", () => {
            sinon.stub(console, "warn")

            return net.train(testData, {callback: console.warn}).then(() => {
                expect(console.warn).to.have.been.called
                expect(console.warn.callCount).to.equal(4)
                expect(console.warn).to.have.been.calledWith(sinon.match.has("iterations"))
                expect(console.warn).to.have.been.calledWith(sinon.match.has("validations"))
                expect(console.warn).to.have.been.calledWith(sinon.match.has("trainingError"))
                expect(console.warn).to.have.been.calledWith(sinon.match.has("validationError"))
                expect(console.warn).to.have.been.calledWith(sinon.match.has("input"))
                expect(console.warn).to.have.been.calledWith(sinon.match.has("elapsed"))
                console.warn.restore()
            })
        })

        it("Calls the initLayers function when the net state is not 'initialised'", () => {
            const network = new Network({updateFn: null})
            sinon.stub(network, "forward").callsFake(() => [1,1])
            sinon.spy(network, "initLayers")

            return network.train(testData).then(() => {
                expect(network.initLayers).to.have.been.called
            })
        })

        it("Calls the initLayers function with the length of the first input and length of first expected", () => {
            const network = new Network({updateFn: null})
            sinon.stub(network, "forward").callsFake(() => [1,1])
            sinon.spy(network, "initLayers")

            return network.train(testData).then(() => {
                expect(network.initLayers).to.have.been.calledWith(2, 2)
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
            return net.train(testData, {miniBatchSize: 2, callback: ()=>{}, validation: {data: testData, rate: 0}}).then(() => {
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
            sinon.stub(NetUtil, "shuffle")
            return net.train(testData, {shuffle: true}).then(() => {
                expect(NetUtil.shuffle).to.be.calledWith(testData)
                NetUtil.shuffle.restore()
            })
        })

        it("Runs validation when validation data is given", () => {
            sinon.spy(net, "validate")
            return net.train(testData, {epochs: 10, validation: {data: testData, interval: 2}}).then(() => {
                expect(net.validate).to.be.called
                net.validate.restore()
            })
        })

        it("Defaults validation interval to 1 epoch (number of training samples)", () => {
            return net.train([...testDataX10, ...testDataX10], {validation: {data: testDataX10}}).then(() => {
                expect(net.validation.interval).to.equal(20)
            })
        })

        it("Allows setting the validation rate to custom value", () => {
            return net.train([...testDataX10, ...testDataX10], {validation: {data: testDataX10, interval: 5}}).then(() => {
                expect(net.validation.interval).to.equal(5)
            })
        })

        describe("Early stopping", () => {

            it("Defaults the threshold to 0.01 when the type is 'threshold'", () => {
                return net.train(testData, {validation: {data: testData, earlyStopping: {
                    type: "threshold"
                }}}).then(() => {
                    expect(net.validation.earlyStopping.threshold).to.equal(0.01)
                })
            })

            it("Allows setting a custom threshold value", () => {
                return net.train(testData, {validation: {data: testData, earlyStopping: {
                    type: "threshold",
                    threshold: 0.2
                }}}).then(() => {
                    expect(net.validation.earlyStopping.threshold).to.equal(0.2)
                })
            })

            it("Checks the early stopping with each validation", () => {
                sinon.stub(net, "checkEarlyStopping")
                return net.train(testDataX10, {validation: {
                    data: testData,
                    interval: 1,
                    earlyStopping: {
                        type: "threshold",
                        threshold: 0.2
                    }
                }}).then(() => {
                    expect(net.checkEarlyStopping.callCount).to.equal(9)
                    net.checkEarlyStopping.restore()
                })
            })


            it("Defaults the patience to 20 when the type is 'patience'", () => {
                for (let l=0; l<net.layers.length; l++) {
                    net.layers[l].restoreValidation = () => {}
                }
                return net.train(testData, {validation: {data: testData, earlyStopping: {
                    type: "patience"
                }}}).then(() => {
                    expect(net.validation.earlyStopping.patience).to.equal(20)
                })
            })

            it("Allows setting a custom threshold value", () => {
                for (let l=0; l<net.layers.length; l++) {
                    net.layers[l].restoreValidation = () => {}
                }
                return net.train(testData, {validation: {data: testData, earlyStopping: {
                    type: "patience",
                    patience: 0.2
                }}}).then(() => {
                    expect(net.validation.earlyStopping.patience).to.equal(0.2)
                })
            })

            it("Sets the bestError to Infinity and patienceCounter to 0", () => {
                for (let l=0; l<net.layers.length; l++) {
                    net.layers[l].restoreValidation = () => {}
                }
                return net.train(testData, {validation: {data: testData, earlyStopping: {
                    type: "patience"
                }}}).then(() => {
                    expect(net.validation.earlyStopping.bestError).to.equal(Infinity)
                    expect(net.validation.earlyStopping.patienceCounter).to.equal(0)
                })
            })

            it("Restores the validation values at the end of the training when using 'patience'", () => {
                for (let l=0; l<net.layers.length; l++) {
                    sinon.stub(net.layers[l], "restoreValidation")
                }

                return net.train(testData, {validation: {data: testData, earlyStopping: {
                    type: "patience"
                }}}).then(() => {

                    expect(net.layers[0].restoreValidation.callCount).to.equal(0)

                    for (let l=1; l<net.layers.length; l++) {
                        expect(net.layers[l].restoreValidation.callCount).to.equal(1)
                    }
                })
            })

            it("Defaults the divergence percent to 30 when the type is 'divergence'", () => {
                for (let l=0; l<net.layers.length; l++) {
                    net.layers[l].restoreValidation = () => {}
                }
                return net.train(testData, {validation: {data: testData, earlyStopping: {
                    type: "divergence"
                }}}).then(() => {
                    expect(net.validation.earlyStopping.percent).to.equal(30)
                })
            })

            it("Allows setting a custom percent value", () => {
                for (let l=0; l<net.layers.length; l++) {
                    net.layers[l].restoreValidation = () => {}
                }
                return net.train(testData, {validation: {data: testData, earlyStopping: {
                    type: "divergence",
                    percent: 0.2
                }}}).then(() => {
                    expect(net.validation.earlyStopping.percent).to.equal(0.2)
                })
            })

            it("Sets the bestError to Infinity", () => {
                for (let l=0; l<net.layers.length; l++) {
                    net.layers[l].restoreValidation = () => {}
                }
                return net.train(testData, {validation: {data: testData, earlyStopping: {
                    type: "divergence"
                }}}).then(() => {
                    expect(net.validation.earlyStopping.bestError).to.equal(Infinity)
                })
            })


            it("Stops the training when checkEarlyStopping returns true", () => {
                let checkEarlyCounter = 0
                const stub = sinon.stub(net, "checkEarlyStopping").callsFake(() => ++checkEarlyCounter>=5)

                return net.train(testDataX10, {validation: {
                    data: testData,
                    interval: 1,
                    earlyStopping: {
                        type: "threshold",
                        threshold: 0.2
                    }
                }}).then(() => {
                    expect(stub.callCount).to.equal(5)
                    stub.restore()
                })
            })

        })
    })

    describe("checkEarlyStopping", () => {

        let net

        before(() => {
            net = new Network()
            net.initLayers([1],[2])
            net.validation = {earlyStopping: {}}
        })

        describe("threshold", () => {
            it("Returns true when the lastValidationError is equal or lower than the threshold", () => {
                net.validation.earlyStopping.type = "threshold"
                net.validation.earlyStopping.threshold = 0.01
                net.lastValidationError = 0.005
                expect(net.checkEarlyStopping([1,2,3])).to.be.true
            })
            it("Returns false when the lastValidationError is not equal or lower than the threshold", () => {
                net.validation.earlyStopping.type = "threshold"
                net.validation.earlyStopping.threshold = 0.01
                net.lastValidationError = 0.5
                expect(net.checkEarlyStopping([1,2,3])).to.be.false
            })
        })

        describe("patience", () => {
            it("Returns true when the patienceCounter is higher than the patience hyperparameter, and new validation is not the best", () => {
                net.validation.earlyStopping.type = "patience"
                net.validation.earlyStopping.patienceCounter = 5
                net.validation.earlyStopping.patience = 4

                expect(net.checkEarlyStopping([1,2,3])).to.be.true
            })
            it("Increments the patienceCounter and returns false, when patienceCounter is lower than patience hyperparameter and new validation is not the best", () => {
                net.validation.earlyStopping.type = "patience"
                net.validation.earlyStopping.patienceCounter = 2
                net.validation.earlyStopping.patience = 4

                expect(net.checkEarlyStopping([1,2,3])).to.be.false
                expect(net.validation.earlyStopping.patienceCounter).to.equal(3)
            })

            it("Backs up the layer weights when a new best validation error is calculated, sets bestError, resets patienceCounter, and returns false", () => {
                net.validation.earlyStopping.type = "patience"
                net.lastValidationError = 2
                net.validation.earlyStopping.patienceCounter = 4
                net.validation.earlyStopping.bestError = 4
                const layer1 = new FCLayer(1)
                const layer2 = new FCLayer(2)
                const layer3 = new FCLayer(3)
                net.layers = [layer1, layer2, layer3]

                sinon.stub(layer1, "backUpValidation")
                sinon.stub(layer2, "backUpValidation")
                sinon.stub(layer3, "backUpValidation")

                expect(net.checkEarlyStopping([1,2,3])).to.be.false
                expect(layer1.backUpValidation).to.not.be.called
                expect(layer2.backUpValidation).to.be.called
                expect(layer3.backUpValidation).to.be.called

                expect(net.validation.earlyStopping.bestError).to.equal(2)
                expect(net.validation.earlyStopping.patienceCounter).to.equal(0)
            })
        })

        describe("divergence", () => {
            it("Returns true when the validation error is higher than the best by at least the defined percent amount", () => {
                net.validation.earlyStopping.type = "divergence"
                net.validation.earlyStopping.bestError = 1
                net.lastValidationError = 1.15
                net.validation.earlyStopping.percent = 15

                expect(net.checkEarlyStopping([1,2,3])).to.be.true
            })
            it("Backs up the layer weights when a new best validation error is calculated, sets bestError, and returns false", () => {
                net.validation.earlyStopping.type = "divergence"
                net.validation.earlyStopping.bestError = 1.16
                net.lastValidationError = 1.15

                const layer1 = new FCLayer(1)
                const layer2 = new FCLayer(2)
                const layer3 = new FCLayer(3)
                net.layers = [layer1, layer2, layer3]

                sinon.stub(layer1, "backUpValidation")
                sinon.stub(layer2, "backUpValidation")
                sinon.stub(layer3, "backUpValidation")

                expect(net.checkEarlyStopping([1,2,3])).to.be.false
                expect(layer1.backUpValidation).to.not.be.called
                expect(layer2.backUpValidation).to.be.called
                expect(layer3.backUpValidation).to.be.called

                expect(net.validation.earlyStopping.bestError).to.equal(1.15)
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

        it("Accepts test data with output key instead of expected", () => {
            return net.test(testDataOutput).then(() => {
                expect(net.cost.callCount).to.equal(4)
            })
        })

        it("Logs to the console twice", () => {
            sinon.spy(console, "log")
            return net.test(testData).then(() => {
                expect(console.log.callCount).to.equal(2)
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

        it("Calls a given callback with an object containing keys: 'elapsed', 'iterations', 'error' and 'input', for each iteration", () => {
            sinon.stub(console, "warn")

            return net.test(testData, {callback: console.warn}).then(() => {
                expect(console.warn).to.have.been.called
                expect(console.warn.callCount).to.equal(4)
                expect(console.warn).to.have.been.calledWith(sinon.match.has("iterations"))
                expect(console.warn).to.have.been.calledWith(sinon.match.has("error"))
                expect(console.warn).to.have.been.calledWith(sinon.match.has("input"))
                expect(console.warn).to.have.been.calledWith(sinon.match.has("elapsed"))
                console.warn.restore()
            })
        })
    })

    describe("toIMG", () => {
        it("Throws an error if IMGArrays is not provided", () => {
            const net = new Network()
            expect(net.toIMG).to.throw("The IMGArrays library must be provided. See the documentation for instructions.")
        })

        it("Calls every layer except the first's toIMG function", () => {
            const net = new Network()
            const l1 = new FCLayer(784)
            const l2 = new FCLayer(100)
            const l3 = new FCLayer(10)
            sinon.stub(l1, "toIMG")
            sinon.stub(l2, "toIMG").callsFake(() => [1,2,3])
            sinon.stub(l3, "toIMG").callsFake(() => [1,2,3])
            net.layers = [l1, l2, l3]
            net.toIMG({toIMG: () => {}})
            expect(l1.toIMG).to.not.be.called
            expect(l2.toIMG).to.be.called
            expect(l3.toIMG).to.be.called
        })
    })

    describe("fromIMG", () => {
        it("Throws an error if IMGArrays is not provided", () => {
            const net = new Network()
            expect(net.fromIMG).to.throw("The IMGArrays library must be provided. See the documentation for instructions.")
        })

        it("Calls every layer except the first's fromIMG function with the data segment matching their size", () => {
            const net = new Network()
            const l1 = new FCLayer(784)
            const l2 = new FCLayer(100)
            const l3 = new FCLayer(10)
            sinon.stub(l1, "fromIMG")
            sinon.stub(l2, "fromIMG")
            sinon.stub(l2, "getDataSize").callsFake(() => 4)
            sinon.stub(l3, "fromIMG")
            sinon.stub(l3, "getDataSize").callsFake(() => 3)
            net.layers = [l1, l2, l3]

            const fakeIMGArrays = {fromIMG: () => [1,2,3,4, 5,6,7]}
            net.fromIMG(null, fakeIMGArrays)

            expect(l1.fromIMG).to.not.be.called
            expect(l2.fromIMG).to.be.calledWith([1,2,3,4])
            expect(l3.fromIMG).to.be.calledWith([5,6,7])
        })
    })
})

describe("FCLayer", () => {

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

        it("Allows setting the activation function to a custom function", () => {
            const customFn = x => x
            const layer = new FCLayer(2, {activation: customFn})
            expect(layer.activation).to.equal(customFn)
        })

        it("Allows setting the activation to false by giving the value false", () => {
            const layer = new FCLayer(5, {activation: false})
            expect(layer.activation).to.be.false
        })

        it("Allows setting the activation function to a function from NetMath using a string", () => {
            const layer = new FCLayer(5, {activation: "relu"})
            expect(layer.activation.name).to.equal("bound relu")
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
        it("Adds a reference to a layer to its prevLayer property", () => {
            const layer1 = new Layer(2)
            const layer2 = new Layer(3)
            layer2.assignPrev(layer1)
            expect(layer2.prevLayer).to.equal(layer1)
        })

        it("Assigns the layer.layerIndex to the value given", () => {
            const layer1 = new Layer(2)
            const layer2 = new Layer(3)
            layer2.assignPrev(layer1, 12345)
            expect(layer2.layerIndex).to.equal(12345)
        })
    })

    describe("init", () => {

        const layer = new Layer()
        let layer1
        let layer2

        beforeEach(() => {
            layer1 = new Layer(2)
            layer2 = new Layer(2)
            layer2.weightsConfig = {limit: 0.1}
            layer2.net = {weightsInitFn: NetMath.xavieruniform}
            sinon.stub(layer2.neurons[0], "init")
            sinon.stub(layer2.neurons[1], "init")
        })

        afterEach(() => {
            layer2.neurons[0].init.restore()
            layer2.neurons[1].init.restore()
        })

        it("Creates the neuron its weights array, with .length the same as given parameter", () => {
            layer2.assignPrev(layer1)
            layer2.init()
            expect(layer2.neurons[0].weights.length).to.equal(2)
        })

        it("Sets the bias to 0", () => {
            layer2.assignPrev(layer1)
            layer2.init()
            expect(layer2.neurons[0].bias).to.equal(1)
        })

        it("Inits all the neurons in the layer's neurons array", () => {
            layer2.assignPrev(layer)
            layer2.init()
            expect(layer2.neurons[0].init).to.have.been.called
            expect(layer2.neurons[1].init).to.have.been.called
        })

        it("Calls the neuron's init function with updateFn and activationConfig", () => {
            layer2.net.updateFn = "test"
            layer2.net.activationConfig = "stuff"
            layer2.assignPrev(layer1)
            layer2.init()
            expect(layer2.neurons[0].init).to.have.been.calledWith(sinon.match({"updateFn": "test"}))
            expect(layer2.neurons[0].init).to.have.been.calledWith(sinon.match({"activationConfig": "stuff"}))
            expect(layer2.neurons[1].init).to.have.been.calledWith(sinon.match({"updateFn": "test"}))
            expect(layer2.neurons[1].init).to.have.been.calledWith(sinon.match({"activationConfig": "stuff"}))
        })

        it("Calls the neuron's init function with eluAlpha", () => {
            layer2.net.eluAlpha = 1
            layer2.assignPrev(layer1)
            layer2.init()
            expect(layer2.neurons[0].init).to.have.been.calledWith(sinon.match({"eluAlpha": 1}))
            expect(layer2.neurons[1].init).to.have.been.calledWith(sinon.match({"eluAlpha": 1}))
        })

        it("Calls the NetMath.xavieruniform function when the weightsInitFn is xavieruniform", () => {
            sinon.stub(layer2.net, "weightsInitFn")
            layer2.assignPrev(layer1)
            layer2.init()
            expect(layer2.net.weightsInitFn).to.be.called
            layer2.net.weightsInitFn.restore()
        })

        it("When prevLayer is Conv, the number of weights is set to match each activation in the activation maps", () => {
            const convLayer = new ConvLayer(3)
            convLayer.filters = [new Filter(),new Filter(),new Filter()]
            convLayer.outMapSize = 7
            layer2.assignPrev(convLayer)
            layer2.init()
            expect(layer2.neurons[0].weights.length).to.equal(147) // 3 * 7 * 7
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

        it("Sets the neuron's activation to its sum when no activation function has been set", () => {
            const net = new Network({
                activation: "sigmoid",
                layers: [layer1, layer2],
                weightsConfig: {limit: 0.1},
                dropout: 1
            })
            layer2.activation = false

            net.forward([1,2])
            expect(layer2.neurons[0].activation).to.equal(layer2.neurons[0].sum)
            expect(layer2.neurons[1].activation).to.equal(layer2.neurons[1].sum)
            expect(layer2.neurons[2].activation).to.equal(layer2.neurons[2].sum)
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

        it("Defaults to dropout value of 1 (and thus, not dropping out any neurons) when net.dropout is not defined", () => {
            const net = new Network({
                activation: "sigmoid",
                layers: [layer1, layer2],
                weightsConfig: {limit: 0.1}
            })

            layer2.net.dropout = undefined
            net.forward([1,2])
            expect(layer2.neurons[0].activation).to.not.equal(0)
            expect(layer2.neurons[1].activation).to.not.equal(0)
            expect(layer2.neurons[2].activation).to.not.equal(0)
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

        it("Sets each neurons' error to the given error, in output layer", () => {
            const net = new Network({layers: [layer1, layer2]})

            layer2.neurons.forEach(neuron => neuron.activation = 0.5)
            net.backward([1,2,3])
            expect(layer2.neurons.map(n => n.error)).to.deep.equal([1, 2, 3])
        })

        it("Sets each neurons' derivative to the layer's activation function prime of the neuron's sum", () => {
            layer2.neurons.forEach(neuron => neuron.sum = 0.5)
            net.backward([1,2,3,4])
            const expectedDerivatives = [...new Array(3)].map(v => NetMath.sigmoid(0.5, true))

            expect(layer2.neurons.map(n => n.derivative)).to.deep.equal(expectedDerivatives)
        })

        it("Sets the neuron derivative to 1 when no activation function has been set", () => {
            layer2.activation = false
            layer2.neurons.forEach(neuron => neuron.sum = 0.5)
            net.backward([1,2,3,4])

            expect(layer2.neurons.map(n => n.derivative)).to.deep.equal([1,1,1])
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
            expect(layer3.neurons[0].deltaWeights).to.deep.equal([0.5, 0.5, 0.5])
            expect(layer3.neurons[1].deltaWeights).to.deep.equal([1, 1, 1])
            expect(layer3.neurons[2].deltaWeights).to.deep.equal([1.5, 1.5, 1.5])
            expect(layer3.neurons[3].deltaWeights).to.deep.equal([2, 2, 2])
        })

        it("Increments each neuron's deltaBias to the its error", () => {
            layer3.neurons.forEach(neuron => {
                neuron.deltaBias = 1
                neuron.activation = 0.5
            })
            layer3.backward([1,2,3,4])
            expect(layer3.neurons.map(n => n.deltaBias)).to.deep.equal([2, 3, 4, 5])
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

            expect(layer2.neurons[0].deltaBias).to.equal("test0")
            expect(layer2.neurons[0].deltaWeights).to.deep.equal(["test", "test"])
            expect(layer2.neurons[0].derivative).to.equal("test")
        })

        // it("Sets weight deltas to the normal delta + the l2 value", () => {
        it("Sets weight deltas to the normal delta", () => {
            layer2.neurons.forEach(neuron => neuron.activation = 0.5)
            layer3.neurons.forEach(neuron => {
                neuron.deltaWeights = [0.25,0.25,0.25,0.25]
                neuron.activation = 0.25
            })
            layer3.net = {l2: 0.001, miniBatchSize: 1}

            layer3.backward([0.3, 0.3, 0.3, 0.3])
            expect(layer3.neurons[0].deltaWeights[0].toFixed(6)).to.equal("0.400000")
        })

        it("Sets weight deltas to the normal delta", () => {
            layer2.neurons.forEach(neuron => neuron.activation = 0.5)
            layer3.neurons.forEach(neuron => {
                neuron.deltaWeights = [0.25,0.25,0.25,0.25]
                neuron.activation = 0.25
            })
            layer3.net = {l1: 0.005, miniBatchSize: 1}

            layer3.backward([0.3, 0.3, 0.3, 0.3])
            expect(layer3.neurons[0].deltaWeights[0].toFixed(6)).to.equal("0.400000")
        })
    })

    describe("resetDeltaWeights", () => {

        it("Clears the neuron deltaBias", () => {
            const layer1 = new Layer(2)
            const layer2 = new Layer(2)
            const net = new Network({layers: [layer1, layer2]})
            layer2.neurons.forEach(neuron => neuron.deltaBias = 1)

            layer2.resetDeltaWeights()
            expect(layer2.neurons[0].deltaBias).to.equal(0)
            expect(layer2.neurons[1].deltaBias).to.equal(0)
        })

        it("Sets the delta weights of all neurons to 0", () => {
            const layer1 = new Layer(2)
            const layer2 = new Layer(2)
            const net = new Network({layers: [layer1, layer2]})
            layer2.neurons.forEach(neuron => neuron.deltaWeights = [1,1])

            layer2.resetDeltaWeights()
            expect(layer2.neurons[0].deltaWeights).to.deep.equal([0,0])
            expect(layer2.neurons[1].deltaWeights).to.deep.equal([0,0])
        })
    })

    describe("applyDeltaWeights", () => {

        it("Increments the weights of all neurons with their respective deltas (when learning rate is 1)", () => {
            const layer1 = new Layer(2)
            const layer2 = new Layer(3)
            const net = new Network({learningRate: 1, l1: false, l2: false, layers: [layer1, layer2], updateFn: "vanillasgd"})
            net.miniBatchSize = 1

            layer2.neurons.forEach(neuron => neuron.weights = [0.25, 0.25])
            layer2.neurons.forEach(neuron => neuron.deltaWeights = [0.5, 0.5])

            layer2.applyDeltaWeights()

            expect(layer2.neurons[0].weights).to.deep.equal([0.75, 0.75])
            expect(layer2.neurons[1].weights).to.deep.equal([0.75, 0.75])
            expect(layer2.neurons[2].weights).to.deep.equal([0.75, 0.75])
        })

        it("Increments the bias of all neurons with their deltaBias", () => {
            const layer1 = new Layer(2)
            const layer2 = new Layer(3)
            const net = new Network({learningRate: 1, layers: [layer1, layer2], updateFn: "vanillasgd"})

            layer2.neurons.forEach(neuron => neuron.bias = 0.25)
            layer2.neurons.forEach(neuron => neuron.deltaBias = 0.5)

            layer2.applyDeltaWeights()

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
            layer2.applyDeltaWeights()

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
            layer2.applyDeltaWeights()

            expect(net.l1Error).to.equal(0.0025)

            net.weightUpdateFn.restore()
        })

        it("Increments the net.maxNormTotal if the net.maxNorm is configured", () => {
            const layer2 = new Layer(1)
            const net = new Network({maxNorm: 3, layers: [new Layer(2), layer2]})

            layer2.neurons.forEach(neuron => neuron.weights = [0.25, 0.25])

            sinon.stub(net, "weightUpdateFn").callsFake(x => x)
            sinon.stub(NetMath, "maxNorm").callsFake(x => x)
            layer2.applyDeltaWeights()

            expect(net.maxNormTotal).to.equal(0.125) // sqrt ( 2 * 0.25**2 )
            net.weightUpdateFn.restore()
            NetMath.maxNorm.restore()
        })

        it("Does not increment net.maxNormTotal if the net.maxNorm is not configured", () => {
            const layer2 = new Layer(1)
            const net = new Network({layers: [new Layer(2), layer2]})

            layer2.neurons.forEach(neuron => neuron.weights = [0.25, 0.25])

            sinon.stub(net, "weightUpdateFn").callsFake(x => x)
            layer2.applyDeltaWeights()

            expect(net.maxNormTotal).to.be.undefined

            net.weightUpdateFn.restore()
        })
    })

    describe("backUpValidation", () => {
        it("Copies the neuron weights to a 'validationWeights' array, for each neuron", () => {
            const layer = new FCLayer(10)
            const prevLayer = new FCLayer(10)
            layer.assignPrev(prevLayer)
            layer.net = {weightsInitFn: x=> [...new Array(x)].map((_,i) => i)}
            layer.init()
            for (let n=0; n<layer.neurons.length; n++) {
                expect(layer.neurons[n].validationWeights).to.be.undefined
            }

            layer.backUpValidation()

            for (let n=0; n<layer.neurons.length; n++) {
                expect(layer.neurons[n].validationWeights).to.deep.equal(layer.neurons[n].weights)
            }
        })

        it("Copies over the neuron biases into a 'validationBias' value", () => {
            const layer = new FCLayer(10)
            const prevLayer = new FCLayer(10)
            layer.assignPrev(prevLayer)
            layer.net = {weightsInitFn: x=> [...new Array(x)].map((_,i) => i)}
            layer.init()

            for (let n=0; n<layer.neurons.length; n++) {
                expect(layer.neurons[n].validationBias).to.be.undefined
            }

            layer.backUpValidation()

            for (let n=0; n<layer.neurons.length; n++) {
                expect(layer.neurons[n].validationBias).to.equal(layer.neurons[n].bias)
            }
        })
    })

    describe("restoreValidation", () => {
        it("Copies backed up 'validationWeights' values into every neuron's weights arrays", () => {
            const layer = new FCLayer(10)
            const prevLayer = new FCLayer(10)
            layer.assignPrev(prevLayer)
            layer.net = {weightsInitFn: x=> [...new Array(x)].map((_,i) => i)}
            layer.init()
            for (let n=0; n<layer.neurons.length; n++) {
                layer.neurons[n].validationWeights = [1,2,3]
                expect(layer.neurons[n].weights).to.not.deep.equal([1,2,3])
            }

            layer.restoreValidation()

            for (let n=0; n<layer.neurons.length; n++) {
                expect(layer.neurons[n].weights).to.deep.equal([1,2,3])
            }
        })
        it("Copies backed up 'validationBias' values into every neuron's bias values", () => {
            const layer = new FCLayer(10)
            const prevLayer = new FCLayer(10)
            layer.assignPrev(prevLayer)
            layer.net = {weightsInitFn: x=> [...new Array(x)].map((_,i) => i)}
            layer.init()
            for (let n=0; n<layer.neurons.length; n++) {
                layer.neurons[n].validationWeights = [1,2,3]
                layer.neurons[n].validationBias = n + 5
                expect(layer.neurons[n].bias).to.not.equal(n+5)
            }

            layer.restoreValidation()

            for (let n=0; n<layer.neurons.length; n++) {
                expect(layer.neurons[n].bias).to.equal(n+5)
            }
        })
    })

    describe("getDataSize", () => {
        it("Returns the correct total number of weights and biases (Example 1)", () => {
            const fc = new FCLayer(5)

            for (let n=0; n<fc.neurons.length; n++) {
                fc.neurons[n].weights = [1,2,3]
                fc.neurons[n].bias = 1
            }

            expect(fc.getDataSize()).to.equal(20)
        })
        it("Returns the correct total number of weights and biases (Example 1)", () => {
            const fc = new FCLayer(15)

            for (let n=0; n<fc.neurons.length; n++) {
                fc.neurons[n].weights = [1,2,3,4]
                fc.neurons[n].bias = 1
            }

            expect(fc.getDataSize()).to.equal(75)
        })
    })

    describe("toIMG", () => {
        it("Returns all neurons' weights and biases as a 1 dimensional array (Example 1)", () => {

            const fc = new FCLayer(5)

            for (let n=0; n<fc.neurons.length; n++) {
                fc.neurons[n].weights = [1,2,3]
                fc.neurons[n].bias = 1
            }

            expect(fc.toIMG()).to.deep.equal([1,1,2,3,1,1,2,3,1,1,2,3,1,1,2,3,1,1,2,3])
        })
        it("Returns all neurons' weights and biases as a 1 dimensional array (Example 2)", () => {

            const fc = new FCLayer(2)

            for (let n=0; n<fc.neurons.length; n++) {
                fc.neurons[n].weights = [1,2,4,5]
                fc.neurons[n].bias = 1
            }

            expect(fc.toIMG()).to.deep.equal([1,1,2,4,5,1,1,2,4,5])
        })
    })

    describe("fromIMG", () => {
        it("Sets the weights and biases to the given 1 dimensional array (Example 1)", () => {
            const testData = [2,1,2,3, 3,1,2,4]

            const fc = new FCLayer(2)

            for (let n=0; n<fc.neurons.length; n++) {
                fc.neurons[n].weights = [1,1,1]
                fc.neurons[n].bias = 1
            }

            fc.fromIMG(testData)
            expect(fc.neurons[0].bias).to.equal(2)
            expect(fc.neurons[0].weights).to.deep.equal([1,2,3])
            expect(fc.neurons[1].bias).to.equal(3)
            expect(fc.neurons[1].weights).to.deep.equal([1,2,4])
        })
        it("Sets the weights and biases to the given 1 dimensional array (Example 2)", () => {
            const testData = [2,1,2, 3,1,2, 4,0,2]

            const fc = new FCLayer(3)

            for (let n=0; n<fc.neurons.length; n++) {
                fc.neurons[n].weights = [1,1]
                fc.neurons[n].bias = 1
            }

            fc.fromIMG(testData)
            expect(fc.neurons[0].bias).to.equal(2)
            expect(fc.neurons[0].weights).to.deep.equal([1,2])
            expect(fc.neurons[1].bias).to.equal(3)
            expect(fc.neurons[1].weights).to.deep.equal([1,2])
            expect(fc.neurons[2].bias).to.equal(4)
            expect(fc.neurons[2].weights).to.deep.equal([0,2])
        })
    })

})

describe("Neuron", () => {

    describe("constructor", () => {
        it("Creates a new neuron instance", () => {
            const neuron = new Neuron()
            expect(neuron).to.not.be.undefined
            expect(neuron).instanceof(Neuron)
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
            neuron.init()
            expect(neuron.deltaWeights).to.not.be.undefined
            expect(neuron.deltaWeights.length).to.equal(neuron.weights.length)
        })

        it("Sets all the delta weights to 0", () => {
            neuron.init()
            expect(neuron.deltaWeights).to.deep.equal([0,0,0,0,0])
        })

        it("Creates a weightGains array if the updateFn parameter is gain, with same size as weights, with 1 values", () => {
            neuron2.init({updateFn: "gain"})
            expect(neuron2.weightGains).to.not.be.undefined
            expect(neuron2.weightGains).to.have.lengthOf(3)
            expect(neuron2.weightGains).to.deep.equal([1,1,1])
        })

        it("Creates a biasGain value of 1 if the updateFn parameter is gain", () => {
            neuron2.init({updateFn: "gain"})
            expect(neuron2.biasGain).to.equal(1)
        })

        it("Does not create the weightGains and biasGain when the updateFn is not gain", () => {
            neuron2.init({updateFn: "not gain"})
            expect(neuron2.weightGains).to.be.undefined
            expect(neuron2.biasGain).to.be.undefined
        })

        it("Creates a weightsCache array, with same dimension as weights, if the updateFn is adagrad, with 0 values", () => {
            neuron2.init({updateFn: "adagrad"})
            expect(neuron2.weightsCache).to.not.be.undefined
            expect(neuron2.weightsCache).to.have.lengthOf(3)
            expect(neuron2.weightsCache).to.deep.equal([0,0,0])
        })

        it("Creates a weightsCache array, with same dimension as weights, if the updateFn is rmsprop, with 0 values", () => {
            neuron2.init({updateFn: "rmsprop"})
            expect(neuron2.weightsCache).to.not.be.undefined
            expect(neuron2.weightsCache).to.have.lengthOf(3)
            expect(neuron2.weightsCache).to.deep.equal([0,0,0])
        })

        it("Creates a biasCache value of 0 if the updateFn parameter is adagrad", () => {
            neuron2.init({updateFn: "adagrad"})
            expect(neuron2.biasCache).to.equal(0)
        })

        it("Creates a biasCache value of 0 if the updateFn parameter is rmsprop", () => {
            neuron2.init({updateFn: "adagrad"})
            expect(neuron2.biasCache).to.equal(0)
        })

        it("Does not create the weightsCache or biasCache if the updateFn is not adagrad", () => {
            neuron2.init({updateFn: "not adagrad"})
            expect(neuron2.weightsCache).to.be.undefined
            expect(neuron2.biasCache).to.be.undefined
        })

        it("Does not create the weightsCache or biasCache if the updateFn is not rmsprop", () => {
            neuron2.init({updateFn: "not rmsprop"})
            expect(neuron2.weightsCache).to.be.undefined
            expect(neuron2.biasCache).to.be.undefined
        })

        it("Creates and sets neuron.m to 0 if the updateFn parameter is adam", () => {
            neuron2.init({updateFn: "adam"})
            expect(neuron2.m).to.not.be.undefined
            expect(neuron2.m).to.equal(0)
        })

        it("Creates and sets neuron.v to 0 if the updateFn parameter is adam", () => {
            neuron2.init({updateFn: "adam"})
            expect(neuron2.v).to.not.be.undefined
            expect(neuron2.v).to.equal(0)
        })

        it("Does not create neuron.m or neuron.v when the updateFn parameter is not adam", () => {
            neuron2.init({updateFn: "not adam"})
            expect(neuron2.m).to.be.undefined
            expect(neuron2.v).to.be.undefined
        })

        it("Creates a weightsCache array, with same dimension as weights, if the updateFn is adadelta, with 0 values", () => {
            neuron2.init({updateFn: "adadelta"})
            expect(neuron2.weightsCache).to.not.be.undefined
            expect(neuron2.weightsCache).to.have.lengthOf(3)
            expect(neuron2.weightsCache).to.deep.equal([0,0,0])
        })

        it("Creates a adadeltaBiasCache value of 0 if the updateFn parameter is adadelta", () => {
            neuron2.init({updateFn: "adadelta"})
            expect(neuron2.adadeltaBiasCache).to.equal(0)
        })

        it("Creates a adadeltaCache array, with same dimension as weights, if the updateFn is adadelta, with 0 values", () => {
            neuron2.init({updateFn: "adadelta"})
            expect(neuron2.adadeltaCache).to.not.be.undefined
            expect(neuron2.adadeltaCache).to.have.lengthOf(3)
            expect(neuron2.adadeltaCache).to.deep.equal([0,0,0])
        })

        it("Does not create adadeltaBiasCache or adadeltaCache when the updateFn is adagrad or rmsprop", () => {
            neuron2.init({updateFn: "adagrad"})
            expect(neuron2.adadeltaCache).to.be.undefined
            expect(neuron2.adadeltaBiasCache).to.be.undefined
            const neuron3 = new Neuron()
            neuron3.weights = [...new Array(3)].map(v => Math.random()*0.2-0.1)
            neuron3.init({updateFn: "rmsprop"})
            expect(neuron3.adadeltaCache).to.be.undefined
            expect(neuron3.adadeltaBiasCache).to.be.undefined
        })

        it("Creates a random neuron.rreluSlope number if the activation is rrelu", () => {
            neuron2.init({activation: "rrelu"})
            expect(neuron2.rreluSlope).to.not.be.undefined
            expect(neuron2.rreluSlope).to.be.a.number
            expect(neuron2.rreluSlope).to.be.at.most(0.0011)
        })

        it("Sets the neuron.eluAlpha to the given value, if given a value", () => {
            neuron2.init({activation: "elu", eluAlpha: 0.5})
            expect(neuron2.eluAlpha).to.equal(0.5)
        })

        it("Creates the neuron.getWeightGain() and neuron.setWeightGain() functions when updateFn is gain", () => {
            neuron2.init({updateFn: "gain"})
            expect(neuron2.getWeightGain).to.not.be.undefined
            expect(neuron2.setWeightGain).to.not.be.undefined
        })

        it("Does not create the neuron.getWeightGain() and neuron.setWeightGain() functions when updateFn is not gain", () => {
            neuron2.init({updateFn: "not gain"})
            expect(neuron2.getWeightGain).to.be.undefined
            expect(neuron2.setWeightGain).to.be.undefined
        })

        it("getWeightGain() returns the neuron.weightGains weight at the given index", () => {
            neuron2.init({updateFn: "gain"})
            neuron2.weightGains = [1,2,3]
            expect(neuron2.getWeightGain(0)).to.equal(1)
            expect(neuron2.getWeightGain(1)).to.equal(2)
            expect(neuron2.getWeightGain(2)).to.equal(3)
        })

        it("setWeightGain() changes the neuron.weightGains weight at the given index", () => {
            neuron2.init({updateFn: "gain"})
            neuron2.weightGains = [1,2,3]
            neuron2.setWeightGain(0, 4)
            neuron2.setWeightGain(1, 5)
            neuron2.setWeightGain(2, 6)
            expect(neuron2.weightGains[0]).to.equal(4)
            expect(neuron2.weightGains[1]).to.equal(5)
            expect(neuron2.weightGains[2]).to.equal(6)
        })

        it("Creates the neuron.getWeightsCache() and neuron.setWeightsCache() function when updateFn is adagrad", () => {
            neuron2.init({updateFn: "adagrad"})
            expect(neuron2.getWeightsCache).to.not.be.undefined
            expect(neuron2.setWeightsCache).to.not.be.undefined
        })

        it("Creates the neuron.getWeightsCache() function when updateFn is rmsprop", () => {
            neuron2.init({updateFn: "rmsprop"})
            expect(neuron2.getWeightsCache).to.not.be.undefined
            expect(neuron2.setWeightsCache).to.not.be.undefined
        })

        it("Creates the neuron.getWeightsCache() function when updateFn is adadelta", () => {
            neuron2.init({updateFn: "adadelta"})
            expect(neuron2.getWeightsCache).to.not.be.undefined
            expect(neuron2.setWeightsCache).to.not.be.undefined
        })

        it("Does not create the neuron.getWeightsCache() and neuron.setWeightsCache() functions when updateFn is something else", () => {
            neuron2.init({updateFn: "something else"})
            expect(neuron2.getWeightsCache).to.be.undefined
            expect(neuron2.setWeightsCache).to.be.undefined
        })

        it("getWeightsCache() returns the neuron.weightsCache weight at the given index", () => {
            neuron2.init({updateFn: "adadelta"})
            neuron2.weightsCache = [1,2,3]
            expect(neuron2.getWeightsCache(0)).to.equal(1)
            expect(neuron2.getWeightsCache(1)).to.equal(2)
            expect(neuron2.getWeightsCache(2)).to.equal(3)
        })

        it("setWeightsCache() changes the neuron.weightsCache weight at the given index", () => {
            neuron2.init({updateFn: "adadelta"})
            neuron2.weightsCache = [1,2,3]
            neuron2.setWeightsCache(0, 4)
            neuron2.setWeightsCache(1, 5)
            neuron2.setWeightsCache(2, 6)
            expect(neuron2.weightsCache[0]).to.equal(4)
            expect(neuron2.weightsCache[1]).to.equal(5)
            expect(neuron2.weightsCache[2]).to.equal(6)
        })

        it("Creates the neuron.getAdadeltaCache() and neuron.setAdadeltaCache() functions when updateFn is adadelta", () => {
            neuron2.init({updateFn: "adadelta"})
            expect(neuron2.getAdadeltaCache).to.not.be.undefined
            expect(neuron2.setAdadeltaCache).to.not.be.undefined
        })

        it("Does not create the neuron.getAdadeltaCache() function when updateFn is not adadelta", () => {
            neuron2.init({updateFn: "not adadelta"})
            expect(neuron2.getAdadeltaCache).to.be.undefined
            expect(neuron2.setAdadeltaCache).to.be.undefined
        })

        it("getAdadeltaCache() returns the neuron.adadeltaCache value at the index given", () => {
            neuron2.init({updateFn: "adadelta"})
            neuron2.adadeltaCache = [1,2,3]
            expect(neuron2.getAdadeltaCache(0)).to.equal(1)
            expect(neuron2.getAdadeltaCache(1)).to.equal(2)
            expect(neuron2.getAdadeltaCache(2)).to.equal(3)
        })

        it("setAdadeltaCache() changes the neuron.adadeltaCache weight at the given index", () => {
            neuron2.init({updateFn: "adadelta"})
            neuron2.adadeltaCache = [1,2,3]
            neuron2.setAdadeltaCache(0, 4)
            neuron2.setAdadeltaCache(1, 5)
            neuron2.setAdadeltaCache(2, 6)
            expect(neuron2.adadeltaCache[0]).to.equal(4)
            expect(neuron2.adadeltaCache[1]).to.equal(5)
            expect(neuron2.adadeltaCache[2]).to.equal(6)
        })
    })

    describe("getWeight", () => {

        const neuron = new Neuron()
        neuron.weights = [1,2,3]

        it("Returns the neuron's weight at that index", () => {
            expect(neuron.getWeight(0)).to.equal(1)
            expect(neuron.getWeight(1)).to.equal(2)
            expect(neuron.getWeight(2)).to.equal(3)
        })
    })

    describe("setWeight", () => {

        const neuron = new Neuron()
        neuron.weights = [1,2,3]

        it("Sets the neuron's weight at that index to the given value", () => {
            neuron.setWeight(0, 4)
            neuron.setWeight(1, 5)
            neuron.setWeight(2, 6)
            expect(neuron.weights[0]).to.equal(4)
            expect(neuron.weights[1]).to.equal(5)
            expect(neuron.weights[2]).to.equal(6)
        })
    })

    describe("getDeltaWeight", () => {

        const neuron = new Neuron()
        neuron.deltaWeights = [4,5,6]

        it("Returns the neuron's weight at that index", () => {
            expect(neuron.getDeltaWeight(0)).to.equal(4)
            expect(neuron.getDeltaWeight(1)).to.equal(5)
            expect(neuron.getDeltaWeight(2)).to.equal(6)
        })
    })

    describe("setDeltaWeight", () => {

        const neuron = new Neuron()
        neuron.deltaWeights = [4,5,6]

        it("Sets the neuron's deltaWeight at that index to the given value", () => {
            neuron.setDeltaWeight(0, 7)
            neuron.setDeltaWeight(1, 8)
            neuron.setDeltaWeight(2, 9)
            expect(neuron.deltaWeights[0]).to.equal(7)
            expect(neuron.deltaWeights[1]).to.equal(8)
            expect(neuron.deltaWeights[2]).to.equal(9)
        })
    })
})

describe("Filter", () => {

    describe("constructor", () => {

        it("Creates a Filter instance", () => {
            const filter = new Filter()
            expect(filter).instanceof(Filter)
        })
    })

    describe("init", () => {

        let filter, filter2

        beforeEach(() => {
            filter = new Filter()
            filter.weights = [[...new Array(3)].map(r => [...new Array(3)].map(v => Math.random()*0.2-0.1))]
            filter2 = new Filter()
            filter2.weights = [[...new Array(3)].map(r => [...new Array(3)].map(v => Math.random()*0.2-0.1))]
        })

        it("Creates a volume of delta weights with depth==channels and the same spacial dimensions as the weights map", () => {
            filter.init()
            expect(filter.deltaWeights).to.not.be.undefined
            expect(filter.deltaWeights).to.have.lengthOf(1)
            expect(filter.deltaWeights[0]).to.have.lengthOf(3)
            expect(filter.deltaWeights[0][0]).to.have.lengthOf(3)
        })

        it("Sets all the delta weight values to 0", () => {
            filter.init()
            expect(filter.deltaWeights[0]).to.deep.equal([[0,0,0], [0,0,0], [0,0,0]])
        })

        it("Sets the filter.deltaBias value to 0", () => {
            filter.deltaBias = undefined
            filter.init()
            expect(filter.deltaBias).to.equal(0)
        })

        describe("weightGains", () => {

            it("Creates a weightGains map if the updateFn parameter is gain, with the same dimensions as weights, with 1 values", () => {
                filter.init({updateFn: "gain"})
                expect(filter.weightGains).to.not.be.undefined
                expect(filter.weightGains).to.deep.equal([[[1,1,1],[1,1,1],[1,1,1]]])
            })

            it("Creates a biasGain value of 1 if the updateFn parameter is gain", () => {
                filter.init({updateFn: "gain"})
                expect(filter.biasGain).to.equal(1)
            })

            it("Does not create the weightGains and biasGain when the updateFn is not gain", () => {
                filter.init({updateFn: "not gain"})
                expect(filter.weightGains).to.be.undefined
                expect(filter.biasGain).to.be.undefined
            })

            it("Creates the filter.getWeightGain() and filter.setWeightGain() functions when updateFn is gain", () => {
                filter.init({updateFn: "gain"})
                expect(filter.getWeightGain).to.not.be.undefined
                expect(filter.setWeightGain).to.not.be.undefined
            })

            it("Does not create the filter.getWeightGain() and filter.setWeightGain() functions when updateFn is not gain", () => {
                filter.init({updateFn: "not gain"})
                expect(filter.getWeightGain).to.be.undefined
                expect(filter.setWeightGain).to.be.undefined
            })

            it("getWeightGain() returns the filter.weightGains weight at the given index", () => {
                filter.init({updateFn: "gain"})
                filter.weightGains = [[[1,2,3],[4,5,6],[7,8,9]]]
                expect(filter.getWeightGain([0,0,0])).to.equal(1)
                expect(filter.getWeightGain([0,0,2])).to.equal(3)
                expect(filter.getWeightGain([0,2,0])).to.equal(7)
                expect(filter.getWeightGain([0,2,2])).to.equal(9)
            })

            it("setWeightGain() changes the filter.weightGains weight at the given index", () => {
                filter.init({updateFn: "gain"})
                filter.weightGains = [[[1,2,3],[4,5,6],[7,8,9]]]
                filter.setWeightGain([0,0,0], 4)
                filter.setWeightGain([0,1,1], 5)
                filter.setWeightGain([0,1,2], "a")
                filter.setWeightGain([0,2,1], "b")
                filter.setWeightGain([0,2,2], "c")
                expect(filter.weightGains[0][0][0]).to.equal(4)
                expect(filter.weightGains[0][1][1]).to.equal(5)
                expect(filter.weightGains[0][1][2]).to.equal("a")
                expect(filter.weightGains[0][2][1]).to.equal("b")
                expect(filter.weightGains[0][2][2]).to.equal("c")
            })
        })

        describe("weightsCache", () => {

            it("Creates a weightsCache map, with same dimensions as weights, with 0 values, if the updateFn is adagrad", () => {
                filter.init({updateFn: "adagrad"})
                expect(filter.weightsCache).to.not.be.undefined
                expect(filter.weightsCache).to.deep.equal([[[0,0,0],[0,0,0],[0,0,0]]])
            })

            it("Creates a weightsCache map, with same dimensions as weights, with 0 values, if the updateFn is rmsprop", () => {
                filter.init({updateFn: "rmsprop"})
                expect(filter.weightsCache).to.not.be.undefined
                expect(filter.weightsCache).to.deep.equal([[[0,0,0],[0,0,0],[0,0,0]]])
            })

            it("Creates a weightsCache map, with same dimensions as weights, with 0 values, if the updateFn is adadelta", () => {
                filter.init({updateFn: "adadelta"})
                expect(filter.weightsCache).to.not.be.undefined
                expect(filter.weightsCache).to.deep.equal([[[0,0,0],[0,0,0],[0,0,0]]])
            })

            it("Creates a biasCache value of 0 if the updateFn parameter is adagrad", () => {
                filter.init({updateFn: "adagrad"})
                expect(filter.biasCache).to.equal(0)
            })

            it("Creates a biasCache value of 0 if the updateFn parameter is rmsprop", () => {
                filter.init({updateFn: "rmsprop"})
                expect(filter.biasCache).to.equal(0)
            })

            it("Creates a biasCache value of 0 if the updateFn parameter is adadelta", () => {
                filter.init({updateFn: "adadelta"})
                expect(filter.biasCache).to.equal(0)
            })

            it("Does not create them if any other updateFn parameter is given", () => {
                filter.init({updateFn: "something else"})
                expect(filter.weightsCache).to.be.undefined
                expect(filter.biasCache).to.be.undefined
            })

            it("Creates the getWeightsCache() and setWeightsCache() functions if the updateFn is adagrad", () => {
                filter.init({updateFn: "adagrad"})
                expect(filter.getWeightsCache).to.not.be.undefined
                expect(filter.setWeightsCache).to.not.be.undefined
            })

            it("Does not create the getWeightsCache and setWeightsCache functions if updateFn is anything else", () => {
                filter.init({updateFn: "whatever"})
                expect(filter.setWeightsCache).to.be.undefined
                expect(filter.setWeightsCache).to.be.undefined
            })

            it("getWeightsCache() returns the filter.weightsCache weight at the given index", () => {
                filter.init({updateFn: "adadelta"})
                filter.weightsCache = [[[1,2,3],[4,5,6],[7,8,9]]]
                expect(filter.getWeightsCache([0,0,0])).to.equal(1)
                expect(filter.getWeightsCache([0,0,2])).to.equal(3)
                expect(filter.getWeightsCache([0,2,0])).to.equal(7)
                expect(filter.getWeightsCache([0,2,2])).to.equal(9)
            })

            it("setWeightsCache() changes the filter.weightsCache weight at the given index", () => {
                filter.init({updateFn: "adadelta"})
                filter.weightsCache = [[[1,2,3],[4,5,6],[7,8,9]]]
                filter.setWeightsCache([0,0,0], 4)
                filter.setWeightsCache([0,1,1], 5)
                filter.setWeightsCache([0,1,2], "a")
                filter.setWeightsCache([0,2,1], "b")
                filter.setWeightsCache([0,2,2], "c")
                expect(filter.weightsCache[0][0][0]).to.equal(4)
                expect(filter.weightsCache[0][1][1]).to.equal(5)
                expect(filter.weightsCache[0][1][2]).to.equal("a")
                expect(filter.weightsCache[0][2][1]).to.equal("b")
                expect(filter.weightsCache[0][2][2]).to.equal("c")
            })
        })

        describe("adadeltaCache", () => {

            it("Creates a adadeltaBiasCache value of 0 if the updateFn parameter is adadelta", () => {
                filter.init({updateFn: "adadelta"})
                expect(filter.adadeltaBiasCache).to.equal(0)
            })

            it("Creates a adadeltaCache map, with same dimensions as weights, with 0 values, if the updateFn is adadelta", () => {
                filter.init({updateFn: "adadelta"})
                expect(filter.adadeltaCache).to.not.be.undefined
                expect(filter.adadeltaCache).to.deep.equal([[[0,0,0],[0,0,0],[0,0,0]]])
            })

            it("Does not create adadeltaBiasCache or adadeltaCache when the updateFn is adagrad or rmsprop", () => {
                filter.init({updateFn: "adagrad"})
                expect(filter.adadeltaCache).to.be.undefined
                expect(filter.adadeltaBiasCache).to.be.undefined

                filter2.init(3, {updateFn: "rmsprop"})
                expect(filter2.adadeltaCache).to.be.undefined
                expect(filter2.adadeltaBiasCache).to.be.undefined
            })

            it("Creates the filter.getAdadeltaCache() and filter.setAdadeltaCache() functions when updateFn is adadelta", () => {
                filter.init({updateFn: "adadelta"})
                expect(filter.getAdadeltaCache).to.not.be.undefined
                expect(filter.setAdadeltaCache).to.not.be.undefined
            })

            it("Does not create the map, bias or getAdadeltaCache() / setAdadeltaCache() functions when updateFn is not adadelta", () => {
                filter.init({updateFn: "not adadelta"})
                expect(filter.getAdadeltaCache).to.be.undefined
                expect(filter.setAdadeltaCache).to.be.undefined
                expect(filter.adadeltaBiasCache).to.be.undefined
                expect(filter.adadeltaCache).to.be.undefined
            })

            it("getAdadeltaCache() returns the filter.adadeltaCache weight at the given index", () => {
                filter.init({updateFn: "adadelta"})
                filter.adadeltaCache = [[[1,2,3],[4,5,6],[7,8,9]]]
                expect(filter.getAdadeltaCache([0,0,0])).to.equal(1)
                expect(filter.getAdadeltaCache([0,0,2])).to.equal(3)
                expect(filter.getAdadeltaCache([0,2,0])).to.equal(7)
                expect(filter.getAdadeltaCache([0,2,2])).to.equal(9)
            })

            it("setAdadeltaCache() changes the filter.adadeltaCache weight at the given index", () => {
                filter.init({updateFn: "adadelta"})
                filter.adadeltaCache = [[[1,2,3],[4,5,6],[7,8,9]]]
                filter.setAdadeltaCache([0,0,0], 4)
                filter.setAdadeltaCache([0,1,1], 5)
                filter.setAdadeltaCache([0,1,2], "a")
                filter.setAdadeltaCache([0,2,1], "b")
                filter.setAdadeltaCache([0,2,2], "c")
                expect(filter.adadeltaCache[0][0][0]).to.equal(4)
                expect(filter.adadeltaCache[0][1][1]).to.equal(5)
                expect(filter.adadeltaCache[0][1][2]).to.equal("a")
                expect(filter.adadeltaCache[0][2][1]).to.equal("b")
                expect(filter.adadeltaCache[0][2][2]).to.equal("c")
            })
        })

        it("Creates and sets filter.m to 0 if the updateFn parameter is adam", () => {
            filter.init({updateFn: "adam"})
            expect(filter.m).to.not.be.undefined
            expect(filter.m).to.equal(0)
        })

        it("Creates and sets filter.v to 0 if the updateFn parameter is adam", () => {
            filter.init({updateFn: "adam"})
            expect(filter.v).to.not.be.undefined
            expect(filter.v).to.equal(0)
        })

        it("Does not create filter.m or filter.v when the updateFn parameter is not adam", () => {
            filter.init({updateFn: "not adam"})
            expect(filter.m).to.be.undefined
            expect(filter.v).to.be.undefined
        })

        it("Creates a random filter.rreluSlope number if the activation is rrelu", () => {
            filter.init({activation: "rrelu"})
            expect(filter.rreluSlope).to.not.be.undefined
            expect(filter.rreluSlope).to.be.a.number
            expect(filter.rreluSlope).to.be.at.most(0.0011)
        })

        it("Sets the filter.eluAlpha to the given value, if given a value", () => {
            filter.init({activation: "elu", eluAlpha: 0.5})
            expect(filter.eluAlpha).to.equal(0.5)
        })
    })

    describe("getWeight", () => {
        it("Gets the weight from the map at given index, as though the map was a 1d array (Example 1)", () => {
            const filter = new Filter()
            filter.weights = [[[1,2,3],[4,5,6],[7,8,9]]]
            expect(filter.getWeight([0,0,0])).to.equal(1)
            expect(filter.getWeight([0,0,2])).to.equal(3)
            expect(filter.getWeight([0,2,0])).to.equal(7)
            expect(filter.getWeight([0,2,2])).to.equal(9)
        })

        it("Gets the weight from the map at given index, as though the map was a 1d array (Example 2)", () => {
            const filter = new Filter()
            filter.weights = [[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]]
            expect(filter.getWeight([0,0,0])).to.equal(1)
            expect(filter.getWeight([0,1,0])).to.equal(3)
            expect(filter.getWeight([1,1,0])).to.equal(7)
            expect(filter.getWeight([2,0,0])).to.equal(9)
            expect(filter.getWeight([2,1,0])).to.equal(11)
        })
    })

    describe("setWeight", () => {
        it("Sets the weight from the map at given index, as though the map was a 1d array", () => {
            const filter = new Filter()
            filter.weights = [[[1,2,3],[4,5,6],[7,8,9]]]
            filter.setWeight([0,0,0], 4)
            filter.setWeight([0,1,1], 5)
            filter.setWeight([0,1,2], "a")
            filter.setWeight([0,2,1], "b")
            filter.setWeight([0,2,2], "c")
            expect(filter.weights[0][0][0]).to.equal(4)
            expect(filter.weights[0][1][1]).to.equal(5)
            expect(filter.weights[0][1][2]).to.equal("a")
            expect(filter.weights[0][2][1]).to.equal("b")
            expect(filter.weights[0][2][2]).to.equal("c")
        })
    })

    describe("getDeltaWeight", () => {
        it("Gets the delta weight from the map at given index, as though the map was a 1d array", () => {
            const filter = new Filter()
            filter.deltaWeights = [[[1,2,3],[4,5,6],[7,8,9]]]
            expect(filter.getDeltaWeight([0,0,0])).to.equal(1)
            expect(filter.getDeltaWeight([0,0,2])).to.equal(3)
            expect(filter.getDeltaWeight([0,2,0])).to.equal(7)
            expect(filter.getDeltaWeight([0,2,2])).to.equal(9)
        })
    })

    describe("setDeltaWeight", () => {
        it("Sets the delta weight from the map at given index, as though the map was a 1d array", () => {
            const filter = new Filter()
            filter.deltaWeights = [[[1,2,3],[4,5,6],[7,8,9]]]
            filter.setDeltaWeight([0,0,0], 4)
            filter.setDeltaWeight([0,1,1], 5)
            filter.setDeltaWeight([0,1,2], "a")
            filter.setDeltaWeight([0,2,1], "b")
            filter.setDeltaWeight([0,2,2], "c")
            expect(filter.deltaWeights[0][0][0]).to.equal(4)
            expect(filter.deltaWeights[0][1][1]).to.equal(5)
            expect(filter.deltaWeights[0][1][2]).to.equal("a")
            expect(filter.deltaWeights[0][2][1]).to.equal("b")
            expect(filter.deltaWeights[0][2][2]).to.equal("c")
        })
    })
})

describe("ConvLayer", () => {

    describe("constructor", () => {

        it("Sets the layer.filterSize to the value given", () => {
            const layer = new ConvLayer(1, {filterSize: 3})
            expect(layer.filterSize).to.equal(3)
        })

        it("Does not set the layer.filterSize to anything if no value is given", () => {
            const layer = new ConvLayer()
            expect(layer.filterSize).to.be.undefined
        })

        it("Sets the layer.zeroPadding to the value given", () => {
            const layer = new ConvLayer(1, {zeroPadding: 1})
            expect(layer.zeroPadding).to.equal(1)
        })

        it("Does not set the layer.zeroPadding to anything if no value is given", () => {
            const layer = new ConvLayer()
            expect(layer.zeroPadding).to.be.undefined
        })

        it("Sets the layer.stride to the value given", () => {
            const layer = new ConvLayer(1, {stride: 1})
            expect(layer.stride).to.equal(1)
        })

        it("Does not set the layer.stride to anything if no value is given", () => {
            const layer = new ConvLayer()
            expect(layer.stride).to.be.undefined
        })

        it("Sets the layer.size to the value given", () => {
            const layer = new ConvLayer(1)
            expect(layer.size).to.equal(1)
        })

        it("Does not set the layer.size to anything if no value is given", () => {
            const layer = new ConvLayer()
            expect(layer.size).to.be.undefined
        })

        it("Sets the state to not-initialised", () => {
            const layer = new ConvLayer()
            expect(layer.state).to.equal("not-initialised")
        })

        it("Doesn't set the layer activation to anything if nothing is provided", () => {
            const layer = new ConvLayer()
            expect(layer.activation).to.be.undefined
        })

        it("Allows setting the activation function to a custom function", () => {
            const customFn = x => x
            const layer = new ConvLayer(2, {activation: customFn})
            expect(layer.activation).to.equal(customFn)
        })

        it("Allows setting the activation to false by giving the value false", () => {
            const layer = new ConvLayer(5, {activation: false})
            expect(layer.activation).to.be.false
        })

        it("Allows setting the activation function to a function from NetMath using a string", () => {
            const layer = new ConvLayer(5, {activation: "relu"})
            expect(layer.activation.name).to.equal("bound relu")
        })

        it("Sets the given activation config value to layer.activationName", () => {
            const layer = new ConvLayer(5, {activation: "relu"})
            expect(layer.activationName).to.equal("relu")
            const layer2 = new ConvLayer(5, {activation: "elu"})
            expect(layer2.activationName).to.equal("elu")
        })
    })

    describe("assignNext", () => {

        it("Assigns a reference to the given layer as this layer's nextLayer", () => {
            const layer1 = new ConvLayer()
            const layer2 = new ConvLayer()
            layer1.net = {}
            layer1.assignNext(layer2)
            expect(layer1.nextLayer).to.equal(layer2)
        })
    })

    describe("assignPrev", () => {

        let layer1, layer2

        beforeEach(() => {
            layer1 = new ConvLayer()
            layer1.outMapSize = 16
            layer1.size = 10
            layer1.neurons = {length: 10}
            layer2 = new ConvLayer(3, {filterSize: 3, stride: 1})
            layer2.net = {conv: {}}
        })

        it("Assigns a reference to the given layer as this layer's prevLayer", () => {
            layer2.assignPrev(layer1)
            expect(layer2.prevLayer).to.equal(layer1)
        })

        it("Sets the layer.layerIndex to the given value", () => {
            const layer = new ConvLayer(3)
            const net = new Network({conv: {filterSize: 5}})
            layer.net = net
            layer.assignPrev(layer1, 12345)
            expect(layer.layerIndex).to.equal(12345)
        })

        it("Defaults the layer.filterSize to the net.filterSize value, if there's no layer.filterSize, but there is one for net", () => {
            const layer = new ConvLayer(3)
            const net = new Network({conv: {filterSize: 5}})
            layer.net = net
            layer.assignPrev(layer1)
            expect(layer.filterSize).to.equal(5)
        })

        it("Defaults the layer.filterSize to 3 if there is no filterSize value for either layer or net", () => {
            const layer = new ConvLayer(3)
            const net = new Network()
            layer.net = net
            layer.assignPrev(layer1)
            expect(layer.filterSize).to.equal(3)
        })

        it("Keeps the same filterSize value if it has already been assigned to layer", () => {
            const net = new Network()
            layer2.filterSize = 7
            layer2.net = net
            layer2.assignPrev(layer1)
            expect(layer2.filterSize).to.equal(7)
        })

        it("Defaults the layer.zeroPadding to the net.zeroPadding value, if there's no layer.zeroPadding, but there is one for net", () => {
            const net = new Network({conv: {zeroPadding: 3}})
            layer2.net = net
            layer2.assignPrev(layer1)
            expect(layer2.zeroPadding).to.equal(3)
        })

        it("Defaults the layer.zeroPadding to rounded down half the filterSize if there is no zeroPadding value for either layer or net", () => {
            const layer3 = new ConvLayer()
            const net = new Network()
            layer2.net = net
            layer3.net = net

            layer2.filterSize = 3
            layer2.assignPrev(layer1)
            expect(layer2.zeroPadding).to.equal(1)

            layer3.filterSize = 5
            layer3.assignPrev(layer1)
            expect(layer3.zeroPadding).to.equal(2)
        })

        it("Keeps the same zeroPadding value if it has already been assigned to layer", () => {
            layer2.size = 1
            layer2.zeroPadding = 2
            const net = new Network()
            layer2.net = net
            layer2.assignPrev(layer1)
            expect(layer2.zeroPadding).to.equal(2)
        })

        it("Allows setting the zero padding to 0", () => {
            layer2.size = 1
            layer2.zeroPadding = 0
            const net = new Network()
            layer2.net = net
            layer2.assignPrev(layer1)
            expect(layer2.zeroPadding).to.equal(0)
        })

        it("Defaults the layer.stride to the net.stride value, if there's no layer.stride, but there is one for net", () => {
            const layer = new ConvLayer(3)
            const net = new Network({conv: {stride: 5}})
            layer.net = net
            layer.assignPrev(layer1)
            expect(layer.stride).to.equal(5)
        })

        it("Defaults the layer.stride to 1 if there is no stride value for either layer or net", () => {
            const net = new Network()
            layer2.net = net
            layer2.assignPrev(layer1)
            expect(layer2.stride).to.equal(1)
        })

        it("Keeps the same stride value if it has already been assigned to layer", () => {
            const net = new Network()
            layer2.stride = 3
            layer2.net = net
            layer2.assignPrev(layer1)
            expect(layer2.stride).to.equal(3)
        })

        it("Defaults the layer.size to 4 if there is no size value for layer", () => {
            const layer = new ConvLayer()
            const net = new Network()
            layer.net = net
            layer.assignPrev(layer1)
            expect(layer.size).to.equal(4)
        })

        it("Keeps the same size value if it has already been assigned to layer", () => {
            const net = new Network()
            layer2.size = 7
            layer2.net = net
            layer2.assignPrev(layer1)
            expect(layer2.size).to.equal(7)
        })

        it("Assigns the layer.channels to the previous layer's size, if it's a Conv Layer", () => {
            layer1.size = 10
            layer2.assignPrev(layer1)
            expect(layer2.channels).to.equal(10)
        })

        it("Assigns the layer.channels to the net.channels if the previous layer is not Conv", () => {
            const layer = new FCLayer(10)
            layer2.net.channels = 69
            layer2.assignPrev(layer)
            expect(layer2.channels).to.equal(69)
        })

        it("Assigns the layer.channels to the net.channels if the previous layer is not Conv and no net.channels is configured", () => {
            const layer = new FCLayer(10)
            layer2.net.channels = undefined
            layer2.assignPrev(layer)
            expect(layer2.channels).to.equal(1)
        })

        it("Assigns the layer.channels to the number of channels in the last layer, if it is a PoolLayer", () => {
            const layer = new PoolLayer(2)
            layer.activations = [[],[],[]]
            layer.outMapSize = 28
            layer2.net.channels = undefined
            layer2.assignPrev(layer)
            expect(layer2.channels).to.equal(3)
        })

        it("Assigns to layer.inMapValuesCount the size of the input map (Example 1)", () => {
            const layer1 = new ConvLayer(4)
            const layer2 = new ConvLayer(3, {filterSize: 3, zeroPadding: 0})
            layer2.net = {conv: {}}
            layer1.outMapSize = 28
            layer2.assignPrev(layer1)
            expect(layer2.inMapValuesCount).to.equal(784)
        })

        it("Assigns to layer.inMapValuesCount the size of the input map (Example 2)", () => {
            const layer1 = new ConvLayer(4)
            const layer2 = new ConvLayer(3, {filterSize: 3, zeroPadding: 1})
            layer2.net = {conv: {}}
            layer1.outMapSize = 28
            layer2.assignPrev(layer1)
            expect(layer2.inMapValuesCount).to.equal(784)
        })

        it("Assigns to layer.inMapValuesCount the size of the input map (Example 3)", () => {
            const layer1 = new FCLayer(75)
            const layer2 = new ConvLayer(3, {zeroPadding: 1, filterSize: 3})
            layer2.net = {conv: {}, channels: 3}
            layer2.assignPrev(layer1)
            expect(layer2.inMapValuesCount).to.equal(25)
        })

        it("Assigns to layer.inZPMapValuesCount the size of the zero padded input map (Example 1)", () => {
            const layer1 = new ConvLayer(4)
            const layer2 = new ConvLayer(3, {filterSize: 3, zeroPadding: 0})
            layer1.outMapSize = 28

            layer2.net = {conv: {}}
            layer2.assignPrev(layer1)
            expect(layer2.inZPMapValuesCount).to.equal(784)
        })

        it("Assigns to layer.inZPMapValuesCount the size of the zero padded input map (Example 2)", () => {
            const layer1 = new ConvLayer(4)
            const layer2 = new ConvLayer(3, {filterSize: 3, zeroPadding: 1})
            layer1.outMapSize = 28
            layer2.net = {conv: {}}
            layer2.assignPrev(layer1)
            expect(layer2.inZPMapValuesCount).to.equal(900)
        })

        it("Assigns to layer.inZPMapValuesCount the size of the zero padded input map (Example 3)", () => {
            const layer1 = new FCLayer(75)
            const layer2 = new ConvLayer(3, {zeroPadding: 1, filterSize: 3})
            layer2.net = {conv: {}, channels: 3}
            layer2.assignPrev(layer1)
            expect(layer2.inZPMapValuesCount).to.equal(49)
        })

        it("Sets the layer.outMapSize to the spacial dimension of the filter activation/sum/error maps (Example 1)", () => {
            const layer1 = new FCLayer(2352) // 784 * 3
            const layer2 = new ConvLayer(4, {filterSize: 3, zeroPadding: 1})
            layer2.net = {conv: {}, channels: 3}
            layer2.assignPrev(layer1)
            expect(layer2.outMapSize).to.equal(28)
        })

        it("Sets the layer.outMapSize to the spacial dimension of the filter activation/sum/error maps (Example 2)", () => {
            const layer1 = new FCLayer(75)
            const layer2 = new ConvLayer()
            layer1.size = 75
            layer2.size = 4
            layer2.stride = 2
            layer2.filterSize = 3
            layer2.zeroPadding = 1
            layer2.net = {conv: {}, channels: 3}
            layer2.assignPrev(layer1)
            expect(layer2.outMapSize).to.equal(3)
        })

        it("Sets the layer.outMapSize to the spacial dimension of the filter activation/sum/error maps (Example 3)", () => {
            const layer1 = new FCLayer(147)
            const layer2 = new ConvLayer()
            layer1.size = 147
            layer2.size = 4
            layer2.stride = 3
            layer2.filterSize = 3
            layer2.zeroPadding = 1
            layer2.net = {conv: {}, channels: 3}
            layer2.assignPrev(layer1)
            expect(layer2.outMapSize).to.equal(3)
        })

        it("Creates a layer.filters array with as many filters as the size of the layer", () => {
            const prevLayer = new FCLayer(147)
            const layer = new ConvLayer(3)
            layer.net = {conv: {}, channels: 3}
            layer.assignPrev(layer1)
            expect(layer.filters).to.not.be.undefined
            expect(layer.filters).to.have.lengthOf(3)
            expect(layer.filters[0]).instanceof(Filter)
            expect(layer.filters[1]).instanceof(Filter)
            expect(layer.filters[2]).instanceof(Filter)
        })

        it("Sets the inMapValuesCount to the square of the prev layer's out map size, if prev layer is Conv", () => {
            const layer1 = new ConvLayer(3)
            const layer2 = new ConvLayer(4, {filterSize: 3, stride: 1, zeroPadding: 1})
            layer1.outMapSize = 100
            layer2.assignPrev(layer1)
            expect(layer2.inMapValuesCount).to.equal(10000)
        })

        it("Throws an error if the hyperparameters don't match the input map properly", () => {
            const layer1 = new ConvLayer(3)
            const layer2 = new ConvLayer(4, {filterSize: 3, stride: 2, zeroPadding: 1})
            layer1.outMapSize = 16
            expect(layer2.assignPrev.bind(layer2, layer1)).to.throw("Misconfigured hyperparameters. Activation volume dimensions would be ")
        })
    })

    describe("init", () => {

        let layer1
        let layer2

        beforeEach(() => {
            layer1 = new FCLayer(2)
            layer2 = new ConvLayer(5, {filterSize: 3})
            layer2.weightsConfig = {weightsInitFn: NetMath.xavieruniform}
            layer2.net = {conv: {}, weightsInitFn: NetMath.xavieruniform}
            layer2.channels = 1
            layer2.assignPrev(layer1)
        })

        it("Initialises the filters' weights to a 3d map", () => {
            layer2.init()
            expect(layer2.filters[0].weights).instanceof(Array)
            expect(layer2.filters[0].weights[0]).instanceof(Array)
            expect(layer2.filters[0].weights[0][0]).instanceof(Array)
        })

        it("Sets the filter weights map to contain as many maps as there are channels in the filter", () => {
            layer2.channels = 9
            layer2.init()
            expect(layer2.filters[0].weights).to.have.lengthOf(9)
        })

        it("Sets the number of weights in each filter's channel weights to a 2D map with dimensions==filterSize (Example 1)", () => {
            layer2.channels = 1
            layer2.filterSize = 3
            layer2.init()
            expect(layer2.filters[0].weights).to.have.lengthOf(1)
            expect(layer2.filters[0].weights[0]).to.have.lengthOf(3)
            expect(layer2.filters[0].weights[0][0]).to.have.lengthOf(3)
        })

        it("Sets the number of weights in each filter's channel weights to a 2D map with dimensions==filterSize (Example 2)", () => {
            layer2.channels = 3
            layer2.filterSize = 3
            layer2.init()
            expect(layer2.filters[0].weights).to.have.lengthOf(3)
            expect(layer2.filters[0].weights[0]).to.have.lengthOf(3)
            expect(layer2.filters[0].weights[0][0]).to.have.lengthOf(3)
        })

        it("Calls the layer weightsInitFn for each weights row in each channel in each filter (Example 1)", () => {
            sinon.spy(layer2.net, "weightsInitFn")
            layer2.channels = 1
            layer2.filterSize = 3
            layer2.init()
            expect(layer2.net.weightsInitFn.callCount).to.equal(15) //5 * 1 * 3
            layer2.net.weightsInitFn.restore()
        })

        it("Calls the layer weightsInitFn for each weights row in each channel in each filter (Example 2)", () => {
            sinon.spy(layer2.net, "weightsInitFn")
            layer2.channels = 3
            layer2.filterSize = 3
            layer2.init()
            expect(layer2.net.weightsInitFn.callCount).to.equal(45) // 5 * 3 * 3
            layer2.net.weightsInitFn.restore()
        })

        it("Sets each filter bias to 0", () => {
            layer2.filters[0].bias = undefined
            layer2.filters[1].bias = undefined
            layer2.init()
            expect(layer2.filters[0].bias).to.equal(1)
            expect(layer2.filters[1].bias).to.equal(1)
        })

        it("Calls the filter's init function with updateFn and activation", () => {
            layer2.net.updateFn = "test"
            layer2.net.activationConfig = "stuff"
            sinon.stub(layer2.filters[0], "init")
            sinon.stub(layer2.filters[1], "init")
            layer2.init()

            expect(layer2.filters[0].init).to.have.been.calledWith(sinon.match({"updateFn": "test"}))
            expect(layer2.filters[0].init).to.have.been.calledWith(sinon.match({"activation": "stuff"}))
            expect(layer2.filters[1].init).to.have.been.calledWith(sinon.match({"updateFn": "test"}))
            expect(layer2.filters[1].init).to.have.been.calledWith(sinon.match({"activation": "stuff"}))
        })

        it("Calls the filter's init function with eluAlpha", () => {
            layer2.net.eluAlpha = 1
            sinon.stub(layer2.filters[0], "init")
            sinon.stub(layer2.filters[1], "init")
            layer2.init()

            expect(layer2.filters[0].init).to.have.been.calledWith(sinon.match({"eluAlpha": 1}))
            expect(layer2.filters[1].init).to.have.been.calledWith(sinon.match({"eluAlpha": 1}))
        })

        it("Sets the filter activationMap with the same dimensions as the expected output size, with 0 values", () => {
            layer2.outMapSize = 5
            layer2.init()
            expect(layer2.filters[0].activationMap).to.deep.equal([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
            expect(layer2.filters[1].activationMap).to.deep.equal([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
            expect(layer2.filters[2].activationMap).to.deep.equal([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
            expect(layer2.filters[3].activationMap).to.deep.equal([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
            expect(layer2.filters[4].activationMap).to.deep.equal([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
        })

        it("Sets the filter errorMap with the same dimensions as the expected output size, with 0 values", () => {
            layer2.outMapSize = 5
            layer2.init()
            expect(layer2.filters[0].errorMap).to.deep.equal([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
            expect(layer2.filters[1].errorMap).to.deep.equal([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
            expect(layer2.filters[2].errorMap).to.deep.equal([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
            expect(layer2.filters[3].errorMap).to.deep.equal([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
            expect(layer2.filters[4].errorMap).to.deep.equal([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
        })

        it("Inits the filter dropoutMap with the same dimensions as the activation map, with only false values", () => {
            layer2.outMapSize = 3
            layer2.init()
            const expected = [[false,false,false], [false,false,false], [false,false,false]]
            expect(layer2.filters[0].dropoutMap).to.deep.equal(expected)
        })

        it("Does not init the dropout map if dropout is not configured", () => {
            layer2.outMapSize = 3
            layer2.net.dropout = 1
            layer2.init()
            expect(layer2.filters[0].dropoutMap).to.be.undefined
        })
    })

    describe("forward", () => {

        let net, layer, prevLayer

        beforeEach(() => {
            prevLayer = new FCLayer(75)
            layer = new ConvLayer(3, {filterSize: 3, zeroPadding: 1, stride: 1})
            net = new Network({channels: 3, layers: [prevLayer, layer]})
            sinon.stub(NetUtil, "getActivations").callsFake(x => [...new Array(75)].map(v => Math.random()))
        })

        afterEach(() => {
            NetUtil.getActivations.restore()
        })

        it("Sets the filter.sumMap of each filter to a map with spacial dimension equal to the output volume's (layer.outMapSize)", () => {
            layer.filters[0].sumMap = undefined
            layer.filters[1].sumMap = undefined
            layer.filters[2].sumMap = undefined
            layer.forward()
            expect(layer.filters[0].sumMap).to.have.lengthOf(5)
            expect(layer.filters[0].sumMap[0]).to.have.lengthOf(5)
        })

        it("Sets the filter.activationMap values to zero if when dropping out", () => {
            layer.net = {dropout: 0}
            layer.state = "training"
            layer.filters.forEach(filter => filter.dropoutMap = filter.activationMap.map(row => row.map(v => false)))
            layer.forward()
            const zeroedOutMap = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
            expect(layer.filters[0].activationMap).to.deep.equal(zeroedOutMap)
            expect(layer.filters[1].activationMap).to.deep.equal(zeroedOutMap)
            expect(layer.filters[2].activationMap).to.deep.equal(zeroedOutMap)
        })

        it("Doesn't do any dropout if the layer state is not training", () => {
            layer.net = {dropout: 0}
            layer.state = "dfkgjdhflgjk"
            layer.filters.forEach(filter => filter.dropoutMap = filter.activationMap.map(row => row.map(v => false)))
            layer.forward()
            expect(layer.filters[0].dropoutMap).to.not.include(true)
            expect(layer.filters[1].dropoutMap).to.not.include(true)
            expect(layer.filters[2].dropoutMap).to.not.include(true)
        })

        it("Doesn't do any dropout if the dropout is set to 1", () => {
            layer.net = {dropout: 1}
            layer.state = "training"
            layer.filters.forEach(filter => filter.dropoutMap = filter.activationMap.map(row => row.map(v => false)))
            layer.forward()
            expect(layer.filters[0].dropoutMap).to.not.include(true)
            expect(layer.filters[1].dropoutMap).to.not.include(true)
            expect(layer.filters[2].dropoutMap).to.not.include(true)
        })

        it("Calls the layer activation function with every sum in the sum map, when no dropout is used", () => {
            const layer1 = new FCLayer(8)
            const layer2 = new ConvLayer(2)
            const net = new Network({depth: 2, layers: [layer1, layer2]})

            sinon.stub(NetUtil, "convolve").callsFake(() => [[1,2],[3,4]])
            sinon.stub(layer2, "activation")

            layer2.net.dropout = 1
            layer2.forward()
            NetUtil.convolve.restore()
            expect(layer2.activation.callCount).to.equal(8)
            expect(layer2.activation).to.be.calledWith(1)
            expect(layer2.activation).to.be.calledWith(2)
            expect(layer2.activation).to.be.calledWith(3)
            expect(layer2.activation).to.be.calledWith(4)
        })


        it("And thus sets the activation values in each filter's activationMap", () => {
            layer.filters.forEach(filter => filter.activationMap = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
            layer.forward()
            layer.filters.forEach(filter => {
                filter.activationMap.forEach(row => {
                    expect(row).to.not.include(0)
                })
            })
        })

        it("It does not call pass sum map values through an activation function if it is set to not be used", () => {
            sinon.stub(NetUtil, "convolve").callsFake(() => [[1,2],[3,4]])
            layer.activation = false
            layer.filters.forEach(filter => {
                filter.activationMap = [[0,0],[0,0]]
            })
            layer.forward()
            expect(layer.filters[0].activationMap).to.deep.equal([[1,2],[3,4]])
            expect(layer.filters[1].activationMap).to.deep.equal([[1,2],[3,4]])
            expect(layer.filters[2].activationMap).to.deep.equal([[1,2],[3,4]])
            NetUtil.convolve.restore()
        })
    })

    describe("backward", () => {

        const prevLayer = new FCLayer(75)
        const nextLayerA = new FCLayer(100)
        const nextLayerB = new ConvLayer(2, {filterSize: 3, stride: 1, zeroPadding: 1})

        let net, layer

        const testDataDeltaWeights = [[[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]]]

        beforeEach(() => {

            layer = new ConvLayer(4, {filterSize: 3, zeroPadding: 1})
            net = new Network({layers: [prevLayer, layer, nextLayerB], channels: 3})

            // Add some test data
            layer.filters.forEach(filter => {
                filter.weights = [[[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]]]
                filter.sumMap = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
                filter.errorMap = [[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3],[3,3,3,3,3]]

                filter.dropoutMap = [[false,false,false,false,false],[false,false,false,false,false],[false,false,false,false,false],[false,false,false,false,false],[false,false,false,false,false]]
            })
            prevLayer.neurons.forEach(neuron => neuron.activation = 0.5)
            layer.net = {l2: 0, l1: 0, miniBatchSize: 1}

            sinon.stub(NetUtil, "buildConvErrorMap")
            sinon.spy(NetUtil, "buildConvDWeights")
            sinon.spy(layer, "activation")
        })

        afterEach(() => {
            NetUtil.buildConvErrorMap.restore()
            NetUtil.buildConvDWeights.restore()
        })

        it("Calls the NetUtil.buildConvErrorMap() function for each filter in this layer when nextLayer is Conv", () => {
            layer.backward()
            expect(NetUtil.buildConvErrorMap.callCount).to.equal(4)
        })

        it("Maps the errors in the PoolLayer, 1 to 1, to the filters' errorMap values", () => {
            const poolLayer = new PoolLayer(3, {stride: 2})
            const convLayer = new ConvLayer(2, {filterSize: 3, stride: 1})
            net = new Network({layers: [prevLayer, convLayer, poolLayer, nextLayerA], channels: 3})
            // Add some test data
            convLayer.filters.forEach(filter => {
                filter.sumMap = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
                filter.errorMap = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
            })
            poolLayer.errors = [[
                [1,2,3,4,5],
                [5,4,3,2,1],
                [1,2,3,4,5],
                [5,4,3,2,1],
                [1,2,3,4,5]
            ], [
                [6,7,8,8,9],
                [6,7,8,8,9],
                [0,9,8,7,6],
                [6,7,8,8,9],
                [0,9,8,7,6]
            ]]
            sinon.stub(convLayer, "activation").callsFake(x => x)
            convLayer.backward()
            expect(convLayer.filters[0].errorMap).to.deep.equal([[1,2,3,4,5],[5,4,3,2,1],[1,2,3,4,5],[5,4,3,2,1],[1,2,3,4,5]])
            expect(convLayer.filters[1].errorMap).to.deep.equal([[6,7,8,8,9],[6,7,8,8,9],[0,9,8,7,6],[6,7,8,8,9],[0,9,8,7,6]])
        })

        it("Does not call NetUtil.buildConvErrorMap() if the next layer is an FCLayer", () => {
            net = new Network({layers: [prevLayer, layer, nextLayerA], channels: 3})

            // Add some test data
            layer.filters.forEach(filter => {
                filter.sumMap = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
                filter.errorMap = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
            })

            layer.backward()
            expect(NetUtil.buildConvErrorMap).to.not.be.called
        })

        it("Calculates the error map correctly when the next layer is an FCLayer", () => {

            // Values were worked out manually
            const fcLayer = new FCLayer(4)

            // const convLayer = new ConvLayer(2, {filterSize: 3, zeroPadding: 0, stride: 1})
            const convLayer = new ConvLayer(2, {filterSize: 3, zeroPadding: 0, stride: 1, activation: false})
            const net = new Network({layers: [new FCLayer(18), convLayer, fcLayer]})
            fcLayer.neurons.forEach((neuron, ni) => {
                neuron.error = (ni+1)/5 // 0.2, 0.4, 0.6, 0.8
                neuron.weights = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2]
            })

            const [filter1, filter2] = convLayer.filters

            filter1.sumMap = [[0,0],[0,0]]
            filter2.sumMap = [[0,0],[0,0]]
            filter1.errorMap = [[0,0],[0,0]]
            filter2.errorMap = [[0,0],[0,0]]
            filter1.activationMap = [[0.1,0.2],[0.3,0.4]]
            filter2.activationMap = [[0.5,0.6],[0.7,0.8]]

            const expectedFilter1ErrorMap = [[1.8, 1.6], [1.4, 1.2]]
            const expectedFilter2ErrorMap = [[1.0, 0.8], [0.6, 0.4]]
            convLayer.outMapSize = 2

            // Avoid tests failing due to tiny precision differences
            const roundMapValues = map => map.map(row => row.map(value => Math.round(value*10)/10))

            convLayer.backward()
            expect(roundMapValues(convLayer.filters[0].errorMap)).to.deep.equal(expectedFilter1ErrorMap)
            expect(roundMapValues(convLayer.filters[1].errorMap)).to.deep.equal(expectedFilter2ErrorMap)
        })

        it("Calls the activation function for each value in the sum maps", () => {
            layer.backward()
            expect(layer.activation.callCount).to.equal(100)
        })

        it("Does not call the activation function if all values are dropped out", () => {
            layer.filters.forEach(filter => filter.dropoutMap = filter.dropoutMap.map(r => r.map(v => true)))
            layer.backward()
            expect(layer.activation).to.not.be.called
        })

        it("Sets the errorMap values to 0 when dropped out", () => {
            layer.filters.forEach(filter => filter.dropoutMap = filter.dropoutMap.map(r => r.map(v => true)))
            layer.backward()
            expect(layer.filters[0].errorMap).to.deep.equal([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
            expect(layer.filters[1].errorMap).to.deep.equal([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
            expect(layer.filters[2].errorMap).to.deep.equal([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
            expect(layer.filters[3].errorMap).to.deep.equal([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]])
        })

        it("Does not increment the deltaBias when all values are dropped out", () => {
            layer.filters.forEach(filter => filter.dropoutMap = filter.dropoutMap.map(r => r.map(v => true)))
            layer.filters[0].deltaBias = 0
            layer.filters[1].deltaBias = 1
            layer.filters[2].deltaBias = 2
            layer.filters[3].deltaBias = 3
            layer.backward()
            expect(layer.filters[0].deltaBias).to.equal(0)
            expect(layer.filters[1].deltaBias).to.equal(1)
            expect(layer.filters[2].deltaBias).to.equal(2)
            expect(layer.filters[3].deltaBias).to.equal(3)
        })

        it("Does not increment the deltaWeights when all values are dropped out", () => {
            layer.filters.forEach(filter => filter.dropoutMap = filter.dropoutMap.map(r => r.map(v => true)))
            layer.filters[0].deltaWeights = testDataDeltaWeights
            layer.filters[1].deltaWeights = testDataDeltaWeights
            layer.filters[2].deltaWeights = testDataDeltaWeights
            layer.filters[3].deltaWeights = testDataDeltaWeights
            layer.backward()
            expect(layer.filters[0].deltaWeights).to.deep.equal(testDataDeltaWeights)
            expect(layer.filters[1].deltaWeights).to.deep.equal(testDataDeltaWeights)
            expect(layer.filters[2].deltaWeights).to.deep.equal(testDataDeltaWeights)
            expect(layer.filters[3].deltaWeights).to.deep.equal(testDataDeltaWeights)
        })

        it("Does otherwise change the deltaBias and deltaWeights values", () => {
            layer.filters.forEach(filter => filter.dropoutMap = filter.dropoutMap.map(r => r.map(v => false)))
            layer.filters[0].deltaBias = 0
            layer.filters[1].deltaBias = 1
            layer.filters[2].deltaBias = 2
            layer.filters[3].deltaBias = 3

            layer.filters[0].deltaWeights = [[[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]]]
            layer.filters[1].deltaWeights = [[[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]]]
            layer.filters[2].deltaWeights = [[[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]]]
            layer.filters[3].deltaWeights = [[[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]]]
            layer.backward()

            expect(layer.filters[0].deltaBias).to.not.equal(0)
            expect(layer.filters[1].deltaBias).to.not.equal(1)
            expect(layer.filters[2].deltaBias).to.not.equal(2)
            expect(layer.filters[3].deltaBias).to.not.equal(3)
            expect(layer.filters[0].deltaWeights).to.not.deep.equal([[[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]]])
            expect(layer.filters[1].deltaWeights).to.not.deep.equal([[[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]]])
            expect(layer.filters[2].deltaWeights).to.not.deep.equal([[[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]]])
            expect(layer.filters[3].deltaWeights).to.not.deep.equal([[[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]], [[1,2,3],[4,5,6],[7,8,9]]])
        })
    })

    describe("resetDeltaWeights", () => {

        let filter1, filter2, filter3, filter1b, filter2b, filter3b, filter4, filter5
        let layer, layer2

        beforeEach(() => {
            layer = new ConvLayer(3)
            filter1 = new Filter()
            filter2 = new Filter()
            filter3 = new Filter()
            layer.filters = [filter1, filter2, filter3]

            layer.filters.forEach(filter => {
                filter.errorMap = [[1,2,3],[1,2,3],[1,2,3]]
                filter.deltaWeights = [[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]
                filter.dropoutMap = [[true,true,true],[true,true,true],[true,true,true]]
            })

            layer2 = new ConvLayer(5)
            filter1b = new Filter()
            filter2b = new Filter()
            filter3b = new Filter()
            filter4 = new Filter()
            filter5 = new Filter()
            layer2.filters = [filter1b, filter2b, filter3b, filter4, filter5]

            layer2.filters.forEach(filter => {
                filter.errorMap = [[1,2,3],[1,2,3],[1,2,3]]
                filter.deltaWeights = [[[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]]
                filter.dropoutMap = [[true,true,true,true,true],[true,true,true,true,true],
                [true,true,true,true,true],[true,true,true,true,true],[true,true,true,true,true]]
            })
        })

        it("Sets all filters' deltaWeights values to 0", () => {
            layer.resetDeltaWeights()
            layer2.resetDeltaWeights()

            expect(filter1.deltaWeights).to.deep.equal([[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]])
            expect(filter2.deltaWeights).to.deep.equal([[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]])
            expect(filter3.deltaWeights).to.deep.equal([[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]])

            expect(filter1b.deltaWeights).to.deep.equal([[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]])
            expect(filter2b.deltaWeights).to.deep.equal([[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]])
            expect(filter3b.deltaWeights).to.deep.equal([[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]])
            expect(filter4.deltaWeights).to.deep.equal([[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]])
            expect(filter5.deltaWeights).to.deep.equal([[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]])
        })

        it("Clears the errorMap values", () => {
            layer.resetDeltaWeights()
            layer2.resetDeltaWeights()

            expect(filter1.errorMap).to.deep.equal([[0,0,0],[0,0,0],[0,0,0]])
            expect(filter2.errorMap).to.deep.equal([[0,0,0],[0,0,0],[0,0,0]])
            expect(filter3.errorMap).to.deep.equal([[0,0,0],[0,0,0],[0,0,0]])
        })

        it("Sets all the filters' dropoutMap values to false", () => {
            layer.resetDeltaWeights()
            layer2.resetDeltaWeights()

            expect(filter1.dropoutMap).to.deep.equal([[false,false,false],[false,false,false],[false,false,false]])
            expect(filter2.dropoutMap).to.deep.equal([[false,false,false],[false,false,false],[false,false,false]])
            expect(filter3.dropoutMap).to.deep.equal([[false,false,false],[false,false,false],[false,false,false]])

            expect(filter1b.dropoutMap).to.deep.equal([[false,false,false,false,false],[false,false,false,false,false],
                [false,false,false,false,false],[false,false,false,false,false],[false,false,false,false,false]])
            expect(filter2b.dropoutMap).to.deep.equal([[false,false,false,false,false],[false,false,false,false,false],
                [false,false,false,false,false],[false,false,false,false,false],[false,false,false,false,false]])
            expect(filter3b.dropoutMap).to.deep.equal([[false,false,false,false,false],[false,false,false,false,false],
                [false,false,false,false,false],[false,false,false,false,false],[false,false,false,false,false]])
            expect(filter4.dropoutMap).to.deep.equal([[false,false,false,false,false],[false,false,false,false,false],
                [false,false,false,false,false],[false,false,false,false,false],[false,false,false,false,false]])
            expect(filter5.dropoutMap).to.deep.equal([[false,false,false,false,false],[false,false,false,false,false],
                [false,false,false,false,false],[false,false,false,false,false],[false,false,false,false,false]])
        })

        it("Sets all the filters' deltaBias to 0", () => {

            layer.resetDeltaWeights()
            layer2.resetDeltaWeights()

            expect(filter1.deltaBias).to.equal(0)
            expect(filter2.deltaBias).to.equal(0)
            expect(filter3.deltaBias).to.equal(0)

            expect(filter1b.deltaBias).to.equal(0)
            expect(filter2b.deltaBias).to.equal(0)
            expect(filter3b.deltaBias).to.equal(0)
            expect(filter4.deltaBias).to.equal(0)
            expect(filter5.deltaBias).to.equal(0)
        })
    })

    describe("applyDeltaWeights", () => {

        let filter1, filter2, filter3, filter4
        let layer

        beforeEach(() => {
            layer = new ConvLayer(5)
            filter1 = new Filter()
            filter2 = new Filter()
            filter3 = new Filter()
            filter4 = new Filter()
            layer.filters = [filter1, filter2, filter3, filter4]

            layer.filters.forEach(filter => {
                filter.weights = [[[-0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]],[[0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,0.5,0.5]]]
                filter.bias = 0.5
                filter.deltaBias = 1
                filter.deltaWeights = [[[1,1,1],[1,1,1],[1,1,1]],[[1,1,1],[1,1,1],[1,1,1]]]
            })

            layer.net = {learningRate: 1, weightUpdateFn: NetMath.vanillasgd, miniBatchSize: 1, l2: 0, l1: 0}
        })


        it("Increments the weights of all filters with their respective deltas (when learning rate is 1)", () => {
            layer.applyDeltaWeights()
            expect(layer.filters[0].weights).to.deep.equal([[[0.5,1.5,1.5],[1.5,1.5,1.5],[1.5,1.5,1.5]],[[1.5,1.5,1.5],[1.5,1.5,1.5],[1.5,1.5,1.5]]])
            expect(layer.filters[1].weights).to.deep.equal([[[0.5,1.5,1.5],[1.5,1.5,1.5],[1.5,1.5,1.5]],[[1.5,1.5,1.5],[1.5,1.5,1.5],[1.5,1.5,1.5]]])
            expect(layer.filters[2].weights).to.deep.equal([[[0.5,1.5,1.5],[1.5,1.5,1.5],[1.5,1.5,1.5]],[[1.5,1.5,1.5],[1.5,1.5,1.5],[1.5,1.5,1.5]]])
            expect(layer.filters[3].weights).to.deep.equal([[[0.5,1.5,1.5],[1.5,1.5,1.5],[1.5,1.5,1.5]],[[1.5,1.5,1.5],[1.5,1.5,1.5],[1.5,1.5,1.5]]])
        })

        it("Increments the bias of all filters with their deltaBias", () => {
            layer.applyDeltaWeights()
            expect(layer.filters[0].bias).to.equal(1.5)
        })

        it("Increments the net.l2Error by each weight, applied to the L2 formula", () => {
            layer.net.l2 = 0.001
            layer.net.l2Error = 0
            layer.applyDeltaWeights()
            expect(Math.round(layer.net.l2Error*1000)/1000).to.equal(0.009) // 4 * 18 * (  0.5 * 0.001 * 0.5*0.5  )
        })

        it("Increments the net.l1Error by each weight, applied to the L1 formula", () => {
            layer.net.l1 = 0.005
            layer.net.l1Error = 0
            layer.applyDeltaWeights()
            expect(Math.round(layer.net.l1Error*1000)/1000).to.equal(0.18) // 4 * 18 * ( 0.005 * |0.5|  )
        })

        it("Increments the net.maxNormTotal if the net.maxNorm is configured", () => {

            sinon.stub(layer.net, "weightUpdateFn").callsFake(() => 0.5) // Return the same weight value
            layer.net.maxNorm = 3
            layer.net.maxNormTotal = 0
            layer.applyDeltaWeights()
            expect(layer.net.maxNormTotal).to.equal(18) // 4 * 18 * 0.5**2
        })
    })

    describe("backUpValidation", () => {
        it("Copies the filter weights to a 'validationWeights' array, for each filter", () => {
            const layer = new ConvLayer(5)
            layer.net = {conv: {}, weightsInitFn: x=> [...new Array(x)].map((_,i) => i)}
            const prevLayer = new ConvLayer(2)
            layer.filters = [...new Array(5)].map(f => new Filter())
            layer.init()

            for (let f=0; f<layer.filters.length; f++) {
                expect(layer.filters[f].validationWeights).to.be.undefined
            }

            layer.backUpValidation()

            for (let f=0; f<layer.filters.length; f++) {
                expect(layer.filters[f].validationWeights).to.deep.equal(layer.filters[f].weights)
            }
        })

        it("Copies the filter weights to a 'validationBias' array, for each filter", () => {
            const layer = new ConvLayer(5)
            layer.net = {conv: {}, weightsInitFn: x=> [...new Array(x)].map((_,i) => i)}
            const prevLayer = new ConvLayer(2)
            layer.filters = [...new Array(5)].map(f => new Filter())
            layer.channels = 2
            layer.filterSize = 3
            layer.init()

            for (let f=0; f<layer.filters.length; f++) {
                expect(layer.filters[f].validationBias).to.be.undefined
            }

            layer.backUpValidation()

            for (let f=0; f<layer.filters.length; f++) {
                expect(layer.filters[f].validationBias).to.equal(layer.filters[f].bias)
            }
        })
    })

    describe("restoreValidation", () => {
        it("Copies backed up 'validationWeights' values into every filter's weights arrays", () => {
            const layer = new ConvLayer(5)
            layer.net = {conv: {}, weightsInitFn: x => [...new Array(x)].map((_,i) => i)}
            const prevLayer = new ConvLayer(2)
            layer.filters = [...new Array(5)].map(f => new Filter())
            layer.channels = 2
            layer.filterSize = 3
            layer.init()

            for (let f=0; f<layer.filters.length; f++) {
                layer.filters[f].validationWeights = [...new Array(layer.channels)].map(channelWeights => {
                    return [...new Array(layer.filterSize)].map((weightsRow, wr) => [...new Array(layer.filterSize)].map((_,i) => i*i))
                })
                expect(layer.filters[f].weights).to.not.deep.equal(layer.filters[f].validationWeights)
            }

            layer.restoreValidation()

            for (let f=0; f<layer.filters.length; f++) {
                expect(layer.filters[f].weights).to.deep.equal(layer.filters[f].validationWeights)
            }
        })
    })

    describe("getDataSize", () => {
        it("Returns the correct total number of weights and biases (Example 1)", () => {
            const conv = new ConvLayer(2)
            conv.filters = [new Filter(), new Filter()]

            for (let n=0; n<conv.filters.length; n++) {
                conv.filters[n].weights = [[[1,2],[3,4]]]
                conv.filters[n].bias = 1
            }

            expect(conv.getDataSize()).to.equal(10)
        })
        it("Returns the correct total number of weights and biases (Example 2)", () => {
            const conv = new ConvLayer(4)
            conv.filters = [new Filter(), new Filter(), new Filter(), new Filter()]

            for (let n=0; n<conv.filters.length; n++) {
                conv.filters[n].weights = [[[1,2],[3,4]]]
                conv.filters[n].bias = 1
            }

            expect(conv.getDataSize()).to.equal(20)
        })
    })

    describe("toIMG", () => {
        it("Returns all filters' weights and biases as a 1 dimensional array (Example 1)", () => {
            const conv = new ConvLayer(2)
            conv.filters = [new Filter(), new Filter()]

            for (let n=0; n<conv.filters.length; n++) {
                conv.filters[n].weights = [[[1,2],[3,4]]]
                conv.filters[n].bias = 1
            }

            expect(conv.toIMG()).to.deep.equal([1,1,2,3,4,1,1,2,3,4])
        })
        it("Returns all filters' weights and biases as a 1 dimensional array (Example 2)", () => {
            const conv = new ConvLayer(4)
            conv.filters = [new Filter(), new Filter(), new Filter(), new Filter()]

            for (let n=0; n<conv.filters.length; n++) {
                conv.filters[n].weights = [[[1,2],[3,4]]]
                conv.filters[n].bias = 1
            }

            expect(conv.toIMG()).to.deep.equal([1,1,2,3,4,1,1,2,3,4,1,1,2,3,4,1,1,2,3,4])
        })
    })

    describe("fromIMG", () => {
        it("Sets the weights and biases to the given 1 dimensional array (Example 1)", () => {
            const testData = [1,1,2,3,4,1,1,2,3,4,1,1,2,3,4,1,1,2,3,4]  

            const conv = new ConvLayer(4)
            conv.filters = [new Filter(), new Filter(), new Filter(), new Filter()]

            for (let n=0; n<conv.filters.length; n++) {
                conv.filters[n].weights = [[[0,0],[0,0]]]
                conv.filters[n].bias = 0
            }

            conv.fromIMG(testData)

            expect(conv.filters[0].bias).to.equal(1)
            expect(conv.filters[1].bias).to.equal(1)
            expect(conv.filters[2].bias).to.equal(1)
            expect(conv.filters[3].bias).to.equal(1)
            expect(conv.filters[0].weights).to.deep.equal([[[1,2],[3,4]]])
            expect(conv.filters[1].weights).to.deep.equal([[[1,2],[3,4]]])
            expect(conv.filters[2].weights).to.deep.equal([[[1,2],[3,4]]])
            expect(conv.filters[3].weights).to.deep.equal([[[1,2],[3,4]]])
        })
        it("Sets the weights and biases to the given 1 dimensional array (Example 2)", () => {
            const testData = [1,1,2,3,4,1,1,2,3,4] 

            const conv = new ConvLayer(4)
            conv.filters = [new Filter(), new Filter()]

            for (let n=0; n<conv.filters.length; n++) {
                conv.filters[n].weights = [[[0,0],[0,0]]]
                conv.filters[n].bias = 0
            }

            conv.fromIMG(testData)

            expect(conv.filters[0].bias).to.equal(1)
            expect(conv.filters[1].bias).to.equal(1)
            expect(conv.filters[0].weights).to.deep.equal([[[1,2],[3,4]]])
            expect(conv.filters[1].weights).to.deep.equal([[[1,2],[3,4]]])
        })
    })

})

describe("PoolLayer", () => {

    describe("constructor", () => {

        it("Sets the layer.size to the size given", () => {
            const layer = new PoolLayer(3)
            expect(layer.size).to.equal(3)
        })

        it("Does not set the size to anything if not given", () => {
            const layer = new PoolLayer()
            expect(layer.size).to.be.undefined
        })

        it("Sets the layer.stride to the value given", () => {
            const layer = new PoolLayer(2, {stride: 3})
            expect(layer.stride).to.equal(3)
        })

        it("Does not set the stride to anything if not given", () => {
            const layer = new PoolLayer(2)
            expect(layer.stride).to.be.undefined
        })

        it("Sets the activation to false if nothing is provided", () => {
            const layer = new PoolLayer()
            expect(layer.activation).to.be.false
        })

        it("Allows setting the activation function to a custom function", () => {
            const customFn = x => x
            const layer = new PoolLayer(2, {activation: customFn})
            expect(layer.activation).to.equal(customFn)
        })

        it("Allows setting the activation function to false by setting it to false", () => {
            const layer = new PoolLayer(5, {activation: false})
            expect(layer.activation).to.be.false
        })

        it("Allows setting the activation function to a function from NetMath using a string", () => {
            const layer = new PoolLayer(5, {activation: "relu"})
            expect(layer.activation.name).to.equal("bound relu")
        })
    })

    describe("init", () => {
        it("Does nothing", () => {
            const layer = new PoolLayer(2)
            expect(layer.init()).to.be.undefined
        })
    })

    describe("assignNext", () => {
        it("Sets the layer.nextLayer to the given layer", () => {
            const layer1 = new ConvLayer()
            const layer2 = new PoolLayer(2)
            layer2.assignNext(layer1)
            expect(layer2.nextLayer).to.equal(layer1)
        })
    })

    describe("assignPrev", () => {

        it("Sets the layer.prevLayer to the given layer", () => {
            const layer1 = new ConvLayer()
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.outMapSize = 16
            layer2.assignPrev(layer1)
            expect(layer2.prevLayer).to.equal(layer1)
        })

        it("Sets the layer.layerIndex to the value given", () => {
            const layer1 = new ConvLayer()
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.outMapSize = 16
            layer2.assignPrev(layer1, 12345)
            expect(layer2.layerIndex).to.equal(12345)
        })


        it("Sets the layer.size to the net.pool.size if not already defined, but existing in net.pool", () => {
            const layer1 = new ConvLayer()
            layer1.outMapSize = 16
            const layer2 = new PoolLayer()
            layer2.net = {pool: {size: 3}}
            layer2.stride = 1
            layer2.assignPrev(layer1)
            expect(layer2.size).to.equal(3)
        })

        it("Defaults the layer.size to 2 if not defined and not present in net.pool", () => {
            const layer1 = new ConvLayer()
            layer1.outMapSize = 16
            const layer2 = new PoolLayer()
            layer2.net = {pool: {}}
            layer2.stride = 1
            layer2.assignPrev(layer1)
            expect(layer2.size).to.equal(2)
        })

        it("Sets the layer.stride to the net.pool.stride if not already defined, but existing in net.pool", () => {
            const layer1 = new ConvLayer()
            layer1.outMapSize = 16
            const layer2 = new PoolLayer()
            layer2.net = {pool: {stride: 3}}
            layer2.size = 1
            layer2.assignPrev(layer1)
            expect(layer2.stride).to.equal(3)
        })

        it("Defaults the layer.stride to the layer.size if not defined and not in net.pool", () => {
            const layer1 = new ConvLayer()
            layer1.outMapSize = 16
            const layer2 = new PoolLayer()
            layer2.net = {pool: {}}
            layer2.size = 1
            layer2.assignPrev(layer1)
            expect(layer2.stride).to.equal(1)
        })

        it("Sets the layer.channels to the last layer's filters count when the last layer is Conv", () => {
            const layer1 = new ConvLayer(5)
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.outMapSize = 16
            layer2.assignPrev(layer1)
            expect(layer2.channels).to.equal(5)
        })

        it("Sets the layer.channels to the net.channels if the prev layer is an FCLayer", () => {
            const layer1 = new FCLayer(108)
            const layer2 = new PoolLayer(2, {stride: 2})
            layer2.net = {channels: 3}
            layer2.assignPrev(layer1)
            expect(layer2.channels).to.equal(3)
        })

        it("Sets the layer.channels to the last layer's channels values if the last layer is Pool", () => {
            const layer1 = new PoolLayer(2, {stride: 2})
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.channels = 34
            layer1.outMapSize = 16
            layer2.assignPrev(layer1)
            expect(layer2.channels).to.equal(34)
        })

        it("Sets the layer.outMapSize to the correctly calculated value (Example 1)", () => {
            const layer1 = new ConvLayer(1)
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.outMapSize = 16
            layer2.assignPrev(layer1)
            expect(layer2.outMapSize).to.equal(8)
        })

        it("Sets the layer.outMapSize to the correctly calculated value (Example 2)", () => {
            const layer1 = new ConvLayer(1)
            const layer2 = new PoolLayer(3, {stride: 3})
            layer1.outMapSize = 15
            layer2.assignPrev(layer1)
            expect(layer2.outMapSize).to.equal(5)
        })

        it("Sets the layer.outMapSize to the correctly calculated value when prevLayer is FCLayer", () => {
            const layer1 = new FCLayer(108)
            const layer2 = new PoolLayer(2, {stride: 2})
            layer2.net = {channels: 3}
            layer2.assignPrev(layer1)
            expect(layer2.outMapSize).to.equal(3)
        })

        it("Sets the layer.outMapSize to the correctly calculated value when prevLayer is PoolLayer", () => {
            const layer1 = new PoolLayer(2, {stride: 2})
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.channels = 34
            layer1.outMapSize = 16
            layer2.assignPrev(layer1)
            expect(layer2.outMapSize).to.equal(8)
        })

        it("Sets the layer.inMapValuesCount to the square value of the input map width value", () => {
            const layer1 = new PoolLayer(2, {stride: 2})
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.channels = 34
            layer1.outMapSize = 16
            layer2.assignPrev(layer1)
            expect(layer2.outMapSize).to.equal(8)
            expect(layer2.inMapValuesCount).to.equal(256)
        })

        it("Throws an error if the hyperparameters are misconfigured to not produce an output volume with integer dimensions", () => {
            const layer1 = new ConvLayer(1)
            const layer2 = new PoolLayer(3, {stride: 3})
            layer1.outMapSize = 16
            expect(layer2.assignPrev.bind(layer2, layer1)).to.throw("Misconfigured hyperparameters. Activation volume dimensions would be ")
        })

        it("Sets the layer.indeces and layer.activations to an array with length equal to the number of channels (prev Conv example)", () => {
            const layer1 = new ConvLayer(6)
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.outMapSize = 16
            layer2.assignPrev(layer1)
            expect(layer2.activations.length).to.equal(6)
            expect(layer2.indeces.length).to.equal(6)
        })

        it("Sets the layer.indeces and layer.activations to an array with length equal to the number of channels (prev FC example)", () => {
            const layer1 = new FCLayer(108)
            const layer2 = new PoolLayer(2, {stride: 2})
            layer2.net = {channels: 3}
            layer2.assignPrev(layer1)
            expect(layer2.activations.length).to.equal(3)
            expect(layer2.indeces.length).to.equal(3)
        })

        it("Sets the layer.indeces and layer.activations to an array with length equal to the number of channels (prev Pool example)", () => {
            const layer1 = new PoolLayer(2, {stride: 2})
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.channels = 34
            layer1.outMapSize = 16
            layer2.assignPrev(layer1)
            expect(layer2.activations.length).to.equal(34)
            expect(layer2.indeces.length).to.equal(34)
        })

        it("Inits the layer.indeces values as maps of with dimensions equal to the outMapSize", () => {
            const layer1 = new ConvLayer(2)
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.outMapSize = 16
            layer2.assignPrev(layer1)
            expect(layer2.indeces[0].length).to.equal(8)
            expect(layer2.indeces[0][0].length).to.equal(8)
            expect(layer2.indeces[1].length).to.equal(8)
            expect(layer2.indeces[1][0].length).to.equal(8)
        })

        it("Inits the layer.indeces values as maps of with dimensions equal to the outMapSize", () => {
            const layer1 = new ConvLayer(2)
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.outMapSize = 16
            layer2.assignPrev(layer1)
            expect(layer2.activations[0].length).to.equal(8)
            expect(layer2.activations[0][0].length).to.equal(8)
            expect(layer2.activations[1].length).to.equal(8)
            expect(layer2.activations[1][0].length).to.equal(8)
        })

        it("Sets the values in the activations to arrays with two 0 values", () => {
            const layer1 = new ConvLayer(2)
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.outMapSize = 4
            layer2.assignPrev(layer1)
            expect(layer2.activations).to.deep.equal([ [[0,0],[0,0]], [[0,0],[0,0]] ])
        })

        it("Sets the values in the indeces to arrays with two 0 values", () => {
            const layer1 = new ConvLayer(2)
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.outMapSize = 4
            layer2.assignPrev(layer1)
            expect(layer2.indeces).to.deep.equal([[[[0,0],[0,0]],[[0,0],[0,0]]],[[[0,0],[0,0]],[[0,0],[0,0]]]])
        })

        it("Sets the layer.errors to an array with dimensions equal to the input map size", () => {
            const layer1 = new ConvLayer(2)
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.outMapSize = 16
            layer2.assignPrev(layer1)
            expect(layer2.errors[0].length).to.equal(16)
            expect(layer2.errors[0][0].length).to.equal(16)
            expect(layer2.errors[1].length).to.equal(16)
            expect(layer2.errors[1][0].length).to.equal(16)
        })

        it("Sets the values in the errors to arrays with two 0 values", () => {
            const layer1 = new ConvLayer(2)
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.outMapSize = 4
            layer2.assignPrev(layer1)
            expect(layer2.errors).to.deep.equal([[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]])
        })
    })

    describe("forward", () => {

        beforeEach(() => sinon.stub(NetMath, "maxPool"))

        afterEach(() => NetMath.maxPool.restore())

        it("Calls the max pool function for each channel in the layer", () => {
            const layer = new PoolLayer(2)
            layer.channels = 14
            layer.forward()
            expect(NetMath.maxPool.callCount).to.equal(14)
        })

        it("Does not apply activation function if not provided", () => {
            const layer = new PoolLayer(2)
            layer.activation = false
            layer.channels = 2
            layer.outMapSize = 2
            layer.activations = [[[1,2],[3,4]],[[5,6],[7,8]]]
            layer.forward()
            expect(layer.activations).to.deep.equal([[[1,2],[3,4]],[[5,6],[7,8]]])
        })

        it("Applies activation function if provided (using sigmoid for test)", () => {
            const layer = new PoolLayer(2, {activation: "sigmoid"})
            layer.channels = 2
            layer.outMapSize = 2
            layer.activations = [[[1,2],[3,4]],[[5,6],[7,8]]]
            layer.forward()

            const expected = [[
                [0.7310585786300049, 0.8807970779778823],
                [0.9525741268224334, 0.9820137900379085]
            ], [
                [0.9933071490757153, 0.9975273768433653],
                [0.9990889488055994, 0.9996646498695336]
            ]]

            expect(layer.activations).to.deep.equal(expected)
        })
    })

    describe("backward", () => {

        const convLayer = new ConvLayer(1, {filterSize: 3, zeroPadding: 1, stride: 1})
        const fcLayer = new FCLayer(36)
        const poolLayer = new PoolLayer(2)
        fcLayer.neurons.forEach((neuron, ni) => {
            neuron.error = ni / 100
            neuron.weights = [...new Array(36)].map((v, vi) => (vi%10))
        })

        // 6 x 6
        convLayer.filters = [new Filter()]
        convLayer.filters[0].errorMap = [
            [1,2,3,4,5,6],
            [7,4,7,2,9,2],
            [1,9,3,7,3,6],
            [2,5,2,6,8,3],
            [8,2,4,9,2,7],
            [1,7,3,7,3,5]
        ]
        convLayer.filters[0].weights = [[
            [1,2,3],
            [4,5,2],
            [3,2,1]
        ]]

        // 2 x 3 x 3
        poolLayer.errors = [[
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ], [
            [5,9,3],
            [8,2,4],
            [6,7,1]
        ]]

        // 12 x 12
        const expectedErrorsFC =  [[
            [0,0, 0,6.3, 0,12.6, 0,18.9, 0,25.2, 0,31.5],
            [0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
            [0,0, 0,0, 0,50.4, 0,0, 0,0, 0,0],
            [0,37.8, 0,44.1, 0,0, 0,56.7, 0,0, 0,6.3],
            [0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
            [0,12.6, 0,18.9, 0,25.2, 0,31.5, 0,37.8, 0,44.1],
            [0,50.4, 0,56.7, 0,0, 0,6.3, 0,12.6, 0,18.9],
            [0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
            [0,0, 0,0, 0,37.8, 0,0, 0,0, 0,0],
            [0,25.2, 0,31.5, 0,0, 0,44.1, 0,50.4, 0,56.7],
            [0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
            [0,0, 0,6.3, 0,12.6, 0,18.9, 0,25.2, 0,31.5]
        ]]

        const expectedErrorsFCWithActivation =  [[
            [0,0, 0, 0.011526349447153203, 0,0.000042487105415310055, 0,1.1702970200399024e-7, 0,2.865355952475672e-10, 0,6.57474075183004e-13],
            [0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
            [0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
            [0,0, 0,0, 0,0, 0,0, 0,0, 0,0.011526349447153203],
            [0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
            [0,0.000042487105415310055, 0,1.1702970200399024e-7, 0,2.865355952475672e-10, 0,6.57474075183004e-13, 0,0, 0,0],
            [0,0, 0,0, 0,0, 0,0.011526349447153203, 0,0.000042487105415310055, 0,1.1702970200399024e-7],
            [0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
            [0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
            [0,2.865355952475672e-10, 0,6.57474075183004e-13, 0,0, 0,0, 0,0, 0,0],
            [0,0, 0,0, 0,0, 0,0, 0,0, 0,0],
            [0,0, 0,0.011526349447153203, 0,0.000042487105415310055, 0,1.1702970200399024e-7, 0,2.865355952475672e-10, 0,6.57474075183004e-13]
        ]]

        // Avoid tests failing due to tiny precision differences
        const roundVolValues = map => map.map(row => row.map(col => col.map(v => Math.round(v*10)/10)))

        // 13 x 13
        const expectedErrorsConv = [[
            [0, 31,  0,  0,  0,  0, 63,  0,  0, 83, 71,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [70,  0,100, 60,111,  0,112,  0,202,  0, 66,  0,  0],
            [0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 76,  0,  0,  0,  0,110,  0,  0,116, 79,  0,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [77,  0, 97,113,103,  0,124,  0,250,  0, 66,  0,  0],
            [0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0, 76,  0,  0, 80,  0,  0,  0,  0,  0,119,  0,  0],
            [0, 0,  0,  0,  0,121,  0,  0,  0,  0,  0,  0,  0],
            [0,  0, 73,  0,  0,  0,125,  0, 83,  0,  0, 72,  0],
            [0, 55,  0,  0,  0,  0,  0, 81,  0,  0,  0, 47,  0],
            [0,  0,  0,  0,  0,  0, 94,  0,  0,  0,  0,  0,  0]
        ]]

        // 2 x 9 x 9
        const expectedErrorsPool = [[
            [0,0,0,0,2,0,0,0,0],
            [0,0,0,0,0,0,0,3,0],
            [0,0,1,0,0,0,0,0,0],
            [0,4,0,0,0,0,0,0,6],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,5,0,0,0,0,0],
            [0,7,0,0,8,0,0,9,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0]
        ], [
            [5,0,0,9,0,0,3,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,4,0],
            [0,0,8,0,0,0,0,0,0],
            [0,0,0,0,2,0,0,0,0],
            [6,0,0,0,0,7,0,1,0],
            [0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0]
        ]]

        it("Creates the error map correctly when the next layer is an FCLayer", () => {
            const layer = new PoolLayer(2, {stride: 2})
            layer.channels = 1
            layer.outMapSize = 6

            // 12 x 12
            layer.errors = [[
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0]
            ]]
            layer.indeces = [[
                [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],
                [[1,1],[1,1],[0,1],[1,1],[1,1],[1,1]],
                [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]],
                [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],
                [[1,1],[1,1],[0,1],[1,1],[1,1],[1,1]],
                [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]
            ]]
            layer.nextLayer = fcLayer
            layer.backward()
            expect(roundVolValues(layer.errors)).to.deep.equal(expectedErrorsFC)
        })

        it("Creates the error map correctly when the next layer is a ConvLayer", () => {
            const layer = new PoolLayer(3, {stride: 2})
            layer.channels = 1
            layer.outMapSize = 6

            // 13 x 13
            layer.errors = [[
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0]
            ]]
            // 6 x 6
            layer.indeces = [[
                [[0,1],[2,1],[0,2],[2,2],[0,1],[0,0]],
                [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
                [[0,1],[2,1],[0,2],[2,2],[0,1],[0,0]],
                [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
                [[0,1],[0,2],[1,1],[2,0],[0,2],[2,1]],

                [[1,1],[0,0],[2,2],[1,1],[0,0],[1,1]]
            ]]

            layer.nextLayer = convLayer
            layer.backward()
            expect(layer.errors).to.deep.equal(expectedErrorsConv)
        })

        it("Creates the error map correctly when the next layer is a PoolLayer", () => {
            const layer = new PoolLayer(3, {stride: 3})
            layer.channels = 2
            layer.outMapSize = 3

            // 2 x 9 x 9
            layer.errors = [[
                [0,0,0, 0,0,0, 0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],

                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],

                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
            ], [
                [0,0,0, 0,0,0, 0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],

                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],

                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0],
            ]]

            // 2 x 3 x 3
            layer.indeces = [[
                [[2,2],[0,1],[1,1]],
                [[0,1],[2,0],[0,2]],
                [[0,1],[0,1],[0,1]]
            ], [
                [[0,0],[0,0],[0,0]],
                [[1,2],[2,1],[0,1]],
                [[0,0],[0,2],[0,1]]
            ]]
            layer.nextLayer = poolLayer
            layer.backward()
            expect(layer.errors).to.deep.equal(expectedErrorsPool)
        })

        it("Applies activation derivative when the activation is sigmoid", () => {
            const layer = new PoolLayer(2, {stride: 2, activation: "sigmoid"})
            layer.channels = 1
            layer.outMapSize = 6

            // 12 x 12
            layer.errors = [[
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0],
            ]]
            layer.indeces = [[
                [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],
                [[1,1],[1,1],[0,1],[1,1],[1,1],[1,1]],
                [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]],
                [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],
                [[1,1],[1,1],[0,1],[1,1],[1,1],[1,1]],
                [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]
            ]]
            layer.nextLayer = fcLayer
            layer.backward()
            expect(layer.errors).to.deep.equal(expectedErrorsFCWithActivation)
        })
    })

    describe("resetDeltaWeights", () => {
        it("Does nothing", () => {
            const layer = new PoolLayer()
            expect(layer.resetDeltaWeights()).to.be.undefined
        })
    })

    describe("applyDeltaWeights", () => {
        it("Does nothing", () => {
            const layer = new PoolLayer()
            expect(layer.applyDeltaWeights()).to.be.undefined
        })
    })

    describe("backUpValidation", () => {
        it("Does nothing", () => {
            const layer = new PoolLayer()
            expect(layer.backUpValidation()).to.be.undefined
        })
    })

    describe("restoreValidation", () => {
        it("Does nothing", () => {
            const layer = new PoolLayer()
            expect(layer.restoreValidation()).to.be.undefined
        })
    })

    describe("toJSON", () => {
        it("Exports an empty object", () => {
            const layer = new PoolLayer()
            expect(layer.toJSON()).to.deep.equal({})
        })
    })

    describe("fromJSON", () => {
        it("Does nothing", () => {
            const layer = new PoolLayer()
            expect(layer.fromJSON()).to.be.undefined
        })
    })

    describe("getDataSize", () => {
        it("Returns 0", () => {
            const pool = new PoolLayer()
            expect(pool.getDataSize()).to.equal(0)
        })
    })

    describe("toIMG", () => {
        it("Returns an empty array", () => {
            const pool = new PoolLayer()
            expect(pool.toIMG()).to.deep.equal([])
        })
    })

    describe("fromIMG", () => {
        it("Does nothing", () => {
            const pool = new PoolLayer()
            expect(pool.fromIMG([])).to.be.undefined
        })
    })
})

describe("Netmath", () => {

    describe("sigmoid", () => {

        it("sigmoid(1.681241237) == 0.8430688214048092", () => {
            expect(NetMath.sigmoid(1.681241237)).to.equal(0.8430688214048092)
        })

        it("sigmoid(0.8430688214048092, true) == 0.21035474941074114", () => {
            expect(NetMath.sigmoid(0.8430688214048092, true)).to.equal(0.21035474941074114)
        })
    })

    describe("tanh", () => {

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

        it("lrelu(-2, true)==-0.0005", () => {
            expect(NetMath.lrelu.bind({lreluSlope:-0.0005}, -2, true)()).to.equal(-0.0005)
        })

        it("Defaults the lreluSlope value if undefined", () => {
            expect(NetMath.lrelu.bind({}, 2)()).to.equal(2)
        })

        it("Defaults the lreluSlope value if undefined (when calculating prime)", () => {
            expect(NetMath.lrelu.bind({}, 2, true)()).to.equal(1)
            expect(NetMath.lrelu.bind({}, -2, true)()).to.equal(-0.0005)
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
            expect(NetMath.elu(2, false, {eluAlpha: 1})).to.equal(2)
        })

        it("elu(-0.25)==-0.22119921692859512", () => {
            expect(NetMath.elu(-0.25, false, {eluAlpha: 1})).to.equal(-0.22119921692859512)
        })

        it("elu(2, true)==1", () => {
            expect(NetMath.elu(2, true, {eluAlpha: 1})).to.equal(1)
        })

        it("elu(-0.5, true)==0.6065306597126334", () => {
            expect(NetMath.elu(-0.5, true, {eluAlpha: 1})).to.equal(0.6065306597126334)
        })
    })

    describe("Cross Entropy", () => {

        it("crossentropy([1,0,0.3], [0,1, 0.8]) == 70.16654147569186", () => {
            expect(NetMath.crossentropy([1,0,0.3], [0,1, 0.8])).to.equal(70.16654147569186)
        })
    })

    describe("Softmax", () => {

        it("softmax([1, 2, 3, 4, 1, 2, 3]) == [0.02364054302159139, 0.06426165851049616, 0.17468129859572226, 0.47483299974438037, 0.02364054302159139, 0.06426165851049616, 0.17468129859572226]", () => {
            expect(NetMath.softmax([1, 2, 3, 4, 1, 2, 3])).to.deep.equal([0.02364054302159139, 0.06426165851049616, 0.17468129859572226, 0.47483299974438037, 0.02364054302159139, 0.06426165851049616, 0.17468129859572226])
        })

        it("softmax([23, 54, 167, 3]) == [2.8946403116483003e-63, 8.408597124803643e-50, 1, 5.96629836401057e-72]", () => {
            expect(NetMath.softmax([23, 54, 167, 3])).to.deep.equal([2.8946403116483003e-63, 8.408597124803643e-50, 1, 5.96629836401057e-72])
        })
    })

    describe("Mean Squared Error", () => {

        it("meansquarederror([13,17,18,20,24], [12,15,20,22,24]) == 2.6", () => {
            expect(NetMath.meansquarederror([13,17,18,20,24], [12,15,20,22,24])).to.equal(2.6)
        })
    })

    describe("Root Mean Squared Error", () => {

        it("rootmeansquarederror([13,17,18,20,24], [12,15,20,22,24]) == 1.61245154965971", () => {
            expect(NetMath.rootmeansquarederror([13,17,18,20,24], [12,15,20,22,24])).to.equal(1.61245154965971)
        })
    })

    describe("vanillasgd", () => {

        const fn = NetMath.vanillasgd.bind({learningRate: 0.5})

        it("Increments a weight with half of its delta weight when the learning rate is 0.5", () => {
            expect(fn(10, 10)).to.equal(15)
            expect(fn(10, 20)).to.equal(20)
            expect(fn(10, -30)).to.equal(-5)
        })
    })

    describe("momentum", () => {

        let neuron

        beforeEach(() => {
            neuron = new Neuron()
            neuron.weights = [1,2,3,4,5]
            neuron.init({updateFn: "momentum"})
        })

        it("Increments the biasCache by the momentum value times learning rate times delta bias", () => {
            neuron.biasCache = 0.123
            NetMath.momentum.bind({learningRate: 0.2, momentum: 0.75}, 1, 3, neuron)()
            expect(neuron.biasCache.toFixed(5)).to.equal("-0.50775")
        })

        it("Increments the neuron's weights cache by the momentum value times learning rate times weight delta", () => {
            neuron.weightsCache = [1,1,1]
            const result1 = NetMath.momentum.bind({learningRate: 0.3, momentum: 0.5}, 1, 3, neuron, 0)()
            const result2 = NetMath.momentum.bind({learningRate: 0.3, momentum: 0.5}, 1, 4, neuron, 1)()
            const result3 = NetMath.momentum.bind({learningRate: 0.3, momentum: 0.5}, 1, 2, neuron, 2)()
            expect(neuron.weightsCache[0]).to.equal(-0.3999999999999999)
            expect(neuron.weightsCache[1]).to.equal(-0.7)
            expect(neuron.weightsCache[2].toFixed(1)).to.equal("-0.1")

            expect(result1).to.equal(1.4)
            expect(result2).to.equal(1.7)
            expect(result3).to.equal(1.1)
        })


    })

    describe("gain", () => {

        let neuron

        beforeEach(() => {
            neuron = new Neuron()
            neuron.weights = [1,2,3,4,5]
            neuron.init({updateFn: "gain"})
        })

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

        beforeEach(() => {
            neuron = new Neuron()
            neuron.weights = [1,2,3,4,5]
            neuron.init({updateFn: "adagrad"})
        })

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

        it("Increments the neuron's weightsCache the same way as the biasCache", () => {
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

        beforeEach(() => {
            neuron = new Neuron()
            neuron.weights = [1,2,3,4,5]
            neuron.init({updateFn: "rmsprop"})
        })

        it("Sets the cache value to the correct value, following the rmsprop formula", () => {
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

        it("Sets the neuron.m to the correct value, following the formula", () => {
            neuron.m = 0.1
            NetMath.adam.bind({learningRate: 0.01}, 1, 0.2, neuron)()
            expect(neuron.m.toFixed(2)).to.equal("0.11") // 0.9 * 0.1 + (1-0.9) * 0.2
        })

        it("Sets the neuron.v to the correct value, following the formula", () => {
            neuron.v = 0.1
            NetMath.adam.bind({learningRate: 0.01}, 1, 0.2, neuron)()
            expect(neuron.v.toFixed(5)).to.equal("0.09994") // 0.999 * 0.1 + (1-0.999) * 0.2*0.2
        })

        it("Calculates a value correctly, following the formula", () => {
            neuron.m = 0.121
            neuron.v = 0.045
            const result = NetMath.adam.bind({learningRate: 0.01, iterations: 2}, -0.3, 0.02, neuron)()
            expect(result.toFixed(6)).to.equal("-0.298943")
        })
    })

    describe("adadelta", () => {

        let neuron

        beforeEach(() => {
            neuron = new Neuron()
            neuron.weights = [1,2,3,4,5]
            neuron.init({updateFn: "adadelta"})
        })

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

        it("Creates a value for the bias correctly, following the formula", () => {
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

        it("The standard deviation of the weights is roughly 0.5 when the fanIn is 5", () => {
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

    describe("maxPool", () => {

        // Hand worked values

        // 12 x 12
        const testMapData = [
            1,2,4,5,7,8,10,11,13,14,16,17,
            1,2,4,5,7,8,10,11,13,14,16,17,

            1,2,4,5,7,9,10,11,13,14,16,17,
            2,3,5,6,8,8,11,12,14,15,17,18,

            1,2,4,5,7,8,10,11,13,14,16,17,
            2,3,5,6,8,9,11,12,14,15,17,18,

            1,2,4,5,7,8,10,11,13,14,16,17,
            1,2,4,5,7,8,10,11,13,14,16,17,

            1,2,4,5,7,9,10,11,13,14,16,17,
            2,3,5,6,8,8,11,12,14,15,17,18,

            1,2,4,5,7,8,10,11,13,14,16,17,
            2,3,5,6,8,9,11,12,14,15,17,18
        ]

        // 6 x 6
        const expectedActivationMap = [
            [2,5,8,11,14,17],
            [3,6,9,12,15,18],
            [3,6,9,12,15,18],
            [2,5,8,11,14,17],
            [3,6,9,12,15,18],
            [3,6,9,12,15,18]
        ]

        // 6 x 6
        const expectedIndeces = [
            [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],
            [[1,1],[1,1],[0,1],[1,1],[1,1],[1,1]],
            [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]],
            [[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],
            [[1,1],[1,1],[0,1],[1,1],[1,1],[1,1]],
            [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]
        ]

        const layer = new PoolLayer(2, {stride: 2})
        layer.prevLayerOutWidth = 12
        layer.outMapSize = 6

        // 6 x 6
        layer.activations = [[
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]
        ]]

        // 6 x 6
        layer.indeces = [[
            [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
        ]]


        before(() => {
            sinon.stub(NetUtil, "getActivations").callsFake(() => testMapData)
        })

        after(() => NetUtil.getActivations.restore())


        it("Performs max pool correctly", () => {
            NetMath.maxPool(layer, 0)
            expect(layer.activations[0]).to.deep.equal(expectedActivationMap)
            expect(layer.indeces[0]).to.deep.equal(expectedIndeces)
        })
    })
})

describe("NetUtil", () => {

    describe("format", () => {

        it("Returns undefined if passed undefined", () => {
            expect(NetUtil.format(undefined)).to.be.undefined
        })

        it("Turns a string to lower case", () => {
            const testString = "aAbB"
            const result = NetUtil.format(testString)
            expect(result).to.equal("aabb")
        })

        it("Removes white spaces", () => {
            const testString = " aA bB "
            const result = NetUtil.format(testString)
            expect(result).to.equal("aabb")
        })

        it("Removes underscores", () => {
            const testString = "_aA_bB_"
            const result = NetUtil.format(testString)
            expect(result).to.equal("aabb")
        })

        it("Formats given milliseconds to milliseconds only when under a second", () => {
            const testMils = 100
            expect(NetUtil.format(testMils, "time")).to.equal("100ms")
        })

        it("Formats given milliseconds to seconds only when under a minute", () => {
            const testMils = 10000
            expect(NetUtil.format(testMils, "time")).to.equal("10.0s")
        })

        it("Formats given milliseconds to minutes and seconds only when under an hour", () => {
            const testMils = 100000
            expect(NetUtil.format(testMils, "time")).to.equal("1m 40s")
        })

        it("Formats given milliseconds to hours, minutes and seconds when over an hour", () => {
            const testMils = 10000000
            expect(NetUtil.format(testMils, "time")).to.equal("2h 46m 40s")
        })
    })

    describe("shuffle", () => {

        const testArr = [1,2,3,4,5, "a", "b", "c"]
        const original = testArr.slice(0)
        NetUtil.shuffle(testArr)

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

    describe("addZeroPadding", () => {

        const testData = [[3,5,2,6,8],
                          [9,6,4,3,2],
                          [2,9,3,4,2],
                          [5,8,1,3,7],
                          [4,8,6,4,3]]


        it("Returns the same data when zero padding of 0 is given", () => {
            const result = NetUtil.addZeroPadding(testData, 0)
            expect(result).to.deep.equal(testData)
        })

        it("Returns a map with 1 level of zeroes padded on the edges when zero padding of 1 is given", () => {
            const result = NetUtil.addZeroPadding(testData, 1)
            expect(result.length).to.equal(7)

            result.forEach(row => expect(row.length).to.equal(7))

            expect(result[0]).to.deep.equal([0,0,0,0,0,0,0])
            expect(result[6]).to.deep.equal([0,0,0,0,0,0,0])
        })

        it("Returns a map with 2 levels of zeroes padded on the edges when zero padding of 2 is given", () => {
            const result = NetUtil.addZeroPadding(testData, 2)
            expect(result.length).to.equal(9)

            result.forEach(row => expect(row.length).to.equal(9))

            expect(result[0]).to.deep.equal([0,0,0,0,0,0,0,0,0])
            expect(result[8]).to.deep.equal([0,0,0,0,0,0,0,0,0])
        })

        it("Returns a map with 3 level of zeroes padded on the edges when zero padding of 3 is given", () => {
            const result = NetUtil.addZeroPadding(testData, 3)
            expect(result.length).to.equal(11)

            result.forEach(row => expect(row.length).to.equal(11))

            expect(result[0].length).to.equal(11)
            expect(result[0]).to.deep.equal([0,0,0,0,0,0,0,0,0,0,0])
        })

        it("Keeps the same data, apart from the zeroes", () => {
            let result = NetUtil.addZeroPadding(testData, 1)
            result = result.splice(1, 5)
            result.forEach((row, ri) => result[ri] = result[ri].splice(1, 5))
            expect(result).to.deep.equal(testData)
        })
    })

    describe("arrayToMap", () => {

        const testArray = [1,2,3,4,5,6,7,8,9]
        const testMap = [[1,2,3],[4,5,6],[7,8,9]]

        const result = NetUtil.arrayToMap(testArray, 3)

        it("Converts the array correctly", () => {
            expect(result).to.deep.equal(testMap)
        })
    })

    describe("arrayToVolume", () => {

        const testArray = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,
                            25,26,27,28,29,30,31,32,33,34,35,36]

        const testVol1 = [[[1,2],[3,4]], [[5,6],[7,8]], [[9,10],[11,12]], [[13,14],[15,16]],
              [[17,18],[19,20]], [[21,22],[23,24]], [[25,26],[27,28]], [[29,30],[31,32]], [[33,34],[35,36]]]

        const testVol2 = [ [[1,2,3],[4,5,6],[7,8,9]], [[10,11,12],[13,14,15],[16,17,18]],
                       [[19,20,21],[22,23,24],[25,26,27]], [[28,29,30],[31,32,33],[34,35,36]] ]

        const testVol3 = [[[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24],
                                    [25,26,27,28,29,30],[31,32,33,34,35,36]]]

        it("Converts the array correctly", () => {
            expect(NetUtil.arrayToVolume(testArray, 9)).to.deep.equal(testVol1)
        })

        it("Converts the array correctly", () => {
            expect(NetUtil.arrayToVolume(testArray, 4)).to.deep.equal(testVol2)
        })

        it("Converts the array correctly", () => {
            expect(NetUtil.arrayToVolume(testArray, 1)).to.deep.equal(testVol3)
        })
    })

    describe("getActivations", () => {

        let layer, layer2, filter, filter2, convLayer, net

        beforeEach(() => {

            layer = new FCLayer(9)
            layer.neurons = [{activation: 1},{activation: 2},{activation: 3},{activation: 4},
            {activation: 5},{activation: 6},{activation: 7},{activation: 8},{activation: 9}]

            layer2 = new FCLayer(64)
            layer2.neurons.forEach((neuron, ni) => neuron.activation = ni+1)

            convLayer = new ConvLayer(2)
            net = new Network({layers: [layer, convLayer]})
            filter = convLayer.filters[0]
            filter2 = convLayer.filters[1]
            filter.activationMap = [[1,2,3],[4,5,6],[7,8,9]]
            filter2.activationMap = [[4,5,6],[7,8,9],[1,2,3]]
        })

        it("Returns all the neurons' activations in an FCLayer, when called with no index parameters", () => {
            const activations = NetUtil.getActivations(layer)
            expect(activations).to.deep.equal([1,2,3,4,5,6,7,8,9])
        })

        it("Returns the FCLayer neuron activations in a square (map) subset of the neurons, indicated by the map index and map size", () => {
            const range1 = NetUtil.getActivations(layer2, 0, 4)
            const range2 = NetUtil.getActivations(layer2, 2, 9)
            const range3 = NetUtil.getActivations(layer2, 1, 16)
            expect(range1).to.deep.equal([1,2,3,4])
            expect(range2).to.deep.equal([19,20,21,22,23,24,25,26,27])
            expect(range3).to.deep.equal([17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])
        })

        it("Returns all the activations from a Conv layer when called with no index parameters", () => {
            const activations = NetUtil.getActivations(convLayer)
            expect(activations).to.deep.equal([1,2,3,4,5,6,7,8,9, 4,5,6,7,8,9,1,2,3])
        })

        it("Returns the activations from a Filter in a ConvLayer if map index is provided as second parameter", () => {
            const activations1 = NetUtil.getActivations(convLayer, 1)
            const activations2 = NetUtil.getActivations(convLayer, 0)
            expect(activations1).to.deep.equal([4,5,6,7,8,9,1,2,3])
            expect(activations2).to.deep.equal([1,2,3,4,5,6,7,8,9])
        })

        it("Returns all the activations from a PoolLayer correctly", () => {
            const layer = new PoolLayer()
            layer.activations = [[
                [2,5,8,11,14,17],
                [3,6,9,12,15,18],
                [3,6,9,12,15,18],
                [2,5,8,11,14,17],
                [3,6,9,12,15,18],
                [3,6,9,12,15,18]
            ]]

            expect(NetUtil.getActivations(layer)).to.deep.equal([2,5,8,11,14,17,3,6,9,12,15,18,3,6,9,12,15,18,2,5,8,11,14,17,3,6,9,12,15,18,3,6,9,12,15,18])
        })

        it("Returns just one activation map from a PoolLayer when called with a map index parameter", () => {
            const layer = new PoolLayer()
            layer.activations = [
                [[1,2],[3,4]],
                [[5,6],[7,8]]
            ]

            expect(NetUtil.getActivations(layer, 0)).to.deep.equal([1,2,3,4])
            expect(NetUtil.getActivations(layer, 1)).to.deep.equal([5,6,7,8])
        })
    })

    describe("convolve", () => {

        // http://cs231n.github.io/convolutional-networks/
        const testInputa = [0,0,2,2,2,  1,1,0,2,0,  1,2,1,1,2,  0,1,2,2,1,  1,2,0,0,1]
        const testInputb = [2,2,1,1,2,  1,1,2,0,0,  2,0,0,2,2,  1,2,2,1,1,  1,1,2,0,1]
        const testInputc = [0,1,1,0,0,  1,2,0,2,0,  2,0,1,2,0,  2,0,1,0,1,  0,1,2,2,1]
        const testInput = [0,0,2,2,2,  1,1,0,2,0,  1,2,1,1,2,  0,1,2,2,1,  1,2,0,0,1,
                           2,2,1,1,2,  1,1,2,0,0,  2,0,0,2,2,  1,2,2,1,1,  1,1,2,0,1,
                           0,1,1,0,0,  1,2,0,2,0,  2,0,1,2,0,  2,0,1,0,1,  0,1,2,2,1]

        const testWeightsa = [[[-1,0,-1],[1,0,1],[1,-1,0]]]
        const testWeightsb = [[[0,1,1],[1,-1,-1],[-1,1,0]]]
        const testWeightsc = [[[-1,0,-1],[1,0,0],[1,0,0]]]

        const testWeights1 = [[[-1,0,-1],[1,0,1],[1,-1,0]],   [[0,1,1],[1,-1,-1],[-1,1,0]],  [[-1,0,-1],[1,0,0],[1,0,0]]]
        const testWeights2 = [[[-1,0,1],[-1,1,1],[1,0,0]],   [[0,-1,1],[1,-1,1],[-1,1,-1]],  [[0,0,0],[0,-1,-1],[0,0,1]]]

        const expecteda = [[0,4,5],[2,0,1],[2,0,-1]]
        const expectedb = [[-2,2,0],[2,1,1],[2,3,1]]
        const expectedc = [[1,4,3],[-1,-3,1],[1,2,3]]

        const expected1 = [[-3,8,6],[1,-4,1],[3,3,1]]
        const expected2 = [[1,9,1],[-1,-2,1],[4,-7,-4]]

        it("Convolves the input correctly (Example a)", () => {
            expect(NetUtil.convolve({
                input: testInputa,
                zeroPadding: 1,
                weights: testWeightsa,
                channels: 1,
                stride: 2,
                bias: 1
            })).to.deep.equal(expecteda)
        })

        it("Convolves the input correctly (Example b)", () => {
            expect(NetUtil.convolve({
                input: testInputb,
                zeroPadding: 1,
                weights: testWeightsb,
                channels: 1,
                stride: 2,
                bias: 1
            })).to.deep.equal(expectedb)
        })

        it("Convolves the input correctly (Example c)", () => {
            expect(NetUtil.convolve({
                input: testInputc,
                zeroPadding: 1,
                weights: testWeightsc,
                channels: 1,
                stride: 2,
                bias: 1
            })).to.deep.equal(expectedc)
        })

        it("Convolves the input correctly (Example 1)", () => {
            expect(NetUtil.convolve({
                input: testInput,
                zeroPadding: 1,
                weights: testWeights1,
                channels: 3,
                stride: 2,
                bias: 1
            })).to.deep.equal(expected1)
        })

        it("Convolves the input correctly (Example 2)", () => {
            expect(NetUtil.convolve({
                input: testInput,
                zeroPadding: 1,
                weights: testWeights2,
                channels: 3,
                stride: 2,
                bias: 0
            })).to.deep.equal(expected2)
        })
    })

    describe("buildConvDWeights", () => {

        // Following values worked out by hand
        const layer = new ConvLayer(2, {filterSize: 3, zeroPadding: 1, stride: 2})
        const prevLayer = new FCLayer(75)

        const net = new Network({
            channels: 3,
            layers: [prevLayer, layer]
        })
        net.miniBatchSize = 1

        // Set the activation of each neuron to 1,2,3,4,....,74,75
        prevLayer.neurons.forEach((neuron, ni) => neuron.activation = ni+1)

        const errorMapFilter1 = [[0.1, 0.6, 0.2], [0.7, 0.3, 0.8], [0.4, 0.9, 0.5]]
        const errorMapFilter2 = [[-0.5, 0, -0.4], [0.1, -0.3, 0.2], [-0.2, 0.3, -0.1]]

        const expectedFilter1Channel1DWeights = [[34.1, 47.2, 31.5], [48.6, 68.1, 45.6], [26.3, 40, 23.7]]
        const expectedFilter1Channel2DWeights = [[96.6, 137.2, 89], [131.1, 180.6, 120.6], [73.8, 107.5, 66.2]]
        const expectedFilter1Channel3DWeights = [[159.1, 227.2, 146.5], [213.6, 293.1, 195.6], [121.3, 175, 108.7]]

        const expectedFilter2Channel1DWeights = [[2.9, 0.4, 0.3], [1.8, -2.1, -1.2], [-4.9, -6.8, -7.5]]
        const expectedFilter2Channel2DWeights = [[5.4, 0.4, -2.2], [-5.7, -24.6, -16.2], [-17.4, -29.3, -25]]
        const expectedFilter2Channel3DWeights = [[7.9, 0.4, -4.7], [-13.2, -47.1, -31.2], [-29.9, -51.8, -42.5]]

        const filter1 = layer.filters[0]
        const filter2 = layer.filters[1]

        filter1.errorMap = errorMapFilter1
        filter2.errorMap = errorMapFilter2

        // Avoid regularization affecting the results
        filter1.weights = filter1.weights.map(channel => channel.map(row => row.map(value => 0)))
        filter2.weights = filter2.weights.map(channel => channel.map(row => row.map(value => 0)))

        // Avoid tests failing due to tiny precision differences
        const roundMapValues = map => map.map(row => row.map(value => Math.round(value*10)/10))

        NetUtil.buildConvDWeights(layer)

        it("Sets the filter 1 deltaBias to 4.5", () => {
            expect(filter1.deltaBias).to.equal(4.5)
        })

        it("Sets the filter 2 deltaBias to -0.9", () => {
            expect(filter2.deltaBias).to.equal(-0.9)
        })

        it("Doesn't change the  deltaWeights data structure", () => {
            expect(filter1.deltaWeights).to.have.lengthOf(3)
            expect(filter1.deltaWeights[0]).to.have.lengthOf(3)
            expect(filter1.deltaWeights[1]).to.have.lengthOf(3)
            expect(filter1.deltaWeights[2]).to.have.lengthOf(3)
        })

        it("Sets the filter 1 channel 1 deltaWeights to hand worked out values", () => {
            expect(roundMapValues(filter1.deltaWeights[0])).to.deep.equal(expectedFilter1Channel1DWeights)
        })

        it("Sets the filter 1 channel 2 deltaWeights to hand worked out values", () => {
            expect(roundMapValues(filter1.deltaWeights[1])).to.deep.equal(expectedFilter1Channel2DWeights)
        })

        it("Sets the filter 1 channel 3 deltaWeights to hand worked out values", () => {
            expect(roundMapValues(filter1.deltaWeights[2])).to.deep.equal(expectedFilter1Channel3DWeights)
        })

        it("Sets the filter 2 channel 1 deltaWeights to hand worked out values", () => {
            expect(roundMapValues(filter2.deltaWeights[0])).to.deep.equal(expectedFilter2Channel1DWeights)
        })

        it("Sets the filter 2 channel 2 deltaWeights to hand worked out values", () => {
            expect(roundMapValues(filter2.deltaWeights[1])).to.deep.equal(expectedFilter2Channel2DWeights)
        })

        it("Sets the filter 2 channel 3 deltaWeights to hand worked out values", () => {
            expect(roundMapValues(filter2.deltaWeights[2])).to.deep.equal(expectedFilter2Channel3DWeights)
        })
    })

    describe("buildConvErrorMap", () => {

        // Following values worked out by hand
        const layer = new ConvLayer(1)
        const nextLayerA = new ConvLayer(1, {filterSize: 3, zeroPadding: 1, stride: 2})
        const nextLayerB = new ConvLayer(2, {filterSize: 3, zeroPadding: 1, stride: 2})
        const nextLayerC = new ConvLayer(1, {filterSize: 3, zeroPadding: 1, stride: 1})

        const filter = new Filter()
        const nlFilterA = new Filter()
        const nlFilterB = new Filter()
        const nlFilterC = new Filter()

        layer.filters = [filter]

        nlFilterA.errorMap = [[0.5, -0.2, 0.1], [0, -0.4, -0.1], [0.2, 0.6, 0.3]]
        nlFilterA.weights = [[[-1, 0, -1], [1, 0, 1], [1, -1, 0]]]

        nlFilterB.errorMap = [[0.1, 0.4, 0.2], [-0.1,0.2,-0.3], [0, -0.4, 0.5]]
        nlFilterB.weights = [[[1, 1, 0], [-1, 1, 0], [1, -1, 1]]]

        nlFilterC.errorMap = [[0.1,0.4,-0.2,0.3,0],[0.9,0.2,-0.7,1.1,0.6],[0.4,0,0.3,-0.8,0.1],[0.2,0.3,0.1,-0.1,0.5],[-0.3,0.4,0.5,-0.2,0.3]]
        nlFilterC.weights = [[[1, 1, 0], [-1, 1, 0], [1, -1, 1]]]

        const expectedA = [[0,0.3,0,-0.1,0],[-0.5,0.2,0.2,0.6,-0.1],[0,-0.4,0,-0.5,0],[0,-1.2,0.4,-1,0.1],[0,0.8,0,0.9,0]]
        const expectedB = [[0.1,-0.4,0.4,-0.2,0.2],[-0.2,0.7,-0.2,0.3,-0.5],[-0.1,-0.2,0.2,0.3,-0.3],[0.1,-0.3,-0.6,0.4,0.8],[0,0.4,-0.4,-0.5,0.5]]
        const expectedC = [[0.1,-0.1,0.4,-0.3,0.2],[-0.7,0.9,0,0.9,-0.6],[-0.1,-0.6,0.2,-0.2,-0.3],[0.1,-1.5,-0.2,-0.6,0.9],[0,1.2,-0.4,0.4,0.5]]
        const expectedD = [[0.8,0.1,-0.1,2.0,0.6],[1.4,0.7,-1.4,-0.7,1],[0.2,0.1,3.1,-1.7,1.1],[-0.4,1.8,-0.6,0.7,-0.1],[-0.6,-0.1,0.8,0.2,-0.3]]

        beforeEach(() => {
            nextLayerA.filters = []
            nextLayerB.filters = []
            nextLayerC.filters = []
            filter.errorMap = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
        })

        // Avoid tests failing due to tiny precision differences
        const roundMapValues = map => map.map(row => row.map(value => Math.round(value*10)/10))

        it("Calculates an error map correctly, using just one channel from 1 filter in next layer (Example 1)", () => {
            nextLayerA.filters = [nlFilterA]
            layer.nextLayer = nextLayerA
            NetUtil.buildConvErrorMap(nextLayerA, filter.errorMap, 0)
            expect(roundMapValues(filter.errorMap)).to.deep.equal(expectedA)
        })

        it("Clears the filter errorMap values first (by getting the same result with different initial errorMap values, using Example 1)", () => {
            filter.errorMap = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]
            nextLayerA.filters = [nlFilterA]
            layer.nextLayer = nextLayerA
            NetUtil.buildConvErrorMap(nextLayerA, filter.errorMap, 0)
            expect(roundMapValues(filter.errorMap)).to.deep.equal(expectedA)
        })

        it("Calculates an error map correctly, using just one channel from 1 filter in next layer (Example 2)", () => {
            nextLayerA.filters = [nlFilterB]
            layer.nextLayer = nextLayerA
            NetUtil.buildConvErrorMap(nextLayerA, filter.errorMap, 0)
            expect(roundMapValues(filter.errorMap)).to.deep.equal(expectedB)
        })

        it("Calculates an error map correctly, using two channels, from 2 filters in the next layer", () => {
            nextLayerB.filters = [nlFilterA, nlFilterB]
            layer.nextLayer = nextLayerB
            NetUtil.buildConvErrorMap(nextLayerB, filter.errorMap, 0)
            expect(roundMapValues(filter.errorMap)).to.deep.equal(expectedC)
        })

        it("Calculates an error map correctly, using 1 channel where the stride is 1, not 2", () => {
            nextLayerC.filters = [nlFilterC]
            layer.nextLayer = nextLayerC
            NetUtil.buildConvErrorMap(nextLayerC, filter.errorMap, 0)
            expect(roundMapValues(filter.errorMap)).to.deep.equal(expectedD)
        })
    })

    describe("splitData", () => {

        const testData = [1,2,3,4,5,6,7,8,9,10]

        it("Returns data split into 3 keys: training, validation, and test", () => {
            const result = NetUtil.splitData(testData)
            expect(result).to.have.keys("training", "validation", "test")
        })

        it("Keeps the same total number of items", () => {
            const {training, validation, test} = NetUtil.splitData(testData)
            expect(training.length + validation.length + test.length).to.equal(testData.length)
        })
    })

    describe("normalize", () => {

        it("Example 1 (Handles negative numbers correctly)", () => {
            const data = [1,2,3,-5,0.4,2]
            const {minVal, maxVal} = NetUtil.normalize(data)
            expect(minVal).to.equal(-5)
            expect(maxVal).to.equal(3)
            expect(data).to.deep.equal([0.75, 0.875, 1, 0, 0.675, 0.875])
        })

        it("Example 2 (Handles arrays with equal values correctly)", () => {
            const data = [3, 3, 3, 3]
            const {minVal, maxVal} = NetUtil.normalize(data)
            expect(minVal).to.equal(3)
            expect(maxVal).to.equal(3)
            expect(data).to.deep.equal([0.5, 0.5, 0.5, 0.5])
        })

        it("Example 3", () => {
            const data = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            const {minVal, maxVal} = NetUtil.normalize(data)
            expect(minVal).to.equal(5)
            expect(maxVal).to.equal(15)
            expect(data).to.deep.equal([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        })

        it("Example 4", () => {
            const data = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 14, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1]
            const {minVal, maxVal} = NetUtil.normalize(data)
            expect(minVal).to.equal(-1)
            expect(maxVal).to.equal(15)
            expect(data).to.deep.equal([0.375,0.4375,0.5,0.5625,0.625,0.6875,0.75,0.8125,0.875,0.9375,1,0.9375,0.875,0.875,0.8125,0.75,0.6875,0.625,0.5625,0.5,0.4375,0.375,0.3125,0.25,0.1875,0.125,0.0625,0])
        })

    })
})
