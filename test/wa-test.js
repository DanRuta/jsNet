"use strict"

const chaiAsPromised = require("chai-as-promised")
const chai = require("chai")
const assert = chai.assert
const expect = chai.expect
const sinonChai = require("sinon-chai")
const sinon = require("sinon")
chai.use(sinonChai)
chai.use(chaiAsPromised)

global.Module = require("./emscriptenTests.js")

const {Network, Layer, FCLayer, ConvLayer, PoolLayer, Neuron, NetUtil, NetMath} = require("../dist/jsNetWebAssembly.concat.js")

describe("Loading", () => {
    it("Network is loaded", () => expect(Network).to.not.be.undefined)
    it("Layer is loaded", () => expect(Layer).to.not.be.undefined)
    it("FCLayer is loaded", () => expect(FCLayer).to.not.be.undefined)
    it("ConvLayer is loaded", () => expect(ConvLayer).to.not.be.undefined)
    it("PoolLayer is loaded", () => expect(PoolLayer).to.not.be.undefined)
    it("Neuron is loaded", () => expect(Neuron).to.not.be.undefined)
    it("NetUtil is loaded", () => expect(NetUtil).to.not.be.undefined)

    it("Loads Layer as an alias of FCLayer", () => {

        const layer = new Layer()
        const fclayer = new FCLayer()

        expect(Layer).to.equal(FCLayer)
        expect(layer).to.deep.equal(fclayer)
    })

    it("Statically returns the Network version when accessing via .version", () => {
        expect(Network.version).to.equal("2.1.1")
    })
})

const fakeModule = {
    ccall: fnName => {
        switch (fnName) {
            case "newNetwork":
                return 0
                break
        }
    },
    cwrap: fnName => fakeModule.cwrapReturnFunction,
    _malloc: () => {},
    _free: () => {},
    HEAPF32: {
        set: () => {}
    },
    HEAPF64: {
        set: () => {}
    },
    // This is here to make testing easier. Not in WASM Modules
    cwrapReturnFunction: param => {}
}

describe("Network", () => {

    describe("constructor", () => {

        describe("Defaults", () => {

            let net
            beforeEach(() => net = new Network({Module: fakeModule}))

            it("Defaults the learning rate to 0.2", () => {
                sinon.spy(fakeModule, "cwrap")
                sinon.spy(fakeModule, "cwrapReturnFunction")

                const net = new Network({Module: fakeModule})
                expect(fakeModule.cwrap).to.be.calledWith("setLearningRate")
                expect(fakeModule.cwrapReturnFunction).to.be.calledWith(0, 0.2)

                fakeModule.cwrapReturnFunction.restore()
                fakeModule.cwrap.restore()
            })

            it("Still allows a custom learningRate value", () => {
                sinon.spy(fakeModule, "cwrap")
                sinon.stub(fakeModule, "cwrapReturnFunction").callsFake(() => 123)

                const net = new Network({Module: fakeModule, learningRate: 123})
                expect(fakeModule.cwrap).to.be.calledWith("setLearningRate")
                expect(fakeModule.cwrapReturnFunction).to.be.calledWith(0, 123)

                fakeModule.cwrapReturnFunction.restore()
                fakeModule.cwrap.restore()
            })

            it("Defaults the learning rate to 0.001 if the updateFn is rmsprop", () => {
                sinon.spy(fakeModule, "cwrap")
                sinon.spy(fakeModule, "cwrapReturnFunction")

                const net2 = new Network({Module: fakeModule, updateFn: "rmsprop"})
                expect(fakeModule.cwrap).to.be.calledWith("setLearningRate")
                expect(fakeModule.cwrapReturnFunction).to.be.calledWith(0, 0.001)
                fakeModule.cwrapReturnFunction.restore()
                fakeModule.cwrap.restore()
            })

            it("Still allows a custom learning rate if the updateFn is rmsprop", () => {
                sinon.spy(fakeModule, "cwrap")
                sinon.stub(fakeModule, "cwrapReturnFunction").callsFake(() => 123)

                const net2 = new Network({Module: fakeModule, updateFn: "rmsprop", learningRate: 123})
                expect(fakeModule.cwrap).to.be.calledWith("setLearningRate")
                expect(fakeModule.cwrapReturnFunction).to.be.calledWith(0, 123)
                fakeModule.cwrapReturnFunction.restore()
                fakeModule.cwrap.restore()
            })

            it("Defaults the learning rate to 0.01 if the updateFn is adam", () => {
                sinon.spy(fakeModule, "cwrap")
                sinon.spy(fakeModule, "cwrapReturnFunction")

                const net2 = new Network({Module: fakeModule, updateFn: "adam"})
                expect(fakeModule.cwrap).to.be.calledWith("setLearningRate")
                expect(fakeModule.cwrapReturnFunction).to.be.calledWith(0, 0.01)
                fakeModule.cwrapReturnFunction.restore()
                fakeModule.cwrap.restore()
            })

            it("Still allows a custom learning rate if the updateFn is adam", () => {
                sinon.spy(fakeModule, "cwrap")
                sinon.stub(fakeModule, "cwrapReturnFunction").callsFake(() => 123)

                const net2 = new Network({Module: fakeModule, updateFn: "adam", learningRate: 123})
                expect(fakeModule.cwrap).to.be.calledWith("setLearningRate")
                expect(fakeModule.cwrapReturnFunction).to.be.calledWith(0, 123)
                fakeModule.cwrapReturnFunction.restore()
                fakeModule.cwrap.restore()
            })

            it("Defaults the activation to sigmoid", () => {
                expect(net.activation).to.equal("WASM sigmoid")
            })

            it("Defaults the cost to meansquarederror", () => {
                expect(net.cost).to.equal("WASM meansquarederror")
            })

            it("Defaults the layers to an empty array", () => {
                expect(net.layers).to.deep.equal([])
            })

            it("Sets the initial epochs value to 0", () => {
                expect(net.epochs).to.equal(0)
            })

            it("Sets the initial iterations value to 0", () => {
                expect(net.iterations).to.equal(0)
            })
        })

        describe("defining properties", () => {

            it("Calls the NetUtil.defineProperty function for updateFn", () => {
                sinon.stub(NetUtil, "defineProperty")

                const net = new Network({Module: fakeModule})
                expect(NetUtil.defineProperty).to.be.calledWith(net, "updateFn")
                net.updateFn

                NetUtil.defineProperty.restore()
            })

            it("Calls the NetUtil.defineProperty function for rho, when updateFn is 'adadelta'", () => {
                sinon.stub(NetUtil, "defineProperty")

                const net = new Network({Module: fakeModule, updateFn: "adadelta", rho: 2})
                expect(NetUtil.defineProperty).to.be.calledWith(net, "rho", ["number"], [0])
                expect(net.rho).to.equal(2)

                NetUtil.defineProperty.restore()
            })

            it("Passes the updateFn index from the WASM module through the updateFnIndeces map", () => {
                const net = new Network({Module: fakeModule})
                sinon.stub(fakeModule, "ccall").callsFake(() => 0)
                const res = net.updateFn
                fakeModule.ccall.restore()
                expect(res).to.equal("vanillaupdatefn")
            })

            it("Defaults the rho value to 0.95 when calling the defineProperty function", () => {
                sinon.stub(NetUtil, "defineProperty")
                const net = new Network({Module: fakeModule, updateFn: "adadelta"})
                expect(net.rho).to.equal(0.95)
                NetUtil.defineProperty.restore()
            })

            it("Sets the rmsDecay property when the updateFn is rmsprop", () => {
                sinon.stub(NetUtil, "defineProperty")
                const net = new Network({Module: fakeModule, updateFn: "rmsprop", rmsDecay: 1})
                expect(net.rmsDecay).to.equal(1)
                expect(NetUtil.defineProperty).to.be.calledWith(net, "rmsDecay")
                NetUtil.defineProperty.restore()
            })

            it("Defaults the rmsDecay value to 0.99 when calling the defineProperty function", () => {
                sinon.stub(NetUtil, "defineProperty")
                const net = new Network({Module: fakeModule, updateFn: "rmsprop"})
                expect(net.rmsDecay).to.equal(0.99)
                NetUtil.defineProperty.restore()
            })
        })

        it("Sets the given Module to NetUtil, also", () => {
            try {
                const net = new Network({Module: "some stuff"})
            } catch (e) {}
            expect(NetUtil.Module).to.equal("some stuff")
        })

        it("Throws an error if the Module is not provided", () => {
            const wrapperFn = () => new Network()
            const wrapperFn2 = () => new Network({})
            expect(wrapperFn).to.throw()
            expect(wrapperFn2).to.throw("WASM module not provided")
        })

        it("Throws an error if configured with custom activation functions", () => {
            const wrapperFn = () => new Network({Module: fakeModule, activation: () => 1})
            expect(wrapperFn).to.throw("Custom functions are not (yet) supported with WASM.")
        })

        it("Throws an error if configured with custom cost functions", () => {
            const wrapperFn = () => new Network({Module: fakeModule, cost: () => 1})
            expect(wrapperFn).to.throw("Custom functions are not (yet) supported with WASM.")
        })

        it("Fills the net.layers array with FCLayer the given size values if they are all numbers", () => {
            const net = new Network({Module: fakeModule, layers: [1,2,3]})
            expect(net.layers[0]).instanceof(FCLayer)
            expect(net.layers[0].size).to.equal(1)
            expect(net.layers[1]).instanceof(FCLayer)
            expect(net.layers[1].size).to.equal(2)
            expect(net.layers[2]).instanceof(FCLayer)
            expect(net.layers[2].size).to.equal(3)
        })

        it("Fills the net.layers array with given FCLayer instances", () => {
            const net = new Network({Module: fakeModule, layers: [new FCLayer(3),new FCLayer(2),new FCLayer(1)]})
            expect(net.layers[0]).instanceof(FCLayer)
            expect(net.layers[0].size).to.equal(3)
            expect(net.layers[1]).instanceof(FCLayer)
            expect(net.layers[1].size).to.equal(2)
            expect(net.layers[2]).instanceof(FCLayer)
            expect(net.layers[2].size).to.equal(1)
        })

        it("Sets the net.state to initialised when layers data is given - otherwise sets it to not-defined ", () => {
            const net1 = new Network({Module: fakeModule, layers: [new FCLayer(3),new FCLayer(2),new FCLayer(1)]})
            const net2 = new Network({Module: fakeModule, layers: [1,2,3]})
            const net3 = new Network({Module: fakeModule})
            expect(net1.state).to.equal("initialised")
            expect(net2.state).to.equal("initialised")
            expect(net3.state).to.equal("not-defined")
        })


        it("Throws an error if mixing numbers and classes, or using anything else as layers config", () => {
            const wrapperFn = () => new Network({Module: fakeModule, layers: [1, new FCLayer(2)]})
            expect(wrapperFn).to.throw("There was an error constructing from the layers given.")
        })

        it("Sets the net.Module to the given module", () => {
            const net = new Network({Module: fakeModule})
            expect(net.Module).to.deep.equal(fakeModule)
        })

        it("CCalls the WASM Module's newNetwork function to set a number value to net.netInstance", () => {
            sinon.spy(fakeModule, "ccall")
            const net = new Network({Module: fakeModule})

            expect(fakeModule.ccall).to.be.calledWith("newNetwork")
            expect(net.netInstance).to.be.a.number

            fakeModule.ccall.restore()
        })

        it("Sets the net.learningRate 'get' function to the cwrap-ed WASM getLearningRate function", () => {
            sinon.spy(fakeModule, "cwrap")
            sinon.spy(fakeModule, "cwrapReturnFunction")

            const net = new Network({Module: fakeModule})

            net.learningRate = 1
            expect(fakeModule.cwrap).to.be.calledWith("getLearningRate")
            expect(fakeModule.cwrapReturnFunction).to.be.calledWith(0, 1)

            fakeModule.cwrapReturnFunction.restore()
            fakeModule.cwrap.restore()
        })

        it("Sets the net.learningRate 'set' function to the cwrap-ed WASM setLearningRate function", () => {
            sinon.spy(fakeModule, "cwrap")
            sinon.spy(fakeModule, "cwrapReturnFunction")

            const net = new Network({Module: fakeModule})
            net.learningRate = 1
            expect(fakeModule.cwrap).to.be.calledWith("setLearningRate")
            expect(fakeModule.cwrapReturnFunction).to.be.calledWith(0, 1)
            expect(fakeModule.cwrap).to.be.calledTwice

            fakeModule.cwrapReturnFunction.restore()
            fakeModule.cwrap.restore()
        })

        it("net.activation returns 'WASM x', where x is the given activation function string", () => {
            const net1 = new Network({Module: fakeModule, activation: "sigmoid"})
            const net2 = new Network({Module: fakeModule, activation: "relu"})
            expect(net1.activation).to.equal("WASM sigmoid")
            expect(net2.activation).to.equal("WASM relu")
        })

        it("Setting a net.activation value that does not exist throws an error", () => {
            const net = new Network({Module: fakeModule, activation: "sigmoid"})
            // Needs to be wrapped as the function tested is a setter
            const wrapperFn = () => net.activation = "test"
            expect(wrapperFn).to.throw("The test function does not exist")
        })

        it("Allows snake_case activation function configuration", () => {
            const net = new Network({Module: fakeModule, activation: "si_gmoid"})
            expect(net.activation).to.equal("WASM sigmoid")
        })

        it("Allows white space activation function configuration", () => {
            const net = new Network({Module: fakeModule, activation: "si gmoid"})
            expect(net.activation).to.equal("WASM sigmoid")
        })

        it("Allows case insensitive activation function configuration", () => {
            const net = new Network({Module: fakeModule, activation: "siGmoid"})
            expect(net.activation).to.equal("WASM sigmoid")
        })

        it("Setting a new net.activation value ccalls the WASM setActivation function", () => {
            const net = new Network({Module: fakeModule, activation: "sigmoid"})
            sinon.spy(fakeModule, "ccall")
            net.activation = "relu"
            expect(fakeModule.ccall).to.be.calledWith("setActivation", null, ["number", "number"], [0, 1])
            fakeModule.ccall.restore()
        })

        it("Setting a new net.activation value changes the getter return string to use the new activation name", () => {
            const net = new Network({Module: fakeModule, activation: "sigmoid"})
            net.activation = "relu"
            expect(net.activation).to.equal("WASM relu")
        })

        it("net.cost returns 'WASM x', where x is the given cost function string", () => {
            const net1 = new Network({Module: fakeModule, cost: "meansquarederror"})
            const net2 = new Network({Module: fakeModule, cost: "crossentropy"})
            expect(net1.cost).to.equal("WASM meansquarederror")
            expect(net2.cost).to.equal("WASM crossentropy")
        })

        it("Setting a net.cost value that does not exist throws an error", () => {
            const net = new Network({Module: fakeModule, cost: "meansquarederror"})
            // Needs to be wrapped as the function tested is a setter
            const wrapperFn = () => net.cost = "something"
            const wrapperFn2 = () => new Network({Module: fakeModule, cost: "something"})
            expect(wrapperFn).to.throw("The something function does not exist")
            expect(wrapperFn2).to.throw("The something function does not exist")
        })

        it("Setting a new net.cost value ccalls the WASM setCostFunction function", () => {
            const net = new Network({Module: fakeModule, cost: "meansquarederror"})
            sinon.spy(fakeModule, "ccall")
            net.cost = "crossentropy"
            expect(fakeModule.ccall).to.be.calledWith("setCostFunction", null, ["number", "number"], [0, 1])
            fakeModule.ccall.restore()
        })

        it("Setting a new net.cost value changes the getter return string to use the new cost name", () => {
            const net = new Network({Module: fakeModule, cost: "meansquarederror"})
            net.cost = "crossentropy"
            expect(net.cost).to.equal("WASM crossentropy")
        })

        it("Allows snake_case cost function configuration", () => {
            const net = new Network({Module: fakeModule, cost: "mean_squared_error"})
            expect(net.cost).to.equal("WASM meansquarederror")
        })

        it("Allows white space cost function configuration", () => {
            const net = new Network({Module: fakeModule, cost: "mean squared error"})
            expect(net.cost).to.equal("WASM meansquarederror")
        })

        it("Allows case insensitive cost function configuration", () => {
            const net = new Network({Module: fakeModule, cost: "MeanSquaredError"})
            expect(net.cost).to.equal("WASM meansquarederror")
        })
    })

    describe("initLayers", () => {

        it("Creates three Layers when state is not-defined. First and last layer sizes respective to input/output, middle is in-between", () => {
            const net = new Network({Module: fakeModule, state: "not-defined", layers: []})
            net.initLayers(3,2)
            expect(net.state).to.equal("initialised")
            expect(net.layers.map(layer => layer.size)).to.deep.equal([3, 5, 2])
        })

        it("Creates three Layers when state is not-defined. (the same, but with big net)", () => {
            const net = new Network({Module: fakeModule, state: "not-defined", layers: []})
            net.initLayers(784, 10)
            expect(net.state).to.equal("initialised")
            expect(net.layers.map(layer => layer.size)).to.deep.equal([784, 204, 10])
        })

        it("CCalls the WASM Module's addFCLayer function for every FCLayer configured", () => {
            const spy = sinon.spy(fakeModule, "ccall")
            const net = new Network({Module: fakeModule, layers: [new FCLayer(2), new FCLayer(3), new FCLayer(1)]})
            spy.withArgs("addFCLayer")
            net.initLayers()
            expect(spy.withArgs("addFCLayer").callCount).to.equal(3)
            fakeModule.ccall.restore()
        })

        it("CCalls the WASM Module's addFCLayer function for every layer configured (when configured with digits)", () => {
            const spy = sinon.spy(fakeModule, "ccall")
            const net = new Network({Module: fakeModule, layers: [2, 3, 1]})
            spy.withArgs("addFCLayer")
            net.initLayers()
            expect(spy.withArgs("addFCLayer").callCount).to.equal(3)
            fakeModule.ccall.restore()
        })

        it("CCalls the WASM Module's addFCLayer function for every layer configured (when layers configured implicitly)", () => {
            const spy = sinon.spy(fakeModule, "ccall")
            const net = new Network({Module: fakeModule})
            spy.withArgs("addFCLayer")
            net.initLayers(2, 1)
            expect(spy.withArgs("addFCLayer").callCount).to.equal(3)
            fakeModule.ccall.restore()
        })

        it("CCalls the WASM Module's addFCLayer function only for FCLayers", () => {
            const spy = sinon.spy(fakeModule, "ccall")
            const net = new Network({Module: fakeModule})
            net.layers = [new FCLayer(2), new ConvLayer(1), new PoolLayer(1), new FCLayer(3), new FCLayer(1)]
            net.state = "constructed"
            spy.withArgs("addFCLayer")
            sinon.stub(net, "joinLayer")

            net.initLayers()
            expect(spy.withArgs("addFCLayer").callCount).to.equal(3)
            fakeModule.ccall.restore()
        })

        it("CCalls the WASM Module's initLayers function", () => {
            const net = new Network({Module: fakeModule})
            const spy = sinon.spy(fakeModule, "ccall")
            spy.withArgs("initLayers")
            net.initLayers(2, 3)
            expect(fakeModule.ccall).to.be.calledWith("initLayers")
            fakeModule.ccall.restore()
        })
    })

    describe("joinLayer", () => {

        let net, layer1, layer2

        beforeEach(() => {
            net = new Network({Module: fakeModule})
            layer1 = new FCLayer(10)
            layer2 = new FCLayer(10)
            sinon.stub(layer1, "init")
            sinon.stub(layer2, "init")
        })

        it("Sets the layer.net to a reference to the network", () => {
            net.layers = [layer1]
            net.joinLayer(layer1, 0)
            net.joinLayer(layer2, 1)
            expect(layer1.net).to.equal(net)
            expect(layer2.net).to.equal(net)
        })

        it("Sets the layer.layerIndex to the given layer index value", () => {
            net.layers = [layer1]
            net.joinLayer(layer1, 0)
            net.joinLayer(layer2, 1)
            expect(layer1.layerIndex).to.equal(0)
            expect(layer2.layerIndex).to.equal(1)
        })

        it("Calls layer1's assignNext function with layer 2", () => {
            net.layers = [layer1]
            sinon.stub(layer1, "assignNext")
            net.joinLayer(layer2, 1)
            expect(layer1.assignNext).to.be.calledWith(layer2)
        })

        it("Calls layer2's assignPrev function with layer 1 and the layer index", () => {
            net.layers = [layer1]
            sinon.stub(layer2, "assignPrev")
            net.netInstance = 123
            net.joinLayer(layer2, 1)
            expect(layer2.assignPrev).to.be.calledWith(123, layer1, 1)
        })

        it("Calls the init function on all layers but the first", () => {
            const layer3 = new FCLayer(10)
            net.layers = [layer1]
            sinon.stub(layer3, "init")
            net.netInstance = 123
            net.joinLayer(layer1, 0)
            net.joinLayer(layer2, 1)
            net.joinLayer(layer3, 1)
            expect(layer1.init).to.not.be.called
            expect(layer2.init).to.be.called
            expect(layer3.init).to.be.called
        })
    })

    describe("forward", () => {

        it("Throws an error if called when the network is not in an 'initialised' state", () => {
            const net = new Network({Module: fakeModule})
            expect(net.forward.bind(net)).to.throw("The network layers have not been initialised.")
        })

        it("Throws an error if called with no data", () => {
            const net = new Network({Module: fakeModule, layers: [new Layer(3)]})
            expect(net.forward.bind(net)).to.throw("No data passed to Network.forward()")
        })

        it("Logs a warning if the given input array length does not match input length", () => {
            sinon.stub(NetUtil, "ccallArrays")
            sinon.spy(console, "warn")
            const net = new Network({Module: fakeModule, layers: [new Layer(3)]})
            net.forward([1,2,3,4])
            expect(console.warn).to.have.been.calledWith("Input data length did not match input layer neurons count.")
            console.warn.restore()
            NetUtil.ccallArrays.restore()
        })

        it("Calls the NetUtil.ccallArrays function with the data, otherwise", () => {
            const net = new Network({Module: fakeModule, layers: [new Layer(3)]})
            sinon.stub(NetUtil, "ccallArrays")
            net.netInstance = 123
            net.forward([1,2,3])
            expect(NetUtil.ccallArrays).to.be.calledWith("forward", "array", ["number", "array"], [123, [1,2,3]], {heapOut: "HEAPF64", returnArraySize: 3})
            NetUtil.ccallArrays.restore()
        })
    })

    describe("train", () => {

        let net

        beforeEach(() => {
            net = new Network({Module: fakeModule, layers: [2,3,2]})
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

        it("Calls the initLayers function when the net state is not 'initialised'", () => {
            const network = new Network({Module: fakeModule})
            sinon.spy(network, "initLayers")

            return network.train(testData).then(() => {
                expect(network.initLayers).to.have.been.called
            })
        })

        it("Calls the initLayers function when the net state is not 'initialised' (When data uses 'output' keys)", () => {
            const network = new Network({Module: fakeModule})
            sinon.spy(network, "initLayers")

            return network.train(testDataWithOutput).then(() => {
                expect(network.initLayers).to.have.been.called
            })
        })

    })

    describe("test", () => {

        let net

        beforeEach(() => {
            sinon.stub(NetUtil, "ccallArrays").callsFake(() => 1)
            sinon.stub(fakeModule, "ccall").callsFake(() => 1)
            net = new Network({Module: fakeModule, layers: [2, 4, 3]})
        })

        afterEach(() => {
            NetUtil.ccallArrays.restore()
            fakeModule.ccall.restore()
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

        it("Resolves with a number, indicating error", () => {
            return net.test(testData).then((result) => {
                expect(typeof result).to.equal("number")
            })
        })

        it("Calls the Module.ccall function with the data", () => {
            net.netInstance = 456
            net.test(testData)

            expect(fakeModule.ccall).to.be.calledWith("test", "number", ["number", "number", "number", "number", "number"])
        })

        it("Accepts test data with output key instead of expected", () => {
            return net.test(testDataOutput).then(() => {
                expect(fakeModule.ccall).to.be.called
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
    })

    describe("toJSON", () => {

        const layer1 = new Layer(2)
        const layer2 = new Layer(3)
        const net = new Network({Module: fakeModule, layers: [layer1, layer2], activation: "sigmoid"})

        it("Exports the correct number of layers", () => {
            sinon.stub(layer1, "toJSON")
            sinon.stub(layer2, "toJSON")
            const json = net.toJSON()
            expect(json.layers).to.not.be.undefined
            expect(json.layers).to.have.lengthOf(2)
            layer1.toJSON.restore()
            layer2.toJSON.restore()
        })

        it("Exports weights correctly", () => {
            sinon.stub(layer1, "toJSON")
            sinon.stub(layer2, "toJSON")
            net.toJSON()
            expect(layer1.toJSON).to.be.called
            expect(layer2.toJSON).to.be.called
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
                },
                {
                    weights: [
                        {bias: 4, weights: [1,1]},
                        {bias: 5, weights: [2,2]},
                        {bias: 6, weights: [3,3]}
                    ]
                }
            ]
        }

        beforeEach(() => sinon.stub(fakeModule, "ccall"))

        afterEach(() => fakeModule.ccall.restore())


        it("Throws an error if no data is given", () => {
            const net = new Network({Module: fakeModule})
            expect(net.fromJSON.bind(net)).to.throw("No JSON data given to import.")
            expect(net.fromJSON.bind(net, null)).to.throw("No JSON data given to import.")
        })

        it("Throws an error if the number of layers in the import data does not match the net's", () => {
            const net = new Network({Module: fakeModule, layers: [2,3,4,5]})
            expect(net.fromJSON.bind(net, testData)).to.throw("Mismatched layers (3 layers in import data, but 4 configured)")
        })

        it("CCalls the WASM Module's resetDeltaWeights function", () => {
            const net = new Network({Module: fakeModule, layers: [2,3,1]})
            net.netInstance = 135
            sinon.stub(net.layers[1], "fromJSON")
            sinon.stub(net.layers[2], "fromJSON")
            net.fromJSON(testData)
            expect(fakeModule.ccall).to.be.calledWith("resetDeltaWeights")//, null, ["number"], 135)
        })

        it("CCalls each layer's (except the first) fromJSON function with specific layer data", () => {
            const net = new Network({Module: fakeModule, layers: [2,3,1]})
            sinon.stub(net.layers[1], "fromJSON")
            sinon.stub(net.layers[2], "fromJSON")

            net.netInstance = 135
            net.fromJSON(testData)
            expect(net.layers[1].fromJSON).to.be.calledWith(testData.layers[1])
            expect(net.layers[2].fromJSON).to.be.calledWith(testData.layers[2])
        })
    })
})

describe("FCLayer", () => {

    describe("constructor", () => {

        it("Sets the layer.size to the given size", () => {
            const fcLayer = new FCLayer(2)
            expect(fcLayer.size).to.equal(2)
        })

        it("Sets the layer.layerIndex to 0", () => {
            const fcLayer = new FCLayer(2)
            expect(fcLayer.layerIndex).to.equal(0)
        })

        it("Creates a list of neurons with the size given as parameter", () => {
            const layer = new Layer(10)
            expect(layer.neurons).to.not.be.undefined
            expect(layer.neurons.length).to.equal(10)
        })
    })

    describe("assignNext", () => {

        it("Sets the layer.nextLayer to the given layer", () => {
            const layer = new FCLayer(10)
            const layer2 = new FCLayer(10)
            layer.assignNext(layer2)
            expect(layer.nextLayer).to.equal(layer2)
        })
    })

    describe("assignPrev", () => {

        it("Sets the layer.netInstance to the given value", () => {
            const layer = new FCLayer(10)
            const layer2 = new FCLayer(10)
            layer.assignPrev(123, layer2, 14)
            expect(layer.netInstance).to.equal(123)
        })

        it("Sets the layer.prevLayer to the given layer", () => {
            const layer = new FCLayer(10)
            const layer2 = new FCLayer(10)
            layer.assignPrev(123, layer2, 14)
            expect(layer.prevLayer).to.equal(layer2)
        })

        it("Assigns the layer.layerIndex to the value given", () => {
            const layer = new FCLayer(10)
            const layer2 = new FCLayer(10)
            layer.assignPrev(123, layer2, 14)
            expect(layer.layerIndex).to.equal(14)
        })
    })

    describe("init", () => {

        it("Calls the init function of all of the layer's neurons with the layerIndex and the neuron index", () => {
            const layer = new FCLayer(3)
            const layer2 = new FCLayer(3)
            layer.net = {}
            layer.netInstance = 456
            layer.prevLayer = layer2
            layer.layerIndex = 4
            sinon.stub(layer.neurons[0], "init")
            sinon.stub(layer.neurons[1], "init")
            sinon.stub(layer.neurons[2], "init")
            layer.init()
            expect(layer.neurons[0].init).to.be.calledWith(456, 4, 0)
            expect(layer.neurons[1].init).to.be.calledWith(456, 4, 1)
            expect(layer.neurons[2].init).to.be.calledWith(456, 4, 2)
        })

        it("Sets the neuron.size to the previous layer's size when the previous layer is an FCLayer", () => {
            const layer = new FCLayer(3)
            const layer2 = new FCLayer(123)
            layer.prevLayer = layer2
            layer.net = {}
            layer.init()
            expect(layer.neurons[0].size).to.equal(123)
            expect(layer.neurons[1].size).to.equal(123)
            expect(layer.neurons[2].size).to.equal(123)
        })
    })

    describe("toJSON", () => {

        let layer1, layer2, net

        beforeEach(() => {
            layer1 = new Layer(2)
            layer2 = new Layer(3)
            net = new Network({Module: fakeModule, layers: [layer1, layer2], activation: "sigmoid"})
        })

        it("Exports the correct number of neurons", () => {
            const json1 = layer1.toJSON()
            expect(json1.weights).to.not.be.undefined
            expect(json1.weights).to.have.lengthOf(2)
        })


        it("Exports both a bias and weights", () => {
            const json = layer1.toJSON()
            expect(json.weights[0]).to.have.keys("bias", "weights")
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
                },
                {
                    weights: [
                        {bias: 4, weights: [1,1]},
                        {bias: 5, weights: [2,2]},
                        {bias: 6, weights: [3,3]}
                    ]
                }
            ]
        }

        it("Throws an error if the weights container shape is mismatched", () => {
            const net = new Network({Module: fakeModule, layers: [new FCLayer(3), new FCLayer(4)]})
            expect(net.layers[1].fromJSON.bind(net.layers[1], testData.layers[1], 1)).to.throw("Mismatched weights count. Given: 2 Existing: 3. At layers[1], neurons[0]")
        })

        it("CCalls the WASM module's set_weights and set_bias functions", () => {
            const net = new Network({Module: fakeModule, layers: [new FCLayer(2), new FCLayer(3)]})
            sinon.stub(fakeModule, "ccall")

            net.layers[1].fromJSON(testData.layers[1], 1)

            expect(fakeModule.ccall).to.be.calledWith("set_weights")
            expect(fakeModule.ccall).to.be.calledWith("set_bias")

            fakeModule.ccall.restore()
        })

    })
})

describe("Neuron", () => {

    describe("init", () => {

        let neuron

        beforeEach(() => {
            neuron = new Neuron()
            neuron.size = 5
            sinon.stub(NetUtil, "defineArrayProperty")
            sinon.stub(NetUtil, "defineProperty")
            neuron.init(789, 1, 13, {updateFn: "vanillaupdatefn"})
        })

        afterEach(() => {
            NetUtil.defineArrayProperty.restore()
            NetUtil.defineProperty.restore()
        })

        it("Calls the NetUtil.defineArrayProperty for neuron.weights", () => {
            expect(NetUtil.defineArrayProperty).to.be.calledWith(neuron, "weights")
        })

        it("Calls the NetUtil.defineProperty for neuron.bias", () => {
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "bias")
        })

        it("Calls the NetUtil.defineArrayProperty for neuron.deltaWeights", () => {
            expect(NetUtil.defineArrayProperty).to.be.calledWith(neuron, "deltaWeights")
        })

        it("Calls the NetUtil.defineProperty for neuron.biasGain when the updateFn is gain", () => {
            NetUtil.defineProperty.restore()
            sinon.stub(NetUtil, "defineProperty")
            const neuron = new Neuron()
            neuron.init(789, 1, 13, {updateFn: "gain"})
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "biasGain")
        })

        it("Doesn't call the NetUtil.defineProperty for neuron.biasGain when the updateFn is not gain", () => {
            expect(NetUtil.defineProperty).to.not.be.calledWith(neuron, "biasGain")
        })

        it("Calls the NetUtil.defineArrayProperty for neuron.weightGain when the updateFn is gain", () => {
            NetUtil.defineArrayProperty.restore()
            sinon.stub(NetUtil, "defineArrayProperty")
            const neuron = new Neuron()
            neuron.init(789, 1, 13, {updateFn: "gain"})
            expect(NetUtil.defineArrayProperty).to.be.calledWith(neuron, "weightGain")
        })

        it("Doesn't call the NetUtil.defineArrayProperty for neuron.weightGain when the updateFn is not gain", () => {
            expect(NetUtil.defineArrayProperty).to.not.be.calledWith(neuron, "weightGain")
        })

        it("Calls the NetUtil.defineProperty for neuron.biasCache when the updateFn is adagrad", () => {
            NetUtil.defineProperty.restore()
            sinon.stub(NetUtil, "defineProperty")
            const neuron = new Neuron()
            neuron.init(789, 1, 13, {updateFn: "adagrad"})
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "biasCache")
        })

        it("Doesn't call the NetUtil.defineProperty for neuron.biasCache when the updateFn is not adagrad", () => {
            expect(NetUtil.defineProperty).to.not.be.calledWith(neuron, "biasGain")
        })

        it("Calls the NetUtil.defineArrayProperty for neuron.weightsCache when the updateFn is adagrad", () => {
            NetUtil.defineArrayProperty.restore()
            sinon.stub(NetUtil, "defineArrayProperty")
            const neuron = new Neuron()
            neuron.init(789, 1, 13, {updateFn: "adagrad"})
            expect(NetUtil.defineArrayProperty).to.be.calledWith(neuron, "weightsCache")
        })

        it("Doesn't call the NetUtil.defineArrayProperty for neuron.weightsCache when the updateFn is not adagrad", () => {
            expect(NetUtil.defineArrayProperty).to.not.be.calledWith(neuron, "weightsCache")
        })

        it("Calls the NetUtil.defineProperty for neuron.m and neuron.v if the updateFn is adam", () => {
            NetUtil.defineProperty.restore()
            sinon.stub(NetUtil, "defineProperty")
            const neuron = new Neuron()
            neuron.init(789, 1, 13, {updateFn: "adam"})
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "m")
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "v")
        })

        it("Doesn't call the NetUtil.defineProperty for neuron.m and neuron.v if the updateFn is not adam", () => {
            NetUtil.defineProperty.restore()
            sinon.stub(NetUtil, "defineProperty")
            const neuron = new Neuron()
            neuron.init(789, 1, 13, {updateFn: "something not adam"})
            expect(NetUtil.defineProperty).to.not.be.calledWith(neuron, "m")
            expect(NetUtil.defineProperty).to.not.be.calledWith(neuron, "v")
        })

        it("Calls the NetUtil.defineProperty for neuron.biasCache and neuron.adadeltaBiasCache when the updateFn is adadelta", () => {
            NetUtil.defineProperty.restore()
            sinon.stub(NetUtil, "defineProperty")
            const neuron = new Neuron()
            neuron.init(789, 1, 13, {updateFn: "adadelta"})
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "biasCache")
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "adadeltaBiasCache")
        })

        it("Doesn't call the NetUtil.defineProperty for neuron.biasCache and neuron.adadeltaBiasCache when the updateFn is not adadelta", () => {
            expect(NetUtil.defineProperty).to.not.be.calledWith(neuron, "biasGain")
            expect(NetUtil.defineProperty).to.not.be.calledWith(neuron, "adadeltaBiasCache")
        })

        it("Calls the NetUtil.defineArrayProperty for neuron.weightsCache and neuron.adadeltaCache when the updateFn is adadelta", () => {
            NetUtil.defineArrayProperty.restore()
            sinon.stub(NetUtil, "defineArrayProperty")
            const neuron = new Neuron()
            neuron.init(789, 1, 13, {updateFn: "adadelta"})
            expect(NetUtil.defineArrayProperty).to.be.calledWith(neuron, "weightsCache")
            expect(NetUtil.defineArrayProperty).to.be.calledWith(neuron, "adadeltaCache")
        })

        it("Doesn't call the NetUtil.defineArrayProperty for neuron.weightsCache and neuron.adadeltaCache when the updateFn is not adadelta", () => {
            expect(NetUtil.defineArrayProperty).to.not.be.calledWith(neuron, "weightsCache")
            expect(NetUtil.defineArrayProperty).to.not.be.calledWith(neuron, "adadeltaCache")
        })
    })
})


describe("NetUtil", () => {

    describe("ccallArrays", () => {

        before(() => {
            NetUtil.Module = global.Module
        })

        it("Example 1", () => {
            const res = NetUtil.ccallArrays("getSetWASMArray", "array", ["array", "number", "array"], [[1,2,3,4,5], 12345, [2, 10]], {heapIn: "HEAPF32", returnArraySize: 5})
            expect(res).to.deep.equal([20,40,60,80,100])
        })

        it("Example 1 - no parameter types", () => {
            const res = NetUtil.ccallArrays("getSetWASMArray", "array", null, [[1,2,3,4,5], 12345, [2, 10]], {heapIn: "HEAPF32", returnArraySize: 5})
            expect(res).to.deep.equal([20,40,60,80,100])
        })

        it("Example 2", () => {
            const res = NetUtil.ccallArrays("get10Nums", "array", null, null, {heapOut: "HEAP32", returnArraySize: 10})
            expect(res).to.deep.equal([1,2,3,4,5,6,7,8,9,10])
        })

        it("Example 3", () => {
            const res = NetUtil.ccallArrays("addNums", "number", ["array"], [[1,2,3,4,5,6,7]])
            expect(res).to.equal(28)
        })

        it("HEAP8 in and out using 'HEAP8' config", () => {
            const res = NetUtil.ccallArrays("testHEAP8", "array", ["array"], [[1,2,3,4,5]], {heapIn: "HEAP8", heapOut: "HEAP8", returnArraySize: 5})
            expect(res).to.deep.equal([2,4,6,8,10])
        })

        it("HEAPU8 in and out using 'HEAPU8' config", () => {
            const res = NetUtil.ccallArrays("testHEAPU8", "array", ["array"], [[1,2,3,4,5]], {heapIn: "HEAPU8", heapOut: "HEAPU8", returnArraySize: 5})
            expect(res).to.deep.equal([2,4,6,8,10])
        })

        it("HEAP16 in and out using 'HEAP16' config", () => {
            const res = NetUtil.ccallArrays("testHEAP16", "array", ["array"], [[1,2,3,4,5]], {heapIn: "HEAP16", heapOut: "HEAP16", returnArraySize: 5})
            expect(res).to.deep.equal([2,4,6,8,10])
        })

        it("HEAPU16 in and out using 'HEAPU16' config", () => {
            const res = NetUtil.ccallArrays("testHEAPU16", "array", ["array"], [[1,2,3,4,5]], {heapIn: "HEAPU16", heapOut: "HEAPU16", returnArraySize: 5})
            expect(res).to.deep.equal([2,4,6,8,10])
        })

        it("HEAP32 in and out using 'HEAP32' config", () => {
            const res = NetUtil.ccallArrays("testHEAP32", "array", ["array"], [[1,2,3,4,5]], {heapIn: "HEAP32", heapOut: "HEAP32", returnArraySize: 5})
            expect(res).to.deep.equal([2,4,6,8,10])
        })

        it("HEAPU32 in and out using 'HEAPU32' config", () => {
            const res = NetUtil.ccallArrays("testHEAPU32", "array", ["array"], [[1,2,3,4,5]], {heapIn: "HEAPU32", heapOut: "HEAPU32", returnArraySize: 5})
            expect(res).to.deep.equal([2,4,6,8,10])
        })

        it("HEAPF32 in and out using 'HEAPF32' config", () => {
            const res = NetUtil.ccallArrays("testHEAPF32", "array", ["array"], [[1,2,3,4,5]], {heapIn: "HEAPF32", heapOut: "HEAPF32", returnArraySize: 5})
            expect(res).to.deep.equal([2,4,6,8,10])
        })

        it("Defaults the HEAP values to HEAPF32", () => {
            const res = NetUtil.ccallArrays("testHEAPF32", "array", ["array"], [[1,2,3,4,5]], {returnArraySize: 5})
            expect(res).to.deep.equal([2,4,6,8,10])
        })

        it("HEAPF64 in and out using 'HEAPF64' config", () => {
            const res = NetUtil.ccallArrays("testHEAPF64", "array", ["array"], [[1,2,3,4,5]], {heapIn: "HEAPF64", heapOut: "HEAPF64", returnArraySize: 5})
            expect(res).to.deep.equal([2,4,6,8,10])
        })

        it("Returns original value when return type is not array", () => {
            const res = NetUtil.ccallArrays("addNums", null, ["array"], [[1,2,3]], {heapIn: "HEAP32"})
            expect(Array.isArray(res)).to.be.false
        })

        it("Throws errors, but first it frees the memory using Module._free", () => {
            sinon.stub(NetUtil.Module, "ccall").callsFake(() => {throw new Error("Fake error")})
            expect(NetUtil.ccallArrays.bind(null, "addNums", "array", ["array"], [[1,2,3]], {heapIn: "HEAP32", heapOut: "HEAP3fdgd2"})).to.throw("Fake error")
            NetUtil.Module.ccall.restore()
        })
    })

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

    describe("defineProperty", () => {

        before(() => NetUtil.Module = fakeModule)

        beforeEach(() => sinon.stub(fakeModule, "ccall").callsFake(() => 1))

        afterEach(() => fakeModule.ccall.restore())

        const net = {}

        it("Sets the given property to the given net instance", () => {
            expect(net.test).to.be.undefined
            NetUtil.defineProperty(net, "test", ["number"], [13579])
            NetUtil.defineProperty(net, "stuff")
            net.test = 1
            net.stuff = 2
            expect(net.test).to.not.be.undefined
            expect(net.stuff).to.not.be.undefined
            expect(fakeModule.ccall).to.be.calledWith("set_test", null, ["number", "number"], [13579, 1])
            expect(fakeModule.ccall).to.be.calledWith("set_stuff", null, ["number"], [2])
        })

        it("CCalls the WASM module's function when setting to the value", () => {
            net.test = 5
            expect(fakeModule.ccall).to.be.calledWith("set_test", null, ["number", "number"], [13579, 5])
        })

        it("The getter returns the returned value from the WASM Module", () => {
            expect(net.test).to.equal(1)
        })
    })

    describe("defineArrayProperty", () => {

        beforeEach(() => {
            sinon.stub(fakeModule, "ccall")
            sinon.spy(NetUtil, "ccallArrays")
        })

        afterEach(() => {
            fakeModule.ccall.restore()
            NetUtil.ccallArrays.restore()
        })

        const scope = {}

        it("Sets the given property to the given scope", () => {
            expect(scope.stuff).to.be.undefined
            NetUtil.defineArrayProperty(scope, "stuff", ["number", "number"], [1,2], 5)
            expect(scope.stuff).to.not.be.undefined
        })

        it("CCalls the NetUtil.ccallArrays with 'stuffGet' and given params when accessing the property", () => {
            scope.stuff
            expect(NetUtil.ccallArrays).to.be.calledWith("get_stuff", "array", ["number", "number"], [1,2])
        })

        it("Returns an array with the correct size", () => {
            const result = scope.stuff
            expect(result).to.have.lengthOf(5)
        })

        it("The getter calls the NetUtil.ccallArrays function with the correct parameters", () => {
            scope.stuff = [1,2,3,4,5]
            expect(NetUtil.ccallArrays).to.be.calledWith("set_stuff", null, ["number", "number", "array"], [1,2,[1,2,3,4,5]])
        })
    })
})

describe("NetMath", () => {
    describe("Softmax", () => {

        it("softmax([23, 54, 167, 3]) == [0.0931174089068826, 0.21862348178137653, 0.6761133603238867, 0.012145748987854251]", () => {
            expect(NetMath.softmax([23, 54, 167, 3])).to.deep.equal([0.0931174089068826, 0.21862348178137653, 0.6761133603238867, 0.012145748987854251])
        })

        it("softmax([0]) == [0]", () => {
            expect(NetMath.softmax([0])).to.deep.equal([0])
        })
    })
})