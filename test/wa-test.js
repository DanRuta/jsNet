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

const {Network, Layer, FCLayer, ConvLayer, PoolLayer, Neuron, Filter, NetUtil, NetMath} = require("../dist/jsNetWebAssembly.concat.js")

describe("Loading", () => {
    it("Network is loaded", () => expect(Network).to.not.be.undefined)
    it("Layer is loaded", () => expect(Layer).to.not.be.undefined)
    it("FCLayer is loaded", () => expect(FCLayer).to.not.be.undefined)
    it("ConvLayer is loaded", () => expect(ConvLayer).to.not.be.undefined)
    it("PoolLayer is loaded", () => expect(PoolLayer).to.not.be.undefined)
    it("Neuron is loaded", () => expect(Neuron).to.not.be.undefined)
    it("Filter is loaded", () => expect(Filter).to.not.be.undefined)
    it("NetUtil is loaded", () => expect(NetUtil).to.not.be.undefined)

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

            it("Defaults the learning rate to 0.001 if the activation is lecuntanh", () => {
                sinon.spy(fakeModule, "cwrap")
                sinon.spy(fakeModule, "cwrapReturnFunction")

                const net2 = new Network({Module: fakeModule, activation: "lecuntanh"})
                expect(fakeModule.cwrap).to.be.calledWith("setLearningRate")
                expect(fakeModule.cwrapReturnFunction).to.be.calledWith(0, 0.001)
                fakeModule.cwrapReturnFunction.restore()
                fakeModule.cwrap.restore()
            })

            it("Defaults the learning rate to 0.001 if the activation is tanh", () => {
                sinon.spy(fakeModule, "cwrap")
                sinon.spy(fakeModule, "cwrapReturnFunction")

                const net2 = new Network({Module: fakeModule, activation: "tanh"})
                expect(fakeModule.cwrap).to.be.calledWith("setLearningRate")
                expect(fakeModule.cwrapReturnFunction).to.be.calledWith(0, 0.001)
                fakeModule.cwrapReturnFunction.restore()
                fakeModule.cwrap.restore()
            })

            it("Defaults the learning rate to 0.01 if the activation is relu", () => {
                sinon.spy(fakeModule, "cwrap")
                sinon.spy(fakeModule, "cwrapReturnFunction")

                const net2 = new Network({Module: fakeModule, activation: "relu"})
                expect(fakeModule.cwrap).to.be.calledWith("setLearningRate")
                expect(fakeModule.cwrapReturnFunction).to.be.calledWith(0, 0.01)
                fakeModule.cwrapReturnFunction.restore()
                fakeModule.cwrap.restore()
            })

            it("Defaults the learning rate to 0.01 if the activation is lrelu", () => {
                sinon.spy(fakeModule, "cwrap")
                sinon.spy(fakeModule, "cwrapReturnFunction")

                const net2 = new Network({Module: fakeModule, activation: "lrelu"})
                expect(fakeModule.cwrap).to.be.calledWith("setLearningRate")
                expect(fakeModule.cwrapReturnFunction).to.be.calledWith(0, 0.01)
                fakeModule.cwrapReturnFunction.restore()
                fakeModule.cwrap.restore()
            })

            it("Defaults the learning rate to 0.01 if the activation is rrelu", () => {
                sinon.spy(fakeModule, "cwrap")
                sinon.spy(fakeModule, "cwrapReturnFunction")

                const net2 = new Network({Module: fakeModule, activation: "rrelu"})
                expect(fakeModule.cwrap).to.be.calledWith("setLearningRate")
                expect(fakeModule.cwrapReturnFunction).to.be.calledWith(0, 0.01)
                fakeModule.cwrapReturnFunction.restore()
                fakeModule.cwrap.restore()
            })

            it("Defaults the learning rate to 0.01 if the activation is elu", () => {
                sinon.spy(fakeModule, "cwrap")
                sinon.spy(fakeModule, "cwrapReturnFunction")

                const net2 = new Network({Module: fakeModule, activation: "elu"})
                expect(fakeModule.cwrap).to.be.calledWith("setLearningRate")
                expect(fakeModule.cwrapReturnFunction).to.be.calledWith(0, 0.01)
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

            it("Defaults the weightsConfig distribution to xavieruniform", () => {
                sinon.stub(fakeModule, "ccall")
                const net = new Network({Module: fakeModule, updateFn: "rmsprop"})
                expect(fakeModule.ccall).to.be.calledWith("set_distribution", null, ["number", "number"], [undefined, 2])
                fakeModule.ccall.restore()

                sinon.stub(fakeModule, "ccall").callsFake(() => 2)
                expect(net.weightsConfig.distribution).to.equal("xavieruniform")
                fakeModule.ccall.restore()
            })

            it("Allows setting a different weightsConfig distribution value", () => {
                sinon.stub(fakeModule, "ccall")
                const net = new Network({Module: fakeModule, updateFn: "rmsprop", weightsConfig: {distribution: "gaussian"}})
                expect(fakeModule.ccall).to.be.calledWith("set_distribution", null, ["number", "number"], [undefined, 1])
                fakeModule.ccall.restore()
            })

            it("Defaults the limit to 0.1", () => {
                sinon.stub(fakeModule, "ccall")
                const net = new Network({Module: fakeModule, updateFn: "rmsprop", weightsConfig: {distribution: "uniform"}})
                expect(fakeModule.ccall).to.be.calledWith("set_limit", null, ["number", "number"], [undefined, 0.1])
                fakeModule.ccall.restore()
            })

            it("Allows setting the limit to own value", () => {
                sinon.stub(fakeModule, "ccall")
                const net = new Network({Module: fakeModule, updateFn: "rmsprop", weightsConfig: {distribution: "uniform", limit: 100}})
                expect(fakeModule.ccall).to.be.calledWith("set_limit", null, ["number", "number"], [undefined, 100])
                fakeModule.ccall.restore()
            })

            it("Allows setting the mean to own value", () => {
                sinon.stub(fakeModule, "ccall")
                const net = new Network({Module: fakeModule, updateFn: "rmsprop", weightsConfig: {distribution: "uniform", mean: 100}})
                expect(fakeModule.ccall).to.be.calledWith("set_mean", null, ["number", "number"], [undefined, 100])
                fakeModule.ccall.restore()
            })

            it("Defaults the stdDeviation to 0.05", () => {
                sinon.stub(fakeModule, "ccall")
                const net = new Network({Module: fakeModule, updateFn: "rmsprop", weightsConfig: {distribution: "uniform"}})
                expect(fakeModule.ccall).to.be.calledWith("set_stdDeviation", null, ["number", "number"], [undefined, 0.05])
                fakeModule.ccall.restore()
            })

            it("Allows setting the stdDeviation to own value", () => {
                sinon.stub(fakeModule, "ccall")
                const net = new Network({Module: fakeModule, updateFn: "rmsprop", weightsConfig: {distribution: "uniform", stdDeviation: 100}})
                expect(fakeModule.ccall).to.be.calledWith("set_stdDeviation", null, ["number", "number"], [undefined, 100])
                fakeModule.ccall.restore()
            })
        })

        describe("defining properties", () => {

            beforeEach(() => sinon.stub(NetUtil, "defineProperty"))

            afterEach(() => NetUtil.defineProperty.restore())

            it("Calls the NetUtil.defineProperty function for updateFn", () => {
                const net = new Network({Module: fakeModule})
                expect(NetUtil.defineProperty).to.be.calledWith(net, "updateFn")
                net.updateFn
            })

            it("Calls the NetUtil.defineProperty function for rho, when updateFn is 'adadelta'", () => {
                const net = new Network({Module: fakeModule, updateFn: "adadelta", rho: 2})
                expect(NetUtil.defineProperty).to.be.calledWith(net, "rho", ["number"], [0])
                expect(net.rho).to.equal(2)
            })

            it("Calls the NetUtil.defineProperty function for momentum, when updateFn is 'momentum'", () => {
                const net = new Network({Module: fakeModule, updateFn: "momentum", momentum: 2})
                expect(NetUtil.defineProperty).to.be.calledWith(net, "momentum", ["number"], [0])
                expect(net.momentum).to.equal(2)
            })

            it("Passes the updateFn index from the WASM module through the updateFnIndeces map", () => {
                const net = new Network({Module: fakeModule})
                sinon.stub(fakeModule, "ccall").callsFake(() => 0)
                const res = net.updateFn
                fakeModule.ccall.restore()
                expect(res).to.equal("vanillasgd")
            })

            it("Defaults the rho value to 0.95 when calling the defineProperty function", () => {
                const net = new Network({Module: fakeModule, updateFn: "adadelta"})
                expect(net.rho).to.equal(0.95)
            })

            it("Defaults the momentum value to 0.9 when calling the defineProperty function", () => {
                const net = new Network({Module: fakeModule, updateFn: "momentum"})
                expect(net.momentum).to.equal(0.9)
            })

            it("Sets the rmsDecay property when the updateFn is rmsprop", () => {
                const net = new Network({Module: fakeModule, updateFn: "rmsprop", rmsDecay: 1})
                expect(net.rmsDecay).to.equal(1)
                expect(NetUtil.defineProperty).to.be.calledWith(net, "rmsDecay")
            })

            it("Defaults the rmsDecay value to 0.99 when calling the defineProperty function", () => {
                const net = new Network({Module: fakeModule, updateFn: "rmsprop"})
                expect(net.rmsDecay).to.equal(0.99)
            })

            it("Defaults the lreluSlope value to -0.0005 if the activation is lrelu", () => {
                const net = new Network({Module: fakeModule, activation: "lrelu"})
                expect(net.lreluSlope).to.equal(-0.0005)
            })

            it("Still allows custom lreluSlope values", () => {
                const net = new Network({Module: fakeModule, activation: "lrelu", lreluSlope: 123})
                expect(net.lreluSlope).to.equal(123)
            })

            it("Defaults the eluAlpha value to 1 if the activation is elu", () => {
                const net = new Network({Module: fakeModule, activation: "elu"})
                expect(net.eluAlpha).to.equal(1)
            })

            it("Still allows custom eluAlpha values", () => {
                const net = new Network({Module: fakeModule, activation: "elu", eluAlpha: 123})
                expect(net.eluAlpha).to.equal(123)
            })

            it("Defaults the dropout to 1", () => {
                const net = new Network({Module: fakeModule})
                expect(net.dropout).to.equal(1)
            })

            it("Allows setting the dropout to false, which sets it to 1", () => {
                const net = new Network({Module: fakeModule, dropout: false})
                expect(net.dropout).to.equal(1)
            })

            it("Allows setting the dropout to custom value", () => {
                const net = new Network({Module: fakeModule, dropout: 0.6})
                expect(net.dropout).to.equal(0.6)
            })

            it("Doesn't set the l2 to anything if not defined", () => {
                const net = new Network({Module: fakeModule})
                expect(net.l2).to.be.undefined
            })

            it("Allows setting l2 to true, which sets it to 0.001", () => {
                const net = new Network({Module: fakeModule, l2: true})
                expect(net.l2).to.equal(0.001)
            })

            it("Allows setting l2 to custom value", () => {
                const net = new Network({Module: fakeModule, l2: 0.6})
                expect(net.l2).to.equal(0.6)
            })

            it("Setting l2 to false does not assign any value", () => {
                const net = new Network({Module: fakeModule, l2: false})
                expect(net.l2).to.be.undefined
            })

            it("Defines the net l2Error", () => {
                const net = new Network({Module: fakeModule, l2: 0.001})
                expect(NetUtil.defineProperty).to.be.calledWith(net, "l2Error")
            })

            it("Doesn't set the l1 to anything if not defined", () => {
                const net = new Network({Module: fakeModule})
                expect(net.l1).to.be.undefined
            })

            it("Allows setting l1 to true, which sets it to 0.005", () => {
                const net = new Network({Module: fakeModule, l1: true})
                expect(net.l1).to.equal(0.005)
            })

            it("Allows setting l1 to custom value", () => {
                const net = new Network({Module: fakeModule, l1: 0.6})
                expect(net.l1).to.equal(0.6)
            })

            it("Setting l1 to false does not assign any value", () => {
                const net = new Network({Module: fakeModule, l1: false})
                expect(net.l1).to.be.undefined
            })

            it("Defines the net l1Error", () => {
                const net = new Network({Module: fakeModule, l1: 0.005})
                expect(NetUtil.defineProperty).to.be.calledWith(net, "l1Error")
            })

            it("Defaults maxNorm to 1000", () => {
                const net = new Network({Module: fakeModule, maxNorm: true})
                expect(net.maxNorm).to.equal(1000)
            })

            it("Allows setting maxNorm to custom value", () => {
                const net = new Network({Module: fakeModule, maxNorm: 0.6})
                expect(net.maxNorm).to.equal(0.6)
            })

            it("Setting maxNorm to false does not assign any value", () => {
                const net = new Network({Module: fakeModule, maxNorm: false})
                expect(net.maxNorm).to.be.undefined
            })

            it("Defines the net maxNormTotal when configured", () => {
                const net = new Network({Module: fakeModule, maxNorm: true})
                expect(NetUtil.defineProperty).to.be.calledWith(net, "maxNormTotal")
            })

            it("Defines the net channels when configured", () => {
                const net = new Network({Module: fakeModule, channels: 3})
                expect(NetUtil.defineProperty).to.be.calledWith(net, "channels")
            })

            it("Sets the net channels to the given value", () => {
                const net = new Network({Module: fakeModule, channels: 3})
                expect(net.channels).to.equal(3)
            })

            it("Creates a net.conv {} object if the conv parameter is given", () => {
                const net = new Network({Module: fakeModule, conv: {}})
                expect(net.conv).to.deep.equal({})
            })

            it("Sets the net conv filterSize to the given value", () => {
                const net = new Network({Module: fakeModule, conv: {filterSize: 1357}})
                expect(net.conv.filterSize).to.equal(1357)
            })

            it("Sets the net conv zeroPadding to the given value", () => {
                const net = new Network({Module: fakeModule, conv: {zeroPadding: 1357}})
                expect(net.conv.zeroPadding).to.equal(1357)
            })

            it("Sets the net conv stride to the given value", () => {
                const net = new Network({Module: fakeModule, conv: {stride: 1357}})
                expect(net.conv.stride).to.equal(1357)
            })

            it("Creates a net.pool {} object if the pool parameter is given", () => {
                const net = new Network({Module: fakeModule, pool: {}})
                expect(net.pool).to.deep.equal({})
            })

            it("Sets the net pool size to the given value", () => {
                const net = new Network({Module: fakeModule, pool: {size: 1357}})
                expect(net.pool.size).to.equal(1357)
            })

            it("Sets the net pool stride to the given value", () => {
                const net = new Network({Module: fakeModule, pool: {stride: 1357}})
                expect(net.pool.stride).to.equal(1357)
            })

            it("Sets the net channels to the given value", () => {
                const net = new Network({Module: fakeModule, pool: {size: 1357}})
                expect(net.pool.size).to.equal(1357)
            })

            it("Defines the net weightsConfig distribution", () => {
                const net = new Network({Module: fakeModule})
                expect(NetUtil.defineProperty).to.be.calledWith(net.weightsConfig, "distribution")
            })

            it("Defines the net weightsConfig limit", () => {
                const net = new Network({Module: fakeModule})
                expect(NetUtil.defineProperty).to.be.calledWith(net.weightsConfig, "limit")
            })

            it("Defines the net weightsConfig mean", () => {
                const net = new Network({Module: fakeModule})
                expect(NetUtil.defineProperty).to.be.calledWith(net.weightsConfig, "mean")
            })

            it("Defines the net weightsConfig stdDeviation", () => {
                const net = new Network({Module: fakeModule})
                expect(NetUtil.defineProperty).to.be.calledWith(net.weightsConfig, "stdDeviation")
            })

            it("Throws an error if setting a custom function as weightsConfig distribution", () => {
                const wrapperFn = () => {
                    const net = new Network({Module: fakeModule, weightsConfig: {distribution: () => {}}})
                }
                expect(wrapperFn).to.throw("Custom weights init functions are not (yet) supported with WASM.")
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
            expect(wrapperFn).to.throw("The test activation function does not exist")
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
            net.activation = "tanh"
            expect(fakeModule.ccall).to.be.calledWith("setActivation", null, ["number", "number"], [0, 1])
            fakeModule.ccall.restore()
        })

        it("Setting a new net.activation value changes the getter return string to use the new activation name", () => {
            const net = new Network({Module: fakeModule, activation: "sigmoid"})
            net.activation = "tanh"
            expect(net.activation).to.equal("WASM tanh")
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
            net.initLayers()
            expect(spy.withArgs("addFCLayer").callCount).to.equal(3)
            fakeModule.ccall.restore()
        })

        it("CCalls the WASM Module's addFCLayer function for every layer configured (when configured with digits)", () => {
            const spy = sinon.spy(fakeModule, "ccall")
            const net = new Network({Module: fakeModule, layers: [2, 3, 1]})
            net.initLayers()
            expect(spy.withArgs("addFCLayer").callCount).to.equal(3)
            fakeModule.ccall.restore()
        })

        it("CCalls the WASM Module's addFCLayer function for every layer configured (when layers configured implicitly)", () => {
            const spy = sinon.spy(fakeModule, "ccall")
            const net = new Network({Module: fakeModule})
            net.initLayers(2, 1)
            expect(spy.withArgs("addFCLayer").callCount).to.equal(3)
            fakeModule.ccall.restore()
        })

        it("CCalls the WASM Module's addFCLayer function only for FCLayers", () => {
            const spy = sinon.spy(fakeModule, "ccall")
            const net = new Network({Module: fakeModule})
            net.layers = [new FCLayer(2), new ConvLayer(1), new PoolLayer(1), new FCLayer(3), new FCLayer(1)]
            net.state = "constructed"
            sinon.stub(net, "joinLayer")

            net.initLayers()
            expect(spy.withArgs("addFCLayer").callCount).to.equal(3)
            fakeModule.ccall.restore()
        })

        it("CCalls the WASM Module's initLayers function", () => {
            const net = new Network({Module: fakeModule})
            const spy = sinon.spy(fakeModule, "ccall")
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
            net.joinLayer(layer2, 1)
            expect(layer2.assignPrev).to.be.calledWith(layer1, 1)
        })

        it("Calls the init function on all layers", () => {
            const layer3 = new FCLayer(10)
            net.layers = [layer1]
            sinon.stub(layer3, "init")
            net.netInstance = 123
            net.joinLayer(layer1, 0)
            net.joinLayer(layer2, 1)
            net.joinLayer(layer3, 1)
            expect(layer1.init).to.be.called
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
            net = new Network({Module: fakeModule, layers: [2,3,2], l1: 0.005, l2: 0.001})
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
            return expect(net.train(badTestData)).to.be.rejectedWith("Data set must be a list of objects with keys: 'input' and 'expected'")
        })

        it("Resolves the promise when you give it data", () => {
            return expect(net.train(testData)).to.be.fulfilled
        })

        it("Does not accept 'output' as an alternative name for expected values", () => {
            return expect(net.train(testDataWithOutput)).to.not.be.fulfilled
        })

        it("CCalls the Module's set_miniBatchSize function with the given miniBatchSize value", () => {
            sinon.stub(fakeModule, "ccall")
            net.netInstance = 99
            return net.train(testData, {miniBatchSize: 1234}).then(() => {
                expect(fakeModule.ccall).to.be.calledWith("set_miniBatchSize", null, ["number", "number"], [99, 1234])
                fakeModule.ccall.restore()
            })
        })

        it("Defaults the miniBatchSize to the number of classifications if set as boolean true", () => {
            sinon.stub(fakeModule, "ccall")
            net.netInstance = 99
            return net.train(testData, {miniBatchSize: true}).then(() => {
                expect(fakeModule.ccall).to.be.calledWith("set_miniBatchSize", null, ["number", "number"], [99, 2])
                fakeModule.ccall.restore()
            })
        })

        it("CCalls the WASM Module's shuffleTrainingData function if the shuffle option is set to true", () => {
            sinon.stub(fakeModule, "ccall")
            net.netInstance = 123
            return net.train(testData, {shuffle: true}).then(() => {
                expect(fakeModule.ccall).to.be.calledWith("shuffleTrainingData", null, ["number"], [123])
                fakeModule.ccall.restore()
            })
        })

        it("Calls the initLayers function when the net state is not 'initialised'", () => {
            const network = new Network({Module: fakeModule})
            sinon.spy(network, "initLayers")

            return network.train(testData).then(() => {
                expect(network.initLayers).to.have.been.called
            })
        })

        it("CCalls the WASM Module's loadTrainingData function", () => {
            sinon.stub(fakeModule, "ccall")
            const network = new Network({Module: fakeModule})
            return network.train(testData).then(() => {

                expect(fakeModule.ccall).to.be.calledWith("loadTrainingData")
                fakeModule.ccall.restore()
            })
        })

        it("CCalls the WASM Module's train function for every iteration when a callback is given", () => {
            const network = new Network({Module: fakeModule})
            const stub = sinon.stub(fakeModule, "ccall").callsFake(() => 0)

            const cb = () => {}

            return network.train(testData, {callback: cb}).then(() => {
                expect(stub.withArgs("train").callCount).to.equal(4)
                stub.restore()
            })
        })

        it("Calls the callback with every iteration, in every epoch", () => {
            let counter = 0
            const cb = () => counter++
            const network = new Network({Module: fakeModule})
            const stub = sinon.stub(fakeModule, "ccall").callsFake(() => 0)

            return network.train(testData, {epochs: 2, callback: cb, validation: {data: testData}}).then(() => {
                expect(counter).to.equal(8)
                stub.restore()
            })
        })

        it("Allows setting a custom validation rate", () => {
            const network = new Network({Module: fakeModule})
            const stub = sinon.stub(fakeModule, "ccall").callsFake(() => 0)

            return network.train(testData, {validation: {interval: 2, data: testData}}).then(() => {
                expect(stub.withArgs("set_validationInterval")).to.be.calledWith("set_validationInterval", null, ["number", "number"], [0, 2])
                stub.restore()
            })
        })

        it("Sets the l2Error to 0 with each epoch", () => {
            const network = new Network({Module: fakeModule, l2: 0.01})
            sinon.stub(fakeModule, "ccall").callsFake((_) => {
                return (_=="get_validationCount" || _=="get_stoppedEarly")?0:1
            })

            network.iterations++
            return network.train(testData, {epochs: 5}).then(() => {
                expect(fakeModule.ccall.withArgs("set_l2Error").callCount).to.equal(5)
                fakeModule.ccall.restore()
            })
        })

        it("Does not set the l2Error to 0 if l2 was not configured", () => {
            const network = new Network({Module: fakeModule})
            sinon.stub(fakeModule, "ccall").callsFake((_) => {
                return (_=="get_validationCount" || _=="get_stoppedEarly")?0:1
            }) // simulates this.l2==false

            return network.train(testData, {epochs: 5}).then(() => {
                expect(fakeModule.ccall.withArgs("set_l2Error")).to.not.be.called
                fakeModule.ccall.restore()
            })
        })

        it("Sets the l2Error to 0 with each epoch (with callbacks)", () => {
            const network = new Network({Module: fakeModule, l2: 0.01})
            network.iterations = 2
            sinon.stub(fakeModule, "ccall").callsFake((_) => {
                return (_=="get_validationCount" || _=="get_stoppedEarly")?0:1
            })
            return network.train(testData, {epochs: 5, callback: () => {}}).then(() => {
                expect(fakeModule.ccall.withArgs("set_l2Error").callCount).to.equal(5)
                fakeModule.ccall.restore()
            })
        })

        it("Does not set the l2Error to 0 if l2 was not configured (with callbacks)", () => {
            const network = new Network({Module: fakeModule})
            sinon.stub(fakeModule, "ccall").callsFake(() => 0) // simulates this.l2==false
            return network.train(testData, {epochs: 5, log: false, callback: () => {}}).then(() => {
                expect(fakeModule.ccall.withArgs("set_l2Error")).to.not.be.called
                fakeModule.ccall.restore()
            })
        })


        it("Sets the l1Error to 0 with each epoch", () => {
            const network = new Network({Module: fakeModule, l1: 0.005})
            sinon.stub(fakeModule, "ccall").callsFake((_) => {
                return (_=="get_validationCount" || _=="get_stoppedEarly")?0:1
            })
            return network.train(testData, {epochs: 5}).then(() => {
                expect(fakeModule.ccall.withArgs("set_l1Error").callCount).to.equal(5)
                fakeModule.ccall.restore()
            })
        })

        it("Does not set the l1Error to 0 if l1 was not configured", () => {
            const network = new Network({Module: fakeModule})
            sinon.stub(fakeModule, "ccall").callsFake(() => 0) // simulates this.l1==false
            return network.train(testData, {epochs: 5}).then(() => {
                expect(fakeModule.ccall.withArgs("set_l1Error")).to.not.be.called
                fakeModule.ccall.restore()
            })
        })

        it("Sets the l1Error to 0 with each epoch (with callbacks)", () => {
            const network = new Network({Module: fakeModule, l1: 0.005})
            sinon.stub(fakeModule, "ccall").callsFake((_) => {
                if (_=="get_validationCount") {
                    return -1
                } else if (_=="get_stoppedEarly") {
                    return 0
                }
                return true
            })
            return network.train(testData, {epochs: 5, validation: {}, callback: () => {}}).then(() => {
                expect(fakeModule.ccall.withArgs("set_l1Error").callCount).to.equal(5)
                fakeModule.ccall.restore()
            })
        })

        it("Does not set the l1Error to 0 if l1 was not configured (with callbacks)", () => {
            const network = new Network({Module: fakeModule})
            sinon.stub(fakeModule, "ccall").callsFake(() => false) // simulates this.l1==false
            return network.train(testData, {epochs: 5, callback: () => {}}).then(() => {
                expect(fakeModule.ccall.withArgs("set_l1Error")).to.not.be.called
                fakeModule.ccall.restore()
            })
        })

        it("console.logs once for each epoch, + 2", () => {
            sinon.stub(console, "log")
            const network = new Network({Module: fakeModule})
            return network.train(testData, {epochs: 4, validation: {}}).then(() => {
                expect(console.log.callCount).to.equal(6)
                console.log.restore()
            })
        })

        it("Does not call console.log if log is set to false", () => {
            sinon.stub(console, "log")
            const network = new Network({Module: fakeModule})
            return network.train(testData, {epochs: 4, log: false}).then(() => {
                expect(console.log).to.not.be.called
                console.log.restore()
            })
        })

        describe("Early stopping", () => {

            it("Defaults the threshold to 0.01 when the type is 'threshold'", () => {
                sinon.stub(fakeModule, "ccall")
                return net.train(testData, {validation: {data: testData, earlyStopping: {
                    type: "threshold"
                }}}).then(() => {
                    expect(fakeModule.ccall).to.be.calledWith("set_earlyStoppingType", null, ["number", "number"], [0, 1])
                    expect(fakeModule.ccall).to.be.calledWith("set_earlyStoppingThreshold")
                    expect(net.validation.earlyStopping.threshold).to.equal(0.01)
                    fakeModule.ccall.restore()
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

            it("Allows setting a custom patience value", () => {
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

            it("Sets the bestError to Infinity and patienceCounter to 0 when early stopping is patience", () => {
                for (let l=0; l<net.layers.length; l++) {
                    net.layers[l].restoreValidation = () => {}
                }
                sinon.stub(fakeModule, "ccall")
                return net.train(testData, {validation: {data: testData, earlyStopping: {
                    type: "patience"
                }}}).then(() => {
                    expect(fakeModule.ccall.withArgs("set_earlyStoppingBestError")).to.be.calledWith("set_earlyStoppingBestError", null, ["number", "number"], [0, Infinity])
                    expect(fakeModule.ccall.withArgs("set_earlyStoppingPatienceCounter")).to.be.calledWith("set_earlyStoppingPatienceCounter", null, ["number", "number"], [0, 0])
                    fakeModule.ccall.restore()
                })
            })


            it("Defaults the percent to 30 when the type is 'divergence'", () => {
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

            it("Sets the bestError to Infinity and earlyStoppingPercent to the percent value, when early stopping is divergence", () => {
                for (let l=0; l<net.layers.length; l++) {
                    net.layers[l].restoreValidation = () => {}
                }
                sinon.stub(fakeModule, "ccall")
                return net.train(testData, {validation: {data: testData, earlyStopping: {
                    type: "divergence"
                }}}).then(() => {
                    expect(fakeModule.ccall.withArgs("set_earlyStoppingBestError")).to.be.calledWith("set_earlyStoppingBestError", null, ["number", "number"], [0, Infinity])
                    expect(fakeModule.ccall.withArgs("set_earlyStoppingPercent")).to.be.calledWith("set_earlyStoppingPercent", null, ["number", "number"], [0, 30])
                    fakeModule.ccall.restore()
                })
            })



            it("Stops ccalling the train function when the stoppedEarly value is true", () => {
                const network = new Network({Module: fakeModule})
                let counter = 0
                const stub = sinon.stub(fakeModule, "ccall").callsFake(_ => {
                    if (_=="get_stoppedEarly") {
                        return ++counter>2
                    }
                    return 0
                })

                return network.train(testData, {epochs: 5}).then(() => {
                    expect(stub.withArgs("train").callCount).to.equal(3)
                    stub.restore()
                })
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
            expect(fakeModule.ccall).to.be.calledWith("loadTestingData", "number", ["number", "number", "number", "number", "number"])
        })

        it("CCalls the WASM Module's test function once", () => {
            net.netInstance = 456
            net.test(testData)
            expect(fakeModule.ccall.withArgs("test").callCount).to.equal(1)
        })

        it("CCalls the WASM Module's test for every test item when a callback is given", () => {
            net.netInstance = 456
            return net.test(testData, {callback: () => {}}).then(() => {
                expect(fakeModule.ccall.withArgs("test").callCount).to.equal(4)
            })
        })

        it("Does not accept test data with output key instead of expected", () => {
            return expect(net.test(testDataOutput)).to.not.be.fulfilled
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
        const convNet = new Network({Module: fakeModule, layers: [new FCLayer(1024), new ConvLayer(2, {filterSize: 3})]})

        it("Exports the correct number of layers", () => {
            sinon.stub(layer1, "toJSON")
            sinon.stub(layer2, "toJSON")
            const json = net.toJSON()
            expect(json.layers).to.not.be.undefined
            expect(json.layers).to.have.lengthOf(2)
            layer1.toJSON.restore()
            layer2.toJSON.restore()
        })

        it("Calls the layers' toJSON functions", () => {
            sinon.stub(layer1, "toJSON")
            sinon.stub(layer2, "toJSON")
            sinon.stub(convNet.layers[1], "toJSON")
            net.toJSON()
            expect(layer1.toJSON).to.be.called
            expect(layer2.toJSON).to.be.called
            convNet.toJSON()
            expect(convNet.layers[1].toJSON).to.be.called
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

    describe("toIMG", () => {
        it("Throws an error if IMGArrays is not provided", () => {
            const net = new Network({Module: fakeModule})
            expect(net.toIMG).to.throw("The IMGArrays library must be provided. See the documentation for instructions.")
        })

        it("Calls every layer except the first's toIMG function", () => {
            const net = new Network({Module: fakeModule})
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
            const net = new Network({Module: fakeModule})
            expect(net.fromIMG).to.throw("The IMGArrays library must be provided. See the documentation for instructions.")
        })

        it("Calls every layer except the first's fromIMG function with the data segment matching their size", () => {
            const net = new Network({Module: fakeModule})
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

        it("Throws an error if setting activation to something other than a string", () => {
            const wrapperFn = () => new FCLayer(1, {activation: x => x})
            const wrapperFn2 = () => new FCLayer(1, {activation: 2})
            expect(wrapperFn).to.throw("Custom activation functions are not available in the WebAssembly version")
            expect(wrapperFn2).to.throw("Custom activation functions are not available in the WebAssembly version")
        })

        it("Otherwise sets the given activation string to the layer", () => {
            const layer = new FCLayer(1, {activation: "test"})
            expect(layer.activationName).to.equal("test")
        })

        it("Sets the activation function to noActivation when configuring it with false", () => {
            const layer = new FCLayer(1, {activation: false})
            expect(layer.activationName).to.equal("noactivation")
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

        let layer1
        let layer2

        beforeEach(() => {
            layer1 = new FCLayer(10)
            layer2 = new FCLayer(10)
            layer1.net = {}
            layer2.net = {netInstance: 132}
            sinon.spy(NetUtil, "defineProperty")
            sinon.stub(NetUtil.Module, "ccall")
        })

        afterEach(() => {
            NetUtil.defineProperty.restore()
            NetUtil.Module.ccall.restore()
        })

        it("Sets the layer.netInstance to the network.netInstance", () => {

            layer1.net = {netInstance: 123}
            layer1.assignPrev(layer2, 14)
            expect(layer1.netInstance).to.equal(123)
        })

        it("Sets the layer.prevLayer to the given layer", () => {
            layer1.net = {}
            layer1.assignPrev(layer2, 14)
            expect(layer1.prevLayer).to.equal(layer2)
        })

        it("Assigns the layer.layerIndex to the value given", () => {
            layer1.net = {}
            layer1.assignPrev(layer2, 14)
            expect(layer1.layerIndex).to.equal(14)
        })

        it("Calls the net Module's ccall function with set_fc_activation with the layer.activationName, if given", () => {
            layer2.activationName = "tanh"
            layer2.assignPrev(layer1, 14)
            expect(layer2.activation).to.equal("WASM tanh")
            expect(NetUtil.defineProperty).to.be.calledWith(layer2, "activation", ["number", "number"], [layer2.netInstance, layer2.layerIndex])
            expect(NetUtil.Module.ccall).to.be.calledWith("set_fc_activation", null, ["number", "number", "number"], [132, 14, 1])
        })

        it("Calls the net Module's ccall function with set_fc_activation with the net.activationName if given", () => {
            layer2.activationName = undefined
            layer2.net.activationName = "relu"
            layer2.assignPrev(layer1, 14)
            expect(layer2.activation).to.equal("WASM relu")
            expect(NetUtil.defineProperty).to.be.calledWith(layer2, "activation", ["number", "number"], [layer2.netInstance, layer2.layerIndex])
            expect(NetUtil.Module.ccall).to.be.calledWith("set_fc_activation", null, ["number", "number", "number"], [132, 14, 3])
        })

        it("Does not call the WASM setConvActivation function if the activation is set to false", () => {
            layer2.activationName = false
            layer2.net.activationName = undefined
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.defineProperty).to.not.be.calledWith(layer2, "activation", ["number", "number"], [layer2.netInstance, layer1.layerIndex])
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

        it("Sets the neuron.size to the number of filters in the last layer, if it is a ConvLayer", () => {
            const convLayer = new ConvLayer(3)
            const layer2 = new FCLayer(2)
            convLayer.filters = [new Filter(),new Filter(),new Filter()]
            convLayer.outMapSize = 7
            layer2.net = {netInstance: 123}
            layer2.assignPrev(convLayer, 14)
            layer2.init()
            expect(layer2.neurons[0].weights.length).to.equal(147) // 3 * 7 * 7
        })

        it("Sets the neuron.size to the number of outgoing activations from the last layer, if it is a ConvLayer", () => {
            const poolLayer = new PoolLayer(2)
            const layer2 = new FCLayer(2)
            poolLayer.channels = 3
            poolLayer.outMapSize = 3
            layer2.net = {netInstance: 123}
            layer2.assignPrev(poolLayer, 14)
            layer2.init()
            expect(layer2.neurons[0].size).to.equal(27) // 3 * 3**2
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

        it("CCalls the WASM module's set_neuron_weights and set_neuron_bias functions", () => {
            const net = new Network({Module: fakeModule, layers: [new FCLayer(2), new FCLayer(3)]})
            sinon.stub(fakeModule, "ccall")

            net.layers[1].fromJSON(testData.layers[1], 1)

            expect(fakeModule.ccall).to.be.calledWith("set_neuron_weights")
            expect(fakeModule.ccall).to.be.calledWith("set_neuron_bias")

            fakeModule.ccall.restore()
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

    describe("init", () => {

        let neuron
        const paramTypes = ["number", "number", "number"]
        const params = [789, 1, 13]

        beforeEach(() => {
            neuron = new Neuron()
            neuron.size = 5
            sinon.stub(NetUtil, "defineArrayProperty")
            sinon.stub(NetUtil, "defineProperty")

        })

        afterEach(() => {
            NetUtil.defineArrayProperty.restore()
            NetUtil.defineProperty.restore()
        })


        it("Calls the NetUtil.defineProperty for neuron.sum", () => {
            neuron.init(789, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "sum", paramTypes, params, {pre: "neuron_"})
        })

        it("Calls the NetUtil.defineProperty for neuron.dropped", () => {
            neuron.init(789, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "dropped", paramTypes, params)
        })

        it("Getting a neuron.dropped value returns a boolean", () => {
            NetUtil.defineProperty.restore()
            neuron.init(789, 1, 13, {updateFn: "vanillasgd"})
            expect(neuron.dropped).to.be.boolean
            sinon.stub(NetUtil, "defineProperty")
        })

        it("Sets a neuron.dropped value as a 1 or 0", () => {
            NetUtil.defineProperty.restore()
            sinon.stub(NetUtil.Module, "ccall")
            neuron.init(789, 1, 13, {updateFn: "vanillasgd"})
            neuron.dropped = true
            expect(NetUtil.Module.ccall).to.be.calledWith("set_neuron_dropped", null, ["number", "number", "number", "number"], [789, 1, 13, 1])
            neuron.dropped = false
            expect(NetUtil.Module.ccall).to.be.calledWith("set_neuron_dropped", null, ["number", "number", "number", "number"], [789, 1, 13, 0])

            NetUtil.Module.ccall.restore()
            sinon.stub(NetUtil, "defineProperty")
        })

        it("Calls the NetUtil.defineProperty for neuron.activation", () => {
            neuron.init(789, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "activation", paramTypes, params, {pre: "neuron_"})
        })

        it("Calls the NetUtil.defineProperty for neuron.error", () => {
            neuron.init(789, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "error", paramTypes, params, {pre: "neuron_"})
        })

        it("Calls the NetUtil.defineProperty for neuron.derivative", () => {
            neuron.init(789, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "derivative", paramTypes, params, {pre: "neuron_"})
        })

        it("Calls the NetUtil.defineArrayProperty for neuron.weights", () => {
            neuron.init(789, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineArrayProperty).to.be.calledWith(neuron, "weights", paramTypes, params, neuron.size, {pre: "neuron_"})
        })

        it("Calls the NetUtil.defineProperty for neuron.bias", () => {
            neuron.init(789, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "bias", paramTypes, params, {pre: "neuron_"})
        })

        it("Calls the NetUtil.defineArrayProperty for neuron.deltaWeights", () => {
            neuron.init(789, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineArrayProperty).to.be.calledWith(neuron, "deltaWeights", paramTypes, params, neuron.size, {pre: "neuron_"})
        })

        it("Calls the NetUtil.defineProperty for neuron.biasGain when the updateFn is gain", () => {
            const neuron = new Neuron()
            neuron.init(789, 1, 13, {updateFn: "gain"})
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "biasGain", paramTypes, params, {pre: "neuron_"})
        })

        it("Doesn't call the NetUtil.defineProperty for neuron.biasGain when the updateFn is not gain", () => {
            neuron.init(789, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineProperty).to.not.be.calledWith(neuron, "biasGain", paramTypes, params, {pre: "neuron_"})
        })

        it("Calls the NetUtil.defineArrayProperty for neuron.weightGain when the updateFn is gain", () => {
            const neuron = new Neuron()
            neuron.init(789, 1, 13, {updateFn: "gain"})
            expect(NetUtil.defineArrayProperty).to.be.calledWith(neuron, "weightGain", paramTypes, params, neuron.size, {pre: "neuron_"})
        })

        it("Doesn't call the NetUtil.defineArrayProperty for neuron.weightGain when the updateFn is not gain", () => {
            neuron.init(789, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineArrayProperty).to.not.be.calledWith(neuron, "weightGain", paramTypes, params, {pre: "neuron_"})
        })

        it("Calls the NetUtil.defineProperty for neuron.biasCache when the updateFn is adagrad", () => {
            const neuron = new Neuron()
            neuron.init(789, 1, 13, {updateFn: "adagrad"})
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "biasCache", paramTypes, params, {pre: "neuron_"})
        })

        it("Doesn't call the NetUtil.defineProperty for neuron.biasCache when the updateFn is not adagrad", () => {
            neuron.init(789, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineProperty).to.not.be.calledWith(neuron, "biasGain", paramTypes, params, {pre: "neuron_"})
        })

        it("Calls the NetUtil.defineArrayProperty for neuron.weightsCache when the updateFn is adagrad", () => {
            const neuron = new Neuron()
            neuron.init(789, 1, 13, {updateFn: "adagrad"})
            expect(NetUtil.defineArrayProperty).to.be.calledWith(neuron, "weightsCache", paramTypes, params, neuron.size, {pre: "neuron_"})
        })

        it("Doesn't call the NetUtil.defineArrayProperty for neuron.weightsCache when the updateFn is not adagrad", () => {
            neuron.init(789, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineArrayProperty).to.not.be.calledWith(neuron, "weightsCache", paramTypes, params, {pre: "neuron_"})
        })

        it("Calls the NetUtil.defineProperty for neuron.m and neuron.v if the updateFn is adam", () => {
            const neuron = new Neuron()
            neuron.init(789, 1, 13, {updateFn: "adam"})
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "m", paramTypes, params, {pre: "neuron_"})
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "v", paramTypes, params, {pre: "neuron_"})
        })

        it("Doesn't call the NetUtil.defineProperty for neuron.m and neuron.v if the updateFn is not adam", () => {
            const neuron = new Neuron()
            neuron.init(789, 1, 13, {updateFn: "something not adam"})
            expect(NetUtil.defineProperty).to.not.be.calledWith(neuron, "m", paramTypes, params, {pre: "neuron_"})
            expect(NetUtil.defineProperty).to.not.be.calledWith(neuron, "v", paramTypes, params, {pre: "neuron_"})
        })

        it("Calls the NetUtil.defineProperty for neuron.biasCache and neuron.adadeltaBiasCache when the updateFn is adadelta", () => {
            const neuron = new Neuron()
            neuron.init(789, 1, 13, {updateFn: "adadelta"})
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "biasCache", paramTypes, params, {pre: "neuron_"})
            expect(NetUtil.defineProperty).to.be.calledWith(neuron, "adadeltaBiasCache", paramTypes, params, {pre: "neuron_"})
        })

        it("Doesn't call the NetUtil.defineProperty for neuron.biasCache and neuron.adadeltaBiasCache when the updateFn is not adadelta", () => {
            neuron.init(789, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineProperty).to.not.be.calledWith(neuron, "biasGain", paramTypes, params, {pre: "neuron_"})
            expect(NetUtil.defineProperty).to.not.be.calledWith(neuron, "adadeltaBiasCache", paramTypes, params, {pre: "neuron_"})
        })

        it("Calls the NetUtil.defineArrayProperty for neuron.weightsCache and neuron.adadeltaCache when the updateFn is adadelta", () => {
            const neuron = new Neuron()
            neuron.init(789, 1, 13, {updateFn: "adadelta"})
            expect(NetUtil.defineArrayProperty).to.be.calledWith(neuron, "weightsCache", paramTypes, params, neuron.size, {pre: "neuron_"})
            expect(NetUtil.defineArrayProperty).to.be.calledWith(neuron, "adadeltaCache", paramTypes, params, neuron.size, {pre: "neuron_"})
        })

        it("Doesn't call the NetUtil.defineArrayProperty for neuron.weightsCache and neuron.adadeltaCache when the updateFn is not adadelta", () => {
            neuron.init(789, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineArrayProperty).to.not.be.calledWith(neuron, "weightsCache", paramTypes, params, neuron.size, {pre: "neuron_"})
            expect(NetUtil.defineArrayProperty).to.not.be.calledWith(neuron, "adadeltaCache", paramTypes, params, neuron.size, {pre: "neuron_"})
        })
    })
})

describe("Filter", () => {

    describe("init", () => {

        let filter

        beforeEach(() => {
            filter = new Filter()
            sinon.stub(NetUtil, "defineVolumeProperty")
            sinon.stub(NetUtil, "defineArrayProperty")
            sinon.stub(NetUtil, "defineProperty")

        })

        afterEach(() => {
            NetUtil.defineVolumeProperty.restore()
            NetUtil.defineArrayProperty.restore()
            NetUtil.defineProperty.restore()
        })

        it("Calls the NetUtil.defineProperty for filter.bias", () => {
            filter.init(0, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineProperty).to.be.calledWith(filter, "bias")
        })

        it("Calls the NetUtil.defineVolumeProperty for filter.weights", () => {
            filter.init(0, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineVolumeProperty).to.be.calledWith(filter, "weights")
        })

        it("Calls the NetUtil.defineVolumeProperty for filter.deltaWeights", () => {
            filter.init(0, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineVolumeProperty).to.be.calledWith(filter, "deltaWeights")
        })

        it("Calls the NetUtil.defineProperty for filter.deltaBias", () => {
            filter.init(0, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineProperty).to.be.calledWith(filter, "deltaBias")
        })

        it("Calls the NetUtil.defineProperty for filter.biasGain when the updateFn is gain", () => {
            filter.init(0, 1, 13, {updateFn: "gain"})
            expect(NetUtil.defineProperty).to.be.calledWith(filter, "biasGain")
        })

        it("Calls the NetUtil.defineVolumeProperty for filter.weightGain when the updateFn is gain", () => {
            filter.init(0, 1, 13, {updateFn: "gain"})
            expect(NetUtil.defineVolumeProperty).to.be.calledWith(filter, "weightGain")
        })

        it("Doesn't call the define property functions for biasGain or weightGain when updateFn is not gain", () => {
            filter.init(0, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineProperty).to.not.be.calledWith(filter, "biasGain")
            expect(NetUtil.defineVolumeProperty).to.not.be.calledWith(filter, "weightGain")
        })

        it("Calls the NetUtil.defineProperty for filter.biasCache when the updateFn is adagrad", () => {
            filter.init(0, 1, 13, {updateFn: "adagrad"})
            expect(NetUtil.defineProperty).to.be.calledWith(filter, "biasCache")
        })

        it("Calls the NetUtil.defineVolumeProperty for filter.weightsCache when the updateFn is adagrad", () => {
            filter.init(0, 1, 13, {updateFn: "adagrad"})
            expect(NetUtil.defineVolumeProperty).to.be.calledWith(filter, "weightsCache")
        })

        it("Doesn't call the define property functions for biasCache or weightsCache when updateFn is not adagrad", () => {
            filter.init(0, 1, 13, {updateFn: "vanillasgd"})
            expect(NetUtil.defineProperty).to.not.be.calledWith(filter, "biasCache")
            expect(NetUtil.defineVolumeProperty).to.not.be.calledWith(filter, "weightsCache")
        })

        it("Calls the NetUtil.defineProperty for filter.biasCache when the updateFn is rmsprop", () => {
            filter.init(0, 1, 13, {updateFn: "rmsprop"})
            expect(NetUtil.defineProperty).to.be.calledWith(filter, "biasCache")
        })

        it("Calls the NetUtil.defineVolumeProperty for filter.weightsCache when the updateFn is rmsprop", () => {
            filter.init(0, 1, 13, {updateFn: "rmsprop"})
            expect(NetUtil.defineVolumeProperty).to.be.calledWith(filter, "weightsCache")
        })

        it("Calls the NetUtil.defineProperty for filter.biasCache when the updateFn is adadelta", () => {
            filter.init(0, 1, 13, {updateFn: "adadelta"})
            expect(NetUtil.defineProperty).to.be.calledWith(filter, "biasCache")
        })

        it("Calls the NetUtil.defineVolumeProperty for filter.weightsCache when the updateFn is adadelta", () => {
            filter.init(0, 1, 13, {updateFn: "adadelta"})
            expect(NetUtil.defineVolumeProperty).to.be.calledWith(filter, "weightsCache")
        })

        it("Calls the NetUtil.defineProperty for filter.adadeltaBiasCache when the updateFn is adadelta", () => {
            filter.init(0, 1, 13, {updateFn: "adadelta"})
            expect(NetUtil.defineProperty).to.be.calledWith(filter, "adadeltaBiasCache")
        })

        it("Calls the NetUtil.defineVolumeProperty for filter.adadeltaWeightsCache when the updateFn is adadelta", () => {
            filter.init(0, 1, 13, {updateFn: "adadelta"})
            expect(NetUtil.defineVolumeProperty).to.be.calledWith(filter, "adadeltaWeightsCache")
        })

        it("Doesn't call the define property functions for adadeltaBiasCache or adadeltaWeightsCache when updateFn is not adadelta", () => {
            filter.init(0, 1, 13, {updateFn: "adagrad"})
            expect(NetUtil.defineProperty).to.not.be.calledWith(filter, "adadeltaBiasCache")
            expect(NetUtil.defineVolumeProperty).to.not.be.calledWith(filter, "adadeltaWeightsCache")
        })

        it("Calls the NetUtil.defineProperty for filter.m when the updateFn is adam", () => {
            filter.init(0, 1, 13, {updateFn: "adam"})
            expect(NetUtil.defineProperty).to.be.calledWith(filter, "m")
        })

        it("Calls the NetUtil.defineProperty for filter.v when the updateFn is adam", () => {
            filter.init(0, 1, 13, {updateFn: "adam"})
            expect(NetUtil.defineProperty).to.be.calledWith(filter, "v")
        })

        it("Doesn't call the NetUtil.defineProperty for m or v when the updateFn is not adam", () => {
            filter.init(0, 1, 13, {updateFn: "gain"})
            expect(NetUtil.defineProperty).to.not.be.calledWith(filter, "m")
            expect(NetUtil.defineProperty).to.not.be.calledWith(filter, "v")

        })

    })
})

describe("ConvLayer", () => {

    describe("constructor", () => {

        beforeEach(() => {
            sinon.spy(NetUtil, "defineProperty")
            sinon.stub(NetUtil.Module, "ccall")
        })
        afterEach(() => {
            NetUtil.defineProperty.restore()
            NetUtil.Module.ccall.restore()
        })

        it("Throws an error if setting activation to something other than a string", () => {
            const wrapperFn = () => new ConvLayer(1, {activation: x => x})
            const wrapperFn2 = () => new ConvLayer(1, {activation: 2})
            expect(wrapperFn).to.throw("Custom activation functions are not available in the WebAssembly version")
            expect(wrapperFn2).to.throw("Custom activation functions are not available in the WebAssembly version")
        })

        it("Otherwise sets the given activation string to the layer", () => {
            const layer = new ConvLayer(1, {activation: "test"})
            expect(layer.activationName).to.equal("test")
        })

        it("Sets the activation to false if nothing is provided", () => {
            const layer = new ConvLayer()
            expect(layer.activation).to.be.false
        })

        it("Sets the layer.layerIndex to 0", () => {
            const layer = new ConvLayer()
            expect(layer.layerIndex).to.equal(0)
        })

        it("Sets the filterSize, stride, size and zeroPadding values to given values", () => {
            const layer = new ConvLayer(1, {filterSize: 3, stride: 2, zeroPadding: 5})
            expect(layer.size).to.equal(1)
            expect(layer.filterSize).to.equal(3)
            expect(layer.stride).to.equal(2)
            expect(layer.zeroPadding).to.equal(5)
        })

        it("Sets the activation function to noActivation when configuring it with false", () => {
            const layer = new ConvLayer(1, {activation: false, filterSize: 3, stride: 2, zeroPadding: 5})
            expect(layer.activationName).to.equal("noactivation")
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

        let layer1, layer2, defaultValues

        beforeEach(() => {
            layer1 = new ConvLayer()
            layer1.outMapSize = 16
            layer1.size = 10
            layer1.neurons = {length: 10}
            layer2 = new ConvLayer(3, {filterSize: 3, stride: 1})
                layer2.size = 3
                layer2.filterSize = 3
                layer2.stride = 1
                layer2.zeroPadding = 0
            layer1.net = {conv: {}, netInstance: 132, Module: {ccall: () => {}}}
            layer2.net = {conv: {}, netInstance: 132, Module: {ccall: () => {}}}
            sinon.spy(NetUtil, "defineProperty")
            defaultValues = {
                get_conv_filterSize: 3,
                get_conv_size: 3,
                get_conv_stride: 1,
                get_conv_zeroPadding: 0
            }
            // sinon.stub(NetUtil.Module, "ccall").callsFake(fn => defaultValues[fn]==undefined?3:defaultValues[fn])
            sinon.stub(NetUtil.Module, "ccall").callsFake(fn => defaultValues[fn])
        })

        afterEach(() => {
            NetUtil.defineProperty.restore()
            NetUtil.Module.ccall.restore()
        })

        it("Sets the layer.netInstance to the net.netInstance", () => {
            layer2.assignPrev(layer1, 14)
            expect(layer2.netInstance).to.equal(132)
        })

        it("Sets the layer2.prevLayer to the given layer", () => {
            layer2.assignPrev(layer1, 14)
            expect(layer2.prevLayer).to.equal(layer1)
        })

        it("Assigns the layer2.layerIndex to the value given", () => {
            layer2.assignPrev(layer1, 14)
            expect(layer2.layerIndex).to.equal(14)
        })

        it("Calls the NetUtil.defineProperty for channels", () => {
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.defineProperty).to.be.calledWith(layer2, "channels", ["number", "number"], [132, 14], {pre: "conv_"})
        })

        it("Calls the NetUtil.defineProperty for filterSize", () => {
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.defineProperty).to.be.calledWith(layer2, "filterSize", ["number", "number"], [132, 14], {pre: "conv_"})
        })

        it("Calls the NetUtil.defineProperty for stride", () => {
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.defineProperty).to.be.calledWith(layer2, "stride", ["number", "number"], [132, 14], {pre: "conv_"})
        })

        it("Calls the NetUtil.defineProperty for zeroPadding", () => {
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.defineProperty).to.be.calledWith(layer2, "zeroPadding", ["number", "number"], [132, 14], {pre: "conv_"})
        })

        it("Defaults the layer.size to 4 if there is no size value for layer, and sets the cpp value", () => {
            layer2.size = undefined
            layer2.assignPrev(layer1, 14)
            expect(layer2.size).to.equal(4)
        })

        it("Keeps the same size value if it has already been assigned to layer", () => {
            layer2.size = 10
            layer2.assignPrev(layer1, 14)
            expect(layer2.size).to.equal(10)
        })

        it("Defaults the layer.filterSize to the net.conv.filterSize value, if there's no layer.filterSize, but there is one for net, and sets the cpp value", () => {
            NetUtil.Module.ccall.restore()
            sinon.stub(NetUtil.Module, "ccall")
            defaultValues["get_conv_filterSize"] = 0
            layer2.filterSize = undefined
            layer2.net.conv.filterSize = 5
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_filterSize", null, ["number", "number", "number"], [132, 14, 5])
        })

        it("Defaults the layer.filterSize to 3 if there is no filterSize value for either layer or net", () => {
            defaultValues["get_conv_filterSize"] = 0
            layer2.filterSize = undefined
            layer2.net.conv.filterSize = undefined
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_filterSize", null, ["number", "number", "number"], [132, 14, 3])
        })

        it("Keeps the same filterSize value if it has already been assigned to layer", () => {
            defaultValues["get_conv_filterSize"] = 101
            layer2.filterSize = 101
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_filterSize", null, ["number", "number", "number"], [132, 14, 101])
        })

        it("Defaults the layer.stride to the net.conv.stride value, if there's no layer.stride, but there is one for net, and it sets the cpp value", () => {
            defaultValues["get_conv_stride"] = 0
            const layer = new ConvLayer(3)
            layer.net = {conv: {stride: 1}, netInstance: 132, Module: {ccall: () => {}}}
            layer.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_stride", null, ["number", "number", "number"], [132, 14, 1])
        })

        it("Defaults the layer.stride to 1 if there is no stride value for either layer or net", () => {
            layer2.net.conv.stride = undefined
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_stride", null, ["number", "number", "number"], [132, 14, 1])
        })

        it("Keeps the same stride value if it has already been assigned to layer", () => {
            defaultValues["get_conv_stride"] = 3
            defaultValues["get_conv_zeroPadding"] = 1
            const layer = new ConvLayer()
            layer.net = {conv: {}, netInstance: 132, Module: {ccall: () => {}}}
            layer.stride = 3
            layer.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_stride", null, ["number", "number", "number"], [132, 14, 3])
        })

        it("Sets the layer channels to the net channels if the previous layer is an FCLayer", () => {
            const fcLayer = new FCLayer(10)
            layer2.net.channels = 69
            layer2.assignPrev(fcLayer, 2)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_channels", null, ["number", "number", "number"], [132, 2, 69])
        })

        it("Defaults the layer channels to 1 if the previous layer is FCLayer, but there is no net.channels configured", () => {
            const fcLayer = new FCLayer(10)
            layer2.net.channels = undefined
            layer2.layerInstance
            layer2.assignPrev(fcLayer, 2)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_channels", null, ["number", "number", "number"], [132, 2, 1])
        })

        it("Sets the layer channels to the prevLayer.size if the prevLayer is a ConvLayer", () => {
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_channels", null, ["number", "number", "number"], [132, 14, 10])
        })

        it("Sets the layer channels to the prevLayer.activations count if the prevLayer is a PoolLayer", () => {
            const poolLayer = new PoolLayer(2)
            poolLayer.activations = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
            poolLayer.outMapSize = 16
            layer2.assignPrev(poolLayer, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_channels", null, ["number", "number", "number"], [132, 14, 16])
        })

        it("Defaults the layer.zeroPadding to the net.zeroPadding value, if there's no layer.zeroPadding, but there is one for net, and sets the cpp value", () => {
            defaultValues["get_conv_zeroPadding"] = undefined
            layer2.zeroPadding = undefined
            layer2.net.conv.zeroPadding = 2
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_zeroPadding", null, ["number", "number", "number"], [132, 14, 2])
        })

        it("Defaults the layer.zeroPadding to rounded down half the filterSize if there is no zeroPadding value for either layer or net (test 1)", () => {
            defaultValues["get_conv_zeroPadding"] = undefined
            layer2.zeroPadding = undefined
            layer2.net.conv.zeroPadding = undefined
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_zeroPadding", null, ["number", "number", "number"], [132, 14, 1])
        })

        it("Defaults the layer.zeroPadding to rounded down half the filterSize if there is no zeroPadding value for either layer or net (test 2)", () => {
            defaultValues["get_conv_zeroPadding"] = undefined
            defaultValues["get_conv_filterSize"] = 5
            layer2.zeroPadding = undefined
            layer2.net.conv.zeroPadding = undefined
            layer2.net.conv.filterSize = 5
            layer2.filterSize = 5
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_zeroPadding", null, ["number", "number", "number"], [132, 14, 2])
        })

        it("Keeps the same zeroPadding value if it has already been assigned to layer", () => {
            defaultValues["get_conv_zeroPadding"] = 1
            layer2.zeroPadding = 1
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_zeroPadding", null, ["number", "number", "number"], [132, 14, 1])
        })

        it("Allows setting the zero padding to 0", () => {
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_zeroPadding", null, ["number", "number", "number"], [132, 14, 0])
        })

        it("Calls the NetUtil.defineProperty for inMapValuesCount", () => {
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.defineProperty).to.be.calledWith(layer2, "inMapValuesCount", ["number", "number"], [132, 14])
        })

        it("Calls the NetUtil.defineProperty for inZPMapValuesCount", () => {
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.defineProperty).to.be.calledWith(layer2, "inZPMapValuesCount", ["number", "number"], [132, 14])
        })

        it("Calls the NetUtil.defineProperty for outMapSize", () => {
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.defineProperty).to.be.calledWith(layer2, "outMapSize", ["number", "number"], [132, 14])
        })

        it("Assigns to layer.inMapValuesCount the size of the input map (Example 1)", () => {
            const layer1 = new ConvLayer(4)
            const layer2 = new ConvLayer(3, {filterSize: 3, zeroPadding: 0})
            layer2.net = {conv: {}, netInstance: 132, Module: {ccall: () => {}}}
            layer1.outMapSize = 28
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_inMapValuesCount", null, ["number", "number", "number"], [132, 14, 784])
        })

        it("Assigns to layer.inMapValuesCount the size of the input map (Example 2)", () => {
            const layer1 = new ConvLayer(4)
            const layer2 = new ConvLayer(3, {filterSize: 3, zeroPadding: 1})
            layer2.net = {conv: {}, netInstance: 132, Module: {ccall: () => {}}}
            layer1.outMapSize = 28
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_inMapValuesCount", null, ["number", "number", "number"], [132, 14, 784])
        })

        it("Assigns to layer.inMapValuesCount the size of the input map (Example 3)", () => {
            const layer1 = new FCLayer(75)
            layer1.size = 75
            const layer2 = new ConvLayer(3, {zeroPadding: 1, filterSize: 3})
            layer2.net = {conv: {}, channels: 3, netInstance: 132, Module: {ccall: () => {}}}
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_inMapValuesCount", null, ["number", "number", "number"], [132, 14, 25])
        })

        it("Sets the inMapValuesCount to the square of the prev layer's out map size, if prev layer is Conv", () => {
            const layer1 = new ConvLayer(3)
            const layer2 = new ConvLayer(4, {filterSize: 3, stride: 1, zeroPadding: 1})
            layer1.outMapSize = 100
            layer2.net = {netInstance: 132, Module: {ccall: () => {}}}
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_inMapValuesCount", null, ["number", "number", "number"], [132, 14, 10000])
        })

        it("Assigns to layer.inZPMapValuesCount the size of the zero padded input map (Example 1)", () => {
            const layer1 = new ConvLayer(4)
            const layer2 = new ConvLayer(3, {filterSize: 3, zeroPadding: 0})
            layer1.outMapSize = 28

            layer2.net = {conv: {}, netInstance: 132, Module: {ccall: () => {}}}
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_inZPMapValuesCount", null, ["number", "number", "number"], [132, 14, 784])
        })

        it("Assigns to layer.inZPMapValuesCount the size of the zero padded input map (Example 2)", () => {
            const layer1 = new ConvLayer(4)
            const layer2 = new ConvLayer(3, {filterSize: 3, zeroPadding: 1})
            defaultValues["get_conv_filterSize"] = 3
            defaultValues["get_conv_zeroPadding"] = 1

            layer1.outMapSize = 28
            layer2.net = {netInstance: 132, conv: {}, Module: {ccall: () => {}}}
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_inZPMapValuesCount", null, ["number", "number", "number"], [132, 14, 900])
        })

        it("Assigns to layer.inZPMapValuesCount the size of the zero padded input map (Example 3)", () => {
            const layer1 = new FCLayer(75)
            const layer2 = new ConvLayer(3, {zeroPadding: 1, filterSize: 3})
            defaultValues["get_conv_zeroPadding"] = 1
            defaultValues["get_conv_filterSize"] = 3

            layer2.net = {netInstance: 132, conv: {}, channels: 3, Module: {ccall: () => {}}}
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_inZPMapValuesCount", null, ["number", "number", "number"], [132, 14, 49])
        })

        it("Sets the layer.outMapSize to the spacial dimension of the filter activation/sum/error maps (Example 1)", () => {
            const layer1 = new FCLayer(2352) // 784 * 3
            const layer2 = new ConvLayer(4, {filterSize: 3, zeroPadding: 1})
            defaultValues["get_conv_filterSize"] = 3
            defaultValues["get_conv_zeroPadding"] = 1
            layer2.net = {netInstance: 132, conv: {}, channels: 3, Module: {ccall: () => {}}}
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_outMapSize", null, ["number", "number", "number"], [132, 14, 28])
        })

        it("Sets the layer.outMapSize to the spacial dimension of the filter activation/sum/error maps (Example 2)", () => {
            const layer1 = new FCLayer(75)
            const layer2 = new ConvLayer()
            layer1.size = 75
            layer2.size = 4
            layer2.stride = 2
            layer2.filterSize = 3
            layer2.zeroPadding = 1
            layer2.net = {netInstance: 132, conv: {}, channels: 3, Module: {ccall: () => {}}}
            layer2.assignPrev(layer1, 14)
            defaultValues["get_conv_filterSize"] = 3
            defaultValues["get_conv_zeroPadding"] = 1
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_outMapSize", null, ["number", "number", "number"], [132, 14, 3])
        })

        it("Sets the layer.outMapSize to the spacial dimension of the filter activation/sum/error maps (Example 3)", () => {
            const layer1 = new FCLayer(147)
            const layer2 = new ConvLayer(4)
            layer1.size = 147
            layer2.size = 4
            layer2.stride = 3
            layer2.filterSize = 3
            layer2.zeroPadding = 1
            layer2.net = {netInstance: 132, conv: {}, channels: 3, Module: {ccall: () => {}}}
            defaultValues["get_conv_zeroPadding"] = 1
            defaultValues["get_conv_filterSize"] = 3
            defaultValues["get_conv_stride"] = 3
            defaultValues["get_conv_channels"] = 3
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_outMapSize", null, ["number", "number", "number"], [132, 14, 3])
        })

        it("Creates a layer.filters array with as many filters as the size of the layer", () => {
            const prevLayer = new FCLayer(147)
            const layer = new ConvLayer(3)
            layer.net = {netInstance: 132, conv: {}, channels: 3, Module: {ccall: () => {}}}
            layer.assignPrev(layer1, 14)
            expect(layer.filters).to.not.be.undefined
            expect(layer.filters).to.have.lengthOf(3)
            expect(layer.filters[0]).instanceof(Filter)
            expect(layer.filters[1]).instanceof(Filter)
            expect(layer.filters[2]).instanceof(Filter)
        })

        it("Calls the net Module's ccall function with set_conv_activation with the layer.activationName, if given", () => {
            layer2.activationName = "tanh"
            layer2.assignPrev(layer1, 14)
            expect(layer2.activation).to.equal("WASM tanh")
            expect(NetUtil.defineProperty).to.be.calledWith(layer2, "activation", ["number", "number"], [layer2.netInstance, layer2.layerIndex])
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_activation", null, ["number", "number", "number"], [132, 14, 1])
        })

        it("Calls the net Module's ccall function with set_conv_activation with the net.activationName if given", () => {
            layer2.activationName = undefined
            layer2.net.activationName = "relu"
            layer2.assignPrev(layer1, 14)
            expect(layer2.activation).to.equal("WASM relu")
            expect(NetUtil.defineProperty).to.be.calledWith(layer2, "activation", ["number", "number"], [layer2.netInstance, layer2.layerIndex])
            expect(NetUtil.Module.ccall).to.be.calledWith("set_conv_activation", null, ["number", "number", "number"], [132, 14, 3])
        })

        it("Does not call the WASM setConvActivation function if the activation is set to false", () => {
            layer2.activationName = false
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.defineProperty).to.not.be.calledWith(layer2, "activation", ["number", "number"], [layer2.netInstance, layer2.layerIndex])
        })

        it("Throws an error if the hyperparameters don't match the input map properly", () => {
            const layer1 = new ConvLayer(3)
            const layer2 = new ConvLayer(4, {filterSize: 3, stride: 2, zeroPadding: 1})
            layer1.outMapSize = 16
            layer2.net = {netInstance: 132, conv: {}, Module: {ccall: () => {}}}

            NetUtil.Module.ccall.restore()
            sinon.stub(NetUtil.Module, "ccall").callsFake(() => 0.5)

            expect(layer2.assignPrev.bind(layer2, 123, layer1, 14)).to.throw("Misconfigured hyperparameters. Activation volume dimensions would be ")
        })
    })

    describe("init", () => {

        let convLayer
        let fcLayer
        const paramTypes = ["number", "number", "number"]

        before(() => {
            convLayer = new ConvLayer(3, {filterSize: 3, stride: 1})
            fcLayer = new FCLayer(21)
            convLayer.net = {netInstance: 123, conv: {}, updateFn: "gain", Module: {ccall: () => {}}}
            convLayer.assignPrev(fcLayer, 2)
            convLayer.outMapSize = 456

            sinon.stub(NetUtil.Module, "ccall").callsFake(fn => ({
                get_conv_filterSize: 3,
                get_conv_size: 3,
                get_conv_stride: 1,
                get_conv_zeroPadding: 0,
                get_conv_channels: 1,
                get_conv_outMapSize: 456
            })[fn])
            sinon.stub(NetUtil, "defineMapProperty")
        })

        after(() => {
            NetUtil.Module.ccall.restore()
            NetUtil.defineMapProperty.restore()
        })

        it("Calls every filter's init function with the layer's netInstance, its layer index, the filter index and layer data", () => {

            sinon.stub(convLayer.filters[0], "init")
            sinon.stub(convLayer.filters[1], "init")
            sinon.stub(convLayer.filters[2], "init")

            convLayer.init()

            expect(convLayer.filters[0].init).to.be.calledWith(123, 2, 0, {updateFn: "gain", filterSize: 3, channels: 1})
            expect(convLayer.filters[1].init).to.be.calledWith(123, 2, 1, {updateFn: "gain", filterSize: 3, channels: 1})
            expect(convLayer.filters[2].init).to.be.calledWith(123, 2, 2, {updateFn: "gain", filterSize: 3, channels: 1})
        })

        it("Defines filters' activationMap property with the correct values", () => {

            convLayer.init()

            for (let f=0; f<3; f++) {
                expect(NetUtil.defineMapProperty).to.be.calledWith(convLayer.filters[f], "activationMap", paramTypes,
                    [123, 2, f], 456, 456, {pre: "filter_"})
            }
        })

        it("Defines filters' errorMap property with the correct values", () => {

            convLayer.init()

            for (let f=0; f<3; f++) {
                expect(NetUtil.defineMapProperty).to.be.calledWith(convLayer.filters[f], "errorMap", paramTypes,
                    [123, 2, f], 456, 456, {pre: "filter_"})
            }
        })

        it("Defines filters' sumMap property with the correct values", () => {

            convLayer.init()

            for (let f=0; f<3; f++) {
                expect(NetUtil.defineMapProperty).to.be.calledWith(convLayer.filters[f], "sumMap", paramTypes,
                    [123, 2, f], 456, 456, {pre: "filter_"})
            }
        })

        it("Defines filters' dropoutMap property with the correct values", () => {

            convLayer.init()

            for (let f=0; f<3; f++) {
                expect(NetUtil.defineMapProperty).to.be.calledWith(convLayer.filters[f], "dropoutMap", paramTypes,
                    [123, 2, f], 456, 456)
            }
        })

        it("Returns an array with 2 items when accessing the layer.dropoutMap", () => {
            NetUtil.defineMapProperty.restore()

            sinon.stub(NetUtil, "ccallVolume").callsFake(() => [[[1, 0, 1]]])

            convLayer.init()
            expect(convLayer.filters[0].dropoutMap).to.deep.equal([[true, false, true]])

            sinon.stub(NetUtil, "defineMapProperty")
            NetUtil.ccallVolume.restore()
        })
    })

    describe("toJSON", () => {

        const convNet = new Network({Module: fakeModule, layers: [new FCLayer(1024), new ConvLayer(2, {filterSize: 3})]})

        it("Exports a conv layer's weights correctly", () => {

            let weights = [[[1,2,3],[4,5,6],[7,8,9]]]

            sinon.stub(NetUtil, "ccallVolume").callsFake(() => {
                let returnValue

                returnValue = weights
                weights = [[[4,5,6],[7,8,9],[1,2,3]]]

                return returnValue
            })

            const convJson = convNet.layers[1].toJSON()

            expect(convJson.weights).to.not.be.undefined
            expect(convJson.weights).to.have.lengthOf(2)
            expect(convJson.weights[0].weights).to.deep.equal([[[1,2,3],[4,5,6],[7,8,9]]])
            expect(convJson.weights[1].weights).to.deep.equal([[[4,5,6],[7,8,9],[1,2,3]]])
            NetUtil.ccallVolume.restore()
        })

        it("Exports a conv layer's bias correctly", () => {

            let v = 1
            sinon.stub(NetUtil.Module, "ccall").callsFake(fn => fn=="get_filter_bias"? v++:undefined)

            const convJson = convNet.layers[1].toJSON()
            expect(convJson.weights[0].bias).to.equal(1)
            expect(convJson.weights[1].bias).to.equal(2)

            NetUtil.Module.ccall.restore()
        })
    })

    describe("fromJSON", () => {

        const testDataConv = {
            weights: [
                {bias: 1, weights: [[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]],[[13,14],[15,16]]]},
                {bias: 2, weights: [[[5,6],[7,8]],[[9,10],[11,12]],[[13,14],[15,16]],[[1,2],[3,4]]]}
            ]
        }

        it("Sets the weights and biases in a conv layer to the import data values", () => {
            sinon.stub(NetUtil.Module, "ccall").callsFake(fn =>  fn=="get_conv_filterSize"?2:undefined)
            sinon.stub(NetUtil, "ccallVolume").callsFake(() => [[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]],[[13,14],[15,16]]])

            const net = new Network({Module: fakeModule, channels: 4, layers: [new FCLayer(16), new ConvLayer(2, {filterSize: 2})]})
            net.layers[1].fromJSON(testDataConv, 1)

            expect(NetUtil.Module.ccall).to.be.calledWith("set_filter_bias", null, ["number", "number", "number", "number"], [net.netInstance, 1, 0, 1])
            expect(NetUtil.Module.ccall).to.be.calledWith("set_filter_bias", null, ["number", "number", "number", "number"], [net.netInstance, 1, 1, 2])

            expect(NetUtil.ccallVolume).to.be.calledWith("set_filter_weights", null, ["number", "number", "number", "array"], [net.netInstance, 1, 0, [[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]],[[13,14],[15,16]]]])
            expect(NetUtil.ccallVolume).to.be.calledWith("set_filter_weights", null, ["number", "number", "number", "array"], [net.netInstance, 1, 1, [[[5,6],[7,8]],[[9,10],[11,12]],[[13,14],[15,16]],[[1,2],[3,4]]]])

            NetUtil.Module.ccall.restore()
            NetUtil.ccallVolume.restore()
        })

        it("Throws an error if the ConvLayer weights depth is mismatched", () => {
            sinon.stub(NetUtil, "ccallVolume").callsFake(() => [[[1,2],[3,4]]])
            const net = new Network({Module: fakeModule, channels: 1, layers: [new FCLayer(16), new ConvLayer(2, {filterSize: 2})]})
            expect(net.layers[1].fromJSON.bind(net.layers[1], testDataConv, 1)).to.throw("Mismatched weights depth. Given: 4 Existing: 1. At: layers[1], filters[0]")
            NetUtil.ccallVolume.restore()
        })

        it("Throws an error if the ConvLayer weights spacial dimension is mismatched", () => {
            sinon.stub(NetUtil, "ccallVolume").callsFake(() => [[[1,2, 2],[3,4,3],[3,4,3]],[[5,6,3],[7,8,3],[3,4,3]],[[9,10,3],[11,12,3],[3,4,3]],[[13,14,3],[15,16,3],[3,4,3]]])
            const net = new Network({Module: fakeModule, channels: 4, layers: [new FCLayer(16), new ConvLayer(2, {filterSize: 3})]})
            expect(net.layers[1].fromJSON.bind(net.layers[1], testDataConv, 1)).to.throw("Mismatched weights size. Given: 2 Existing: 3. At: layers[1], filters[0]")
            NetUtil.ccallVolume.restore()
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

        it("Throws an error if setting activation to something other than a string", () => {
            const wrapperFn = () => new PoolLayer(1, {activation: x => x})
            const wrapperFn2 = () => new PoolLayer(1, {activation: 2})
            expect(wrapperFn).to.throw("Custom activation functions are not available in the WebAssembly version")
            expect(wrapperFn2).to.throw("Custom activation functions are not available in the WebAssembly version")
        })

        it("Otherwise sets the given activation string to the layer", () => {
            const layer = new PoolLayer(1, {activation: "test"})
            expect(layer.activationName).to.equal("test")
        })

        it("Sets the activation to false if nothing is provided", () => {
            const layer = new PoolLayer()
            expect(layer.activation).to.be.false
        })

        it("Sets the activation function to noActivation when configuring it with false", () => {
            const layer = new PoolLayer(5, {activation: false})
            expect(layer.activationName).to.equal("noactivation")
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

        let layer1, layer2

        beforeEach(() => {
            layer1 = new ConvLayer(5)
            layer1.outMapSize = 16
            layer2 = new PoolLayer(2, {stride: 2})
            layer2.net = {netInstance: 123, pool: {}}
            sinon.spy(NetUtil, "defineProperty")
            sinon.spy(NetUtil, "defineMapProperty")
            sinon.stub(NetUtil, "defineVolumeProperty")

            sinon.stub(NetUtil.Module, "ccall")//.callsFake(fn => defaultValues[fn])
        })

        afterEach(() => {
            NetUtil.defineProperty.restore()
            NetUtil.defineMapProperty.restore()
            NetUtil.defineVolumeProperty.restore()
            NetUtil.Module.ccall.restore()
        })

        it("Sets the layer.netInstance to the net.netInstance", () => {
            layer2.assignPrev(layer1, 14)
            expect(layer2.netInstance).to.equal(123)
        })

        it("Sets the layer.prevLayer to the given layer", () => {
            layer1.outMapSize = 16
            layer2.assignPrev(layer1)
            expect(layer2.prevLayer).to.equal(layer1)
        })

        it("Assigns the layer2.layerIndex to the value given", () => {
            layer2.assignPrev(layer1, 5)
            expect(layer2.layerIndex).to.equal(5)
        })

        it("Calls the NetUtil.defineProperty for channels", () => {
            layer2.assignPrev(layer1, 5)
            expect(NetUtil.defineProperty).to.be.calledWith(layer2, "channels", ["number", "number"], [123, 5], {pre: "pool_"})
        })

        it("Calls the NetUtil.defineProperty for stride", () => {
            layer2.assignPrev(layer1, 5)
            expect(NetUtil.defineProperty).to.be.calledWith(layer2, "stride", ["number", "number"], [123, 5], {pre: "pool_"})
        })

        it("Sets the layer.size to the net.pool.size if not already defined, but existing in net.pool", () => {
            layer2.size = undefined
            layer2.stride = 1
            layer2.net.pool.size = 3
            layer2.assignPrev(layer1, 5)
            expect(layer2.size).to.equal(3)
        })

        it("Defaults the layer.size to 2 if not defined and not present in net.pool", () => {
            layer2.size = undefined
            layer2.net.pool.size = undefined
            layer2.assignPrev(layer1, 5)
            expect(layer2.size).to.equal(2)
        })

        it("Keeps the same size value if it has already been assigned to layer", () => {
            layer2.size = 10
            layer2.assignPrev(layer1, 5)
            expect(layer2.size).to.equal(10)
        })

        it("Sets the layer.stride to the net.pool.stride if not already defined, but existing in net.pool", () => {
            layer2.size = 1
            layer2.stride = undefined
            layer2.net.pool.stride = 3
            layer2.assignPrev(layer1, 5)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_pool_stride", null, ["number", "number", "number"], [123, 5, 3])
        })

        it("Defaults the layer.stride to the layer.size if not defined and not in net.pool", () => {
            layer2.size = 1
            layer2.stride = undefined
            layer2.net.pool.stride = undefined
            layer2.assignPrev(layer1, 5)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_pool_stride", null, ["number", "number", "number"], [123, 5, 1])
        })

        it("Sets the layer.channels to the last layer's filters count when the last layer is Conv", () => {
            layer2.assignPrev(layer1, 5)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_pool_channels", null, ["number", "number", "number"], [123, 5, 5])
        })

        it("Sets the layer.channels to the net.channels if the prev layer is an FCLayer", () => {
            layer2.size = 2
            layer2.stride = 2
            layer2.net.channels = 3
            layer2.assignPrev(new FCLayer(108), 5)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_pool_channels", null, ["number", "number", "number"], [123, 5, 3])
        })

        it("Sets the layer.channels to the last layer's channels values if the last layer is Pool", () => {
            const layer1 = new PoolLayer(2, {stride: 2})
            layer1.channels = 34
            layer1.outMapSize = 16
            layer2.assignPrev(layer1, 5)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_pool_channels", null, ["number", "number", "number"], [123, 5, 34])
        })

        it("Sets the layer.outMapSize to the correctly calculated value (Example 1)", () => {
            const layer1 = new ConvLayer(1)
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.outMapSize = 16
            layer2.net = {netInstance: 123}
            layer2.assignPrev(layer1, 5)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_pool_outMapSize", null, ["number", "number", "number"], [123, 5, 8])
        })

        it("Sets the layer.outMapSize to the correctly calculated value (Example 2)", () => {
            const layer1 = new ConvLayer(1)
            const layer2 = new PoolLayer(3, {stride: 3})
            layer1.outMapSize = 15
            layer2.net = {netInstance: 123}
            layer2.assignPrev(layer1, 5)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_pool_outMapSize", null, ["number", "number", "number"], [123, 5, 5])
        })

        it("Sets the layer.outMapSize to the correctly calculated value when prevLayer is FCLayer", () => {
            const layer1 = new FCLayer(108)
            const layer2 = new PoolLayer(2, {stride: 2})
            layer2.net = {channels: 3, netInstance: 123}
            layer2.assignPrev(layer1, 5 )
            expect(NetUtil.Module.ccall).to.be.calledWith("set_pool_outMapSize", null, ["number", "number", "number"], [123, 5, 3])
        })

        it("Sets the layer.outMapSize to the correctly calculated value when prevLayer is PoolLayer", () => {
            const layer1 = new PoolLayer(2, {stride: 2})
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.channels = 34
            layer1.outMapSize = 16
            layer2.net = {channels: 3, netInstance: 123}
            layer2.assignPrev(layer1, 5 )
            expect(NetUtil.Module.ccall).to.be.calledWith("set_pool_outMapSize", null, ["number", "number", "number"], [123, 5, 8])
        })

        it("Calls the NetUtil.defineProperty for prevLayerOutWidth", () => {
            layer2.assignPrev(layer1, 5)
            expect(NetUtil.defineProperty).to.be.calledWith(layer2, "prevLayerOutWidth", ["number", "number"], [123, 5], {pre: "pool_"})
        })

        it("Sets the layer.inMapValuesCount to the square value of the input map width value", () => {
            const layer1 = new PoolLayer(2, {stride: 2})
            const layer2 = new PoolLayer(2, {stride: 2})
            layer1.channels = 34
            layer1.outMapSize = 16
            layer2.net = {channels: 3, netInstance: 123}
            layer2.assignPrev(layer1, 5)
            expect(NetUtil.Module.ccall).to.be.calledWith("set_pool_outMapSize", null, ["number", "number", "number"], [123, 5, 8])
            expect(NetUtil.Module.ccall).to.be.calledWith("set_pool_inMapValuesCount", null, ["number", "number", "number"], [123, 5, 256])
        })

        it("Throws an error if the hyperparameters are misconfigured to not produce an output volume with integer dimensions", () => {
            const layer1 = new ConvLayer(1)
            const layer2 = new PoolLayer(3, {stride: 3})
            layer2.net = {channels: 3, netInstance: 123}
            layer1.outMapSize = 16
            expect(layer2.assignPrev.bind(layer2, layer1)).to.throw("Misconfigured hyperparameters. Activation volume dimensions would be ")
        })

        it("Calls the NetUtil.defineVolumeProperty function for layer.errors", () => {
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.defineVolumeProperty).to.be.calledWith(layer2, "errors", ["number", "number"], [123, 14], 5, 16, 16, {pre: "pool_"})
        })

        it("Calls the NetUtil.defineVolumeProperty function for layer.activations", () => {
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.defineVolumeProperty).to.be.calledWith(layer2, "activations", ["number", "number"], [123, 14], 5, 8, 8, {pre: "pool_"})
        })

        it("Calls the NetUtil.defineVolumeProperty function for layer.indeces", () => {
            layer1.outMapSize = 14
            layer2.assignPrev(layer1, 14)
            expect(NetUtil.defineVolumeProperty).to.be.calledWith(layer2, "indeces", ["number", "number"], [123, 14], 5, 7, 7)
        })

        it("Returns layer.indeces values as arrays with two elements", () => {
            NetUtil.defineVolumeProperty.restore()
            sinon.stub(NetUtil, "ccallVolume").callsFake(() => [[[3]]])
            layer1.outMapSize = 14
            layer2.assignPrev(layer1, 14)

            expect(layer2.indeces).to.deep.equal([[[[1,1]]]])

            NetUtil.ccallVolume.restore()
            sinon.stub(NetUtil, "defineVolumeProperty")
        })

        it("Sets layer.indeces values as single digits", () => {
            NetUtil.defineVolumeProperty.restore()
            sinon.stub(NetUtil, "ccallVolume")
            layer1.outMapSize = 14
            layer2.assignPrev(layer1, 14)

            layer2.indeces = [[[[1,1]]]]
            expect(NetUtil.ccallVolume).to.be.calledWith("set_pool_indeces", null, ["number", "number", "array"], [123, 14, [[[3]]]])

            NetUtil.ccallVolume.restore()
            sinon.stub(NetUtil, "defineVolumeProperty")
        })

        it("Calls the net Module's ccall function with set_pool_activation with the layer.activationName, if given", () => {
            layer2.activationName = "tanh"
            layer2.net = {netInstance: 123, pool: {}}
            layer2.assignPrev(layer1, 14)
            expect(layer2.activation).to.equal("WASM tanh")
            expect(NetUtil.defineProperty).to.be.calledWith(layer2, "activation", ["number", "number"], [123, layer2.layerIndex])
            expect(NetUtil.Module.ccall).to.be.calledWith("set_pool_activation", null, ["number", "number", "number"], [123, 14, 1])
        })

        it("Doesn't call the net Module's ccall function with set_pool_activation, if the activation is not given", () => {
            layer2.activationName = undefined
            layer2.net = {netInstance: 123, pool: {}}
            layer2.assignPrev(layer1, 14)
            expect(layer2.activation).to.be.false
            expect(NetUtil.defineProperty).to.not.be.calledWith(layer2, "activation", ["number", "number"], [123, layer2.layerIndex])
            expect(NetUtil.Module.ccall).to.not.be.calledWith("set_pool_activation", null, ["number", "number", "number"], [123, 14, 1])
        })

    })

    describe("init", () => {
        it("Does nothing", () => {
            const layer = new PoolLayer()
            expect(layer.init()).to.be.undefined
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

    describe("ccallVolume", () => {

        const testData = [[[1,2,3],[4,5,6]]]
        const expected = [1,2,3,4,5,6]

        const ccallVolume = NetUtil.ccallVolume

        beforeEach(() => {
            sinon.stub(NetUtil, "ccallArrays").callsFake(() => [1,2,3,4,5,6])
        })

        afterEach(() => {
            NetUtil.ccallArrays.restore()
        })

        it("Calls ccallArrays when no parameters are given", () => {
            ccallVolume("test")
            expect(NetUtil.ccallArrays).to.be.calledWith("test")
        })

        it("Calls ccallArrays with the same parameters when no volume data is given", () => {
            ccallVolume("test", "array", ["number"], [1])
            expect(NetUtil.ccallArrays).to.be.calledWith("test", "array", ["number"], [1])
        })

        it("Calls ccallArrays with a flattened array representing the given volume", () => {
            ccallVolume("test", "array", ["volume"], [testData])
            expect(NetUtil.ccallArrays).to.be.calledWith("test", "array", ["array", "number", "number", "number"], [expected, 1, 2, 3])
        })

        it("Calls ccallArrays with a multiplied total of the return dimensions", () => {
            ccallVolume("test", "array", ["volume"], [testData], {depth: 1, rows: 2, columns: 3})
            expect(NetUtil.ccallArrays).to.be.calledWith("test", "array", ["array", "number", "number", "number"], [expected, 1, 2, 3], {
                heapIn: "HEAPF32",
                heapOut: "HEAPF32",
                returnArraySize: 6
            })
        })

        it("Builds a returned array into a volume when the return value is volume", () => {
            const result = ccallVolume("test", "volume", ["volume"], testData, {depth: 1, rows: 2, columns: 3})
            expect(result).to.deep.equal(testData)
        })

        it("Returns a volume of data correctly from WASM", () => {
            NetUtil.ccallArrays.restore()
            sinon.spy(NetUtil, "ccallArrays")

            const expected = [ [[1,2],[3,4]], [[5,6],[7,8]]]
            const res = ccallVolume("testGetVolume", "volume", [], [], {depth: 2, rows: 2, columns: 2})
            expect(NetUtil.ccallArrays).to.be.calledWith("testGetVolume", "array", [], [], {
                heapIn: "HEAPF32",
                heapOut: "HEAPF32",
                returnArraySize: 8
            })
            expect(res).to.deep.equal(expected)
        })

        it("Passes and received volume parameters correctly", () => {
            NetUtil.ccallArrays.restore()
            sinon.spy(NetUtil, "ccallArrays")

            const volIn = [ [[1,2,0], [2,2,3]] ]
            const expected = [ [[1,4,0], [8,10,18]] ]
            const res = ccallVolume("testGetSetVolume", "volume", ["volume"], [volIn], {
                depth: 1,
                rows: 2,
                columns: 3,
                heapIn:"HEAPU8",
                heapOut: "HEAPU8"
            })
            expect(NetUtil.ccallArrays).to.be.calledWith("testGetSetVolume", "array", ["array", "number", "number", "number"], [[1,2,0,2,2,3], 1, 2, 3], {
                returnArraySize: 6,
                heapIn:"HEAPU8",
                heapOut: "HEAPU8"
            })
            expect(res).to.deep.equal(expected)
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

        it("Sets the given property to the given scope, calling with the provided 'pre' string as a prefix", () => {
            expect(scope.stuff2).to.be.undefined
            NetUtil.defineArrayProperty(scope, "stuff2", ["number", "number"], [1,2], 5, {pre: "TEST_"})
            expect(scope.stuff2).to.not.be.undefined
            expect(NetUtil.ccallArrays).to.be.calledWith("get_TEST_stuff2")
        })
    })

    describe("defineMapProperty", () => {

        beforeEach(() => {
            sinon.stub(fakeModule, "ccall")
            sinon.spy(NetUtil, "ccallArrays")
            sinon.spy(NetUtil, "ccallVolume")
        })

        afterEach(() => {
            fakeModule.ccall.restore()
            NetUtil.ccallArrays.restore()
            NetUtil.ccallVolume.restore()
        })

        const scope = {}

        it("Sets the given property to the given scope", () => {
            expect(scope.stuff).to.be.undefined
            NetUtil.defineMapProperty(scope, "stuff", ["number", "number"], [1,2], 5, 5)
            expect(scope.stuff).to.not.be.undefined
        })

        it("Setting a value passes the value through the setCallback first", () => {
            expect(scope.stuff2).to.be.undefined
            NetUtil.defineMapProperty(scope, "stuff2", ["number", "number"], [1,2], 5, 5, {setCallback: x => 20})
            scope.stuff2 = 10
            expect(NetUtil.ccallVolume).to.be.calledWith("set_stuff2", null, ["number", "number", "array"], [1,2,20])
        })

        it("Getting a value passes it through the getCallback first", () => {
            expect(scope.stuff3).to.be.undefined
            NetUtil.defineMapProperty(scope, "stuff3", ["number", "number"], [1,2], 5, 5, {getCallback: x => 20})
            expect(scope.stuff3).to.equal(20)
        })

        it("Not providing a getCallback function just returns the value as is", () => {
            expect(scope.stuff4).to.be.undefined
            NetUtil.defineMapProperty(scope, "stuff4", ["number", "number"], [1,2], 5, 5)
            scope.stuff4 = 123
            expect(scope.stuff4).to.have.lengthOf(5)
        })
    })

    describe("defineVolumeProperty", () => {

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
            NetUtil.defineVolumeProperty(scope, "stuff", ["number", "number"], [1,2], 5)
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

        it("Setting a value calls the setCallback function with the value first", () => {
            NetUtil.defineVolumeProperty(scope, "stuff2", ["number", "number"], [1,2], 5, 5, 5, {setCallback: x => 123})
            scope.stuff2 = [1,2,3,4,5]
            expect(NetUtil.ccallArrays).to.be.calledWith("set_stuff2", null, ["number", "number", "array"], [1,2,123])
        })

        it("Getting a value passes it through the getCallback first", () => {
            NetUtil.defineVolumeProperty(scope, "stuff3", ["number", "number"], [1,2], 5, 5, 5, {getCallback: x => 321})
            expect(scope.stuff3).to.equal(321)
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