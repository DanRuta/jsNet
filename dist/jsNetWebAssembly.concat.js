"use strict"

class ConvLayer {

}

typeof window=="undefined" && (exports.ConvLayer = ConvLayer)

"use strict"

class FCLayer {

    constructor (size) {
        this.size = size
        this.neurons = [...new Array(size)].map(n => new Neuron())
        this.layerIndex = 0
    }

    assignNext (layer) {
        this.nextLayer = layer
    }

    assignPrev (netInstance, layer, layerIndex) {
        this.netInstance = netInstance
        this.prevLayer = layer
        this.layerIndex = layerIndex

    }

    init () {
        this.neurons.forEach((neuron, ni) => {
            switch (true) {
                case this.prevLayer instanceof FCLayer:
                    neuron.size = this.prevLayer.size
                    break
            }

            neuron.init(this.netInstance, this.layerIndex, ni, {
                updateFn: this.net.updateFn
            })
        })
    }

    toJSON () {
        return {
            weights: this.neurons.map(neuron => {
                return {
                    bias: neuron.bias,
                    weights: neuron.weights
                }
            })
        }
    }

    fromJSON (data, layerIndex) {

        this.neurons.forEach((neuron, ni) => {

            if (data.weights[ni].weights.length!=(neuron.weights).length) {
                throw new Error(`Mismatched weights count. Given: ${data.weights[ni].weights.length} Existing: ${neuron.weights.length}. At layers[${layerIndex}], neurons[${ni}]`)
            }

            neuron.bias = data.weights[ni].bias
            neuron.weights = data.weights[ni].weights
        })
    }
}

const Layer = FCLayer

typeof window=="undefined" && (exports.FCLayer = exports.Layer = FCLayer)
"use strict"

class NetMath {
    static softmax (values) {
        let total = 0

        for (let i=0; i<values.length; i++) {
            total += values[i]
        }

        for (let i=0; i<values.length; i++) {
            if (total) {
                values[i] /= total
            }
        }

        return values
    }
}

typeof window=="undefined" && (exports.NetMath = NetMath)
"use strict"

class NetUtil {

    static ccallArrays (func, returnType, paramTypes, params, {heapIn="HEAPF32", heapOut="HEAPF32", returnArraySize=1}={}) {

        const heapMap = {}
        heapMap.HEAP8 = Int8Array // int8_t
        heapMap.HEAPU8 = Uint8Array // uint8_t
        heapMap.HEAP16 = Int16Array // int16_t
        heapMap.HEAPU16 = Uint16Array // uint16_t
        heapMap.HEAP32 = Int32Array // int32_t
        heapMap.HEAPU32 = Uint32Array // uint32_t
        heapMap.HEAPF32 = Float32Array // float
        heapMap.HEAPF64 = Float64Array // double

        let res
        let error
        paramTypes = paramTypes || []
        const returnTypeParam = returnType=="array" ? "number" : returnType
        const parameters = []
        const parameterTypes = []
        const bufs = []

        try {
            if (params) {
                for (let p=0; p<params.length; p++) {

                    if (paramTypes[p] == "array" || Array.isArray(params[p])) {

                        const typedArray = new heapMap[heapIn](params[p].length)

                        for (let i=0; i<params[p].length; i++) {
                            typedArray[i] = params[p][i]
                        }

                        const buf = NetUtil.Module._malloc(typedArray.length * typedArray.BYTES_PER_ELEMENT)

                        switch (heapIn) {
                            case "HEAP8": case "HEAPU8":
                                NetUtil.Module[heapIn].set(typedArray, buf)
                                break
                            case "HEAP16": case "HEAPU16":
                                NetUtil.Module[heapIn].set(typedArray, buf >> 1)
                                break
                            case "HEAP32": case "HEAPU32": case "HEAPF32":
                                NetUtil.Module[heapIn].set(typedArray, buf >> 2)
                                break
                            case "HEAPF64":
                                NetUtil.Module[heapIn].set(typedArray, buf >> 3)
                                break
                        }

                        bufs.push(buf)
                        parameters.push(buf)
                        parameters.push(params[p].length)
                        parameterTypes.push("number")
                        parameterTypes.push("number")

                    } else {
                        parameters.push(params[p])
                        parameterTypes.push(paramTypes[p]==undefined ? "number" : paramTypes[p])
                    }
                }
            }

            res = NetUtil.Module.ccall(func, returnTypeParam, parameterTypes, parameters)
        } catch (e) {
            error = e
        } finally {
            for (let b=0; b<bufs.length; b++) {
                NetUtil.Module._free(bufs[b])
            }
        }

        if (error) throw error


        if (returnType=="array") {
            const returnData = []

            for (let v=0; v<returnArraySize; v++) {
                returnData.push(NetUtil.Module[heapOut][res/heapMap[heapOut].BYTES_PER_ELEMENT+v])
            }

            return returnData
        } else {
            return res
        }
    }

    static format (value, type="string") {
        switch (true) {

            case type=="string" && typeof value=="string":
                value = value.replace(/(_|\s)/g, "").toLowerCase()
                break

            case type=="time" && typeof value=="number":
                const date = new Date(value)
                const formatted = []

                if (value < 1000) {
                    formatted.push(`${date.getMilliseconds()}ms`)

                } else if (value < 60000) {
                    formatted.push(`${date.getSeconds()}.${date.getMilliseconds()}s`)

                } else {

                    if (value >= 3600000) formatted.push(`${date.getHours()}h`)

                    formatted.push(`${date.getMinutes()}m`)
                    formatted.push(`${date.getSeconds()}s`)
                }

                value = formatted.join(" ")
                break
        }

        return value
    }

    static defineProperty (self, prop, valTypes=[], values=[], {getCallback=x=>x, setCallback=x=>x}={}) {
        Object.defineProperty(self, prop, {
            get: () => getCallback(this.Module.ccall(`get_${prop}`, "number", valTypes, values)),
            set: val => this.Module.ccall(`set_${prop}`, null, valTypes.concat("number"), values.concat(setCallback(val)))
        })
    }

    static defineArrayProperty (self, prop, valTypes, values, returnSize) {
        Object.defineProperty(self, prop, {
            get: () => NetUtil.ccallArrays(`get_${prop}`, "array", valTypes, values, {returnArraySize: returnSize, heapOut: "HEAPF64"}),
            set: (value) => NetUtil.ccallArrays(`set_${prop}`, null, valTypes.concat("array"), values.concat([value]), {heapIn: "HEAPF64"})
        })
    }

}

typeof window=="undefined" && (exports.NetUtil = NetUtil)
"use strict"

class Network {

    constructor ({Module, learningRate, activation="sigmoid", updateFn="vanillaupdatefn", cost="meansquarederror", layers=[],
        rmsDecay, rho}) {

        if (!Module) {
            throw new Error("WASM module not provided")
        }

        if (typeof activation == "function" || typeof cost == "function") {
            throw new Error("Custom functions are not (yet) supported with WASM.")
        }

        NetUtil.Module = Module
        this.Module = Module
        this.netInstance = this.Module.ccall("newNetwork", null, null, null)
        this.state = "not-defined"

        // Learning Rate get / set
        Object.defineProperty(this, "learningRate", {
            get: this.Module.cwrap("getLearningRate", null, null).bind(this, this.netInstance),
            set: this.Module.cwrap("setLearningRate", "number", null).bind(this, this.netInstance)
        })

        if (learningRate) this.learningRate = learningRate

        // Activation function get / set
        const activationsIndeces = {
            sigmoid: 0,
            tanh: 1,
            lecuntanh: 2,
            relu: 3
        }
        let activationName = NetUtil.format(activation)
        Object.defineProperty(this, "activation", {
            get: () => `WASM ${activationName}`,
            set: activation => {

                if (activationsIndeces[activation] == undefined) {
                    throw new Error(`The ${activation} function does not exist`)
                }
                activationName = activation
                this.Module.ccall("setActivation", null, ["number", "number"], [this.netInstance, activationsIndeces[activation]])
            }
        })
        this.activation = activationName

        // Cost function get / set
        const costIndeces = {
            meansquarederror: 0,
            crossentropy: 1
        }
        let costFunctionName = NetUtil.format(cost)
        Object.defineProperty(this, "cost", {
            get: () => `WASM ${costFunctionName}`,
            set: cost => {
                if (costIndeces[cost] == undefined) {
                    throw new Error(`The ${cost} function does not exist`)
                }
                costFunctionName = cost
                this.Module.ccall("setCostFunction", null, ["number", "number"], [this.netInstance, costIndeces[cost]])
            }
        })
        this.cost = costFunctionName

        const updateFnIndeces = {
            vanillaupdatefn: 0,
            gain: 1,
            adagrad: 2,
            rmsprop: 3,
            adam: 4,
            adadelta: 5
        }
        NetUtil.defineProperty(this, "updateFn", ["number"], [this.netInstance], {
            getCallback: index => Object.keys(updateFnIndeces).find(key => updateFnIndeces[key]==index),
            setCallback: name => updateFnIndeces[name]
        })
        this.updateFn = NetUtil.format(updateFn)

        switch (NetUtil.format(updateFn)) {

            case "rmsprop":
                this.learningRate = this.learningRate==undefined ? 0.001 : this.learningRate
                break

            case "adam":
                this.learningRate = this.learningRate==undefined ? 0.01 : this.learningRate
                break

            case "adadelta":
                NetUtil.defineProperty(this, "rho", ["number"], [this.netInstance])
                this.rho = rho==null ? 0.95 : rho
                break

            default:

                if (learningRate==undefined) {

                    switch (activationName) {
                        case "tanh":
                        case "lecuntanh":
                            this.learningRate = 0.01
                            break

                        default:
                            this.learningRate = 0.2
                    }
                }
        }

        if (this.updateFn=="rmsprop") {
            NetUtil.defineProperty(this, "rmsDecay", ["number"], [this.netInstance])
            this.rmsDecay = rmsDecay===undefined ? 0.99 : rmsDecay
        }

        this.layers = []
        this.epochs = 0
        this.iterations = 0


        if (layers.length) {

            this.state = "constructed"

            switch (true) {
                case layers.every(item => Number.isInteger(item)):
                    this.layers = layers.map(size => new FCLayer(size))
                    this.initLayers()
                    break

                case layers.every(layer => layer instanceof FCLayer || layer instanceof ConvLayer || layer instanceof PoolLayer):
                    this.layers = layers
                    this.initLayers()
                    break

                default:
                    throw new Error("There was an error constructing from the layers given.")

            }
        }
    }

    initLayers (input, expected) {

        if (this.state == "initialised") {
            return
        }

        if (this.state == "not-defined") {
            this.layers[0] = new FCLayer(input)
            this.layers[1] = new FCLayer(Math.ceil(input/expected > 5 ? expected + (Math.abs(input-expected))/4
                                                                      : input + expected))
            this.layers[2] = new FCLayer(Math.ceil(expected))
        }

        this.state = "initialised"

        for (let l=0; l<this.layers.length; l++) {

            const layer = this.layers[l]

            if (layer instanceof FCLayer) {
                this.Module.ccall("addFCLayer", null, ["number", "number"], [this.netInstance, layer.size])
                this.joinLayer(layer, l)
            }
        }

        this.Module.ccall("initLayers", null, ["number"], [this.netInstance])
    }

    joinLayer (layer, layerIndex) {

        layer.net = this
        layer.layerIndex = layerIndex

        if (layerIndex) {
            this.layers[layerIndex-1].assignNext(layer)
            layer.assignPrev(this.netInstance, this.layers[layerIndex-1], layerIndex)
            layer.init()
        }
    }

    forward (data) {

        if (this.state!="initialised") {
            throw new Error("The network layers have not been initialised.")
        }

        if (data === undefined || data === null) {
            throw new Error("No data passed to Network.forward()")
        }

        if (data.length != this.layers[0].neurons.length) {
            console.warn("Input data length did not match input layer neurons count.")
        }

        return NetUtil.ccallArrays("forward", "array", ["number", "array"], [this.netInstance, data], {
            heapOut: "HEAPF64",
            returnArraySize: this.layers[this.layers.length-1].neurons.length
        })
    }

    train (data, {epochs=1}={}) {
        return new Promise((resolve, reject) => {

            if (data === undefined || data === null) {
                return void reject("No data provided")
            }

            if (this.state != "initialised") {
                this.initLayers(data[0].input.length, (data[0].expected || data[0].output).length)
            }

            const startTime = Date.now()

            const dimension = data[0].input.length
            const itemSize = dimension + (data[0].expected || data[0].output).length
            const itemsCount = itemSize * data.length

            const typedArray = new Float32Array(itemsCount)

            for (let di=0; di<data.length; di++) {

                if (!data[di].hasOwnProperty("input") || (!data[di].hasOwnProperty("expected") && !data[di].hasOwnProperty("output"))) {
                    return void reject("Data set must be a list of objects with keys: 'input' and 'expected' (or 'output')")
                }

                let index = itemSize*di

                for (let ii=0; ii<data[di].input.length; ii++) {
                    typedArray[index] = data[di].input[ii]
                    index++
                }

                for (let ei=0; ei<(data[di].expected || data[di].output).length; ei++) {
                    typedArray[index] = (data[di].expected || data[di].output)[ei]
                    index++
                }
            }

            const buf = this.Module._malloc(typedArray.length*typedArray.BYTES_PER_ELEMENT)
            this.Module.HEAPF32.set(typedArray, buf >> 2)

            for (let e=0; e<epochs; e++) {
                this.Module.ccall("train", "number", ["number", "number", "number", "number", "number"],
                                                [this.netInstance, buf, itemsCount, itemSize, dimension])
            }

            this.Module._free(buf)
            const elapsed = Date.now() - startTime
            console.log(`Training finished. Total time: ${NetUtil.format(elapsed, "time")}`)
            resolve()
        })
    }

    test (data, {log=true}={}) {
        return new Promise((resolve, reject) => {

            if (data === undefined || data === null) {
                reject("No data provided")
            }

            if (log) {
                console.log("Testing started")
            }

            const startTime = Date.now()
            const dimension = data[0].input.length
            const itemSize = dimension + (data[0].expected || data[0].output).length
            const itemsCount = itemSize * data.length
            const typedArray = new Float32Array(itemsCount)

            for (let di=0; di<data.length; di++) {

                let index = itemSize*di

                for (let ii=0; ii<data[di].input.length; ii++) {
                    typedArray[index] = data[di].input[ii]
                    index++
                }

                for (let ei=0; ei<(data[di].expected || data[di].output).length; ei++) {
                    typedArray[index] = (data[di].expected || data[di].output)[ei]
                    index++
                }
            }

            const buf = this.Module._malloc(typedArray.length*typedArray.BYTES_PER_ELEMENT)
            this.Module.HEAPF32.set(typedArray, buf >> 2)
            const avgError = this.Module.ccall("test", "number", ["number", "number", "number", "number", "number"],
                                            [this.netInstance, buf, itemsCount, itemSize, dimension])
            this.Module._free(buf)

            const elapsed = Date.now() - startTime

            if (log) {
                console.log(`Testing finished. Total time: ${NetUtil.format(elapsed, "time")}  Average iteration time: ${NetUtil.format(elapsed/data.length, "time")}`)
            }

            resolve(avgError)
        })
    }

    toJSON () {
        return {
            layers: this.layers.map(layer => layer.toJSON())
        }
    }

    fromJSON (data) {

        if (data === undefined || data === null) {
            throw new Error("No JSON data given to import.")
        }

        if (data.layers.length != this.layers.length) {
            throw new Error(`Mismatched layers (${data.layers.length} layers in import data, but ${this.layers.length} configured)`)
        }

        this.Module.ccall("resetDeltaWeights", null, ["number"], [this.netInstance])
        this.layers.forEach((layer, li) => li && layer.fromJSON(data.layers[li], li))
    }

    static get version () {
        return "2.1.1"
    }
}

typeof window=="undefined" && (exports.Network = Network)
"use strict"

class Neuron {

    constructor () {}

    init (netInstance, layerIndex, neuronIndex, {updateFn}) {

        NetUtil.defineArrayProperty(this, "weights", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex], this.size)
        NetUtil.defineProperty(this, "bias", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex])
        NetUtil.defineArrayProperty(this, "deltaWeights", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex], this.size)

        switch (updateFn) {
            case "gain":
                NetUtil.defineProperty(this, "biasGain", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex])
                NetUtil.defineArrayProperty(this, "weightGain", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex], this.size)
                break
            case "adagrad":
            case "rmsprop":
            case "adadelta":
                NetUtil.defineProperty(this, "biasCache", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex])
                NetUtil.defineArrayProperty(this, "weightsCache", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex], this.size)

                if (updateFn=="adadelta") {
                    NetUtil.defineProperty(this, "adadeltaBiasCache", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex])
                    NetUtil.defineArrayProperty(this, "adadeltaCache", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex], this.size)
                }
                break

            case "adam":
                NetUtil.defineProperty(this, "m", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex])
                NetUtil.defineProperty(this, "v", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex])
                break
        }

    }

}

typeof window=="undefined" && (exports.Neuron = Neuron)
"use strict"

class PoolLayer {

}

typeof window=="undefined" && (exports.PoolLayer = PoolLayer)

//# sourceMappingURL=jsNetWebAssembly.concat.js.map