"use strict"

class ConvLayer {

    constructor (size, {filterSize, zeroPadding, stride, activation}={}) {

        this.size = size
        this.stride = stride
        this.filterSize = filterSize
        this.layerIndex = 0
        this.zeroPadding = zeroPadding

        if (activation != undefined) {
            if (typeof activation != "string") {
                throw new Error("Only string activation functions available in the WebAssembly version")
            }
            this.activation = NetUtil.format(activation)
        }
    }

    assignNext (layer) {
        this.nextLayer = layer
    }

    assignPrev (layer, layerIndex) {

        this.netInstance = this.net.netInstance
        this.prevLayer = layer
        this.layerIndex = layerIndex

        const stride = this.stride || this.net.conv.stride || 1
        const filterSize = this.filterSize || this.net.conv.filterSize || 3
        let zeroPadding = this.zeroPadding

        NetUtil.defineProperty(this, "channels", ["number", "number"], [this.netInstance, layerIndex], {pre: "conv_"})
        NetUtil.defineProperty(this, "filterSize", ["number", "number"], [this.netInstance, layerIndex], {pre: "conv_"})
        NetUtil.defineProperty(this, "stride", ["number", "number"], [this.netInstance, layerIndex], {pre: "conv_"})
        NetUtil.defineProperty(this, "zeroPadding", ["number", "number"], [this.netInstance, layerIndex], {pre: "conv_"})

        this.size = this.size || 4
        let channels

        switch (true) {
            case layer instanceof FCLayer:
                channels = this.net.channels || 1
                break

            case layer instanceof ConvLayer:
                channels = layer.size
                break

            case layer instanceof PoolLayer:
                channels = layer.activations.length
                break
        }

        if (zeroPadding == undefined) {
            zeroPadding = this.net.conv.zeroPadding==undefined ? Math.floor(filterSize/2) : this.net.conv.zeroPadding
        }

        this.channels = channels
        this.filterSize = filterSize
        this.stride = stride
        this.zeroPadding = zeroPadding

        // Caching calculations
        const prevLayerOutWidth = layer instanceof FCLayer ? Math.max(Math.floor(Math.sqrt(layer.size/channels)), 1)
                                                           : layer.outMapSize

        NetUtil.defineProperty(this, "inMapValuesCount", ["number", "number"], [this.netInstance, layerIndex], {pre: "conv_"})
        NetUtil.defineProperty(this, "inZPMapValuesCount", ["number", "number"], [this.netInstance, layerIndex], {pre: "conv_"})
        NetUtil.defineProperty(this, "outMapSize", ["number", "number"], [this.netInstance, layerIndex], {pre: "conv_"})

        const outSize = (prevLayerOutWidth - filterSize + 2*zeroPadding) / stride + 1
        this.inMapValuesCount = Math.pow(prevLayerOutWidth, 2)
        this.inZPMapValuesCount = Math.pow(prevLayerOutWidth + zeroPadding*2, 2)
        this.outMapSize = outSize

        if (outSize%1!=0) {
            throw new Error(`Misconfigured hyperparameters. Activation volume dimensions would be ${outSize} in conv layer at index ${layerIndex}`)
        }

        if (this.activation != false) {
            this.net.Module.ccall("setConvActivation", null, ["number", "number", "number"],
                [this.netInstance, layerIndex, NetUtil.activationsIndeces[this.activation||this.net.activationName]])
        }

        this.filters = [...new Array(this.size)].map(f => new Filter())
    }

    init () {
        this.filters.forEach((filter, fi) => {

            const paramTypes = ["number", "number", "number"]
            const params = [this.netInstance, this.layerIndex, fi]

            NetUtil.defineMapProperty(filter, "activationMap", paramTypes, params, this.outMapSize, this.outMapSize, {pre: "filter_"})
            NetUtil.defineMapProperty(filter, "errorMap", paramTypes, params, this.outMapSize, this.outMapSize, {pre: "filter_"})
            NetUtil.defineMapProperty(filter, "sumMap", paramTypes, params, this.outMapSize, this.outMapSize, {pre: "filter_"})
            NetUtil.defineMapProperty(filter, "dropoutMap", paramTypes, params, this.outMapSize, this.outMapSize, {
                pre: "filter_",
                getCallback: m => m.map(row => row.map(v => v==1))
            })

            filter.init(this.netInstance, this.layerIndex, fi, {
                updateFn: this.net.updateFn,
                filterSize: this.filterSize,
                channels: this.channels
            })
        })
    }

    toJSON () {
        return {
            weights: this.filters.map(filter => {
                return {
                    bias: filter.bias,
                    weights: filter.weights
                }
            })
        }
    }

    fromJSON (data, layerIndex) {
        this.filters.forEach((filter, fi) => {

            if (data.weights[fi].weights.length != filter.weights.length) {
                throw new Error(`Mismatched weights depth. Given: ${data.weights[fi].weights.length} Existing: ${filter.weights.length}. At: layers[${layerIndex}], filters[${fi}]`)
            }

            if (data.weights[fi].weights[0].length != filter.weights[0].length) {
                throw new Error(`Mismatched weights size. Given: ${data.weights[fi].weights[0].length} Existing: ${filter.weights[0].length}. At: layers[${layerIndex}], filters[${fi}]`)
            }

            filter.bias = data.weights[fi].bias
            filter.weights = data.weights[fi].weights
        })
    }

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

    assignPrev (layer, layerIndex) {
        this.netInstance = this.net.netInstance
        this.prevLayer = layer
        this.layerIndex = layerIndex
    }

    init () {
        this.neurons.forEach((neuron, ni) => {
            switch (true) {

                case this.prevLayer instanceof FCLayer:
                    neuron.size = this.prevLayer.size
                    break

                case this.prevLayer instanceof ConvLayer:
                    neuron.size = this.prevLayer.filters.length * this.prevLayer.outMapSize**2
                    break

                case this.prevLayer instanceof PoolLayer:
                    neuron.size = this.prevLayer.channels * this.prevLayer.outMapSize**2
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

class Filter {

    constructor () {}

    init (netInstance, layerIndex, filterIndex, {updateFn, channels, filterSize}) {

        const paramTypes = ["number", "number", "number"]
        const params = [netInstance, layerIndex, filterIndex]

        NetUtil.defineProperty(this, "bias", paramTypes, params, {pre: "filter_"})
        NetUtil.defineVolumeProperty(this, "weights", paramTypes, params, channels, filterSize, filterSize, {pre: "filter_"})
        NetUtil.defineProperty(this, "deltaBias", paramTypes, params, {pre: "filter_"})
        NetUtil.defineVolumeProperty(this, "deltaWeights", paramTypes, params, channels, filterSize, filterSize, {pre: "filter_"})

        switch (updateFn) {
            case "gain":
                NetUtil.defineProperty(this, "biasGain", paramTypes, params, {pre: "filter_"})
                NetUtil.defineVolumeProperty(this, "weightGain", paramTypes, params, channels, filterSize, filterSize, {pre: "filter_"})
                break
            case "adagrad":
            case "rmsprop":
            case "adadelta":
                NetUtil.defineProperty(this, "biasCache", paramTypes, params, {pre: "filter_"})
                NetUtil.defineVolumeProperty(this, "weightsCache", paramTypes, params, channels, filterSize, filterSize, {pre: "filter_"})

                if (updateFn == "adadelta") {
                    NetUtil.defineProperty(this, "adadeltaBiasCache", paramTypes, params, {pre: "filter_"})
                    NetUtil.defineVolumeProperty(this, "adadeltaWeightsCache", paramTypes, params, channels, filterSize, filterSize, {pre: "filter_"})
                }
                break
            case "adam":
                NetUtil.defineProperty(this, "m", paramTypes, params, {pre: "filter_"})
                NetUtil.defineProperty(this, "v", paramTypes, params, {pre: "filter_"})
                break
        }
    }
}

typeof window=="undefined" && (exports.Filter = Filter)
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

    static ccallVolume (func, returnType, paramTypes=[], params=[], {heapIn="HEAPF32", heapOut="HEAPF32", depth=1, rows=1, columns=rows}={}) {

        const totalValues = depth * rows * columns
        const parameters = []
        const parameterTypes = []

        // Loop through parameters, check if they are volumes, flatten them, and send them along with their dimensions
        for (let p=0; p<params.length; p++) {

            let parameter = params[p]
            const isVolume = Array.isArray(parameter) && Array.isArray(parameter[0]) && Array.isArray(parameter[0][0])

            if (paramTypes[p] == "volume" || isVolume) {
                const flat = []

                for (let d=0; d<parameter.length; d++) {
                    for (let r=0; r<parameter[d].length; r++) {
                        for (let c=0; c<parameter[d][r].length; c++) {
                            flat.push(parameter[d][r][c])
                        }
                    }
                }

                parameters.splice(parameters.length, 0, flat, parameter.length, parameter[0].length, parameter[0][0].length)
                parameterTypes.splice(parameterTypes.length, 0, "array", "number", "number", "number")

            } else {
                parameters.push(parameter)
                parameterTypes.push(paramTypes[p])
            }
        }

        const res = NetUtil.ccallArrays(func, returnType=="volume" ? "array" : returnType, parameterTypes, parameters, {heapIn, heapOut, returnArraySize: totalValues})
        const vol = []

        if (returnType == "volume") {
            for (let d=0; d<depth; d++) {
                const map = []

                for (let r=0; r<rows; r++) {
                    const row = []

                    for (let c=0; c<columns; c++) {
                        row.push(res[d * rows * columns + r * columns + c])
                    }
                    map.push(row)
                }
                vol.push(map)
            }
            return vol
        }

        return res
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

    static defineProperty (self, prop, valTypes=[], values=[], {getCallback=x=>x, setCallback=x=>x, pre=""}={}) {
        Object.defineProperty(self, prop, {
            get: () => getCallback(this.Module.ccall(`get_${pre}${prop}`, "number", valTypes, values)),
            set: val => this.Module.ccall(`set_${pre}${prop}`, null, valTypes.concat("number"), values.concat(setCallback(val)))
        })
    }

    static defineArrayProperty (self, prop, valTypes, values, returnSize, {pre=""}={}) {
        Object.defineProperty(self, prop, {
            get: () => NetUtil.ccallArrays(`get_${pre}${prop}`, "array", valTypes, values, {returnArraySize: returnSize, heapOut: "HEAPF64"}),
            set: value => NetUtil.ccallArrays(`set_${pre}${prop}`, null, valTypes.concat("array"), values.concat([value]), {heapIn: "HEAPF64"})
        })
    }

    static defineMapProperty (self, prop, valTypes, values, rows, columns, {getCallback=x=>x, setCallback=x=>x, pre=""}={}) {
        Object.defineProperty(self, prop, {
            get: () => getCallback(NetUtil.ccallVolume(`get_${pre}${prop}`, "volume", valTypes, values, {depth: 1, rows, columns, heapOut: "HEAPF64"})[0]),
            set: value => NetUtil.ccallVolume(`set_${pre}${prop}`, null, valTypes.concat("array"), values.concat([setCallback(value)]), {heapIn: "HEAPF64"})
        })
    }

    static defineVolumeProperty (self, prop, valTypes, values, depth, rows, columns, {getCallback=x=>x, setCallback=x=>x, pre=""}={}) {
        Object.defineProperty(self, prop, {
            get: () => getCallback(NetUtil.ccallVolume(`get_${pre}${prop}`, "volume", valTypes, values, {depth, rows, columns, heapOut: "HEAPF64"})),
            set: value => NetUtil.ccallVolume(`set_${pre}${prop}`, null, valTypes.concat("array"), values.concat([setCallback(value)]), {heapIn: "HEAPF64"})
        })
    }
}

NetUtil.activationsIndeces = {
    sigmoid: 0,
    tanh: 1,
    lecuntanh: 2,
    relu: 3,
    lrelu: 4,
    rrelu: 5,
    elu: 6
}

typeof window=="undefined" && (exports.NetUtil = NetUtil)
"use strict"

class Network {

    constructor ({Module, learningRate, activation="sigmoid", updateFn="vanillaupdatefn", cost="meansquarederror", layers=[],
        rmsDecay, rho, lreluSlope, eluAlpha, dropout=1, l2=true, l1=true, maxNorm, weightsConfig, channels, conv}) {

        if (!Module) {
            throw new Error("WASM module not provided")
        }

        if (typeof activation == "function" || typeof cost == "function") {
            throw new Error("Custom functions are not (yet) supported with WASM.")
        }

        NetUtil.Module = Module
        this.Module = Module
        this.conv = {}
        this.netInstance = this.Module.ccall("newNetwork", null, null, null)
        this.state = "not-defined"

        // Learning Rate get / set
        Object.defineProperty(this, "learningRate", {
            get: this.Module.cwrap("getLearningRate", null, null).bind(this, this.netInstance),
            set: this.Module.cwrap("setLearningRate", "number", null).bind(this, this.netInstance)
        })

        if (learningRate) this.learningRate = learningRate

        NetUtil.defineProperty(this, "dropout", ["number"], [this.netInstance])
        this.dropout = dropout==false ? 1 : dropout

        if (l2) {
            NetUtil.defineProperty(this, "l2", ["number"], [this.netInstance])
            NetUtil.defineProperty(this, "l2Error", ["number"], [this.netInstance])
            this.l2 = typeof l2=="boolean" ? 0.001 : l2
        }

        if (l1) {
            NetUtil.defineProperty(this, "l1", ["number"], [this.netInstance])
            NetUtil.defineProperty(this, "l1Error", ["number"], [this.netInstance])
            this.l1 = typeof l1=="boolean" ? 0.005 : l1
        }

        if (maxNorm) {
            NetUtil.defineProperty(this, "maxNorm", ["number"], [this.netInstance])
            NetUtil.defineProperty(this, "maxNormTotal", ["number"], [this.netInstance])
            this.maxNorm = typeof maxNorm=="boolean" && maxNorm ? 1000 : maxNorm
        }

        if (channels) {
            NetUtil.defineProperty(this, "channels", ["number"], [this.netInstance])
            this.channels = channels
        }

        if (conv) {

            if (conv.filterSize != undefined) {
                NetUtil.defineProperty(this.conv, "filterSize", ["number"], [this.netInstance])
                this.conv.filterSize = conv.filterSize
            }

            if (conv.zeroPadding != undefined) {
                NetUtil.defineProperty(this.conv, "zeroPadding", ["number"], [this.netInstance])
                this.conv.zeroPadding = conv.zeroPadding
            }

            if (conv.stride != undefined) {
                NetUtil.defineProperty(this.conv, "stride", ["number"], [this.netInstance])
                this.conv.stride = conv.stride
            }
        }

        Object.defineProperty(this, "error", {
            get: () => Module.ccall("getError", "number", ["number"], [this.netInstance])
        })

        // Activation function get / set
        this.activationName = NetUtil.format(activation)
        Object.defineProperty(this, "activation", {
            get: () => `WASM ${this.activationName}`,
            set: activation => {

                if (NetUtil.activationsIndeces[activation] == undefined) {
                    throw new Error(`The ${activation} activation function does not exist`)
                }
                this.activationName = activation
                this.Module.ccall("setActivation", null, ["number", "number"], [this.netInstance, NetUtil.activationsIndeces[activation]])
            }
        })
        this.activation = this.activationName

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


        // Weights init configs
        const weightsConfigFns = {
            uniform: 0,
            gaussian: 1,
            xavieruniform: 2,
            xaviernormal: 3,
            lecununiform: 4,
            lecunnormal: 5
        }
        this.weightsConfig = {}

        NetUtil.defineProperty(this.weightsConfig, "distribution", ["number"], [this.netInstance], {
            getCallback: index => Object.keys(weightsConfigFns).find(key => weightsConfigFns[key]==Math.round(index)),
            setCallback: name => weightsConfigFns[name]
        })
        NetUtil.defineProperty(this.weightsConfig, "limit", ["number"], [this.netInstance])
        NetUtil.defineProperty(this.weightsConfig, "mean", ["number"], [this.netInstance])
        NetUtil.defineProperty(this.weightsConfig, "stdDeviation", ["number"], [this.netInstance])

        this.weightsConfig.distribution = "xavieruniform"

        if (weightsConfig!=undefined && weightsConfig.distribution) {

            if (typeof weightsConfig.distribution == "function") {
                throw new Error("Custom weights init functions are not (yet) supported with WASM.")
            }

            this.weightsConfig.distribution = NetUtil.format(weightsConfig.distribution)
        }

        this.weightsConfig.limit = weightsConfig && weightsConfig.limit!=undefined ? weightsConfig.limit : 0.1
        this.weightsConfig.mean = weightsConfig && weightsConfig.mean!=undefined ? weightsConfig.mean : 0
        this.weightsConfig.stdDeviation = weightsConfig && weightsConfig.stdDeviation!=undefined ? weightsConfig.stdDeviation : 0.05

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

                    switch (this.activationName) {
                        case "relu":
                        case "lrelu":
                        case "rrelu":
                        case "elu":
                            this.learningRate = 0.01
                            break

                        case "tanh":
                        case "lecuntanh":
                            this.learningRate = 0.001
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

        if (this.activationName=="lrelu") {
            NetUtil.defineProperty(this, "lreluSlope", ["number"], [this.netInstance])
            this.lreluSlope = lreluSlope==undefined ? -0.0005 : lreluSlope
        } else if (this.activationName=="elu") {
            NetUtil.defineProperty(this, "eluAlpha", ["number"], [this.netInstance])
            this.eluAlpha = eluAlpha==undefined ? 1 : eluAlpha
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

            switch (true) {
                case layer instanceof FCLayer:
                    this.Module.ccall("addFCLayer", null, ["number", "number"], [this.netInstance, layer.size])
                    break

                case layer instanceof ConvLayer:
                    this.Module.ccall("addConvLayer", null, ["number", "number"], [this.netInstance, layer.size])
                    break

                case layer instanceof PoolLayer:
                    this.Module.ccall("addPoolLayer", null, ["number", "number"], [this.netInstance, layer.size])
                    break
            }

            this.joinLayer(layer, l)
        }

        this.Module.ccall("initLayers", null, ["number"], [this.netInstance])
    }

    joinLayer (layer, layerIndex) {

        layer.net = this
        layer.layerIndex = layerIndex

        if (layerIndex) {
            this.layers[layerIndex-1].assignNext(layer)
            layer.assignPrev(this.layers[layerIndex-1], layerIndex)
        }
        layer.init()
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

    train (data, {epochs=1, callback, miniBatchSize=1, log=true, shuffle=false}={}) {

        miniBatchSize = typeof miniBatchSize=="boolean" && miniBatchSize ? data[0].expected.length : miniBatchSize
        this.Module.ccall("set_miniBatchSize", null, ["number", "number"], [this.netInstance, miniBatchSize])

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

            if (log) {
                console.log(`Training started. Epochs: ${epochs} Batch size: ${miniBatchSize}`)
            }

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

            let elapsed

            this.Module.ccall("loadTrainingData", "number", ["number", "number", "number", "number", "number"],
                                            [this.netInstance, buf, itemsCount, itemSize, dimension])

            if (shuffle) {
                this.Module.ccall("shuffleTrainingData", null, ["number"], [this.netInstance])
            }

            if (callback) {

                let epochIndex = 0
                let iterationIndex = 0

                const doEpoch = () => {

                    if (this.l2) this.l2Error = 0
                    if (this.l1) this.l1Error = 0

                    iterationIndex = 0
                    doIteration()
                }

                const doIteration = () => {

                    this.Module.ccall("train", "number", ["number", "number", "number"], [this.netInstance, miniBatchSize, iterationIndex])

                    callback({
                        iterations: (iterationIndex+1),
                        error: this.error,
                        elapsed: Date.now() - startTime,
                        input: data[this.iterations].input
                    })

                    iterationIndex += miniBatchSize

                    if (iterationIndex < data.length) {
                        setTimeout(doIteration.bind(this), 0)
                    } else {
                        epochIndex++

                        elapsed = Date.now() - startTime

                        log && console.log(`Epoch ${epochIndex} Error: ${this.error}${this.l2==undefined ? "": ` L2 Error: ${this.l2Error/iterationIndex}`}`,
                                    `\nElapsed: ${NetUtil.format(elapsed, "time")} Average Duration: ${NetUtil.format(elapsed/epochIndex, "time")}`)

                        if (epochIndex < epochs) {
                            doEpoch()
                        } else {
                            this.Module._free(buf)
                            resolve()
                        }
                    }
                }
                doEpoch()

            } else {
                for (let e=0; e<epochs; e++) {

                    if (this.l2) this.l2Error = 0
                    if (this.l1) this.l1Error = 0

                    this.Module.ccall("train", "number", ["number", "number", "number"], [this.netInstance, -1, 0])
                    elapsed = Date.now() - startTime
                    if (log) {
                        console.log(`Epoch ${e+1} Error: ${this.error}${this.l2==undefined ? "": ` L2 Error: ${this.l2Error/data.length}`}`,
                                    `\nElapsed: ${NetUtil.format(elapsed, "time")} Average Duration: ${NetUtil.format(elapsed/(e+1), "time")}`)
                    }
                }
                this.Module._free(buf)
                if (log) {
                    console.log(`Training finished. Total time: ${NetUtil.format(elapsed, "time")}`)
                }
                resolve()
            }
        })
    }

    test (data, {log=true, callback}={}) {
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

            this.Module.ccall("loadTestingData", "number", ["number", "number", "number", "number", "number"],
                                            [this.netInstance, buf, itemsCount, itemSize, dimension])

            if (callback) {

                let iterationIndex = 0
                let totalError = 0

                const doIteration = () => {

                    totalError += this.Module.ccall("test", "number", ["number", "number", "number"], [this.netInstance, 1, iterationIndex])

                    callback({
                        iterations: (iterationIndex+1),
                        error: totalError/(iterationIndex+1),
                        elapsed: Date.now() - startTime,
                        input: data[iterationIndex].input
                    })

                    if (++iterationIndex < data.length) {
                        setTimeout(doIteration.bind(this), 0)
                    } else {
                        iterationIndex

                        const elapsed = Date.now() - startTime
                        log && console.log(`Testing finished. Total time: ${NetUtil.format(elapsed, "time")}  Average iteration time: ${NetUtil.format(elapsed/iterationIndex, "time")}`)

                        this.Module._free(buf)
                        resolve(totalError/data.length)
                    }
                }

                doIteration()

            } else {

                const avgError = this.Module.ccall("test", "number", ["number", "number"], [this.netInstance, -1, 0])
                this.Module._free(buf)

                const elapsed = Date.now() - startTime

                if (log) {
                    console.log(`Testing finished. Total time: ${NetUtil.format(elapsed, "time")}  Average iteration time: ${NetUtil.format(elapsed/data.length, "time")}`)
                }

                resolve(avgError)
            }
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

        const paramTypes = ["number", "number", "number"]
        const params = [netInstance, layerIndex, neuronIndex]

        NetUtil.defineProperty(this, "sum", paramTypes, params, {pre: "neuron_"})
        NetUtil.defineProperty(this, "dropped", paramTypes, params, {
            pre: "neuron_",
            getCallback: v => v==1,
            setCallback: v => v ? 1 : 0
        })
        NetUtil.defineProperty(this, "activation", paramTypes, params, {pre: "neuron_"})
        NetUtil.defineProperty(this, "error", paramTypes, params, {pre: "neuron_"})
        NetUtil.defineProperty(this, "derivative", paramTypes, params, {pre: "neuron_"})

        NetUtil.defineProperty(this, "bias", paramTypes, params, {pre: "neuron_"})
        NetUtil.defineArrayProperty(this, "weights", paramTypes, params, this.size, {pre: "neuron_"})

        NetUtil.defineProperty(this, "deltaBias", paramTypes, params, {pre: "neuron_"})
        NetUtil.defineArrayProperty(this, "deltaWeights", paramTypes, params, this.size, {pre: "neuron_"})

        switch (updateFn) {
            case "gain":
                NetUtil.defineProperty(this, "biasGain", paramTypes, params, {pre: "neuron_"})
                NetUtil.defineArrayProperty(this, "weightGain", paramTypes, params, this.size, {pre: "neuron_"})
                break
            case "adagrad":
            case "rmsprop":
            case "adadelta":
                NetUtil.defineProperty(this, "biasCache", paramTypes, params, {pre: "neuron_"})
                NetUtil.defineArrayProperty(this, "weightsCache", paramTypes, params, this.size, {pre: "neuron_"})

                if (updateFn=="adadelta") {
                    NetUtil.defineProperty(this, "adadeltaBiasCache", paramTypes, params, {pre: "neuron_"})
                    NetUtil.defineArrayProperty(this, "adadeltaCache", paramTypes, params, this.size, {pre: "neuron_"})
                }
                break

            case "adam":
                NetUtil.defineProperty(this, "m", paramTypes, params, {pre: "neuron_"})
                NetUtil.defineProperty(this, "v", paramTypes, params, {pre: "neuron_"})
                break
        }
    }
}

typeof window=="undefined" && (exports.Neuron = Neuron)
"use strict"

class PoolLayer {

    constructor (size, {stride, activation}={}) {

        if (size)   this.size = size
        if (stride) this.stride = stride

        if (activation != undefined && activation!=false) {
            if (typeof activation != "string") {
                throw new Error("Only string activation functions available in the WebAssembly version")
            }
            this.activation = NetUtil.format(activation)
        } else {
            this.activation = false
        }
    }

    assignNext (layer) {
        this.nextLayer = layer
    }

    assignPrev (layer, layerIndex) {

        this.netInstance = this.net.netInstance
        this.prevLayer = layer
        this.layerIndex = layerIndex

        let channels
        let prevLayerOutWidth = layer.outMapSize
        const size = this.size || this.net.pool.size || 2
        const stride = this.stride || this.net.pool.stride || this.size

        NetUtil.defineProperty(this, "channels", ["number", "number"], [this.netInstance, layerIndex], {pre: "pool_"})
        NetUtil.defineProperty(this, "stride", ["number", "number"], [this.netInstance, layerIndex], {pre: "pool_"})
        this.size = size
        this.stride = stride

        switch (true) {

            case layer instanceof FCLayer:
                channels = this.net.channels
                prevLayerOutWidth = Math.max(Math.floor(Math.sqrt(layer.size/channels)), 1)
                break

            case layer instanceof ConvLayer:
                channels = layer.size
                break

            case layer instanceof PoolLayer:
                channels = layer.channels
                break
        }

        this.channels = channels

        NetUtil.defineProperty(this, "prevLayerOutWidth", ["number", "number"], [this.netInstance, layerIndex], {pre: "pool_"})
        NetUtil.defineProperty(this, "inMapValuesCount", ["number", "number"], [this.netInstance, layerIndex], {pre: "pool_"})
        NetUtil.defineProperty(this, "outMapSize", ["number", "number"], [this.netInstance, layerIndex], {pre: "pool_"})
        NetUtil.defineVolumeProperty(this, "errors", ["number", "number"], [this.netInstance, layerIndex], channels, prevLayerOutWidth, prevLayerOutWidth, {pre: "pool_"})

        const outMapSize = (prevLayerOutWidth - size) / stride + 1
        this.outMapSize = outMapSize
        this.inMapValuesCount = prevLayerOutWidth ** 2

        NetUtil.defineVolumeProperty(this, "activations", ["number", "number"], [this.netInstance, layerIndex], channels, outMapSize, outMapSize, {pre: "pool_"})
        NetUtil.defineVolumeProperty(this, "indeces", ["number", "number"], [this.netInstance, layerIndex], channels, outMapSize, outMapSize, {
            pre: "pool_",
            getCallback: vol => vol.map(map => map.map(row => row.map(val => [parseInt(val/2), val%2]))),
            setCallback: vol => vol.map(map => map.map(row => row.map(([x,y]) => 2*x+y)))
        })

        if (outMapSize%1 != 0) {
            throw new Error(`Misconfigured hyperparameters. Activation volume dimensions would be ${outMapSize} in pool layer at index ${layerIndex}`)
        }
    }

    init () {}

    toJSON () {
        return {}
    }

    fromJSON() {}

}

typeof window=="undefined" && (exports.PoolLayer = PoolLayer)

//# sourceMappingURL=jsNetWebAssembly.concat.js.map