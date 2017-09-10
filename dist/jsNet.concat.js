"use strict"

class ConvLayer {

    constructor (size, {filterSize, zeroPadding, stride, activation}={}) {

        if (filterSize)     this.filterSize = filterSize
        if (stride)         this.stride = stride
        if (size)           this.size = size

        this.zeroPadding = zeroPadding

        if (activation!=undefined) {

            if (typeof activation=="boolean" && !activation) {
                this.activation = false
            } else {
                this.activation = typeof activation=="function" ? activation : NetMath[NetUtil.format(activation)].bind(this)
            }
        }

        this.state = "not-initialised"
    }

    assignNext (layer) {
        this.nextLayer = layer
    }

    assignPrev (layer, layerIndex) {

        this.prevLayer = layer

        this.size = this.size || 4
        this.filterSize = this.filterSize || this.net.conv.filterSize || 3
        this.stride = this.stride || this.net.conv.stride || 1

        switch (true) {
            case layer instanceof FCLayer:
                this.channels = this.net.channels ||1
                break

            case layer instanceof ConvLayer:
                this.channels = layer.size
                break

            case layer instanceof PoolLayer:
                this.channels = layer.activations.length
                break
        }

        if (this.zeroPadding==undefined) {
            this.zeroPadding = this.net.conv.zeroPadding==undefined ? Math.floor(this.filterSize/2) : this.net.conv.zeroPadding
        }

        // Caching calculations
        const prevLayerOutWidth = layer instanceof FCLayer ? Math.max(Math.floor(Math.sqrt(layer.size/this.channels)), 1)
                                                           : layer.outMapSize

        this.inMapValuesCount = Math.pow(prevLayerOutWidth, 2)
        this.inZPMapValuesCount = Math.pow(prevLayerOutWidth + this.zeroPadding*2, 2)
        this.outMapSize = (prevLayerOutWidth - this.filterSize + 2*this.zeroPadding) / this.stride + 1

        if (this.outMapSize%1!=0) {
            throw new Error(`Misconfigured hyperparameters. Activation volume dimensions would be ${this.outMapSize} in conv layer at index ${layerIndex}`)
        }

        this.filters = [...new Array(this.size)].map(f => new Filter())
    }

    init () {
        this.filters.forEach(filter => {

            filter.weights = [...new Array(this.channels)].map(channelWeights => {
                return [...new Array(this.filterSize)].map(weightsRow => this.net.weightsInitFn(this.filterSize * (this.prevLayer.channels||1), this.weightsConfig))
            })

            filter.activationMap = [...new Array(this.outMapSize)].map(row => [...new Array(this.outMapSize)].map(v => 0))
            filter.errorMap = [...new Array(this.outMapSize)].map(row => [...new Array(this.outMapSize)].map(v => 0))
            filter.dropoutMap = filter.activationMap.map(row => row.map(v => false))
            filter.bias = Math.random()*0.2-0.1

            filter.init({
                updateFn: this.net.updateFn,
                activation: this.net.activationConfig,
                eluAlpha: this.net.eluAlpha
            })
        })
    }

    forward () {

        const activations = NetUtil.getActivations(this.prevLayer)

        for (let filterI=0; filterI<this.size; filterI++) {

            const filter = this.filters[filterI]

            filter.sumMap = NetUtil.convolve({
                input: activations,
                zeroPadding: this.zeroPadding,
                weights: filter.weights,
                channels: this.channels,
                stride: this.stride,
                bias: filter.bias
            })

            for (let sumY=0; sumY<filter.sumMap.length; sumY++) {
                for (let sumX=0; sumX<filter.sumMap.length; sumX++) {
                    if (this.state=="training" && (filter.dropoutMap[sumY][sumX] = Math.random() > this.net.dropout)) {
                        filter.activationMap[sumY][sumX] = 0
                    } else if (this.activation) {
                        filter.activationMap[sumY][sumX] = this.activation(filter.sumMap[sumY][sumX], false, filter) / (this.net.dropout||1)
                    } else {
                        filter.activationMap[sumY][sumX] = filter.sumMap[sumY][sumX]
                    }
                }
            }
        }
    }

    backward () {

        // First, get the filters' error maps
        if (this.nextLayer instanceof FCLayer) {

            // For each filter, build the errorMap from the weighted neuron errors in the next FCLayer corresponding to each value in the activation map
            for (let filterI=0; filterI<this.filters.length; filterI++) {

                const filter = this.filters[filterI]

                for (let emY=0; emY<filter.errorMap.length; emY++) {
                    for (let emX=0; emX<filter.errorMap.length; emX++) {

                        const weightIndex = filterI * this.outMapSize**2 + emY * filter.errorMap.length + emX

                        for (let neuronI=0; neuronI<this.nextLayer.neurons.length; neuronI++) {

                            const neuron = this.nextLayer.neurons[neuronI]
                            filter.errorMap[emY][emX] += neuron.error * neuron.weights[weightIndex]
                        }
                    }
                }
            }

        } else if (this.nextLayer instanceof ConvLayer) {

            for (let filterI=0; filterI<this.filters.length; filterI++) {
                NetUtil.buildConvErrorMap(this.nextLayer, this.filters[filterI].errorMap, filterI)
            }

        } else {

            for (let filterI=0; filterI<this.filters.length; filterI++) {

                const filter = this.filters[filterI]

                for (let row=0; row<filter.errorMap.length; row++) {
                    for (let col=0; col<filter.errorMap.length; col++) {
                        filter.errorMap[row][col] = this.nextLayer.errors[filterI][row][col]
                    }
                }
            }
        }

        // Apply derivative to each error value
        for (let filterI=0; filterI<this.filters.length; filterI++) {

            const filter = this.filters[filterI]

            for (let row=0; row<filter.errorMap.length; row++) {
                for (let col=0; col<filter.errorMap[0].length; col++) {

                    if (filter.dropoutMap[row][col]) {
                        filter.errorMap[row][col] = 0
                    } else if (this.activation){
                        filter.errorMap[row][col] *= this.activation(filter.sumMap[row][col], true, filter)
                    }
                }
            }
        }

        // Then use the error map values to build the delta weights
        NetUtil.buildConvDWeights(this)
    }

    resetDeltaWeights () {
        for (let filterI=0; filterI<this.filters.length; filterI++) {

            const filter = this.filters[filterI]

            for (let channel=0; channel<filter.deltaWeights.length; channel++) {
                for (let row=0; row<filter.deltaWeights[0].length; row++) {
                    for (let col=0; col<filter.deltaWeights[0][0].length; col++) {
                        filter.deltaWeights[channel][row][col] = 0
                    }
                }
            }

            for (let row=0; row<filter.dropoutMap.length; row++) {
                for (let col=0; col<filter.dropoutMap[0].length; col++) {
                    filter.dropoutMap[row][col] = false
                }
            }
        }
    }

    applyDeltaWeights () {
        for (let filterI=0; filterI<this.filters.length; filterI++) {

            const filter = this.filters[filterI]

            for (let channel=0; channel<filter.deltaWeights.length; channel++) {
                for (let row=0; row<filter.deltaWeights[0].length; row++) {
                    for (let col=0; col<filter.deltaWeights[0][0].length; col++) {

                        if (this.net.l2!=undefined) this.net.l2Error += 0.5 * this.net.l2 * filter.weights[channel][row][col]**2
                        if (this.net.l1!=undefined) this.net.l1Error += this.net.l1 * Math.abs(filter.weights[channel][row][col])

                        filter.weights[channel][row][col] = this.net.weightUpdateFn.bind(this.net, filter.weights[channel][row][col],
                                                                filter.deltaWeights[channel][row][col], filter, [channel, row, col])()

                        if (this.net.maxNorm!=undefined) this.net.maxNormTotal += filter.weights[channel][row][col]**2
                    }
                }
            }

            filter.bias = this.net.weightUpdateFn.bind(this.net, filter.bias, filter.deltaBias, filter)()
        }
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
        this.state = "not-initialised"
    }

    assignNext (layer) {
        this.nextLayer = layer
    }

    assignPrev (layer) {
        this.prevLayer = layer
    }

    init () {
        this.neurons.forEach(neuron => {

            let weightsCount

            switch (this.prevLayer.constructor.name) {
                case "FCLayer":
                    weightsCount = this.prevLayer.size
                    break

                case "ConvLayer":
                    weightsCount = this.prevLayer.filters.length * this.prevLayer.outMapSize**2
                    break

                case "PoolLayer":
                    weightsCount = this.prevLayer.activations.length * this.prevLayer.outMapSize**2
                    break
            }

            neuron.weights = this.net.weightsInitFn(weightsCount, this.weightsConfig)
            neuron.bias = Math.random()*0.2-0.1

            neuron.init({
                updateFn: this.net.updateFn,
                activationConfig: this.net.activationConfig,
                eluAlpha: this.net.eluAlpha
            })
        })
    }

    forward () {
        this.neurons.forEach((neuron, ni) => {
            if (this.state=="training" && (neuron.dropped = Math.random() > this.net.dropout)) {
                neuron.activation = 0
            } else {
                neuron.sum = neuron.bias

                const activations = NetUtil.getActivations(this.prevLayer)

                for (let ai=0; ai<activations.length; ai++) {
                    neuron.sum += activations[ai] * neuron.weights[ai]
                }

                neuron.activation = this.activation(neuron.sum, false, neuron) / (this.net.dropout||1)
            }
        })
    }

    backward (expected) {
        this.neurons.forEach((neuron, ni) => {

            if (neuron.dropped) {
                neuron.error = 0
                neuron.deltaBias = 0
            } else {
                if (typeof expected !== "undefined") {
                    neuron.error = expected[ni] - neuron.activation
                } else {
                    neuron.derivative = this.activation(neuron.sum, true, neuron)
                    neuron.error = neuron.derivative * this.nextLayer.neurons.map(n => n.error * (n.weights[ni]|0))
                                                                             .reduce((p,c) => p+c, 0)
                }

                const activations = NetUtil.getActivations(this.prevLayer)

                for (let wi=0; wi<neuron.weights.length; wi++) {
                    neuron.deltaWeights[wi] += (neuron.error * activations[wi]) *
                        (1 + (((this.net.l2||0)+(this.net.l1||0))/this.net.miniBatchSize) * neuron.deltaWeights[wi])
                }

                neuron.deltaBias = neuron.error
            }
        })
    }

    resetDeltaWeights () {
        this.neurons.forEach(neuron => neuron.deltaWeights = neuron.weights.map(dw => 0))
    }

    applyDeltaWeights () {
        this.neurons.forEach(neuron => {
            neuron.deltaWeights.forEach((dw, dwi) => {

                if (this.net.l2!=undefined) this.net.l2Error += 0.5 * this.net.l2 * neuron.weights[dwi]**2
                if (this.net.l1!=undefined) this.net.l1Error += this.net.l1 * Math.abs(neuron.weights[dwi])

                neuron.weights[dwi] = this.net.weightUpdateFn.bind(this.net, neuron.weights[dwi], dw, neuron, dwi)()

                if (this.net.maxNorm!=undefined) this.net.maxNormTotal += neuron.weights[dwi]**2
            })

            neuron.bias = this.net.weightUpdateFn.bind(this.net, neuron.bias, neuron.deltaBias, neuron)()
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

            if (data.weights[ni].weights.length!=neuron.weights.length) {
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

    init ({updateFn, activation, eluAlpha}={}) {

        const size = this.weights.length

        this.deltaWeights = this.weights.map(channel => channel.map(wRow => wRow.map(w => 0)))
        this.deltaBias = 0

        switch (updateFn) {

            case "gain":
                this.biasGain = 1
                this.weightGains = this.weights.map(channel => channel.map(wRow => wRow.map(w => 1)))
                this.getWeightGain = ([channel, row, column]) => this.weightGains[channel][row][column]
                this.setWeightGain = ([channel, row, column], v) => this.weightGains[channel][row][column] = v
                break

            case "adagrad":
            case "rmsprop":
            case "adadelta":
                this.biasCache = 0
                this.weightsCache = this.weights.map(channel => channel.map(wRow => wRow.map(w => 0)))
                this.getWeightsCache = ([channel, row, column]) => this.weightsCache[channel][row][column]
                this.setWeightsCache = ([channel, row, column], v) => this.weightsCache[channel][row][column] = v

                if (updateFn=="adadelta") {
                    this.adadeltaBiasCache = 0
                    this.adadeltaCache = this.weights.map(channel => channel.map(wRow => wRow.map(w => 0)))
                    this.getAdadeltaCache = ([channel, row, column]) => this.adadeltaCache[channel][row][column]
                    this.setAdadeltaCache = ([channel, row, column], v) => this.adadeltaCache[channel][row][column] = v
                }
                break

            case "adam":
                this.m = 0
                this.v = 0
        }

        if (activation=="rrelu") {
            this.rreluSlope = Math.random() * 0.001

        } else if (activation=="elu") {
            this.eluAlpha = eluAlpha
        }
    }

    getWeight ([channel, row, column]) {
        return this.weights[channel][row][column]
    }

    setWeight ([channel, row, column], v) {
        this.weights[channel][row][column] = v
    }

    getDeltaWeight ([channel, row, column]) {
        return this.deltaWeights[channel][row][column]
    }

    setDeltaWeight ([channel, row, column], v) {
        this.deltaWeights[channel][row][column] = v
    }
}

typeof window=="undefined" && (exports.Filter = Filter)



"use strict"

class NetMath {

    // Activation functions
    static sigmoid (value, prime) {
        const val = 1/(1+Math.exp(-value))
        return prime ? val*(1-val)
                     : val
    }

    static tanh (value, prime) {
        const exp = Math.exp(2*value)
        return prime ? 4/Math.pow(Math.exp(value)+Math.exp(-value), 2) || 1e-18
                     : (exp-1)/(exp+1) || 1e-18
    }

    static relu (value, prime) {
        return prime ? value > 0 ? 1 : 0
                     : Math.max(value, 0)
    }

    static lrelu (value, prime) {
        return prime ? value > 0 ? 1 : (this.lreluSlope || -0.0005)
                     : Math.max((this.lreluSlope || -0.0005)*Math.abs(value), value)
    }

    static rrelu (value, prime, neuron) {
        return prime ? value > 0 ? 1 : neuron.rreluSlope
                     : Math.max(neuron.rreluSlope, value)
    }

    static lecuntanh (value, prime) {
        return prime ? 1.15333 * Math.pow(NetMath.sech((2/3) * value), 2)
                     : 1.7159 * NetMath.tanh((2/3) * value)
    }

    static elu (value, prime, neuron) {
        return prime ? value >=0 ? 1 : NetMath.elu(value, false, neuron) + neuron.eluAlpha
                     : value >=0 ? value : neuron.eluAlpha * (Math.exp(value) - 1)
    }

    // Cost functions
    static crossentropy (target, output) {
        return output.map((value, vi) => target[vi] * Math.log(value+1e-15) + ((1-target[vi]) * Math.log((1+1e-15)-value)))
                     .reduce((p,c) => p-c, 0)
    }

    static meansquarederror (calculated, desired) {
        return calculated.map((output, index) => Math.pow(output - desired[index], 2))
                         .reduce((prev, curr) => prev+curr, 0) / calculated.length
    }

    // Weight updating functions
    static vanillaupdatefn (value, deltaValue) {
        return value + this.learningRate * deltaValue
    }

    static gain (value, deltaValue, neuron, weightI) {

        const newVal = value + this.learningRate * deltaValue * (weightI==null ? neuron.biasGain : neuron.getWeightGain(weightI))

        if (newVal<=0 && value>0 || newVal>=0 && value<0){
            if (weightI!=null) {
                neuron.setWeightGain(weightI, Math.max(neuron.getWeightGain(weightI)*0.95, 0.5))
            } else {
                neuron.biasGain = Math.max(neuron.biasGain*0.95, 0.5)
            }
        } else {
            if (weightI!=null) {
                neuron.setWeightGain(weightI, Math.min(neuron.getWeightGain(weightI)+0.05, 5))
            } else {
                neuron.biasGain = Math.min(neuron.biasGain+0.05, 5)
            }
        }

        return newVal
    }

    static adagrad (value, deltaValue, neuron, weightI) {

        if (weightI!=null) {
            neuron.setWeightsCache(weightI, neuron.getWeightsCache(weightI) + Math.pow(deltaValue, 2))
        } else {
            neuron.biasCache += Math.pow(deltaValue, 2)
        }

        return value + this.learningRate * deltaValue / (1e-6 + Math.sqrt(weightI!=null ? neuron.getWeightsCache(weightI)
                                                                                        : neuron.biasCache))
    }

    static rmsprop (value, deltaValue, neuron, weightI) {

        if (weightI!=null) {
            neuron.setWeightsCache(weightI, this.rmsDecay * neuron.getWeightsCache(weightI) + (1 - this.rmsDecay) * Math.pow(deltaValue, 2))
        } else {
            neuron.biasCache = this.rmsDecay * neuron.biasCache + (1 - this.rmsDecay) * Math.pow(deltaValue, 2)
        }

        return value + this.learningRate * deltaValue / (1e-6 + Math.sqrt(weightI!=null ? neuron.getWeightsCache(weightI)
                                                                                        : neuron.biasCache))
    }

    static adam (value, deltaValue, neuron) {

        neuron.m = 0.9*neuron.m + (1-0.9) * deltaValue
        const mt = neuron.m / (1-Math.pow(0.9, this.iterations + 1))

        neuron.v = 0.999*neuron.v + (1-0.999)*(Math.pow(deltaValue, 2))
        const vt = neuron.v / (1-Math.pow(0.999, this.iterations + 1))

        return value + this.learningRate * mt / (Math.sqrt(vt) + 1e-8)
    }

    static adadelta (value, deltaValue, neuron, weightI) {

        if (weightI!=null) {
            neuron.setWeightsCache(weightI, this.rho * neuron.getWeightsCache(weightI) + (1-this.rho) * Math.pow(deltaValue, 2))
            const newVal = value + Math.sqrt((neuron.getAdadeltaCache(weightI) + 1e-6)/(neuron.getWeightsCache(weightI) + 1e-6)) * deltaValue
            neuron.setAdadeltaCache(weightI, this.rho * neuron.getAdadeltaCache(weightI) + (1-this.rho) * Math.pow(deltaValue, 2))
            return newVal

        } else {
            neuron.biasCache = this.rho * neuron.biasCache + (1-this.rho) * Math.pow(deltaValue, 2)
            const newVal = value + Math.sqrt((neuron.adadeltaBiasCache + 1e-6)/(neuron.biasCache + 1e-6)) * deltaValue
            neuron.adadeltaBiasCache = this.rho * neuron.adadeltaBiasCache + (1-this.rho) * Math.pow(deltaValue, 2)
            return newVal
        }
    }

    // Weights init
    static uniform (size, {limit}) {
        const values = []

        for (let i=0; i<size; i++) {
            values.push(Math.random()*2*limit-limit)
        }

        return values
    }

    static gaussian (size, {mean, stdDeviation}) {
        const values = []

        // Polar Box Muller
        for (let i=0; i<size; i++) {
            let x1, x2, r, y

            do {
                x1 = 2 * Math.random() -1
                x2 = 2 * Math.random() -1
                r = x1**2 + x2**2
            } while (r >= 1 || !r)

            values.push(mean + (x1 * (Math.sqrt(-2 * Math.log(r) / r))) * stdDeviation)
        }

        return values
    }

    static xaviernormal (size, {fanIn, fanOut}) {
        return fanOut || fanOut==0 ? NetMath.gaussian(size, {mean: 0, stdDeviation: Math.sqrt(2/(fanIn+fanOut))})
                                   : NetMath.lecunnormal(size, {fanIn})
    }

    static xavieruniform (size, {fanIn, fanOut}) {
        return fanOut || fanOut==0 ? NetMath.uniform(size, {limit: Math.sqrt(6/(fanIn+fanOut))})
                                   : NetMath.lecununiform(size, {fanIn})
    }

    static lecunnormal (size, {fanIn}) {
        return NetMath.gaussian(size, {mean: 0, stdDeviation: Math.sqrt(1/fanIn)})
    }

    static lecununiform (size, {fanIn}) {
        return NetMath.uniform(size, {limit: Math.sqrt(3/fanIn)})
    }

    // Pool
    static maxPool (layer, channel) {

        const activations = NetUtil.getActivations(layer.prevLayer, channel, layer.inMapValuesCount)

        for (let row=0; row<layer.outMapSize; row++) {
            for (let col=0; col<layer.outMapSize; col++) {

                const rowStart = row * layer.stride
                const colStart = col * layer.stride

                // The first value
                let activation = activations[rowStart*layer.prevLayerOutWidth + colStart]

                for (let filterRow=0; filterRow<layer.size; filterRow++) {
                    for (let filterCol=0; filterCol<layer.size; filterCol++) {

                        const value = activations[ ((rowStart+filterRow) * layer.prevLayerOutWidth) + (colStart+filterCol) ]

                        if (value > activation) {
                            activation = value
                            layer.indeces[channel][row][col] = [filterRow, filterCol]
                        }
                    }
                }

                layer.activations[channel][row][col] = activation
            }
        }
    }

    // Other
    static softmax (values) {
        const total = values.reduce((prev, curr) => prev+curr, 0)
        return values.map(value => value/total)
    }

    static sech (value) {
        return (2*Math.exp(-value))/(1+Math.exp(-2*value))
    }

    static standardDeviation (arr) {
        const avg = arr.reduce((p,c) => p+c) / arr.length
        const diffs = arr.map(v => v - avg).map(v => v**2)
        return Math.sqrt(diffs.reduce((p,c) => p+c) / diffs.length)
    }

    static maxNorm () {

        if (this.maxNormTotal > this.maxNorm) {

            const multiplier = this.maxNorm / (1e-18 + this.maxNormTotal)

            this.layers.forEach((layer, li) => {
                li && layer.neurons.forEach(neuron => {
                    neuron.weights.forEach((w, wi) => neuron.setWeight(wi, neuron.getWeight(wi) * multiplier))
                })
            })
        }

        this.maxNormTotal = 0
    }
}

typeof window=="undefined" && (exports.NetMath = NetMath)
"use strict"

class NetUtil {

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

                } else {

                    if (value >= 3600000) formatted.push(`${date.getHours()}h`)
                    if (value >= 60000)   formatted.push(`${date.getMinutes()}m`)

                    formatted.push(`${date.getSeconds()}s`)
                }

                value = formatted.join(" ")
                break
        }

        return value
    }

    static shuffle (arr) {
        for (let i=arr.length; i; i--) {
            const j = Math.floor(Math.random() * i)
            const x = arr[i-1]
            arr[i-1] = arr[j]
            arr[j] = x
        }
    }

    static addZeroPadding (map, zP) {

        const data = []

        for (let row=0; row<map.length; row++) {
            data.push(map[row].slice(0))
        }

        const extraRows = []

        for (let i=0; i<data.length+2*zP; i++) {
            extraRows.push(0)
        }

        for (let col=0; col<data.length; col++) {
            for (let i=0; i<zP; i++) {
                data[col].splice(0, 0, 0)
                data[col].splice(data.length+1, data.length, 0)
            }
        }

        for (let i=0; i<zP; i++) {
            data.splice(0, 0, extraRows.slice(0))
            data.splice(data.length, data.length-1, extraRows.slice(0))
        }

        return data
    }

    static arrayToMap (arr, size) {
        const map = []

        for (let i=0; i<size; i++) {
            map[i] = []

            for (let j=0; j<size; j++) {
                map[i][j] = arr[i*size+j]
            }
        }

        return map
    }

    static arrayToVolume (arr, channels) {

        const vol = []
        const size = Math.sqrt(arr.length/channels)
        const mapValues = size**2

        for (let d=0; d<Math.floor(arr.length/mapValues); d++) {

            const map = []

            for (let i=0; i<size; i++) {
                map[i] = []

                for (let j=0; j<size; j++) {
                    map[i][j] = arr[d*mapValues  + i*size+j]
                }
            }

            vol[d] = map
        }

        return vol
    }

    static convolve ({input, zeroPadding, weights, channels, stride, bias}) {

        const inputVol = NetUtil.arrayToVolume(input, channels)
        const outputMap = []

        const paddedLength = inputVol[0].length + zeroPadding*2
        const fSSpread = Math.floor(weights[0].length / 2)

        // For each input channels,
        for (let di=0; di<channels; di++) {
            inputVol[di] = NetUtil.addZeroPadding(inputVol[di], zeroPadding)
            // For each inputY without ZP
            for (let inputY=fSSpread; inputY<paddedLength-fSSpread; inputY+=stride) {
                outputMap[(inputY-fSSpread)/stride] = outputMap[(inputY-fSSpread)/stride] || []
                // For each inputX without zP
                for (let inputX=fSSpread; inputX<paddedLength-fSSpread; inputX+=stride) {
                    let sum = 0
                    // For each weightsY on input
                    for (let weightsY=0; weightsY<weights[0].length; weightsY++) {
                        // For each weightsX on input
                        for (let weightsX=0; weightsX<weights[0].length; weightsX++) {
                            sum += inputVol[di][inputY+(weightsY-fSSpread)][inputX+(weightsX-fSSpread)] * weights[di][weightsY][weightsX]
                        }
                    }

                    outputMap[(inputY-fSSpread)/stride][(inputX-fSSpread)/stride] = (outputMap[(inputY-fSSpread)/stride][(inputX-fSSpread)/stride]||0) + sum
                }
            }
        }

        // Then add bias
        for (let outY=0; outY<outputMap.length; outY++) {
            for (let outX=0; outX<outputMap.length; outX++) {
                outputMap[outY][outX] += bias
            }
        }

        return outputMap
    }

    static buildConvErrorMap (nextLayer, errorMap, filterI) {

        // Cache / convenience
        const zeroPadding = nextLayer.zeroPadding
        const paddedLength = errorMap.length + zeroPadding*2
        const fSSpread = Math.floor(nextLayer.filterSize / 2)

        // Zero pad and clear the error map, to allow easy convoling
        const paddedRow = []

        for (let val=0; val<paddedLength; val++) {
            paddedRow.push(0)
        }

        for (let row=0; row<paddedLength; row++) {
            errorMap[row] = paddedRow.slice(0)
        }

        // For each channel in filter in the next layer which corresponds to this filter
        for (let nlFilterI=0; nlFilterI<nextLayer.size; nlFilterI++) {

            const weights = nextLayer.filters[nlFilterI].weights[filterI]
            const errMap = nextLayer.filters[nlFilterI].errorMap

            // Unconvolve their error map using the weights
            for (let inputY=fSSpread; inputY<paddedLength - fSSpread; inputY+=nextLayer.stride) {
                for (let inputX=fSSpread; inputX<paddedLength - fSSpread; inputX+=nextLayer.stride) {

                    for (let weightsY=0; weightsY<nextLayer.filterSize; weightsY++) {
                        for (let weightsX=0; weightsX<nextLayer.filterSize; weightsX++) {
                            errorMap[inputY+(weightsY-fSSpread)][inputX+(weightsX-fSSpread)] += weights[weightsY][weightsX]
                                * errMap[(inputY-fSSpread)/nextLayer.stride][(inputX-fSSpread)/nextLayer.stride]
                        }
                    }
                }
            }
        }

        // Take out the zero padding. Rows:
        errorMap.splice(0, zeroPadding)
        errorMap.splice(errorMap.length-zeroPadding, errorMap.length)

        // Columns:
        for (let emXI=0; emXI<errorMap.length; emXI++) {
            errorMap[emXI] = errorMap[emXI].splice(zeroPadding, errorMap[emXI].length - zeroPadding*2)
        }
    }

    static buildConvDWeights (layer) {

        const weightsCount = layer.filters[0].weights[0].length
        const fSSpread = Math.floor(weightsCount / 2)
        const channelsCount = layer.filters[0].weights.length

        // Adding an intermediary step to allow regularization to work
        const deltaDeltaWeights = []

        // Filling the deltaDeltaWeights with 0 values
        for (let weightsY=0; weightsY<weightsCount; weightsY++) {
            deltaDeltaWeights[weightsY] = []
            for (let weightsX=0; weightsX<weightsCount; weightsX++) {
                deltaDeltaWeights[weightsY][weightsX] = 0
            }
        }

        // For each filter
        for (let filterI=0; filterI<layer.filters.length; filterI++) {

            const filter = layer.filters[filterI]

            // Each channel will take the error map and the corresponding inputMap from the input...
            for (let channelI=0; channelI<channelsCount; channelI++) {

                const inputValues = NetUtil.getActivations(layer.prevLayer, channelI, layer.inMapValuesCount)
                const inputMap = NetUtil.addZeroPadding(NetUtil.arrayToMap(inputValues, Math.sqrt(layer.inMapValuesCount)), layer.zeroPadding)

                // ...slide the filter with correct stride across the zero-padded inputMap...
                for (let inputY=fSSpread; inputY<inputMap.length-fSSpread; inputY+=layer.stride) {
                    for (let inputX=fSSpread; inputX<inputMap.length-fSSpread; inputX+=layer.stride) {

                        // ...and at each location...
                        for (let weightsY=0; weightsY<weightsCount; weightsY++) {
                            for (let weightsX=0; weightsX<weightsCount; weightsX++) {

                                const activation = inputMap[inputY-fSSpread+weightsY][inputX-fSSpread+weightsX]

                                // Increment and regularize the delta delta weights by the input activation (later multiplied by the error)
                                deltaDeltaWeights[weightsY][weightsX] += activation *
                                     (1 + (((layer.net.l2||0)+(layer.net.l1||0))/layer.net.miniBatchSize) * filter.weights[channelI][weightsY][weightsX])
                            }
                        }

                        const error = filter.errorMap[(inputY-fSSpread)/layer.stride][(inputX-fSSpread)/layer.stride]

                        // Applying and resetting the deltaDeltaWeights
                        for (let weightsY=0; weightsY<weightsCount; weightsY++) {
                            for (let weightsX=0; weightsX<weightsCount; weightsX++) {
                                filter.deltaWeights[channelI][weightsY][weightsX] += deltaDeltaWeights[weightsY][weightsX] * error
                                deltaDeltaWeights[weightsY][weightsX] = 0
                            }
                        }
                    }
                }
            }

            // Increment the deltaBias by the sum of all errors in the filter
            for (let eY=0; eY<filter.errorMap.length; eY++) {
                for (let eX=0; eX<filter.errorMap.length; eX++) {
                    filter.deltaBias += filter.errorMap[eY][eX]
                }
            }
        }
    }

    static getActivations (layer, mapStartI, mapSize){

        const returnArr = []

        if (arguments.length==1) {

            if (layer instanceof FCLayer) {

                for (let ni=0; ni<layer.neurons.length; ni++) {
                    returnArr.push(layer.neurons[ni].activation)
                }

            } else if (layer instanceof ConvLayer) {

                for (let fi=0; fi<layer.filters.length; fi++) {
                    for (let rowI=0; rowI<layer.filters[fi].activationMap.length; rowI++) {
                        for (let colI=0; colI<layer.filters[fi].activationMap[rowI].length; colI++) {
                            returnArr.push(layer.filters[fi].activationMap[rowI][colI])
                        }
                    }
                }

            } else {

                for (let channel=0; channel<layer.activations.length; channel++) {
                    for (let row=0; row<layer.activations[0].length; row++) {
                        for (let col=0; col<layer.activations[0].length; col++) {
                            returnArr.push(layer.activations[channel][row][col])
                        }
                    }
                }
            }

        } else {

            if (layer instanceof FCLayer) {

                for (let i=mapStartI*mapSize; i<(mapStartI+1)*mapSize; i++) {
                    returnArr.push(layer.neurons[i].activation)
                }

            } else if (layer instanceof ConvLayer) {

                for (let row=0; row<layer.filters[mapStartI].activationMap.length; row++) {
                    for (let col=0; col<layer.filters[mapStartI].activationMap[row].length; col++) {
                        returnArr.push(layer.filters[mapStartI].activationMap[row][col])
                    }
                }

            } else {

                for (let row=0; row<layer.activations[mapStartI].length; row++) {
                    for (let col=0; col<layer.activations[mapStartI].length; col++) {
                        returnArr.push(layer.activations[mapStartI][row][col])
                    }
                }
            }
        }

        return returnArr
    }
}

typeof window=="undefined" && (exports.NetUtil = NetUtil)
"use strict"

class Network {

    constructor ({learningRate, layers=[], updateFn="vanillaupdatefn", activation="sigmoid", cost="meansquarederror",
        rmsDecay, rho, lreluSlope, eluAlpha, dropout=1, l2=true, l1=true, maxNorm, weightsConfig, channels, conv, pool}={}) {

        this.state = "not-defined"
        this.layers = []
        this.conv = {}
        this.pool = {}
        this.epochs = 0
        this.iterations = 0
        this.dropout = dropout==false ? 1 : dropout
        this.error = 0
        activation = NetUtil.format(activation)
        updateFn = NetUtil.format(updateFn)
        cost = NetUtil.format(cost)

        if (l2) {
            this.l2 = typeof l2=="boolean" ? 0.001 : l2
            this.l2Error = 0
        }

        if (l1) {
            this.l1 = typeof l1=="boolean" ? 0.005 : l1
            this.l1Error = 0
        }

        if (maxNorm) {
            this.maxNorm = typeof maxNorm=="boolean" && maxNorm ? 1000 : maxNorm
            this.maxNormTotal = 0
        }

        if (learningRate)   this.learningRate = learningRate
        if (channels)       this.channels = channels

        if (conv) {
            if (conv.filterSize!=undefined)     this.conv.filterSize = conv.filterSize
            if (conv.zeroPadding!=undefined)    this.conv.zeroPadding = conv.zeroPadding
            if (conv.stride!=undefined)         this.conv.stride = conv.stride
        }

        if (pool) {
            if (pool.size)      this.pool.size = pool.size
            if (pool.stride)    this.pool.stride = pool.stride
        }

        // Activation function / Learning Rate
        switch (updateFn) {

            case "rmsprop":
                this.learningRate = this.learningRate==undefined ? 0.001 : this.learningRate
                break

            case "adam":
                this.learningRate = this.learningRate==undefined ? 0.01 : this.learningRate
                break

            case "adadelta":
                this.rho = rho==null ? 0.95 : rho
                break

            default:

                if (this.learningRate==undefined) {

                    switch (activation) {

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

        this.updateFn = [false, null, undefined].includes(updateFn) ? "vanillaupdatefn" : updateFn
        this.weightUpdateFn = NetMath[this.updateFn]
        this.activation = typeof activation=="function" ? activation : NetMath[activation].bind(this)
        this.activationConfig = activation
        this.cost = typeof cost=="function" ? cost : NetMath[cost]

        if (this.updateFn=="rmsprop") {
            this.rmsDecay = rmsDecay==undefined ? 0.99 : rmsDecay
        }

        this.lreluSlope = lreluSlope==undefined ? -0.0005 : lreluSlope
        this.eluAlpha = eluAlpha==undefined ? 1 : eluAlpha

        // Weights distributiom
        this.weightsConfig = {distribution: "xavieruniform"}

        if (weightsConfig != undefined && weightsConfig.distribution) {
            this.weightsConfig.distribution = NetUtil.format(weightsConfig.distribution)
        }

        if (this.weightsConfig.distribution == "uniform") {
            this.weightsConfig.limit = weightsConfig && weightsConfig.limit!=undefined ? weightsConfig.limit : 0.1

        } else if (this.weightsConfig.distribution == "gaussian") {
            this.weightsConfig.mean = weightsConfig.mean || 0
            this.weightsConfig.stdDeviation = weightsConfig.stdDeviation || 0.05
        }

        if (typeof this.weightsConfig.distribution=="function") {
            this.weightsInitFn = this.weightsConfig.distribution
        } else {
            this.weightsInitFn = NetMath[this.weightsConfig.distribution]
        }

        // State
        if (layers.length) {

            switch (true) {

                case layers.every(item => Number.isInteger(item)):
                    this.layers = layers.map(size => new FCLayer(size))
                    this.state = "constructed"
                    this.initLayers()
                    break

                case layers.every(layer => layer instanceof FCLayer || layer instanceof ConvLayer || layer instanceof PoolLayer):
                    this.state = "constructed"
                    this.layers = layers
                    this.initLayers()
                    break

                default:
                    throw new Error("There was an error constructing from the layers given.")
            }
        }
    }

    initLayers (input, expected) {

        switch (this.state) {

            case "initialised":
                return

            case "not-defined":
                this.layers[0] = new FCLayer(input)
                this.layers[1] = new FCLayer(Math.ceil(input/expected > 5 ? expected + (Math.abs(input-expected))/4
                                                                          : input + expected))
                this.layers[2] = new FCLayer(Math.ceil(expected))
                break
        }

        this.layers.forEach(this.joinLayer.bind(this))
        this.state = "initialised"
    }

    joinLayer (layer, layerIndex) {

        layer.net = this
        layer.activation = layer.activation==undefined ? this.activation : layer.activation

        layer.weightsConfig = {}
        Object.assign(layer.weightsConfig, this.weightsConfig)

        if (layerIndex) {
            this.layers[layerIndex-1].assignNext(layer)
            layer.assignPrev(this.layers[layerIndex-1], layerIndex)

            layer.weightsConfig.fanIn = layer.prevLayer.size
            layer.prevLayer.weightsConfig.fanOut = layer.size

            layer.init()
            layer.state = "initialised"
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

        this.layers[0].neurons.forEach((neuron, ni) => neuron.activation = data[ni])
        this.layers.forEach((layer, li) => li && layer.forward(data))
        return this.layers[this.layers.length-1].neurons.map(n => n.activation)
    }

    backward (expected) {

        if (expected === undefined) {
            throw new Error("No data passed to Network.backward()")
        }

        if (expected.length != this.layers[this.layers.length-1].neurons.length) {
            console.warn("Expected data length did not match output layer neurons count.", expected)
        }

        this.layers[this.layers.length-1].backward(expected)

        for (let layerIndex=this.layers.length-2; layerIndex>0; layerIndex--) {
            this.layers[layerIndex].backward()
        }
    }

    train (dataSet, {epochs=1, callback, log=true, miniBatchSize=1, shuffle=false}={}) {

        this.miniBatchSize = typeof miniBatchSize=="boolean" && miniBatchSize ? dataSet[0].expected.length : miniBatchSize

        return new Promise((resolve, reject) => {

            if (shuffle) {
                NetUtil.shuffle(dataSet)
            }

            if (log) {
                console.log(`Training started. Epochs: ${epochs} Batch Size: ${this.miniBatchSize}`)
            }

            if (dataSet === undefined || dataSet === null) {
                return void reject("No data provided")
            }

            if (this.state != "initialised") {
                this.initLayers(dataSet[0].input.length, (dataSet[0].expected || dataSet[0].output).length)
            }

            this.layers.forEach(layer => layer.state = "training")

            let iterationIndex = 0
            let epochsCounter = 0
            const startTime = Date.now()

            const doEpoch = () => {
                this.epochs++
                this.error = 0
                iterationIndex = 0

                if (this.l2Error!=undefined) this.l2Error = 0
                if (this.l1Error!=undefined) this.l1Error = 0

                doIteration()
            }

            const doIteration = () => {

                if (!dataSet[iterationIndex].hasOwnProperty("input") || (!dataSet[iterationIndex].hasOwnProperty("expected") && !dataSet[iterationIndex].hasOwnProperty("output"))) {
                    return void reject("Data set must be a list of objects with keys: 'input' and 'expected' (or 'output')")
                }

                const input = dataSet[iterationIndex].input
                const output = this.forward(input)
                const target = dataSet[iterationIndex].expected || dataSet[iterationIndex].output

                this.backward(target)

                if (++iterationIndex%this.miniBatchSize==0) {
                    this.applyDeltaWeights()
                    this.resetDeltaWeights()
                } else if (iterationIndex >= dataSet.length) {
                    this.applyDeltaWeights()
                }

                const iterationError = this.cost(target, output)
                const elapsed = Date.now() - startTime
                this.error += iterationError
                this.iterations++

                if (typeof callback=="function") {
                    callback({
                        iterations: this.iterations,
                        error: iterationError,
                        elapsed, input
                    })
                }

                if (iterationIndex < dataSet.length) {
                    setTimeout(doIteration.bind(this), 0)

                } else {
                    epochsCounter++

                    if (log) {
                        console.log(`Epoch: ${this.epochs} Error: ${this.error/iterationIndex}${this.l2==undefined ? "": ` L2 Error: ${this.l2Error/iterationIndex}`}`,
                                    `\nElapsed: ${NetUtil.format(elapsed, "time")} Average Duration: ${NetUtil.format(elapsed/epochsCounter, "time")}`)
                    }

                    if (epochsCounter < epochs) {
                        doEpoch()
                    } else {
                        this.layers.forEach(layer => layer.state = "initialised")

                        if (log) {
                            console.log(`Training finished. Total time: ${NetUtil.format(elapsed, "time")}  Average iteration time: ${NetUtil.format(elapsed/iterationIndex, "time")}`)
                        }
                        resolve()
                    }
                }
            }

            this.resetDeltaWeights()
            doEpoch()
        })
    }

    test (testSet, {log=true, callback}={}) {
        return new Promise((resolve, reject) => {

            if (testSet === undefined || testSet === null) {
                reject("No data provided")
            }

            if (log) {
                console.log("Testing started")
            }

            let totalError = 0
            let iterationIndex = 0
            const startTime = Date.now()

            const testInput = () => {

                const input = testSet[iterationIndex].input
                const output = this.forward(input)
                const target = testSet[iterationIndex].expected || testSet[iterationIndex].output
                const elapsed = Date.now() - startTime

                const iterationError = this.cost(target, output)
                totalError += iterationError
                iterationIndex++

                if (typeof callback=="function") {
                    callback({
                        iterations: iterationIndex,
                        error: iterationError,
                        elapsed, input
                    })
                }

                if (iterationIndex < testSet.length) {
                    setTimeout(testInput.bind(this), 0)

                } else {

                    if (log) {
                        console.log(`Testing finished. Total time: ${NetUtil.format(elapsed, "time")}  Average iteration time: ${NetUtil.format(elapsed/iterationIndex, "time")}`)
                    }

                    resolve(totalError/testSet.length)
                }
            }
            testInput()
        })
    }

    resetDeltaWeights () {
        this.layers.forEach((layer, li) => li && layer.resetDeltaWeights())
    }

    applyDeltaWeights () {

        this.layers.forEach((layer, li) => li && layer.applyDeltaWeights())

        if (this.maxNorm!=undefined) {
            this.maxNormTotal = Math.sqrt(this.maxNormTotal)
            NetMath.maxNorm.bind(this)()
        }
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

        this.resetDeltaWeights()
        this.layers.forEach((layer, li) => li && layer.fromJSON(data.layers[li], li))
    }

    static get version () {
        return "2.0.0"
    }
}

typeof window=="undefined" && (exports.Network = Network)
"use strict"

class Neuron {

    constructor () {}

    init ({updateFn, activation, eluAlpha}={}) {

        const size = this.weights.length
        this.deltaWeights = this.weights.map(v => 0)

        switch (updateFn) {

            case "gain":
                this.biasGain = 1
                this.weightGains = [...new Array(size)].map(v => 1)
                this.getWeightGain = i => this.weightGains[i]
                this.setWeightGain = (i,v) => this.weightGains[i] = v
                break

            case "adagrad":
            case "rmsprop":
            case "adadelta":
                this.biasCache = 0
                this.weightsCache = [...new Array(size)].map(v => 0)
                this.getWeightsCache = i => this.weightsCache[i]
                this.setWeightsCache = (i,v) => this.weightsCache[i] = v

                if (updateFn=="adadelta") {
                    this.adadeltaBiasCache = 0
                    this.adadeltaCache = [...new Array(size)].map(v => 0)
                    this.getAdadeltaCache = i => this.adadeltaCache[i]
                    this.setAdadeltaCache = (i,v) => this.adadeltaCache[i] = v
                }
                break

            case "adam":
                this.m = 0
                this.v = 0
                break
        }

        if (activation=="rrelu") {
            this.rreluSlope = Math.random() * 0.001

        } else if (activation=="elu") {
            this.eluAlpha = eluAlpha
        }
    }

    getWeight (i) {
        return this.weights[i]
    }

    setWeight (i, v) {
        this.weights[i] = v
    }

    getDeltaWeight (i) {
        return this.deltaWeights[i]
    }

    setDeltaWeight (i, v) {
        this.deltaWeights[i] = v
    }
}

typeof window=="undefined" && (exports.Neuron = Neuron)
"use strict"

class PoolLayer {

    constructor (size, {stride, activation}={}) {

        if (size)   this.size = size
        if (stride) this.stride = stride

        if (activation!=undefined && activation!=false) {
            this.activation = typeof activation=="function" ? activation : NetMath[NetUtil.format(activation)].bind(this)
        } else {
            this.activation = false
        }
    }

    init () {}

    assignNext (layer) {
        this.nextLayer = layer
    }

    assignPrev (layer, layerIndex) {

        this.prevLayer = layer
        this.size = this.size || this.net.pool.size || 2
        this.stride = this.stride || this.net.pool.stride || this.size

        let prevLayerOutWidth = layer.outMapSize

        switch (layer.constructor.name) {

            case "FCLayer":
                this.channels = this.net.channels
                prevLayerOutWidth = Math.max(Math.floor(Math.sqrt(layer.size/this.channels)), 1)
                break

            case "ConvLayer":
                this.channels = layer.size
                break

            case "PoolLayer":
                this.channels = layer.channels
                break
        }

        this.prevLayerOutWidth = prevLayerOutWidth
        this.outMapSize = (prevLayerOutWidth - this.size) / this.stride + 1
        this.inMapValuesCount = prevLayerOutWidth ** 2

        if (this.outMapSize%1 != 0) {
            throw new Error(`Misconfigured hyperparameters. Activation volume dimensions would be ${this.outMapSize} in pool layer at index ${layerIndex}`)
        }

        this.activations = [...new Array(this.channels)].map(channel => {
            return [...new Array(this.outMapSize)].map(row => [...new Array(this.outMapSize)].map(v => 0))
        })
        this.errors = [...new Array(this.channels)].map(channel => {
            return [...new Array(prevLayerOutWidth)].map(row => [...new Array(prevLayerOutWidth)].map(v => 0))
        })
        this.indeces = this.activations.map(channel => channel.map(row => row.map(v => [0,0])))
    }

    forward () {
        for (let channel=0; channel<this.channels; channel++) {

            NetMath.maxPool(this, channel)

            // Apply activations
            if (this.activation) {
                for (let row=0; row<this.outMapSize; row++) {
                    for (let col=0; col<this.outMapSize; col++) {
                        this.activations[channel][row][col] = this.activation(this.activations[channel][row][col], false, this.net)
                    }
                }
            }
        }
    }

    backward () {

        // Clear the existing error values, first
        for (let channel=0; channel<this.channels; channel++) {
            for (let row=0; row<this.errors[0].length; row++) {
                for (let col=0; col<this.errors[0].length; col++) {
                    this.errors[channel][row][col] = 0
                }
            }
        }

        if (this.nextLayer instanceof FCLayer) {

            for (let channel=0; channel<this.channels; channel++) {
                for (let row=0; row<this.outMapSize; row++) {
                    for (let col=0; col<this.outMapSize; col++) {

                        const rowI = this.indeces[channel][row][col][0] + row * this.stride
                        const colI = this.indeces[channel][row][col][1] + col * this.stride
                        const neuronI = channel * this.outMapSize**2 + row * this.outMapSize + col

                        const weightIndex = channel * this.outMapSize**2 + row * this.outMapSize + col

                        for (let neuron=0; neuron<this.nextLayer.neurons.length; neuron++) {
                            this.errors[channel][rowI][colI] += this.nextLayer.neurons[neuron].error
                                                                * this.nextLayer.neurons[neuron].weights[weightIndex]
                        }
                    }
                }
            }

        } else if (this.nextLayer instanceof ConvLayer) {

            for (let channel=0; channel<this.channels; channel++) {

                const errs = []

                for (let col=0; col<this.outMapSize; col++) {
                    errs[col] = 0
                }

                // Convolve on the error map
                NetUtil.buildConvErrorMap(this.nextLayer, errs, channel)

                for (let row=0; row<this.outMapSize; row++) {
                    for (let col=0; col<this.outMapSize; col++) {

                        const rowI = this.indeces[channel][row][col][0] + row * this.stride
                        const colI = this.indeces[channel][row][col][1] + col * this.stride

                        this.errors[channel][rowI][colI] += errs[row][col]
                    }
                }
            }

        } else {

            for (let channel=0; channel<this.channels; channel++) {
                for (let row=0; row<this.outMapSize; row++) {
                    for (let col=0; col<this.outMapSize; col++) {

                        const rowI = this.indeces[channel][row][col][0] + row * this.stride
                        const colI = this.indeces[channel][row][col][1] + col * this.stride

                        this.errors[channel][rowI][colI] += this.nextLayer.errors[channel][row][col]
                    }
                }
            }
        }

        // Apply derivatives
        if (this.activation) {
            for (let channel=0; channel<this.channels; channel++) {

                for (let row=0; row<this.indeces[channel].length; row++) {
                    for (let col=0; col<this.indeces[channel].length; col++) {

                        const rowI = this.indeces[channel][row][col][0] + row * this.stride
                        const colI = this.indeces[channel][row][col][1] + col * this.stride

                        this.errors[channel][rowI][colI] *= this.activation(this.errors[channel][rowI][colI], true, this.net)
                    }
                }
            }
        }
    }

    resetDeltaWeights () {}

    applyDeltaWeights () {}

    toJSON () {return {}}

    fromJSON () {}
}

typeof window=="undefined" && (exports.PoolLayer = PoolLayer)
//# sourceMappingURL=jsNet.concat.js.map