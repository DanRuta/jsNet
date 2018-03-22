"use strict"

class ConvLayer {

    constructor (size, {filterSize, zeroPadding, stride, activation}={}) {

        if (filterSize)     this.filterSize = filterSize
        if (stride)         this.stride = stride
        if (size)           this.size = size

        this.zeroPadding = zeroPadding
        this.activationName = activation

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

        this.layerIndex = layerIndex
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
                return [...new Array(this.filterSize)].map(weightsRow => this.net.weightsInitFn(this.filterSize, this.weightsConfig))
            })

            filter.activationMap = [...new Array(this.outMapSize)].map(row => [...new Array(this.outMapSize)].map(v => 0))
            filter.errorMap = [...new Array(this.outMapSize)].map(row => [...new Array(this.outMapSize)].map(v => 0))
            filter.bias = 1

            if (this.net.dropout != 1) {
                filter.dropoutMap = filter.activationMap.map(row => row.map(v => false))
            }

            filter.init({
                updateFn: this.net.updateFn,
                activation: this.activationName || this.net.activationConfig,
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
                    if (this.state=="training" && filter.dropoutMap && (filter.dropoutMap[sumY][sumX] = Math.random() > this.net.dropout)) {
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

                    if (filter.dropoutMap && filter.dropoutMap[row][col]) {
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
            filter.deltaBias = 0

            for (let channel=0; channel<filter.deltaWeights.length; channel++) {
                for (let row=0; row<filter.deltaWeights[0].length; row++) {
                    for (let col=0; col<filter.deltaWeights[0][0].length; col++) {
                        filter.deltaWeights[channel][row][col] = 0
                    }
                }
            }

            for (let row=0; row<filter.errorMap.length; row++) {
                for (let col=0; col<filter.errorMap.length; col++) {
                    filter.errorMap[row][col] = 0
                }
            }

            if (filter.dropoutMap) {
                for (let row=0; row<filter.dropoutMap.length; row++) {
                    for (let col=0; col<filter.dropoutMap[0].length; col++) {
                        filter.dropoutMap[row][col] = false
                    }
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

                        if (this.net.l2Error!=undefined) this.net.l2Error += 0.5 * this.net.l2 * filter.weights[channel][row][col]**2
                        if (this.net.l1Error!=undefined) this.net.l1Error += this.net.l1 * Math.abs(filter.weights[channel][row][col])

                        const regularized = (filter.deltaWeights[channel][row][col]
                            + this.net.l2 * filter.weights[channel][row][col]
                            + this.net.l1 * (filter.weights[channel][row][col] > 0 ? 1 : -1)) / this.net.miniBatchSize

                        filter.weights[channel][row][col] = this.net.weightUpdateFn.bind(this.net, filter.weights[channel][row][col],
                                                                regularized, filter, [channel, row, col])()

                        if (this.net.maxNorm!=undefined) this.net.maxNormTotal += filter.weights[channel][row][col]**2
                    }
                }
            }

            filter.bias = this.net.weightUpdateFn.bind(this.net, filter.bias, filter.deltaBias, filter)()
        }
    }

    backUpValidation () {
        for (let f=0; f<this.filters.length; f++) {
            const filter = this.filters[f]

            filter.validationBias = filter.bias
            filter.validationWeights = []

            for (let wd=0; wd<filter.weights.length; wd++) {
                const channel = []
                for (let wy=0; wy<filter.weights[wd].length; wy++) {
                    channel[wy] = filter.weights[wd][wy].slice(0)
                }
                filter.validationWeights[wd] = channel
            }
        }
    }

    restoreValidation () {
        for (let f=0; f<this.filters.length; f++) {
            const filter = this.filters[f]

            filter.bias = filter.validationBias

            for (let wd=0; wd<filter.weights.length; wd++) {
                for (let wy=0; wy<filter.weights[wd].length; wy++) {
                    filter.weights[wd][wy] = filter.validationWeights[wd][wy].slice(0)
                }
            }
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

    // Used for importing data
    getDataSize () {

        let size = 0

        for (let f=0; f<this.filters.length; f++) {

            const filter = this.filters[f]

            for (let c=0; c<filter.weights.length; c++) {
                for (let r=0; r<filter.weights[c].length; r++) {
                    size += filter.weights[c][r].length
                }
            }

            size += 1
        }

        return size
    }

    toIMG () {

        const data = []

        for (let f=0; f<this.filters.length; f++) {
            const filter = this.filters[f]

            data.push(filter.bias)

            for (let c=0; c<filter.weights.length; c++) {
                for (let r=0; r<filter.weights[c].length; r++) {
                    for (let v=0; v<filter.weights[c][r].length; v++) {
                        data.push(filter.weights[c][r][v])
                    }
                }
            }
        }

        return data
    }

    fromIMG (data) {

        let valI = 0

        for (let f=0; f<this.filters.length; f++) {

            const filter = this.filters[f]
            filter.bias = data[valI]
            valI++

            for (let c=0; c<filter.weights.length; c++) {
                for (let r=0; r<filter.weights[c].length; r++) {
                    for (let v=0; v<filter.weights[c][r].length; v++) {
                        filter.weights[c][r][v] = data[valI]
                        valI++
                    }
                }
            }
        }
    }
}

// https://github.com/DanRuta/jsNet/issues/33
/* istanbul ignore next */
typeof window!="undefined" && (window.exports = window.exports || {})
/* istanbul ignore next */
typeof window!="undefined" && (window.ConvLayer = ConvLayer)
exports.ConvLayer = ConvLayer
"use strict"

class FCLayer {

    constructor (size, {activation}={}) {
        this.size = size
        this.neurons = [...new Array(size)].map(n => new Neuron())
        this.state = "not-initialised"

        if (activation!=undefined) {
            if (typeof activation=="boolean" && !activation) {
                this.activation = false
            } else {
                this.activation = typeof activation=="function" ? activation : NetMath[NetUtil.format(activation)].bind(this)
            }
        }
    }

    assignNext (layer) {
        this.nextLayer = layer
    }

    assignPrev (layer, layerIndex) {
        this.prevLayer = layer
        this.layerIndex = layerIndex
    }

    init () {
        this.neurons.forEach(neuron => {

            let weightsCount

            switch (true) {
                case this.prevLayer instanceof FCLayer:
                    weightsCount = this.prevLayer.size
                    break

                case this.prevLayer instanceof ConvLayer:
                    weightsCount = this.prevLayer.filters.length * this.prevLayer.outMapSize**2
                    break

                case this.prevLayer instanceof PoolLayer:
                    weightsCount = this.prevLayer.activations.length * this.prevLayer.outMapSize**2
                    break
            }

            neuron.weights = this.net.weightsInitFn(weightsCount, this.weightsConfig)
            neuron.bias = 1

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

                neuron.activation = (this.activation ? this.activation(neuron.sum, false, neuron) : neuron.sum) / (this.net.dropout||1)
            }
        })
    }

    backward (errors) {
        this.neurons.forEach((neuron, ni) => {

            if (neuron.dropped) {
                neuron.error = 0
                neuron.deltaBias += 0
            } else {
                if (typeof errors !== "undefined") {
                    neuron.error = errors[ni]
                } else {
                    neuron.derivative = this.activation ? this.activation(neuron.sum, true, neuron) : 1
                    neuron.error = neuron.derivative * this.nextLayer.neurons.map(n => n.error * (n.weights[ni]||0))
                                                                             .reduce((p,c) => p+c, 0)
                }

                const activations = NetUtil.getActivations(this.prevLayer)

                for (let wi=0; wi<neuron.weights.length; wi++) {
                    neuron.deltaWeights[wi] += (neuron.error * activations[wi])
                }

                neuron.deltaBias += neuron.error
            }
        })
    }

    resetDeltaWeights () {
        for (let n=0; n<this.neurons.length; n++) {

            this.neurons[n].deltaBias = 0

            for (let dwi=0; dwi<this.neurons[n].deltaWeights.length; dwi++) {
                this.neurons[n].deltaWeights[dwi] = 0
            }
        }
    }

    applyDeltaWeights () {
        for (let n=0; n<this.neurons.length; n++) {

            const neuron = this.neurons[n]

            for (let dwi=0; dwi<this.neurons[n].deltaWeights.length; dwi++) {

                if (this.net.l2Error!=undefined) this.net.l2Error += 0.5 * this.net.l2 * neuron.weights[dwi]**2
                if (this.net.l1Error!=undefined) this.net.l1Error += this.net.l1 * Math.abs(neuron.weights[dwi])

                const regularized = (neuron.deltaWeights[dwi]
                    + this.net.l2 * neuron.weights[dwi]
                    + this.net.l1 * (neuron.weights[dwi] > 0 ? 1 : -1)) / this.net.miniBatchSize

                neuron.weights[dwi] = this.net.weightUpdateFn.bind(this.net, neuron.weights[dwi], regularized, neuron, dwi)()

                if (this.net.maxNorm!=undefined) this.net.maxNormTotal += neuron.weights[dwi]**2
            }

            neuron.bias = this.net.weightUpdateFn.bind(this.net, neuron.bias, neuron.deltaBias, neuron)()
        }
    }

    backUpValidation () {
        for (let n=0; n<this.neurons.length; n++) {
            const neuron = this.neurons[n]
            neuron.validationBias = neuron.bias
            neuron.validationWeights = neuron.weights.slice(0)
        }
    }

    restoreValidation () {
        for (let n=0; n<this.neurons.length; n++) {
            const neuron = this.neurons[n]
            neuron.bias = neuron.validationBias
            neuron.weights = neuron.validationWeights.slice(0)
        }
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

    // Used for importing data
    getDataSize () {

        let size = 0

        for (let n=0; n<this.neurons.length; n++) {
            size += this.neurons[n].weights.length + 1
        }

        return size
    }

    toIMG () {
        const data = []

        for (let n=0; n<this.neurons.length; n++) {
            data.push(this.neurons[n].bias)

            for (let w=0; w<this.neurons[n].weights.length; w++) {
                data.push(this.neurons[n].weights[w])
            }
        }

        return data
    }

    fromIMG (data) {

        let valI = 0

        for (let n=0; n<this.neurons.length; n++) {

            const neuron = this.neurons[n]
            neuron.bias = data[valI]
            valI++

            for (let w=0; w<neuron.weights.length; w++) {
                neuron.weights[w] = data[valI]
                valI++
            }
        }
    }
}

const Layer = FCLayer

/* istanbul ignore next */
typeof window!="undefined" && (window.FCLayer = window.Layer = FCLayer)
exports.FCLayer = exports.Layer = FCLayer
"use strict"

class Filter {

    constructor () {}

    init ({updateFn, activation, eluAlpha}={}) {

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
            case "momentum":
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
                break
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

/* istanbul ignore next */
typeof window!="undefined" && (window.Filter = Filter)
exports.Filter = Filter
"use strict"

class InputLayer extends FCLayer {
    constructor (size, {span=1}={}) {
        super(size * span*span)
    }
}

/* istanbul ignore next */
typeof window!="undefined" && (window.InputLayer = InputLayer)
exports.InputLayer = InputLayer

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

    static rootmeansquarederror (calculated, desired) {
        return Math.sqrt(NetMath.meansquarederror(calculated, desired))
    }

    // Weight updating functions
    static vanillasgd (value, deltaValue) {
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

        neuron.v = 0.999*neuron.v + (1-0.999) * Math.pow(deltaValue, 2)
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

    static momentum (value, deltaValue, neuron, weightI) {

        let v

        if (weightI!=null) {
            v = this.momentum * (neuron.getWeightsCache(weightI)) - this.learningRate * deltaValue
            neuron.setWeightsCache(weightI, v)
        } else {
            v = this.momentum * (neuron.biasCache) - this.learningRate * deltaValue
            neuron.biasCache = v
        }

        return value - v
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
            let x1, x2, r

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
    static softmax (v) {

        const values = v.slice(0)
        let maxValue = values[0]

        for (let i=1; i<values.length; i++) {
            if (values[i] > maxValue) {
                maxValue = values[i]
            }
        }

        // Exponentials
        const exponentials = new Array(values.length)
        let exponentialsSum = 0.0

        for (let i=0; i<values.length; i++) {
            let e = Math.exp(values[i] - maxValue)
            exponentialsSum += e
            exponentials[i] = e
        }

        for (let i=0; i<values.length; i++) {
            exponentials[i] /= exponentialsSum
            values[i] = exponentials[i]
        }

        return values
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

/* istanbul ignore next */
typeof window!="undefined" && (window.NetMath = NetMath)
exports.NetMath = NetMath
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
                data[col].splice(data[col].length+1, data[col].length, 0)
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

        // For each input channel,
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
        for (let emYI=0; emYI<errorMap.length; emYI++) {
            errorMap[emYI] = errorMap[emYI].splice(zeroPadding, errorMap[emYI].length - zeroPadding*2)
        }
    }

    static buildConvDWeights (layer) {

        const weightsCount = layer.filters[0].weights[0].length
        const fSSpread = Math.floor(weightsCount / 2)
        const channelsCount = layer.filters[0].weights.length

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

                        const error = filter.errorMap[(inputY-fSSpread)/layer.stride][(inputX-fSSpread)/layer.stride]

                        // ...and at each location...
                        for (let weightsY=0; weightsY<weightsCount; weightsY++) {
                            for (let weightsX=0; weightsX<weightsCount; weightsX++) {
                                const activation = inputMap[inputY-fSSpread+weightsY][inputX-fSSpread+weightsX]
                                filter.deltaWeights[channelI][weightsY][weightsX] += activation * error
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

    static getActivations (layer, mapStartI, mapSize) {

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

    static splitData (data, {training=0.7, validation=0.15, test=0.15}={}) {

        const split = {
            training: [],
            validation: [],
            test: []
        }

        // Define here splits, for returning at the end
        for (let i=0; i<data.length; i++) {
            let x = Math.random()

            if (x > 1-training) {
                split.training.push(data[i])
            } else {

                if (x<validation) {
                    split.validation.push(data[i])
                } else {
                    split.test.push(data[i])
                }

            }
        }

        return split
    }

    static normalize (data) {
        let minVal = Infinity
        let maxVal = -Infinity

        for (let i=0; i<data.length; i++) {
            if (data[i] < minVal) {
                minVal = data[i]
            }
            if (data[i] > maxVal) {
                maxVal = data[i]
            }
        }

        if ((-1*minVal + maxVal) != 0) {
            for (let i=0; i<data.length; i++) {
                data[i] = (data[i] + -1*minVal) / (-1*minVal + maxVal)
            }
        } else {
            for (let i=0; i<data.length; i++) {
                data[i] = 0.5
            }
        }

        return {minVal, maxVal}
    }

    static makeConfusionMatrix (originalData) {
        let total = 0
        let totalCorrect = 0
        const data = []

        for (let r=0; r<originalData.length; r++) {
            const row = []
            for (let c=0; c<originalData[r].length; c++) {
                row.push(originalData[r][c])
            }
            data.push(row)
        }


        for (let r=0; r<data.length; r++) {
            for (let c=0; c<data[r].length; c++) {
                total += data[r][c]
            }
        }

        for (let r=0; r<data.length; r++) {

            let rowTotal = 0
            totalCorrect += data[r][r]

            for (let c=0; c<data[r].length; c++) {
                rowTotal += data[r][c]
                data[r][c] = {count: data[r][c], percent: (data[r][c] / total * 100)||0}
            }

            const correctPercent = data[r][r].count / rowTotal * 100

            data[r].total = {
                correct: (correctPercent||0),
                wrong: (100 - correctPercent)||0
            }
        }

        // Collect bottom row percentages
        const bottomRow = []

        for (let c=0; c<data[0].length; c++) {

            let columnTotal = 0

            for (let r=0; r<data.length; r++) {
                columnTotal += data[r][c].count
            }

            const correctPercent = data[c][c].count / columnTotal * 100

            bottomRow.push({
                correct: (correctPercent)||0,
                wrong: (100 - correctPercent)||0
            })
        }

        data.total = bottomRow

        // Calculate final matrix percentage
        data.total.total = {
            correct: (totalCorrect / total * 100)||0,
            wrong: (100 - (totalCorrect / total * 100))||0
        }

        return data
    }

    /* istanbul ignore next */
    static printConfusionMatrix (data) {
        if (typeof window!="undefined") {

            for (let r=0; r<data.length; r++) {
                for (let c=0; c<data[r].length; c++) {
                    data[r][c] = `${data[r][c].count} (${data[r][c].percent.toFixed(1)}%)`
                }
                data[r].total = `${data[r].total.correct.toFixed(1)}% / ${data[r].total.wrong.toFixed(1)}%`
                data.total[r] = `${data.total[r].correct.toFixed(1)}% / ${data.total[r].wrong.toFixed(1)}%`
            }

            data.total.total = `${data.total.total.correct.toFixed(1)}% / ${data.total.total.wrong.toFixed(1)}%`

            console.table(data)
            return
        }


        const padNum = (num, percent) => {
            num = percent ? num.toFixed(1) + "%" : num.toString()
            const leftPad = Math.max(Math.floor((3*2+1 - num.length) / 2), 0)
            const rightPad = Math.max(3*2+1 - (num.length + leftPad), 0)
            return " ".repeat(leftPad)+num+" ".repeat(rightPad)
        }

        let colourText
        let colourBackground

        // Bright
        process.stdout.write("\n\x1b[1m")

        for (let r=0; r<data.length; r++) {

            // Bright white text
            colourText = "\x1b[2m\x1b[37m"

            // Count
            for (let c=0; c<data[r].length; c++) {
                colourBackground =  r==c ? "\x1b[42m" : "\x1b[41m"
                process.stdout.write(`${colourText}${colourBackground}\x1b[1m${padNum(data[r][c].count)}\x1b[22m`)
            }

            // Dim green text on white background
            colourText = "\x1b[2m\x1b[32m"
            colourBackground = "\x1b[47m"
            process.stdout.write(`${colourText}${colourBackground}${padNum(data[r].total.correct, true)}`)

            // Bright white text
            colourText = "\x1b[2m\x1b[37m"
            process.stdout.write(`${colourText}\n`)

            // Percent
            for (let c=0; c<data[r].length; c++) {
                colourBackground =  r==c ? "\x1b[42m" : "\x1b[41m"
                process.stdout.write(`${colourText}${colourBackground}${padNum(data[r][c].percent, true)}`)
            }

            // Dim red
            colourText = "\x1b[2m\x1b[31m"
            colourBackground = "\x1b[47m"
            process.stdout.write(`${colourText}${colourBackground}${padNum(data[r].total.wrong, true)}`)

            // Bright
            process.stdout.write("\x1b[1m\x1b[30m\n")
        }

        // Dim green text
        colourText = "\x1b[22m\x1b[32m"

        // Bottom row correct percentages
        for (const col of data.total) {
            process.stdout.write(`${colourText}${colourBackground}${padNum(col.correct, true)}`)
        }
        // Total correct percentages
        // Blue background
        colourBackground = "\x1b[1m\x1b[44m"
        process.stdout.write(`${colourText}${colourBackground}${padNum(data.total.total.correct, true)}\n`)

        // Dim red on white background
        colourText = "\x1b[22m\x1b[31m"
        colourBackground = "\x1b[47m"

        // Bottom row wrong percentages
        for (const col of data.total) {
            process.stdout.write(`${colourText}${colourBackground}${padNum(col.wrong, true)}`)
        }

        // Bright red on blue background
        colourText = "\x1b[1m\x1b[31m"
        colourBackground = "\x1b[44m"
        process.stdout.write(`${colourText}${colourBackground}${padNum(data.total.total.wrong, true)}\n`)

        // Reset
        process.stdout.write("\x1b[0m\n")
    }
}

/* istanbul ignore next */
typeof window!="undefined" && (window.NetUtil = NetUtil)
exports.NetUtil = NetUtil
"use strict"

class Network {

    constructor ({learningRate, layers=[], updateFn="vanillasgd", activation="sigmoid", cost="meansquarederror", momentum=0.9,
        rmsDecay, rho, lreluSlope, eluAlpha, dropout=1, l2, l1, maxNorm, weightsConfig, channels, conv, pool}={}) {

        this.state = "not-defined"
        this.layers = []
        this.conv = {}
        this.pool = {}
        this.epochs = 0
        this.iterations = 0
        this.validations = 0
        this.dropout = dropout==false ? 1 : dropout
        this.error = 0
        activation = NetUtil.format(activation)
        updateFn = NetUtil.format(updateFn)
        cost = NetUtil.format(cost)
        this.l1 = 0
        this.l2 = 0

        if (l1) {
            this.l1 = typeof l1=="boolean" ? 0.005 : l1
            this.l1Error = 0
        }

        if (l2) {
            this.l2 = typeof l2=="boolean" ? 0.001 : l2
            this.l2Error = 0
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

            case "momentum":
                this.learningRate = this.learningRate==undefined ? 0.2 : this.learningRate
                this.momentum = momentum
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

        this.updateFn = [false, null, undefined].includes(updateFn) ? "vanillasgd" : updateFn
        this.weightUpdateFn = NetMath[this.updateFn]
        this.activation = typeof activation=="function" ? activation : NetMath[activation].bind(this)
        this.activationConfig = activation
        this.cost = typeof cost=="function" ? cost : NetMath[cost]

        if (this.updateFn=="rmsprop") {
            this.rmsDecay = rmsDecay==undefined ? 0.99 : rmsDecay
        }

        this.lreluSlope = lreluSlope==undefined ? -0.0005 : lreluSlope
        this.rreluSlope = Math.random() * 0.001
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

        this.collectedErrors = {training: [], validation: [], test: []}
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

        const outSize = this.layers[this.layers.length-1].size
        this.trainingConfusionMatrix = [...new Array(outSize)].map(r => [...new Array(outSize)].map(v => 0))
        this.testConfusionMatrix = [...new Array(outSize)].map(r => [...new Array(outSize)].map(v => 0))
        this.validationConfusionMatrix = [...new Array(outSize)].map(r => [...new Array(outSize)].map(v => 0))

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

            if (layerIndex<this.layers.length-1) {
                layer.weightsConfig.fanOut = this.layers[layerIndex+1].size
            }

            layer.init()

        } else if (this.layers.length > 1) {
            layer.weightsConfig.fanOut = this.layers[1].size
        }

        layer.state = "initialised"
    }

    forward (data) {

        if (this.state!="initialised") {
            throw new Error("The network layers have not been initialised.")
        }

        if (data === undefined || data === null) {
            throw new Error("No data passed to Network.forward()")
        }

        // Flatten volume inputs
        if (Array.isArray(data[0])) {
            const flat = []

            for (let c=0; c<data.length; c++) {
                for (let r=0; r<data[0].length; r++) {
                    for (let v=0; v<data[0].length; v++) {
                        flat.push(data[c][r][v])
                    }
                }
            }
            data = flat
        }

        if (data.length != this.layers[0].neurons.length) {
            console.warn("Input data length did not match input layer neurons count.")
        }

        this.layers[0].neurons.forEach((neuron, ni) => neuron.activation = data[ni])
        this.layers.forEach((layer, li) => li && layer.forward())

        return this.layers[this.layers.length-1].neurons.map(n => n.activation)
    }

    backward (errors) {

        if (errors === undefined) {
            throw new Error("No data passed to Network.backward()")
        }

        if (errors.length != this.layers[this.layers.length-1].neurons.length) {
            console.warn("Expected data length did not match output layer neurons count.", errors)
        }

        this.layers[this.layers.length-1].backward(errors)

        for (let layerIndex=this.layers.length-2; layerIndex>0; layerIndex--) {
            this.layers[layerIndex].backward()
        }
    }

    train (dataSet, {epochs=1, callback, callbackInterval=1, collectErrors, log=true, miniBatchSize=1, shuffle=false, validation}={}) {

        this.miniBatchSize = typeof miniBatchSize=="boolean" && miniBatchSize ? dataSet[0].expected.length : miniBatchSize
        this.validation = validation

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
                this.initLayers.bind(this, dataSet[0].input.length, dataSet[0].expected.length)()
            }

            this.layers.forEach(layer => layer.state = "training")

            if (this.validation) {
                this.validation.interval = this.validation.interval || dataSet.length // Default to 1 epoch

                if (this.validation.earlyStopping) {
                    switch (this.validation.earlyStopping.type) {
                        case "threshold":
                            this.validation.earlyStopping.threshold = this.validation.earlyStopping.threshold || 0.01
                            break
                        case "patience":
                            this.validation.earlyStopping.patienceCounter = 0
                            this.validation.earlyStopping.bestError = Infinity
                            this.validation.earlyStopping.patience = this.validation.earlyStopping.patience || 20
                            break
                        case "divergence":
                            this.validation.earlyStopping.percent = this.validation.earlyStopping.percent || 30
                            this.validation.earlyStopping.bestError = Infinity
                            break
                    }
                }
            }

            let iterationIndex = 0
            let epochsCounter = 0
            let elapsed
            const startTime = Date.now()

            const logAndResolve = () => {
                this.layers.forEach(layer => layer.state = "initialised")

                if (this.validation && this.validation.earlyStopping && (this.validation.earlyStopping.type == "patience" || this.validation.earlyStopping.type == "divergence")) {
                    for (let l=1; l<this.layers.length; l++) {
                        this.layers[l].restoreValidation()
                    }
                }

                if (log) {
                    console.log(`Training finished. Total time: ${NetUtil.format(elapsed, "time")}  Average iteration time: ${NetUtil.format(elapsed/iterationIndex, "time")}`)
                }
                resolve()
            }

            const doEpoch = () => {
                this.epochs++
                this.error = 0
                this.validationError = 0
                iterationIndex = 0

                if (this.l2Error!=undefined) this.l2Error = 0
                if (this.l1Error!=undefined) this.l1Error = 0

                doIteration()
            }

            const doIteration = async () => {

                if (!dataSet[iterationIndex].hasOwnProperty("input") || !dataSet[iterationIndex].hasOwnProperty("expected")) {
                    return void reject("Data set must be a list of objects with keys: 'input' and 'expected'")
                }

                let trainingError
                let validationError

                const input = dataSet[iterationIndex].input
                const output = this.forward(input)
                const target = dataSet[iterationIndex].expected

                let classification = -Infinity
                const errors = []
                for (let n=0; n<output.length; n++) {
                    errors[n] = (target[n]==1 ? 1 : 0) - output[n]
                    classification = Math.max(classification, output[n])
                }

                if (this.trainingConfusionMatrix[target.indexOf(1)]) {
                    this.trainingConfusionMatrix[target.indexOf(1)][output.indexOf(classification)]++
                }

                // Do validation
                if (this.validation && iterationIndex && iterationIndex%this.validation.interval==0) {

                    validationError = await this.validate(this.validation.data)

                    if (this.validation.earlyStopping && this.checkEarlyStopping(errors)) {
                        log && console.log("Stopping early")
                        return logAndResolve()
                    }
                }

                this.backward(errors)

                if (++iterationIndex%this.miniBatchSize==0) {
                    this.applyDeltaWeights()
                    this.resetDeltaWeights()
                } else if (iterationIndex >= dataSet.length) {
                    this.applyDeltaWeights()
                }

                trainingError = this.cost(target, output)
                this.error += trainingError
                this.iterations++

                elapsed = Date.now() - startTime

                if (collectErrors) {
                    this.collectedErrors.training.push(trainingError)

                    if (validationError) {
                        this.collectedErrors.validation.push(validationError)
                    }
                }

                if ((iterationIndex%callbackInterval == 0 || validationError) && typeof callback=="function") {
                    callback({
                        iterations: this.iterations,
                        validations: this.validations,
                        validationError, trainingError,
                        elapsed, input
                    })
                }

                if (iterationIndex < dataSet.length) {

                    if (iterationIndex%callbackInterval == 0) {
                        setTimeout(doIteration.bind(this), 0)
                    } else {
                        doIteration()
                    }

                } else {
                    epochsCounter++

                    if (log) {
                        let text = `Epoch: ${this.epochs}\nTraining Error: ${this.error/iterationIndex}`

                        if (validation) {
                            text += `\nValidation Error: ${this.validationError}`
                        }

                        if (this.l2Error!=undefined) {
                            text += `\nL2 Error: ${this.l2Error/iterationIndex}`
                        }

                        text += `\nElapsed: ${NetUtil.format(elapsed, "time")} Average Duration: ${NetUtil.format(elapsed/epochsCounter, "time")}`
                        console.log(text)
                    }

                    if (epochsCounter < epochs) {
                        doEpoch()
                    } else {
                        logAndResolve()
                    }
                }
            }

            this.resetDeltaWeights()
            doEpoch()
        })
    }

    validate (data) {
        return new Promise((resolve, reject) => {
            let validationIndex = 0
            let totalValidationErrors = 0

            const validateItem = (item) => {

                const output = this.forward(data[validationIndex].input)
                const target = data[validationIndex].expected

                let classification = -Infinity
                for (let i=0; i<output.length; i++) {
                    classification = Math.max(classification, output[i])
                }

                if (this.validationConfusionMatrix[target.indexOf(1)]) {
                    this.validationConfusionMatrix[target.indexOf(1)][output.indexOf(classification)]++
                }

                this.validations++
                totalValidationErrors += this.cost(target, output)
                // maybe do this only once, as there's no callback anyway
                this.validationError = totalValidationErrors / (validationIndex+1)

                if (++validationIndex<data.length) {
                    setTimeout(() => validateItem(validationIndex), 0)
                } else {
                    this.lastValidationError = totalValidationErrors / data.length
                    resolve(totalValidationErrors / data.length)
                }
            }
            validateItem(validationIndex)
        })
    }

    checkEarlyStopping (errors) {

        let stop = false

        switch (this.validation.earlyStopping.type) {
            case "threshold":
                stop = this.lastValidationError <= this.validation.earlyStopping.threshold

                // Do the last backward pass
                if (stop) {
                    this.backward(errors)
                    this.applyDeltaWeights()
                }

                return stop

            case "patience":
                if (this.lastValidationError < this.validation.earlyStopping.bestError) {
                    this.validation.earlyStopping.patienceCounter = 0
                    this.validation.earlyStopping.bestError = this.lastValidationError

                    for (let l=1; l<this.layers.length; l++) {
                        this.layers[l].backUpValidation()
                    }

                } else {
                    this.validation.earlyStopping.patienceCounter++
                    stop = this.validation.earlyStopping.patienceCounter>=this.validation.earlyStopping.patience
                }
                return stop

            case "divergence":
                if (this.lastValidationError < this.validation.earlyStopping.bestError) {
                    this.validation.earlyStopping.bestError = this.lastValidationError

                    for (let l=1; l<this.layers.length; l++) {
                        this.layers[l].backUpValidation()
                    }
                } else {
                    stop = this.lastValidationError / this.validation.earlyStopping.bestError >= (1+this.validation.earlyStopping.percent/100)
                }

                return stop
        }
    }

    test (testSet, {log=true, callback, collectErrors}={}) {
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
                const target = testSet[iterationIndex].expected
                const elapsed = Date.now() - startTime

                let classification = -Infinity
                for (let i=0; i<output.length; i++) {
                    classification = Math.max(classification, output[i])
                }

                if (this.testConfusionMatrix[target.indexOf(1)]) {
                    this.testConfusionMatrix[target.indexOf(1)][output.indexOf(classification)]++
                }

                const iterationError = this.cost(target, output)
                totalError += iterationError
                iterationIndex++

                if (collectErrors) {
                    this.collectedErrors.test.push(iterationError)
                }

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

    toIMG (IMGArrays, opts={}) {

        if (!IMGArrays) {
            throw new Error("The IMGArrays library must be provided. See the documentation for instructions.")
        }

        const data = []

        for (let l=1; l<this.layers.length; l++) {

            const layerData = this.layers[l].toIMG()
            for (let v=0; v<layerData.length; v++) {
                data.push(layerData[v])
            }
        }

        return IMGArrays.toIMG(data, opts)
    }

    fromIMG (rawData, IMGArrays, opts={}) {

        if (!IMGArrays) {
            throw new Error("The IMGArrays library must be provided. See the documentation for instructions.")
        }

        let valI = 0
        const data = IMGArrays.fromIMG(rawData, opts)

        for (let l=1; l<this.layers.length; l++) {

            const dataCount = this.layers[l].getDataSize()
            this.layers[l].fromIMG(data.splice(0, dataCount))
        }
    }

    printConfusionMatrix (type) {
        if (type) {
            NetUtil.printConfusionMatrix(NetUtil.makeConfusionMatrix(this[`${type}ConfusionMatrix`]))
        } else {
            // Total all data
            const data = []

            for (let r=0; r<this.trainingConfusionMatrix.length; r++) {
                const row = []
                for (let c=0; c<this.trainingConfusionMatrix.length; c++) {
                    row.push(this.trainingConfusionMatrix[r][c] + this.testConfusionMatrix[r][c] + this.validationConfusionMatrix[r][c])
                }
                data.push(row)
            }
            NetUtil.printConfusionMatrix(NetUtil.makeConfusionMatrix(data))
        }
    }

    static get version () {
        return "3.3.1"
    }
}

/* istanbul ignore next */
typeof window!="undefined" && (window.Network = Network)
exports.Network = Network
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
            case "momentum":
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

/* istanbul ignore next */
typeof window!="undefined" && (window.Neuron = Neuron)
exports.Neuron = Neuron
"use strict"

class OutputLayer extends FCLayer {

    constructor (size, {activation, softmax}={}) {

        super(size, {activation})

        if (softmax) {
            this.softmax = true
        }
    }

    forward () {

        super.forward()

        if (this.softmax) {

            const softmax = NetMath.softmax(this.neurons.map(n => n.activation))

            for (let s=0; s<softmax.length; s++) {
                this.neurons[s].activation = softmax[s]
            }
        }
    }
}

/* istanbul ignore next */
typeof window!="undefined" && (window.OutputLayer = OutputLayer)
exports.OutputLayer = OutputLayer

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
        this.layerIndex = layerIndex

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

    backUpValidation () {}

    restoreValidation () {}

    toJSON () {return {}}

    fromJSON () {}

    getDataSize () {return 0}

    toIMG () {return []}

    fromIMG () {}
}

/* istanbul ignore next */
typeof window!="undefined" && (window.PoolLayer = PoolLayer)
exports.PoolLayer = PoolLayer
//# sourceMappingURL=jsNetJS.concat.js.map