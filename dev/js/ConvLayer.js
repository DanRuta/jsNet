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

// https://github.com/DanRuta/jsNet/issues/33
/* istanbul ignore next */
typeof window!="undefined" && (window.exports = window.exports || {})
/* istanbul ignore next */
typeof window!="undefined" && (window.ConvLayer = ConvLayer)
exports.ConvLayer = ConvLayer