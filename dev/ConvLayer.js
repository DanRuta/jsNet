"use strict"

class ConvLayer {

    constructor (size, {filterSize, zeroPadding, stride, activation}={}) {

        if (filterSize)     this.filterSize = filterSize
        if (stride)         this.stride = stride
        if (size)           this.size = size

        this.zeroPadding = zeroPadding

        if (activation!=undefined) {

            if (typeof activation=="boolean" && !activation) {
                this.activation = NetMath.noactivation
            } else {
                this.activation = typeof activation=="function" ? activation : NetMath[NetUtil.format(activation)].bind(this)
            }
        }

        this.state = "not-initialised"
    }

    assignNext (layer) {
        this.nextLayer = layer
    }

    assignPrev (layer) {

        this.prevLayer = layer

        this.filterSize = this.filterSize || this.net.filterSize || 3
        this.stride = this.stride || this.net.stride || 1
        this.size = this.size || this.net.filterCount || 4
        this.channels = layer instanceof ConvLayer ? layer.size : (this.net.channels || 1)

        if (this.zeroPadding==undefined) {
            this.zeroPadding = this.net.zeroPadding==undefined ? Math.floor(this.filterSize/2) : this.net.zeroPadding
        }

        // Caching calculations
        const prevLayerMapWidth = layer instanceof ConvLayer ? layer.outMapSize
                                                             : Math.max(Math.floor(Math.sqrt(layer.size/this.channels)), 1)

        this.inMapValuesCount = Math.pow(prevLayerMapWidth, 2)
        this.inZPMapValuesCount = Math.pow(prevLayerMapWidth + this.zeroPadding*2, 2)
        this.outMapSize = (prevLayerMapWidth - this.filterSize + 2*this.zeroPadding) / this.stride + 1

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
                adaptiveLR: this.net.adaptiveLR,
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
                    } else {
                        filter.activationMap[sumY][sumX] = this.activation(filter.sumMap[sumY][sumX], false, filter) / (this.net.dropout||1)
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

        } else {
            for (let filterI=0; filterI<this.filters.length; filterI++) {
                NetUtil.buildConvErrorMap(this, this.filters[filterI], filterI)
            }
        }

        // Apply derivative to each error value
        for (let filterI=0; filterI<this.filters.length; filterI++) {

            const filter = this.filters[filterI]

            for (let row=0; row<filter.errorMap.length; row++) {
                for (let col=0; col<filter.errorMap[0].length; col++) {

                    if (filter.dropoutMap[row][col]) {
                        filter.errorMap[row][col] = 0
                    } else {
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