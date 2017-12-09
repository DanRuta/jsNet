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

    toJSON () {return {}}

    fromJSON () {}
}

typeof window=="undefined" && (exports.PoolLayer = PoolLayer)