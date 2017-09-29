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

            neuron.init(this.netInstance, this.layerIndex, ni)
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