"use strict"

class FCLayer {

    constructor (size, {activation}={}) {
        this.size = size
        this.neurons = [...new Array(size)].map(n => new Neuron())
        this.layerIndex = 0

        if (activation != undefined) {
            if (typeof activation == "boolean" && !activation) {
                activation = "noactivation"
            }
            if (typeof activation != "string") {
                throw new Error("Custom activation functions are not available in the WebAssembly version")
            }
            this.activationName = NetUtil.format(activation)
        }
    }

    assignNext (layer) {
        this.nextLayer = layer
    }

    assignPrev (layer, layerIndex) {
        this.netInstance = this.net.netInstance
        this.prevLayer = layer
        this.layerIndex = layerIndex

        if (this.activationName || this.net.activationName) {
            NetUtil.defineProperty(this, "activation", ["number", "number"], [this.netInstance, layerIndex], {
                pre: "fc_",
                getCallback: _ => `WASM ${this.activationName||this.net.activationName}`
            })
            this.activation = NetUtil.activationsIndeces[this.activationName||this.net.activationName]
        }
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

            neuron.weights = data.slice(valI, valI+neuron.weights.length)
            valI += neuron.weights.length
        }
    }
}

const Layer = FCLayer

/* istanbul ignore next */
typeof window!="undefined" && (window.FCLayer = window.Layer = FCLayer)
exports.FCLayer = exports.Layer = FCLayer