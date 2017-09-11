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
        for (let n=0; n<this.neurons.length; n++) {
            for (let dwi=0; dwi<this.neurons[n].deltaWeights.length; dwi++) {
                this.neurons[n].deltaWeights[dwi] = 0
            }
        }
    }

    applyDeltaWeights () {
        for (let n=0; n<this.neurons.length; n++) {

            const neuron = this.neurons[n]

            for (let dwi=0; dwi<this.neurons[n].deltaWeights.length; dwi++) {

                if (this.net.l2!=undefined) this.net.l2Error += 0.5 * this.net.l2 * neuron.weights[dwi]**2
                if (this.net.l1!=undefined) this.net.l1Error += this.net.l1 * Math.abs(neuron.weights[dwi])

                neuron.weights[dwi] = this.net.weightUpdateFn.bind(this.net, neuron.weights[dwi], neuron.deltaWeights[dwi], neuron, dwi)()

                if (this.net.maxNorm!=undefined) this.net.maxNormTotal += neuron.weights[dwi]**2
            }

            neuron.bias = this.net.weightUpdateFn.bind(this.net, neuron.bias, neuron.deltaBias, neuron)()
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
}

const Layer = FCLayer

typeof window=="undefined" && (exports.FCLayer = exports.Layer = FCLayer)