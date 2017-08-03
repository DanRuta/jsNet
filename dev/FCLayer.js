"use strict"

class FCLayer {
    
    constructor (size, importedData) {
        this.size = size
        this.neurons = [...new Array(size)].map((n, ni) => new Neuron(importedData ? importedData[ni] : undefined))
        this.state = "not-initialised"
    }

    assignNext (layer) {
        this.nextLayer = layer
    }

    assignPrev (layer) {

        this.prevLayer = layer
        this.neurons.forEach(neuron => {

            if (!neuron.imported) {
                neuron.weights = this.net.weightsInitFn(layer.size, this.weightsConfig)
                neuron.bias = Math.random()*0.2-0.1
            }

            neuron.init(layer.size, {
                adaptiveLR: this.net.adaptiveLR,
                activationConfig: this.net.activationConfig,
                eluAlpha: this.net.eluAlpha
            })
        }) 
        this.state = "initialised"
    }

    forward (data) {

        this.neurons.forEach((neuron, ni) => {

            if (this.state=="training" && (neuron.dropped = Math.random() > this.net.dropout)) {
                neuron.activation = 0
            } else {
                neuron.sum = neuron.bias
                this.prevLayer.neurons.forEach((pNeuron, pni) => neuron.sum += pNeuron.activation * neuron.weights[pni])
                neuron.activation = this.activation(neuron.sum, false, neuron) / (this.net.dropout|1)
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

                neuron.weights.forEach((weight, wi) => {
                    neuron.deltaWeights[wi] += (neuron.error * this.prevLayer.neurons[wi].activation) * 
                                               (1 + (((this.net.l2||0)+(this.net.l1||0))/this.net.miniBatchSize) * neuron.deltaWeights[wi])
                })

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
}

const Layer = FCLayer

typeof window=="undefined" && (exports.FCLayer = exports.Layer = FCLayer)