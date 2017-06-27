"use strict"

class Layer {
    
    constructor (size, importedData) {
        this.size = size
        this.neurons = [...new Array(size)].map((n, ni) => new Neuron(importedData ? importedData[ni] : undefined))
    }

    assignNext (layer) {
        this.nextLayer = layer
    }

    assignPrev (layer) {
        this.prevLayer = layer
        this.neurons.forEach(neuron => neuron.init(layer.size, {
            adaptiveLR: this.adaptiveLR,
            activationConfig: this.activationConfig,
            eluAlpha: this.eluAlpha
        }))
    }

    forward (data) {

        this.neurons.forEach((neuron, ni) => {

            neuron.sum = neuron.bias
            this.prevLayer.neurons.forEach((pNeuron, pni) => neuron.sum += pNeuron.activation * neuron.weights[pni])
            neuron.activation = this.activation(neuron.sum, false, neuron)
        })
    }

    backward (expected) {
        this.neurons.forEach((neuron, ni) => {

            if(typeof expected !== "undefined") {
                neuron.error = expected[ni] - neuron.activation
            }else {
                neuron.derivative = this.activation(neuron.sum, true, neuron)
                neuron.error = neuron.derivative * this.nextLayer.neurons.map(n => n.error * n.weights[ni])
                                                                         .reduce((p,c) => p+c, 0)
            }

            neuron.weights.forEach((weight, wi) => {
                neuron.deltaWeights[wi] += neuron.error * this.prevLayer.neurons[wi].activation
            })

            neuron.deltaBias = neuron.error
        })
    }
}

typeof window=="undefined" && (global.Layer = Layer) 