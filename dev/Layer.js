"use strict"

class Layer {
    
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
        this.neurons.forEach(neuron => neuron.init(layer.size, {
            adaptiveLR: this.adaptiveLR,
            activationConfig: this.activationConfig,
            eluAlpha: this.eluAlpha
        }))
        this.state = "initialised"
    }

    forward (data) {

        this.neurons.forEach((neuron, ni) => {

            if(this.state=="training" && (neuron.dropped = Math.random() > this.dropout)) {
                neuron.activation = 0
            }else {
                neuron.sum = neuron.bias
                this.prevLayer.neurons.forEach((pNeuron, pni) => neuron.sum += pNeuron.activation * neuron.weights[pni])
                neuron.activation = this.activation(neuron.sum, false, neuron) / (this.dropout|1)
            }
        })
    }

    backward (expected) {
        this.neurons.forEach((neuron, ni) => {

            if(neuron.dropped) {
                neuron.error = 0
                neuron.deltaBias = 0
            }else {
                if(typeof expected !== "undefined") {
                    neuron.error = expected[ni] - neuron.activation
                }else {
                    neuron.derivative = this.activation(neuron.sum, true, neuron)
                    neuron.error = neuron.derivative * this.nextLayer.neurons.map(n => n.error * (n.weights[ni]|0))
                                                                             .reduce((p,c) => p+c, 0)
                }

                neuron.weights.forEach((weight, wi) => {
                    neuron.deltaWeights[wi] += (neuron.error * this.prevLayer.neurons[wi].activation) * 
                                               (1 + (this.l2||0) * neuron.deltaWeights[wi])
                })

                neuron.deltaBias = neuron.error
            }            
        })
    }
}

typeof window=="undefined" && (global.Layer = Layer) 