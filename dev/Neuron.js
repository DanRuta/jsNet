"use strict"

class Neuron {
    
    constructor (importedData) {
        if(importedData){
            this.imported = true
            this.weights = importedData.weights || []
            this.bias = importedData.bias
        }
    }

    init (size) {
        if(!this.imported){
            this.weights = [...new Array(size)].map(v => Math.random()*0.2-0.1)
            this.bias = Math.random()*0.2-0.1
        }

        this.deltaWeights = this.weights.map(v => 0)
        this.weightGains = [...new Array(size)].map(v => 1)
        this.biasGain = 1
    }
}

typeof window=="undefined" && (global.Neuron = Neuron)