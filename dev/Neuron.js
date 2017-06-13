"use strict"

class Neuron {
    
    constructor (importedData) {
        if(importedData){
            this.imported = true
            this.weights = importedData.weights || []
            this.bias = importedData.bias
        }
    }

    init (size, adaptiveLR) {
        if(!this.imported){
            this.weights = [...new Array(size)].map(v => Math.random()*0.2-0.1)
            this.bias = Math.random()*0.2-0.1
        }

        this.deltaWeights = this.weights.map(v => 0)

        if(adaptiveLR=="gain"){
            this.weightGains = [...new Array(size)].map(v => 1)
            this.biasGain = 1

        }else if(adaptiveLR=="adagrad" || adaptiveLR=="RMSProp"){
            this.weightsCache = [...new Array(size)].map(v => 0)
            this.biasCache = 0
        }
    }
}

typeof window=="undefined" && (global.Neuron = Neuron)