"use strict"

class Neuron {
    
    constructor (importedData) {
        if (importedData) {
            this.imported = true
            this.weights = importedData.weights || []
            this.bias = importedData.bias
        }
    }

    init (size, {adaptiveLR, activationConfig, eluAlpha}={}) {

        this.deltaWeights = this.weights.map(v => 0)

        switch (adaptiveLR) {
            
            case "gain":
                this.biasGain = 1
                this.weightGains = [...new Array(size)].map(v => 1)
                this.getWeightGain = i => this.weightGains[i]
                this.setWeightGain = (i,v) => this.weightGains[i] = v
                break

            case "adagrad":
            case "rmsprop":
            case "adadelta":
                this.biasCache = 0
                this.weightsCache = [...new Array(size)].map(v => 0)
                this.getWeightsCache = i => this.weightsCache[i]
                this.setWeightsCache = (i,v) => this.weightsCache[i] = v

                if (adaptiveLR=="adadelta") {
                    this.adadeltaBiasCache = 0
                    this.adadeltaCache = [...new Array(size)].map(v => 0)
                    this.getAdadeltaCache = i => this.adadeltaCache[i]
                    this.setAdadeltaCache = (i,v) => this.adadeltaCache[i] = v
                }
                break

            case "adam":
                this.m = 0
                this.v = 0
                break
        }

        if (activationConfig=="rrelu") {
            this.rreluSlope = Math.random() * 0.001
            
        } else if (activationConfig=="elu") {
            this.eluAlpha = eluAlpha
        }
    }

    getWeight (i) {
        return this.weights[i]
    }

    setWeight (i,v) {
        this.weights[i] = v
    }

    getDeltaWeight (i) {
        return this.deltaWeights[i]
    }

    setDeltaWeight (i,v) {
        this.deltaWeights[i] = v
    }
}

typeof window=="undefined" && (exports.Neuron = Neuron)