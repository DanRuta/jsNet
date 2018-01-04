"use strict"

class Neuron {

    constructor () {}

    init ({updateFn, activation, eluAlpha}={}) {

        const size = this.weights.length
        this.deltaWeights = this.weights.map(v => 0)

        switch (updateFn) {

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

                if (updateFn=="adadelta") {
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

        if (activation=="rrelu") {
            this.rreluSlope = Math.random() * 0.001

        } else if (activation=="elu") {
            this.eluAlpha = eluAlpha
        }
    }

    getWeight (i) {
        return this.weights[i]
    }

    setWeight (i, v) {
        this.weights[i] = v
    }

    getDeltaWeight (i) {
        return this.deltaWeights[i]
    }

    setDeltaWeight (i, v) {
        this.deltaWeights[i] = v
    }
}

/* istanbul ignore next */
typeof window!="undefined" && (window.Neuron = Neuron)
exports.Neuron = Neuron