"use strict"

class Filter {

    constructor () {}

    init ({updateFn, activation, eluAlpha}={}) {

        this.deltaWeights = this.weights.map(channel => channel.map(wRow => wRow.map(w => 0)))
        this.deltaBias = 0

        switch (updateFn) {

            case "gain":
                this.biasGain = 1
                this.weightGains = this.weights.map(channel => channel.map(wRow => wRow.map(w => 1)))
                this.getWeightGain = ([channel, row, column]) => this.weightGains[channel][row][column]
                this.setWeightGain = ([channel, row, column], v) => this.weightGains[channel][row][column] = v
                break

            case "adagrad":
            case "rmsprop":
            case "adadelta":
                this.biasCache = 0
                this.weightsCache = this.weights.map(channel => channel.map(wRow => wRow.map(w => 0)))
                this.getWeightsCache = ([channel, row, column]) => this.weightsCache[channel][row][column]
                this.setWeightsCache = ([channel, row, column], v) => this.weightsCache[channel][row][column] = v

                if (updateFn=="adadelta") {
                    this.adadeltaBiasCache = 0
                    this.adadeltaCache = this.weights.map(channel => channel.map(wRow => wRow.map(w => 0)))
                    this.getAdadeltaCache = ([channel, row, column]) => this.adadeltaCache[channel][row][column]
                    this.setAdadeltaCache = ([channel, row, column], v) => this.adadeltaCache[channel][row][column] = v
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

    getWeight ([channel, row, column]) {
        return this.weights[channel][row][column]
    }

    setWeight ([channel, row, column], v) {
        this.weights[channel][row][column] = v
    }

    getDeltaWeight ([channel, row, column]) {
        return this.deltaWeights[channel][row][column]
    }

    setDeltaWeight ([channel, row, column], v) {
        this.deltaWeights[channel][row][column] = v
    }
}

/* istanbul ignore next */
typeof window!="undefined" && (window.Filter = Filter)
exports.Filter = Filter