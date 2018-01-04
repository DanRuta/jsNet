"use strict"

class Neuron {

    constructor () {}

    init (netInstance, layerIndex, neuronIndex, {updateFn}) {

        const paramTypes = ["number", "number", "number"]
        const params = [netInstance, layerIndex, neuronIndex]

        NetUtil.defineProperty(this, "sum", paramTypes, params, {pre: "neuron_"})
        NetUtil.defineProperty(this, "dropped", paramTypes, params, {
            pre: "neuron_",
            getCallback: v => v==1,
            setCallback: v => v ? 1 : 0
        })
        NetUtil.defineProperty(this, "activation", paramTypes, params, {pre: "neuron_"})
        NetUtil.defineProperty(this, "error", paramTypes, params, {pre: "neuron_"})
        NetUtil.defineProperty(this, "derivative", paramTypes, params, {pre: "neuron_"})

        NetUtil.defineProperty(this, "bias", paramTypes, params, {pre: "neuron_"})
        NetUtil.defineArrayProperty(this, "weights", paramTypes, params, this.size, {pre: "neuron_"})

        NetUtil.defineProperty(this, "deltaBias", paramTypes, params, {pre: "neuron_"})
        NetUtil.defineArrayProperty(this, "deltaWeights", paramTypes, params, this.size, {pre: "neuron_"})

        switch (updateFn) {
            case "gain":
                NetUtil.defineProperty(this, "biasGain", paramTypes, params, {pre: "neuron_"})
                NetUtil.defineArrayProperty(this, "weightGain", paramTypes, params, this.size, {pre: "neuron_"})
                break
            case "adagrad":
            case "rmsprop":
            case "adadelta":
                NetUtil.defineProperty(this, "biasCache", paramTypes, params, {pre: "neuron_"})
                NetUtil.defineArrayProperty(this, "weightsCache", paramTypes, params, this.size, {pre: "neuron_"})

                if (updateFn=="adadelta") {
                    NetUtil.defineProperty(this, "adadeltaBiasCache", paramTypes, params, {pre: "neuron_"})
                    NetUtil.defineArrayProperty(this, "adadeltaCache", paramTypes, params, this.size, {pre: "neuron_"})
                }
                break

            case "adam":
                NetUtil.defineProperty(this, "m", paramTypes, params, {pre: "neuron_"})
                NetUtil.defineProperty(this, "v", paramTypes, params, {pre: "neuron_"})
                break
        }
    }
}

/* istanbul ignore next */
typeof window!="undefined" && (window.Neuron = Neuron)
exports.Neuron = Neuron