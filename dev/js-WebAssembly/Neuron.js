"use strict"

class Neuron {

    constructor () {}

    init (netInstance, layerIndex, neuronIndex, {updateFn}) {

        NetUtil.defineArrayProperty(this, "weights", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex], this.size)
        NetUtil.defineProperty(this, "bias", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex])
        NetUtil.defineArrayProperty(this, "deltaWeights", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex], this.size)

        switch (updateFn) {
            case "gain":
                NetUtil.defineProperty(this, "biasGain", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex])
                NetUtil.defineArrayProperty(this, "weightGain", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex], this.size)
                break
            case "adagrad":
            case "rmsprop":
                NetUtil.defineProperty(this, "biasCache", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex])
                NetUtil.defineArrayProperty(this, "weightsCache", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex], this.size)
                break

            case "adam":
                NetUtil.defineProperty(this, "m", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex])
                NetUtil.defineProperty(this, "v", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex])
                break
        }

    }

}

typeof window=="undefined" && (exports.Neuron = Neuron)