"use strict"

class Neuron {

    constructor () {}

    init (netInstance, layerIndex, neuronIndex) {


        NetUtil.defineArrayProperty(this, "weights", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex], this.size)
        NetUtil.defineProperty(this, "bias", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex])
        NetUtil.defineArrayProperty(this, "deltaWeights", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex], this.size)

        NetUtil.defineProperty(this, "biasGain", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex])
        NetUtil.defineArrayProperty(this, "weightGain", ["number", "number", "number"], [netInstance, layerIndex, neuronIndex], this.size)

    }

}

typeof window=="undefined" && (exports.Neuron = Neuron)