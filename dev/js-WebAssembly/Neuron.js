"use strict"

class Neuron {

    constructor () {}

    init (netInstance, layerIndex, neuronIndex) {

        Object.defineProperty(this, "weights", {
            get: () => NetUtil.ccallArrays("getNeuronWeights", "array", ["number", "number", "number"],
                    [netInstance, layerIndex, neuronIndex], {returnArraySize: this.size, heapOut: "HEAPF64"}),
            set: weights => NetUtil.ccallArrays("setNeuronWeights", null, ["number", "number", "number", "array"],
                    [netInstance, layerIndex, neuronIndex, weights], {heapIn: "HEAPF64"})
        })
        Object.defineProperty(this, "bias", {
            get: () => NetUtil.ccallArrays("getNeuronBias", "number", ["number", "number", "number"],
                [netInstance, layerIndex, neuronIndex]),
            set: value => NetUtil.ccallArrays("setNeuronBias", null, ["number", "number", "number", "number"],
                [netInstance, layerIndex, neuronIndex, value])
        })
        Object.defineProperty(this, "deltaWeights", {
            get: () => NetUtil.ccallArrays("getNeuronDeltaWeights", "array", ["number", "number", "number"],
                [netInstance, layerIndex, neuronIndex], {returnArraySize: this.size, heapOut: "HEAPF64"}),
            set: deltaWeights => NetUtil.ccallArrays("setNeuronDeltaWeights", null, ["number", "number", "number", "array"],
                    [netInstance, layerIndex, neuronIndex, deltaWeights], {heapIn: "HEAPF64"})
        })
    }

}

typeof window=="undefined" && (exports.Neuron = Neuron)