"use strict"

class Filter {

    constructor () {}

    init (netInstance, layerIndex, filterIndex, {updateFn, channels, filterSize}) {

        const paramTypes = ["number", "number", "number"]
        const params = [netInstance, layerIndex, filterIndex]

        NetUtil.defineProperty(this, "bias", paramTypes, params, {pre: "filter_"})
        NetUtil.defineVolumeProperty(this, "weights", paramTypes, params, channels, filterSize, filterSize, {pre: "filter_"})
        NetUtil.defineProperty(this, "deltaBias", paramTypes, params, {pre: "filter_"})
        NetUtil.defineVolumeProperty(this, "deltaWeights", paramTypes, params, channels, filterSize, filterSize, {pre: "filter_"})

        switch (updateFn) {
            case "gain":
                NetUtil.defineProperty(this, "biasGain", paramTypes, params, {pre: "filter_"})
                NetUtil.defineVolumeProperty(this, "weightGain", paramTypes, params, channels, filterSize, filterSize, {pre: "filter_"})
                break
            case "adagrad":
            case "rmsprop":
            case "adadelta":
                NetUtil.defineProperty(this, "biasCache", paramTypes, params, {pre: "filter_"})
                NetUtil.defineVolumeProperty(this, "weightsCache", paramTypes, params, channels, filterSize, filterSize, {pre: "filter_"})

                if (updateFn == "adadelta") {
                    NetUtil.defineProperty(this, "adadeltaBiasCache", paramTypes, params, {pre: "filter_"})
                    NetUtil.defineVolumeProperty(this, "adadeltaWeightsCache", paramTypes, params, channels, filterSize, filterSize, {pre: "filter_"})
                }
                break
            case "adam":
                NetUtil.defineProperty(this, "m", paramTypes, params, {pre: "filter_"})
                NetUtil.defineProperty(this, "v", paramTypes, params, {pre: "filter_"})
                break
        }
    }
}

typeof window=="undefined" && (exports.Filter = Filter)