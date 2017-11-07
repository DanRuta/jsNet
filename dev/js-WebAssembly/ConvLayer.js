"use strict"

class ConvLayer {

    constructor (size, {filterSize, zeroPadding, stride, activation}={}) {

        this.size = size
        this.stride = stride
        this.filterSize = filterSize
        this.layerIndex = 0
        this.zeroPadding = zeroPadding

        if (activation != undefined) {
            if (typeof activation != "string") {
                throw new Error("Only string activation functions available in the WebAssembly version")
            }
            this.activation = NetUtil.format(activation)
        }
    }

    assignNext (layer) {
        this.nextLayer = layer
    }

    assignPrev (layer, layerIndex) {

        this.netInstance = this.net.netInstance
        this.prevLayer = layer
        this.layerIndex = layerIndex

        NetUtil.defineProperty(this, "channels", ["number", "number"], [this.netInstance, layerIndex], {pre: "conv_"})
        NetUtil.defineProperty(this, "filterSize", ["number", "number"], [this.netInstance, layerIndex], {pre: "conv_"})
        NetUtil.defineProperty(this, "stride", ["number", "number"], [this.netInstance, layerIndex], {pre: "conv_"})
        NetUtil.defineProperty(this, "zeroPadding", ["number", "number"], [this.netInstance, layerIndex], {pre: "conv_"})

        const stride = this.stride || this.net.conv.stride || 1
        const filterSize = this.filterSize || this.net.conv.filterSize || 3
        let zeroPadding = this.zeroPadding

        this.size = this.size || 4
        let channels

        switch (true) {
            case layer instanceof FCLayer:
                channels = this.net.channels || 1
                break

            case layer instanceof ConvLayer:
                channels = layer.size
                break
        }

        if (zeroPadding == undefined) {
            zeroPadding = this.net.conv.zeroPadding==undefined ? Math.floor(filterSize/2) : this.net.conv.zeroPadding
        }

        this.channels = channels
        this.filterSize = filterSize
        this.stride = stride
        this.zeroPadding = zeroPadding

        // Caching calculations
        const prevLayerOutWidth = layer instanceof FCLayer ? Math.max(Math.floor(Math.sqrt(layer.size/channels)), 1)
                                                           : layer.outMapSize

        NetUtil.defineProperty(this, "inMapValuesCount", ["number", "number"], [this.netInstance, layerIndex], {pre: "conv_"})
        NetUtil.defineProperty(this, "inZPMapValuesCount", ["number", "number"], [this.netInstance, layerIndex], {pre: "conv_"})
        NetUtil.defineProperty(this, "outMapSize", ["number", "number"], [this.netInstance, layerIndex], {pre: "conv_"})

        const outSize = (prevLayerOutWidth - filterSize + 2*zeroPadding) / stride + 1
        this.inMapValuesCount = Math.pow(prevLayerOutWidth, 2)
        this.inZPMapValuesCount = Math.pow(prevLayerOutWidth + zeroPadding*2, 2)
        this.outMapSize = outSize

        if (outSize%1!=0) {
            throw new Error(`Misconfigured hyperparameters. Activation volume dimensions would be ${outSize} in conv layer at index ${layerIndex}`)
        }

        if (this.activation != false) {
            this.net.Module.ccall("setConvActivation", null, ["number", "number", "number"],
                [this.netInstance, NetUtil.activationsIndeces[this.activation||this.net.activationName], layerIndex])
        }

        this.filters = [...new Array(this.size)].map(f => new Filter())
    }

    init () {
        this.filters.forEach((filter, fi) => {
            filter.init(this.netInstance, this.layerIndex, fi, {
                updateFn: this.net.updateFn,
                filterSize: this.filterSize,
                channels: this.channels
            })
        })
    }

    toJSON () {
        return {
            weights: this.filters.map(filter => {
                return {
                    bias: filter.bias,
                    weights: filter.weights
                }
            })
        }
    }

    fromJSON (data, layerIndex) {
        this.filters.forEach((filter, fi) => {

            if (data.weights[fi].weights.length != filter.weights.length) {
                throw new Error(`Mismatched weights depth. Given: ${data.weights[fi].weights.length} Existing: ${filter.weights.length}. At: layers[${layerIndex}], filters[${fi}]`)
            }

            if (data.weights[fi].weights[0].length != filter.weights[0].length) {
                throw new Error(`Mismatched weights size. Given: ${data.weights[fi].weights[0].length} Existing: ${filter.weights[0].length}. At: layers[${layerIndex}], filters[${fi}]`)
            }

            filter.bias = data.weights[fi].bias
            filter.weights = data.weights[fi].weights
        })
    }

}

typeof window=="undefined" && (exports.ConvLayer = ConvLayer)
