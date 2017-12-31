"use strict"

class PoolLayer {

    constructor (size, {stride, activation}={}) {

        if (size)   this.size = size
        if (stride) this.stride = stride

        this.activation = false
        this.activationName = activation

        if (activation) {
            if (typeof activation != "string") {
                throw new Error("Custom activation functions are not available in the WebAssembly version")
            }
            this.activationName = NetUtil.format(activation)
        }
    }

    assignNext (layer) {
        this.nextLayer = layer
    }

    assignPrev (layer, layerIndex) {

        this.netInstance = this.net.netInstance
        this.prevLayer = layer
        this.layerIndex = layerIndex

        let channels
        let prevLayerOutWidth = layer.outMapSize
        const size = this.size || this.net.pool.size || 2
        const stride = this.stride || this.net.pool.stride || this.size

        NetUtil.defineProperty(this, "channels", ["number", "number"], [this.netInstance, layerIndex], {pre: "pool_"})
        NetUtil.defineProperty(this, "stride", ["number", "number"], [this.netInstance, layerIndex], {pre: "pool_"})
        this.size = size
        this.stride = stride

        switch (true) {

            case layer instanceof FCLayer:
                channels = this.net.channels
                prevLayerOutWidth = Math.max(Math.floor(Math.sqrt(layer.size/channels)), 1)
                break

            case layer instanceof ConvLayer:
                channels = layer.size
                break

            case layer instanceof PoolLayer:
                channels = layer.channels
                break
        }

        this.channels = channels

        NetUtil.defineProperty(this, "prevLayerOutWidth", ["number", "number"], [this.netInstance, layerIndex], {pre: "pool_"})
        NetUtil.defineProperty(this, "inMapValuesCount", ["number", "number"], [this.netInstance, layerIndex], {pre: "pool_"})
        NetUtil.defineProperty(this, "outMapSize", ["number", "number"], [this.netInstance, layerIndex], {pre: "pool_"})
        NetUtil.defineVolumeProperty(this, "errors", ["number", "number"], [this.netInstance, layerIndex], channels, prevLayerOutWidth, prevLayerOutWidth, {pre: "pool_"})

        const outMapSize = (prevLayerOutWidth - size) / stride + 1
        this.outMapSize = outMapSize
        this.inMapValuesCount = prevLayerOutWidth ** 2

        NetUtil.defineVolumeProperty(this, "activations", ["number", "number"], [this.netInstance, layerIndex], channels, outMapSize, outMapSize, {pre: "pool_"})
        NetUtil.defineVolumeProperty(this, "indeces", ["number", "number"], [this.netInstance, layerIndex], channels, outMapSize, outMapSize, {
            pre: "pool_",
            getCallback: vol => vol.map(map => map.map(row => row.map(val => [parseInt(val/2), val%2]))),
            setCallback: vol => vol.map(map => map.map(row => row.map(([x,y]) => 2*x+y)))
        })

        if (outMapSize%1 != 0) {
            throw new Error(`Misconfigured hyperparameters. Activation volume dimensions would be ${outMapSize} in pool layer at index ${layerIndex}`)
        }

        if (this.activationName) {
            NetUtil.defineProperty(this, "activation", ["number", "number"], [this.netInstance, layerIndex], {
                pre: "pool_",
                getCallback: _ => `WASM ${this.activationName}`
            })
            this.activation = NetUtil.activationsIndeces[this.activationName]
        }
    }

    init () {}

    toJSON () {
        return {}
    }

    fromJSON() {}

}

typeof window=="undefined" && (exports.PoolLayer = PoolLayer)
