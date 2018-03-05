"use strict"

class FCLayer {

    constructor (size, {activation}={}) {
        this.size = size
        this.neurons = [...new Array(size)].map(n => new Neuron())
        this.state = "not-initialised"

        if (activation!=undefined) {
            if (typeof activation=="boolean" && !activation) {
                this.activation = false
            } else {
                this.activation = typeof activation=="function" ? activation : NetMath[NetUtil.format(activation)].bind(this)
            }
        }
    }

    assignNext (layer) {
        this.nextLayer = layer
    }

    assignPrev (layer, layerIndex) {
        this.prevLayer = layer
        this.layerIndex = layerIndex
    }

    init () {
        this.neurons.forEach(neuron => {

            let weightsCount

            switch (true) {
                case this.prevLayer instanceof FCLayer:
                    weightsCount = this.prevLayer.size
                    break

                case this.prevLayer instanceof ConvLayer:
                    weightsCount = this.prevLayer.filters.length * this.prevLayer.outMapSize**2
                    break

                case this.prevLayer instanceof PoolLayer:
                    weightsCount = this.prevLayer.activations.length * this.prevLayer.outMapSize**2
                    break
            }

            neuron.weights = this.net.weightsInitFn(weightsCount, this.weightsConfig)
            neuron.bias = 1

            neuron.init({
                updateFn: this.net.updateFn,
                activationConfig: this.net.activationConfig,
                eluAlpha: this.net.eluAlpha
            })
        })
    }

    forward () {
        this.neurons.forEach((neuron, ni) => {
            if (this.state=="training" && (neuron.dropped = Math.random() > this.net.dropout)) {
                neuron.activation = 0
            } else {
                neuron.sum = neuron.bias

                const activations = NetUtil.getActivations(this.prevLayer)

                for (let ai=0; ai<activations.length; ai++) {
                    neuron.sum += activations[ai] * neuron.weights[ai]
                }

                neuron.activation = (this.activation ? this.activation(neuron.sum, false, neuron) : neuron.sum) / (this.net.dropout||1)
            }
        })
    }

    backward (errors) {
        this.neurons.forEach((neuron, ni) => {

            if (neuron.dropped) {
                neuron.error = 0
                neuron.deltaBias += 0
            } else {
                if (typeof errors !== "undefined") {
                    neuron.error = errors[ni]
                } else {
                    neuron.derivative = this.activation ? this.activation(neuron.sum, true, neuron) : 1
                    neuron.error = neuron.derivative * this.nextLayer.neurons.map(n => n.error * (n.weights[ni]||0))
                                                                             .reduce((p,c) => p+c, 0)
                }

                const activations = NetUtil.getActivations(this.prevLayer)

                for (let wi=0; wi<neuron.weights.length; wi++) {
                    neuron.deltaWeights[wi] += (neuron.error * activations[wi])
                }

                neuron.deltaBias += neuron.error
            }
        })
    }

    resetDeltaWeights () {
        for (let n=0; n<this.neurons.length; n++) {

            this.neurons[n].deltaBias = 0

            for (let dwi=0; dwi<this.neurons[n].deltaWeights.length; dwi++) {
                this.neurons[n].deltaWeights[dwi] = 0
            }
        }
    }

    applyDeltaWeights () {
        for (let n=0; n<this.neurons.length; n++) {

            const neuron = this.neurons[n]

            for (let dwi=0; dwi<this.neurons[n].deltaWeights.length; dwi++) {

                if (this.net.l2Error!=undefined) this.net.l2Error += 0.5 * this.net.l2 * neuron.weights[dwi]**2
                if (this.net.l1Error!=undefined) this.net.l1Error += this.net.l1 * Math.abs(neuron.weights[dwi])

                const regularized = (neuron.deltaWeights[dwi]
                    + this.net.l2 * neuron.weights[dwi]
                    + this.net.l1 * (neuron.weights[dwi] > 0 ? 1 : -1)) / this.net.miniBatchSize

                neuron.weights[dwi] = this.net.weightUpdateFn.bind(this.net, neuron.weights[dwi], regularized, neuron, dwi)()

                if (this.net.maxNorm!=undefined) this.net.maxNormTotal += neuron.weights[dwi]**2
            }

            neuron.bias = this.net.weightUpdateFn.bind(this.net, neuron.bias, neuron.deltaBias, neuron)()
        }
    }

    backUpValidation () {
        for (let n=0; n<this.neurons.length; n++) {
            const neuron = this.neurons[n]
            neuron.validationBias = neuron.bias
            neuron.validationWeights = neuron.weights.slice(0)
        }
    }

    restoreValidation () {
        for (let n=0; n<this.neurons.length; n++) {
            const neuron = this.neurons[n]
            neuron.bias = neuron.validationBias
            neuron.weights = neuron.validationWeights.slice(0)
        }
    }

    toJSON () {
        return {
            weights: this.neurons.map(neuron => {
                return {
                    bias: neuron.bias,
                    weights: neuron.weights
                }
            })
        }
    }

    fromJSON (data, layerIndex) {
        this.neurons.forEach((neuron, ni) => {

            if (data.weights[ni].weights.length!=neuron.weights.length) {
                throw new Error(`Mismatched weights count. Given: ${data.weights[ni].weights.length} Existing: ${neuron.weights.length}. At layers[${layerIndex}], neurons[${ni}]`)
            }

            neuron.bias = data.weights[ni].bias
            neuron.weights = data.weights[ni].weights
        })
    }

    // Used for importing data
    getDataSize () {

        let size = 0

        for (let n=0; n<this.neurons.length; n++) {
            size += this.neurons[n].weights.length + 1
        }

        return size
    }

    toIMG () {
        const data = []

        for (let n=0; n<this.neurons.length; n++) {
            data.push(this.neurons[n].bias)

            for (let w=0; w<this.neurons[n].weights.length; w++) {
                data.push(this.neurons[n].weights[w])
            }
        }

        return data
    }

    fromIMG (data) {

        let valI = 0

        for (let n=0; n<this.neurons.length; n++) {

            const neuron = this.neurons[n]
            neuron.bias = data[valI]
            valI++

            for (let w=0; w<neuron.weights.length; w++) {
                neuron.weights[w] = data[valI]
                valI++
            }
        }
    }
}

const Layer = FCLayer

/* istanbul ignore next */
typeof window!="undefined" && (window.FCLayer = window.Layer = FCLayer)
exports.FCLayer = exports.Layer = FCLayer