"use strict"

class OutputLayer extends FCLayer {

    constructor (size, {activation, softmax}={}) {

        super(size, {activation})

        if (softmax) {
            this.softmax = true
        }
    }

    forward () {

        super.forward()

        if (this.softmax) {

            const softmax = NetMath.softmax(this.neurons.map(n => n.activation))

            for (let s=0; s<softmax.length; s++) {
                this.neurons[s].activation = softmax[s]
            }
        }
    }
}

/* istanbul ignore next */
typeof window!="undefined" && (window.OutputLayer = OutputLayer)
exports.OutputLayer = OutputLayer
