"use strict"

class OutputLayer extends FCLayer {

    constructor (size, {activation, softmax}={}) {

        super(size, {activation})

        if (softmax) {
            this.softmax = true
        }
    }
}

/* istanbul ignore next */
typeof window!="undefined" && (window.OutputLayer = OutputLayer)
exports.OutputLayer = OutputLayer
