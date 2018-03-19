"use strict"

class InputLayer extends FCLayer {
    constructor (size, {span=1}={}) {
        super(size * span*span)
    }
}

/* istanbul ignore next */
typeof window!="undefined" && (window.InputLayer = InputLayer)
exports.InputLayer = InputLayer
