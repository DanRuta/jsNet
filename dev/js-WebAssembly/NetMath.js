"use strict"

class NetMath {
    static softmax (values) {
        let total = 0

        for (let i=0; i<values.length; i++) {
            total += values[i]
        }

        for (let i=0; i<values.length; i++) {
            if (total) {
                values[i] /= total
            }
        }

        return values
    }
}

/* istanbul ignore next */
typeof window!="undefined" && (window.NetMath = NetMath)
exports.NetMath = NetMath