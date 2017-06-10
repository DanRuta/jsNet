"use strict"

class NetMath {
    
    static sigmoid (value, prime) {
        return prime ? NetMath.sigmoid(value)*(1-NetMath.sigmoid(value))
                     : 1/(1+Math.exp(-value))
    }

    static crossEntropy (target, output) {
        return output.map((value, vi) => target[vi] * Math.log(value+1e-15) + ((1-target[vi]) * Math.log((1+1e-15)-value)))
                     .reduce((p,c) => p-c, 0)
    }

    static softmax (values) {
        const total = values.reduce((prev, curr) => prev+curr, 0)
        return values.map(value => value/total)
    }

    static meanSquaredError (calculated, desired) {
        return calculated.map((output, index) => Math.pow(output - desired[index], 2))
                         .reduce((prev, curr) => prev+curr, 0) / calculated.length
    }
}

typeof window=="undefined" && (global.NetMath = NetMath)