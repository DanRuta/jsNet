"use strict"

class NetMath {

    // Activation functions
    static sigmoid (value, prime) {
        const val = 1/(1+Math.exp(-value))
        return prime ? val*(1-val)
                     : val
    }

    static tanh (value, prime) {
        const exp = Math.exp(2*value)
        return prime ? 4/Math.pow(Math.exp(value)+Math.exp(-value), 2) || 1e-18
                     : (exp-1)/(exp+1) || 1e-18
    }

    static relu (value, prime) {
        return prime ? value > 0 ? 1 : 0
                     : Math.max(value, 0)
    }

    static lrelu (value, prime) {
        return prime ? value > 0 ? 1 : this.lreluSlope
                     : Math.max(this.lreluSlope*Math.abs(value), value)
    }

    static rrelu (value, prime, neuron) {
        return prime ? value > 0 ? 1 : neuron.rreluSlope
                     : Math.max(neuron.rreluSlope, value)
    }

    static lecuntanh (value, prime) {
        return prime ? 1.15333 * Math.pow(NetMath.sech((2/3) * value), 2)
                     : 1.7159 * NetMath.tanh((2/3) * value)
    }

    static elu (value, prime, neuron) {
        return prime ? value >=0 ? 1 : NetMath.elu(value, false, neuron) + neuron.eluAlpha
                     : value >=0 ? value : neuron.eluAlpha * (Math.exp(value) - 1)
    }

    // Cost functions
    static crossentropy (target, output) {
        return output.map((value, vi) => target[vi] * Math.log(value+1e-15) + ((1-target[vi]) * Math.log((1+1e-15)-value)))
                     .reduce((p,c) => p-c, 0)
    }

    static meansquarederror (calculated, desired) {
        return calculated.map((output, index) => Math.pow(output - desired[index], 2))
                         .reduce((prev, curr) => prev+curr, 0) / calculated.length
    }

    // Weight updating functions
    static noadaptivelr (value, deltaValue) {
        return value + this.learningRate * deltaValue
    }

    static gain (value, deltaValue, neuron, weightI) {

        const newVal = value + this.learningRate * deltaValue * (weightI==null ? neuron.biasGain : neuron.getWeightGain(weightI))

        if (newVal<=0 && value>0 || newVal>=0 && value<0){
            if (weightI!=null) {
                neuron.setWeightGain(weightI, Math.max(neuron.getWeightGain(weightI)*0.95, 0.5))
            } else {
                neuron.biasGain = Math.max(neuron.biasGain*0.95, 0.5)
            }
        } else {
            if (weightI!=null) {
                neuron.setWeightGain(weightI, Math.min(neuron.getWeightGain(weightI)+0.05, 5))
            } else {
                neuron.biasGain = Math.min(neuron.biasGain+0.05, 5)
            }
        }

        return newVal
    }

    static adagrad (value, deltaValue, neuron, weightI) {

        if (weightI!=null) {
            neuron.setWeightsCache(weightI, neuron.getWeightsCache(weightI) + Math.pow(deltaValue, 2))
        } else {
            neuron.biasCache += Math.pow(deltaValue, 2)
        }

        return value + this.learningRate * deltaValue / (1e-6 + Math.sqrt(weightI!=null ? neuron.getWeightsCache(weightI)
                                                                                        : neuron.biasCache))
    }

    static rmsprop (value, deltaValue, neuron, weightI) {

        if (weightI!=null) {
            neuron.setWeightsCache(weightI, this.rmsDecay * neuron.getWeightsCache(weightI) + (1 - this.rmsDecay) * Math.pow(deltaValue, 2))
        } else {
            neuron.biasCache = this.rmsDecay * neuron.biasCache + (1 - this.rmsDecay) * Math.pow(deltaValue, 2)
        }

        return value + this.learningRate * deltaValue / (1e-6 + Math.sqrt(weightI!=null ? neuron.getWeightsCache(weightI)
                                                                                        : neuron.biasCache))
    }

    static adam (value, deltaValue, neuron) {

        neuron.m = 0.9*neuron.m + (1-0.9) * deltaValue
        const mt = neuron.m / (1-Math.pow(0.9, this.iterations + 1))

        neuron.v = 0.999*neuron.v + (1-0.999)*(Math.pow(deltaValue, 2))
        const vt = neuron.v / (1-Math.pow(0.999, this.iterations + 1))

        return value + this.learningRate * mt / (Math.sqrt(vt) + 1e-8)
    }

    static adadelta (value, deltaValue, neuron, weightI) {

        if (weightI!=null) {
            neuron.setWeightsCache(weightI, this.rho * neuron.getWeightsCache(weightI) + (1-this.rho) * Math.pow(deltaValue, 2))
            const newVal = value + Math.sqrt((neuron.getAdadeltaCache(weightI) + 1e-6)/(neuron.getWeightsCache(weightI) + 1e-6)) * deltaValue
            neuron.setAdadeltaCache(weightI, this.rho * neuron.getAdadeltaCache(weightI) + (1-this.rho) * Math.pow(deltaValue, 2))
            return newVal

        } else {
            neuron.biasCache = this.rho * neuron.biasCache + (1-this.rho) * Math.pow(deltaValue, 2)
            const newVal = value + Math.sqrt((neuron.adadeltaBiasCache + 1e-6)/(neuron.biasCache + 1e-6)) * deltaValue
            neuron.adadeltaBiasCache = this.rho * neuron.adadeltaBiasCache + (1-this.rho) * Math.pow(deltaValue, 2)
            return newVal
        }
    }

    // Weights init
    static uniform (size, {limit}) {
        return [...new Array(size)].map(v => Math.random()*2*limit-limit)
    }

    static gaussian (size, {mean, stdDeviation}) {
        return [...new Array(size)].map(() => {
            // Polar Box Muller
            let x1, x2, r, y

            do {
                x1 = 2 * Math.random() -1
                x2 = 2 * Math.random() -1
                r = x1**2 + x2**2
            } while (r >= 1 || !r)

            return mean + (x1 * (Math.sqrt(-2 * Math.log(r) / r))) * stdDeviation
        })
    }

    static xaviernormal (size, {fanIn, fanOut}) {
        return fanOut || fanOut==0 ? NetMath.gaussian(size, {mean: 0, stdDeviation: Math.sqrt(2/(fanIn+fanOut))})
                                   : NetMath.lecunnormal(size, {fanIn})
    }

    static xavieruniform (size, {fanIn, fanOut}) {
        return fanOut || fanOut==0 ? NetMath.uniform(size, {limit: Math.sqrt(6/(fanIn+fanOut))})
                                   : NetMath.lecununiform(size, {fanIn})
    }

    static lecunnormal (size, {fanIn}) {
        return NetMath.gaussian(size, {mean: 0, stdDeviation: Math.sqrt(1/fanIn)})
    }

    static lecununiform (size, {fanIn}) {
        return NetMath.uniform(size, {limit: Math.sqrt(3/fanIn)})
    }

    // Other
    static softmax (values) {
        const total = values.reduce((prev, curr) => prev+curr, 0)
        return values.map(value => value/total)
    }

    static sech (value) {
        return (2*Math.exp(-value))/(1+Math.exp(-2*value))
    }

    static standardDeviation (arr) {
        const avg = arr.reduce((p,c) => p+c) / arr.length
        const diffs = arr.map(v => v - avg).map(v => v**2)
        return Math.sqrt(diffs.reduce((p,c) => p+c) / diffs.length)
    }

    static maxNorm () {

        if (this.maxNormTotal > this.maxNorm) {

            const multiplier = this.maxNorm / (1e-18 + this.maxNormTotal)

            this.layers.forEach((layer, li) => {
                li && layer.neurons.forEach(neuron => {
                    neuron.weights.forEach((w, wi) => neuron.setWeight(wi, neuron.getWeight(wi) * multiplier))
                })
            })
        }

        this.maxNormTotal = 0
    }
}

typeof window=="undefined" && (exports.NetMath = NetMath)