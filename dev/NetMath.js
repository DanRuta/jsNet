"use strict"

class NetMath {
    
    // Activation functions
    static sigmoid (value, prime) {
        return prime ? NetMath.sigmoid(value)*(1-NetMath.sigmoid(value))
                     : 1/(1+Math.exp(-value))
    }

    // Cost functions
    static crossEntropy (target, output) {
        return output.map((value, vi) => target[vi] * Math.log(value+1e-15) + ((1-target[vi]) * Math.log((1+1e-15)-value)))
                     .reduce((p,c) => p-c, 0)
    }

    static meanSquaredError (calculated, desired) {
        return calculated.map((output, index) => Math.pow(output - desired[index], 2))
                         .reduce((prev, curr) => prev+curr, 0) / calculated.length
    }

    // Weight updating functions
    static noAdaptiveLR (value, deltaValue) {
        return value + this.learningRate * deltaValue
    }

    static gain (value, deltaValue, gain, neuron, weightI) {

        const newVal = value + this.learningRate * deltaValue * gain

        if(newVal<=0 && value>0 || newVal>=0 && value<0){
            if(weightI!=null){
                neuron.weightGains[weightI] = Math.max(neuron.weightGains[weightI]*0.95, 0.5)
            }else{
                neuron.biasGain = Math.max(neuron.biasGain*0.95, 0.5)
            }
        }else {
            if(weightI!=null){
                neuron.weightGains[weightI] = Math.min(neuron.weightGains[weightI]+0.05, 5)
            }else {
                neuron.biasGain = Math.min(neuron.biasGain+0.05, 5)
            }
        }

        return newVal
    }

    // Other
    static softmax (values) {
        const total = values.reduce((prev, curr) => prev+curr, 0)
        return values.map(value => value/total)
    }
}

typeof window=="undefined" && (global.NetMath = NetMath)