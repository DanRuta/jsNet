"use strict"

class Layer {
    
    constructor (size, importedData) {
        this.size = size
        this.neurons = [...new Array(size)].map((n, ni) => new Neuron(importedData ? importedData[ni] : undefined))
        this.state = "not-initialised"
    }

    assignNext (layer) {
        this.nextLayer = layer
    }

    assignPrev (layer) {

        this.prevLayer = layer
        this.neurons.forEach(neuron => {

            if(!neuron.imported) {
                neuron.weights = this.weightsInitFn(layer.size, this.weightsConfig)
                neuron.bias = Math.random()*0.2-0.1
            }

            neuron.init(layer.size, {
                adaptiveLR: this.adaptiveLR,
                activationConfig: this.activationConfig,
                eluAlpha: this.eluAlpha
            })
        }) 
        this.state = "initialised"
    }

    forward (data) {

        this.neurons.forEach((neuron, ni) => {

            if(this.state=="training" && (neuron.dropped = Math.random() > this.dropout)) {
                neuron.activation = 0
            }else {
                neuron.sum = neuron.bias
                this.prevLayer.neurons.forEach((pNeuron, pni) => neuron.sum += pNeuron.activation * neuron.weights[pni])
                neuron.activation = this.activation(neuron.sum, false, neuron) / (this.dropout|1)
            }
        })
    }

    backward (expected) {
        this.neurons.forEach((neuron, ni) => {

            if(neuron.dropped) {
                neuron.error = 0
                neuron.deltaBias = 0
            }else {
                if(typeof expected !== "undefined") {
                    neuron.error = expected[ni] - neuron.activation
                }else {
                    neuron.derivative = this.activation(neuron.sum, true, neuron)
                    neuron.error = neuron.derivative * this.nextLayer.neurons.map(n => n.error * (n.weights[ni]|0))
                                                                             .reduce((p,c) => p+c, 0)
                }

                neuron.weights.forEach((weight, wi) => {
                    neuron.deltaWeights[wi] += (neuron.error * this.prevLayer.neurons[wi].activation) * 
                                               (1 + ((this.l2||0)+(this.l1||0)) * neuron.deltaWeights[wi])
                })

                neuron.deltaBias = neuron.error
            }            
        })
    }
}

typeof window=="undefined" && (global.Layer = Layer) 
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

        const newVal = value + this.learningRate * deltaValue * (weightI==null ? neuron.biasGain : neuron.weightGains[weightI])

        if(newVal<=0 && value>0 || newVal>=0 && value<0){
            if(weightI!=null)
                 neuron.weightGains[weightI] = Math.max(neuron.weightGains[weightI]*0.95, 0.5)
            else neuron.biasGain = Math.max(neuron.biasGain*0.95, 0.5)
        }else {
            if(weightI!=null)
                 neuron.weightGains[weightI] = Math.min(neuron.weightGains[weightI]+0.05, 5)
            else neuron.biasGain = Math.min(neuron.biasGain+0.05, 5)
        }

        return newVal
    }

    static adagrad (value, deltaValue, neuron, weightI) {

        if(weightI!=null)
             neuron.weightsCache[weightI] += Math.pow(deltaValue, 2)
        else neuron.biasCache += Math.pow(deltaValue, 2)

        return value + this.learningRate * deltaValue / (1e-6 + Math.sqrt(weightI!=null ? neuron.weightsCache[weightI]
                                                                                        : neuron.biasCache))
    }

    static rmsprop (value, deltaValue, neuron, weightI) {

        if(weightI!=null)
             neuron.weightsCache[weightI] = this.rmsDecay * neuron.weightsCache[weightI] + (1 - this.rmsDecay) * Math.pow(deltaValue, 2)
        else neuron.biasCache = this.rmsDecay * neuron.biasCache + (1 - this.rmsDecay) * Math.pow(deltaValue, 2)

        return value + this.learningRate * deltaValue / (1e-6 + Math.sqrt(weightI!=null ? neuron.weightsCache[weightI]
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

        if(weightI!=null) {
            neuron.weightsCache[weightI] = this.rho * neuron.weightsCache[weightI] + (1-this.rho) * Math.pow(deltaValue, 2)
            const newVal = value + Math.sqrt((neuron.adadeltaCache[weightI] + 1e-6)/(neuron.weightsCache[weightI] + 1e-6)) * deltaValue
            neuron.adadeltaCache[weightI] = this.rho * neuron.adadeltaCache[weightI] + (1-this.rho) * Math.pow(deltaValue, 2)
            return newVal

        }else {
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

        if(this.maxNormTotal > this.maxNorm) {

            const multiplier = this.maxNorm / (1e-18 + this.maxNormTotal)

            this.layers.forEach((layer, li) => {
                li && layer.neurons.forEach(neuron => {
                    neuron.weights.forEach((w, wi) => {
                        neuron.weights[wi] *= multiplier
                    })
                })
            })
        }

        this.maxNormTotal = 0
    }
}

typeof window=="undefined" && (global.NetMath = NetMath)
"use strict"

class Network {

    constructor ({learningRate, layers=[], adaptiveLR="noadaptivelr", activation="sigmoid", cost="crossentropy", 
        rmsDecay, rho, lreluSlope, eluAlpha, dropout=0.5, l2, l1, maxNorm, weightsConfig}={}) {
        this.state = "not-defined"
        this.layers = []
        this.epochs = 0
        this.iterations = 0
        this.dropout = dropout==false ? 1 : dropout
        this.error = 0
        activation = this.format(activation)
        adaptiveLR = this.format(adaptiveLR)
        cost = this.format(cost)

        if(learningRate!=null){    
            this.learningRate = learningRate
        }

        if(l2){
            this.l2 = typeof l2=="boolean" && l2 ? 0.001 : l2
            this.l2Error = 0
        }

        if(l1){
            this.l1 = typeof l1=="boolean" && l1 ? 0.005 : l1
            this.l1Error = 0
        }

        if(maxNorm){
            this.maxNorm = typeof maxNorm=="boolean" && maxNorm ? 1000 : maxNorm
            this.maxNormTotal = 0
        }

        // Activation function / Learning Rate
        switch(true) {

            case adaptiveLR=="rmsprop":
                this.learningRate = this.learningRate==undefined ? 0.001 : this.learningRate
                break

            case adaptiveLR=="adam":
                this.learningRate = this.learningRate==undefined ? 0.01 : this.learningRate
                break

            case adaptiveLR=="adadelta":
                this.rho = rho==null ? 0.95 : rho
                break

            default:

                if(this.learningRate==undefined){
                    switch(activation) {
                        case "relu":
                        case "lrelu":
                        case "rrelu":
                        case "elu":
                            this.learningRate = 0.01
                            break
                        case "tanh":
                        case "lecuntanh":
                            this.learningRate = 0.001
                            break
                        default:
                            this.learningRate = 0.2
                    }
                }
        }
        
        this.adaptiveLR = [false, null, undefined].includes(adaptiveLR) ? "noadaptivelr" : adaptiveLR
        this.weightUpdateFn = NetMath[this.adaptiveLR]
        this.activation = typeof activation=="function" ? activation : NetMath[activation].bind(this)
        this.activationConfig = activation
        this.cost = typeof cost=="function" ? cost : NetMath[cost]

        if(this.adaptiveLR=="rmsprop"){
            this.rmsDecay = rmsDecay==undefined ? 0.99 : rmsDecay
        }

        if(activation=="lrelu"){
            this.lreluSlope = lreluSlope==undefined ? -0.0005 : lreluSlope
        }else if(activation=="elu") {
            this.eluAlpha = eluAlpha==undefined ? 1 : eluAlpha
        }

        // Weights distributiom
        this.weightsConfig = {distribution: "uniform"}

        if(weightsConfig != undefined) {
            if(weightsConfig.distribution) {
                this.weightsConfig.distribution = this.format(weightsConfig.distribution) 
            }
        }

        if(this.weightsConfig.distribution == "uniform") {
            this.weightsConfig.limit = weightsConfig && weightsConfig.limit!=undefined ? weightsConfig.limit : 0.1

        } else if(this.weightsConfig.distribution == "gaussian") {

            this.weightsConfig.mean = weightsConfig.mean || 0
            this.weightsConfig.stdDeviation = weightsConfig.stdDeviation || 0.05        
        }

        // Status
        if(layers.length) {

            switch(true) {

                case layers.every(item => Number.isInteger(item)):
                    this.layers = layers.map(size => new Layer(size))
                    this.state = "constructed"
                    this.initLayers()
                    break

                case layers.every(item => item instanceof Layer):
                    this.state = "constructed"
                    this.layers = layers
                    this.initLayers()
                    break

                case layers.every(item => item === Layer):
                    this.state = "defined"
                    this.definedLayers = layers
                    break

                default:
                    throw new Error("There was an error constructing from the layers given.")
            }
        }
    }

    initLayers (input, expected) {

        switch(this.state){

            case "initialised":
                return

            case "defined":
                this.layers = this.definedLayers.map((layer, li) => {
                    
                    if(!li)
                        return new layer(input)

                    if(li==this.definedLayers.length-1) 
                        return new layer(expected)

                    const hidden = this.definedLayers.length-2
                    const size = input/expected > 5 ? expected + (expected + (Math.abs(input-expected))/4) * (hidden-li+1)/(hidden/2)
                                                    : input >= expected ? input + expected * (hidden-li)/(hidden/2)
                                                                        : expected + input * (hidden-li)/(hidden/2)

                    return new layer(Math.max(Math.round(size), 0))
                })
                break

            case "not-defined":
                this.layers[0] = new Layer(input)
                this.layers[1] = new Layer(Math.ceil(input/expected > 5 ? expected + (Math.abs(input-expected))/4
                                                                        : input + expected))
                this.layers[2] = new Layer(Math.ceil(expected))
                break
        }

        this.layers.forEach(this.joinLayer.bind(this))
        this.state = "initialised"
    }

    joinLayer (layer, layerIndex) {

        layer.activation = this.activation
        layer.adaptiveLR = this.adaptiveLR
        layer.activationConfig = this.activationConfig
        layer.dropout = this.dropout

        layer.weightsConfig = {}
        Object.assign(layer.weightsConfig, this.weightsConfig)
        layer.weightsInitFn = NetMath[layer.weightsConfig.distribution]

        if(this.rho!=undefined) {
            layer.rho = this.rho
        }
        
        if(this.eluAlpha!=undefined) {
            layer.eluAlpha = this.eluAlpha
        }

        if(this.l2!=undefined) {
            layer.l2 = this.l2
        }

        if(this.l1!=undefined) {
            layer.l1 = this.l1
        }

        if(layerIndex) {
            layer.weightsConfig.fanIn = this.layers[layerIndex-1].size
            this.layers[layerIndex-1].weightsConfig.fanOut = layer.size
            this.layers[layerIndex-1].assignNext(layer)
            layer.assignPrev(this.layers[layerIndex-1])
        }
    }

    forward (data) {

        if(this.state!="initialised"){
            throw new Error("The network layers have not been initialised.")
        }

        if(data === undefined){
            throw new Error("No data passed to Network.forward()")
        }

        if(data.length != this.layers[0].neurons.length){
            console.warn("Input data length did not match input layer neurons count.")
        }

        this.layers[0].neurons.forEach((neuron, ni) => neuron.activation = data[ni])
        this.layers.forEach((layer, li) => li && layer.forward(data))
        return this.layers[this.layers.length-1].neurons.map(n => n.activation)
    }

    backward (expected) {
        if(expected === undefined){
            throw new Error("No data passed to Network.backward()")
        }

        if(expected.length != this.layers[this.layers.length-1].neurons.length){
            console.warn("Expected data length did not match output layer neurons count.")
        }

        this.layers[this.layers.length-1].backward(expected)

        for(let layerIndex=this.layers.length-2; layerIndex>0; layerIndex--){
            this.layers[layerIndex].backward()
        }
    }

    train (dataSet, {epochs=1, callback}={}) {
        return new Promise((resolve, reject) => {
            
            if(dataSet === undefined || dataSet === null) {
                reject("No data provided")
            }

            if(this.state != "initialised"){
                this.initLayers(dataSet[0].input.length, (dataSet[0].expected || dataSet[0].output).length)
            }

            this.layers.forEach(layer => layer.state = "training")

            let iterationIndex = 0
            let epochsCounter = 0

            const doEpoch = () => {
                this.epochs++
                this.error = 0
                iterationIndex = 0

                if(this.l2Error!=undefined){
                    this.l2Error = 0
                }

                if(this.l1Error!=undefined){
                    this.l1Error = 0
                }

                doIteration()               
            }

            const doIteration = () => {
                
                if(!dataSet[iterationIndex].hasOwnProperty("input") || (!dataSet[iterationIndex].hasOwnProperty("expected") && !dataSet[iterationIndex].hasOwnProperty("output"))){
                    return reject("Data set must be a list of objects with keys: 'input' and 'expected' (or 'output')")
                }

                this.resetDeltaWeights()

                const input = dataSet[iterationIndex].input
                const output = this.forward(input)
                const target = dataSet[iterationIndex].expected || dataSet[iterationIndex].output

                this.backward(target)
                this.applyDeltaWeights()

                const iterationError = this.cost(target, output)
                this.error += iterationError

                if(typeof callback=="function") {
                    callback({
                        iterations: this.iterations,
                        error: iterationError,
                        input
                    })
                }

                this.iterations++
                iterationIndex++

                if(iterationIndex < dataSet.length){
                    setTimeout(doIteration.bind(this), 0)
                }else {

                    epochsCounter++
                    console.log(`Epoch: ${this.epochs} Error: ${this.error/iterationIndex}${this.l2==undefined ? "": ` L2 Error: ${this.l2Error/iterationIndex}`}`)

                    if(epochsCounter < epochs){
                        doEpoch()
                    }else {
                        this.layers.forEach(layer => layer.state = "initialised")
                        resolve()
                    }
                }
            }

            doEpoch()
        })
    }

    test (testSet) {
        return new Promise((resolve, reject) => {

            if(testSet === undefined || testSet === null) {
                reject("No data provided")
            }

            let totalError = 0
            let testIteration = 0

            const testInput = () => {

                const output = this.forward(testSet[testIteration].input)
                const target = testSet[testIteration].expected || testSet[testIteration].output

                totalError += this.cost(target, output)

                console.log("Testing iteration", testIteration+1, totalError/(testIteration+1))

                testIteration++

                if(testIteration < testSet.length)
                    setTimeout(testInput.bind(this), 0)
                else resolve(totalError/testSet.length)
            }
            testInput()
        })
    }

    resetDeltaWeights () {
        this.layers.forEach((layer, li) => {
            li && layer.neurons.forEach(neuron => {
                neuron.deltaWeights = neuron.weights.map(dw => 0)
            })
        })
    }

    applyDeltaWeights () {
        this.layers.forEach((layer, li) => {
            li && layer.neurons.forEach(neuron => {
                neuron.deltaWeights.forEach((dw, dwi) => {

                    if(this.l2!=undefined) {
                        this.l2Error += 0.5 * this.l2 * neuron.weights[dwi]**2
                    }

                    if(this.l1!=undefined) {
                        this.l1Error += this.l1 * Math.abs(neuron.weights[dwi])
                    }

                    neuron.weights[dwi] = this.weightUpdateFn.bind(this, neuron.weights[dwi], dw, neuron, dwi)()

                    if(this.maxNorm!=undefined) {
                        this.maxNormTotal += neuron.weights[dwi]**2
                    }
                })
                neuron.bias = this.weightUpdateFn.bind(this, neuron.bias, neuron.deltaBias, neuron)()
            })
        })

        if(this.maxNorm!=undefined) {
            this.maxNormTotal = Math.sqrt(this.maxNormTotal)
            NetMath.maxNorm.bind(this)()
        }
    }

    toJSON () {
        return {
            layers: this.layers.map(layer => {
                return {
                    neurons: layer.neurons.map(neuron => {
                        return {
                            bias: neuron.bias,
                            weights: neuron.weights
                        }
                    })
                }
            })
        }
    }

    fromJSON (data) {

        if(data === undefined || data === null) {
            throw new Error("No JSON data given to import.")
        }

        this.layers = data.layers.map(layer => new Layer(layer.neurons.length, layer.neurons))
        this.state = "constructed"
        this.initLayers()
    }

    format (string) {
        return string && typeof string=="string" ? string.replace(/(_|\s)/g, "").toLowerCase() : string
    }
}

typeof window=="undefined" && (global.Network = Network)
"use strict"

class Neuron {
    
    constructor (importedData) {
        if(importedData){
            this.imported = true
            this.weights = importedData.weights || []
            this.bias = importedData.bias
        }
    }

    init (size, {adaptiveLR, activationConfig, eluAlpha}={}) {

        this.deltaWeights = this.weights.map(v => 0)

        switch(adaptiveLR) {
            case "gain":
                this.weightGains = [...new Array(size)].map(v => 1)
                this.biasGain = 1
                break

            case "adagrad":
            case "rmsprop":
            case "adadelta":
                this.biasCache = 0
                this.weightsCache = [...new Array(size)].map(v => 0)

                if(adaptiveLR=="adadelta"){
                    this.adadeltaCache = [...new Array(size)].map(v => 0)
                    this.adadeltaBiasCache = 0
                }
                break

            case "adam":
                this.m = 0
                this.v = 0
                break
        }

        if(activationConfig=="rrelu") {
            this.rreluSlope = Math.random() * 0.001
            
        }else if(activationConfig=="elu") {
            this.eluAlpha = eluAlpha
        }
    }
}

typeof window=="undefined" && (global.Neuron = Neuron)
//# sourceMappingURL=Network.concat.js.map