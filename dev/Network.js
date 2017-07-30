"use strict"

class Network {

    constructor ({learningRate, layers=[], adaptiveLR="noAdaptiveLR", activation="sigmoid", cost="crossEntropy", 
        rmsDecay, rho, lreluSlope, eluAlpha, dropout=0.5, l2, l1, maxNorm, weightsConfig}={}) {
        this.state = "not-defined"
        this.layers = []
        this.epochs = 0
        this.iterations = 0
        this.dropout = dropout==false ? 1 : dropout
        this.error = 0

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

            case adaptiveLR=="RMSProp":
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
        
        this.adaptiveLR = [false, null, undefined].includes(adaptiveLR) ? "noAdaptiveLR" : adaptiveLR
        this.weightUpdateFn = NetMath[this.adaptiveLR]
        this.activation = NetMath[activation].bind(this)
        this.activationConfig = activation
        this.cost = NetMath[cost]

        if(this.adaptiveLR=="RMSProp"){
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
                this.weightsConfig.distribution = weightsConfig.distribution 
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
}

typeof window=="undefined" && (global.Network = Network)