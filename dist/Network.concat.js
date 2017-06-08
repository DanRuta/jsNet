"use strict"

class Layer {
    
    constructor (size, importedData) {
        this.size = size
        this.neurons = [...new Array(size)].map((n, ni) => new Neuron(importedData ? importedData[ni] : undefined))
    }

    assignNext (layer) {
        this.nextLayer = layer
    }

    assignPrev (layer) {
        this.prevLayer = layer
        this.neurons.forEach(neuron => neuron.init(layer.size))   
    }

    forward (data) {

        this.neurons.forEach((neuron, ni) => {

            neuron.sum = neuron.bias
            this.prevLayer.neurons.forEach((pNeuron, pni) => neuron.sum += pNeuron.activation * neuron.weights[pni])
            neuron.activation = this.activation(neuron.sum)
        })
    }

    backward (expected) {
        this.neurons.forEach((neuron, ni) => {

            if(typeof expected !== "undefined") {
                neuron.error = expected[ni] - neuron.activation
            }else {
                neuron.derivative = this.activation(neuron.sum, true)
                neuron.error = neuron.derivative * this.nextLayer.neurons.map(n => n.error * n.weights[ni])
                                                                         .reduce((p,c) => p+c, 0)
            }

            neuron.weights.forEach((weight, wi) => {
                neuron.deltaWeights[wi] += neuron.error * this.prevLayer.neurons[wi].activation
            })

            neuron.deltaBias = neuron.error
        })
    }
}

typeof window=="undefined" && (global.Layer = Layer) 
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
}

typeof window=="undefined" && (global.NetMath = NetMath)
"use strict"

class Network {

    constructor ({learningRate=0.2, layers=[], activation="sigmoid", cost="crossEntropy"}={}) {
        this.learningRate = learningRate
        this.state = "not-defined"
        this.layers = []
        this.epochs = 0
        this.iterations = 0
        this.activation = NetMath[activation]
        this.cost = NetMath[cost]

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

        if(layerIndex){
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

            let iterationIndex = 0
            let epochsCounter = 0
            let error = 0

            const doEpoch = () => {
                this.epochs++
                iterationIndex = 0

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
                error += iterationError

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
                    console.log(`Epoch: ${epochsCounter} Error: ${error/100}`)

                    if(epochsCounter < epochs){
                        doEpoch()
                    }else resolve()
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

                console.log("Testing iteration", testIteration+1, totalError/(testIteration+1)/100)

                const output = this.forward(testSet[testIteration].input)
                const target = testSet[testIteration].expected || testSet[testIteration].output

                totalError += this.cost(target, output)

                testIteration++

                if(testIteration < testSet.length)
                    setTimeout(testInput.bind(this), 0)
                else resolve(totalError/testSet.length/100)
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
                    const newWeight = neuron.weights[dwi] + this.learningRate * dw
                    neuron.weights[dwi] = newWeight
                })

                const newBias = neuron.bias + this.learningRate * neuron.deltaBias
                neuron.bias = newBias
            })
        })
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

        if(data === undefined || data === null){
            throw new Error("No JSON data given to import.")
        }

        this.layers = data.layers.map(layer => new Layer(layer.neurons.length, layer.neurons))
        this.state = "constructed"
        this.initLayers()
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

    init (size) {
        if(!this.imported){
            this.weights = [...new Array(size)].map(v => Math.random()*0.2-0.1)
            this.bias = Math.random()*0.2-0.1
        }
        this.deltaWeights = this.weights.map(v => 0)
    }
}

typeof window=="undefined" && (global.Neuron = Neuron)
//# sourceMappingURL=Network.concat.js.map