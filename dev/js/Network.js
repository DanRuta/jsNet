"use strict"

class Network {

    constructor ({learningRate, layers=[], updateFn="vanillaupdatefn", activation="sigmoid", cost="meansquarederror",
        rmsDecay, rho, lreluSlope, eluAlpha, dropout=1, l2, l1, maxNorm, weightsConfig, channels, conv, pool}={}) {

        this.state = "not-defined"
        this.layers = []
        this.conv = {}
        this.pool = {}
        this.epochs = 0
        this.iterations = 0
        this.validations = 0
        this.dropout = dropout==false ? 1 : dropout
        this.error = 0
        activation = NetUtil.format(activation)
        updateFn = NetUtil.format(updateFn)
        cost = NetUtil.format(cost)
        this.l1 = 0
        this.l2 = 0

        if (l1) {
            this.l1 = typeof l1=="boolean" ? 0.005 : l1
            this.l1Error = 0
        }

        if (l2) {
            this.l2 = typeof l2=="boolean" ? 0.001 : l2
            this.l2Error = 0
        }

        if (maxNorm) {
            this.maxNorm = typeof maxNorm=="boolean" && maxNorm ? 1000 : maxNorm
            this.maxNormTotal = 0
        }

        if (learningRate)   this.learningRate = learningRate
        if (channels)       this.channels = channels

        if (conv) {
            if (conv.filterSize!=undefined)     this.conv.filterSize = conv.filterSize
            if (conv.zeroPadding!=undefined)    this.conv.zeroPadding = conv.zeroPadding
            if (conv.stride!=undefined)         this.conv.stride = conv.stride
        }

        if (pool) {
            if (pool.size)      this.pool.size = pool.size
            if (pool.stride)    this.pool.stride = pool.stride
        }

        // Activation function / Learning Rate
        switch (updateFn) {

            case "rmsprop":
                this.learningRate = this.learningRate==undefined ? 0.001 : this.learningRate
                break

            case "adam":
                this.learningRate = this.learningRate==undefined ? 0.01 : this.learningRate
                break

            case "adadelta":
                this.rho = rho==null ? 0.95 : rho
                break

            default:

                if (this.learningRate==undefined) {

                    switch (activation) {

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

        this.updateFn = [false, null, undefined].includes(updateFn) ? "vanillaupdatefn" : updateFn
        this.weightUpdateFn = NetMath[this.updateFn]
        this.activation = typeof activation=="function" ? activation : NetMath[activation].bind(this)
        this.activationConfig = activation
        this.cost = typeof cost=="function" ? cost : NetMath[cost]

        if (this.updateFn=="rmsprop") {
            this.rmsDecay = rmsDecay==undefined ? 0.99 : rmsDecay
        }

        this.lreluSlope = lreluSlope==undefined ? -0.0005 : lreluSlope
        this.rreluSlope = Math.random() * 0.001
        this.eluAlpha = eluAlpha==undefined ? 1 : eluAlpha

        // Weights distributiom
        this.weightsConfig = {distribution: "xavieruniform"}

        if (weightsConfig != undefined && weightsConfig.distribution) {
            this.weightsConfig.distribution = NetUtil.format(weightsConfig.distribution)
        }

        if (this.weightsConfig.distribution == "uniform") {
            this.weightsConfig.limit = weightsConfig && weightsConfig.limit!=undefined ? weightsConfig.limit : 0.1

        } else if (this.weightsConfig.distribution == "gaussian") {
            this.weightsConfig.mean = weightsConfig.mean || 0
            this.weightsConfig.stdDeviation = weightsConfig.stdDeviation || 0.05
        }

        if (typeof this.weightsConfig.distribution=="function") {
            this.weightsInitFn = this.weightsConfig.distribution
        } else {
            this.weightsInitFn = NetMath[this.weightsConfig.distribution]
        }

        if (layers.length) {

            switch (true) {

                case layers.every(item => Number.isInteger(item)):
                    this.layers = layers.map(size => new FCLayer(size))
                    this.state = "constructed"
                    this.initLayers()
                    break

                case layers.every(layer => layer instanceof FCLayer || layer instanceof ConvLayer || layer instanceof PoolLayer):
                    this.state = "constructed"
                    this.layers = layers
                    this.initLayers()
                    break

                default:
                    throw new Error("There was an error constructing from the layers given.")
            }
        }
    }

    initLayers (input, expected) {

        switch (this.state) {

            case "initialised":
                return

            case "not-defined":
                this.layers[0] = new FCLayer(input)
                this.layers[1] = new FCLayer(Math.ceil(input/expected > 5 ? expected + (Math.abs(input-expected))/4
                                                                          : input + expected))
                this.layers[2] = new FCLayer(Math.ceil(expected))
                break
        }

        this.layers.forEach(this.joinLayer.bind(this))
        this.state = "initialised"
    }

    joinLayer (layer, layerIndex) {

        layer.net = this
        layer.activation = layer.activation==undefined ? this.activation : layer.activation

        layer.weightsConfig = {}
        Object.assign(layer.weightsConfig, this.weightsConfig)

        if (layerIndex) {
            this.layers[layerIndex-1].assignNext(layer)
            layer.assignPrev(this.layers[layerIndex-1], layerIndex)

            layer.weightsConfig.fanIn = layer.prevLayer.size

            if (layerIndex<this.layers.length-1) {
                layer.weightsConfig.fanOut = this.layers[layerIndex+1].size
            }

            layer.init()

        } else if (this.layers.length > 1) {
            layer.weightsConfig.fanOut = this.layers[1].size
        }

        layer.state = "initialised"
    }

    forward (data) {

        if (this.state!="initialised") {
            throw new Error("The network layers have not been initialised.")
        }

        if (data === undefined || data === null) {
            throw new Error("No data passed to Network.forward()")
        }

        if (data.length != this.layers[0].neurons.length) {
            console.warn("Input data length did not match input layer neurons count.")
        }

        this.layers[0].neurons.forEach((neuron, ni) => neuron.activation = data[ni])
        this.layers.forEach((layer, li) => li && layer.forward())
        const output = this.layers[this.layers.length-1].neurons.map(n => n.sum)
        return output.length > 1 ? NetMath.softmax(output) : output
    }

    backward (errors) {

        if (errors === undefined) {
            throw new Error("No data passed to Network.backward()")
        }

        if (errors.length != this.layers[this.layers.length-1].neurons.length) {
            console.warn("Expected data length did not match output layer neurons count.", errors)
        }

        this.layers[this.layers.length-1].backward(errors)

        for (let layerIndex=this.layers.length-2; layerIndex>0; layerIndex--) {
            this.layers[layerIndex].backward()
        }
    }

    train (dataSet, {epochs=1, callback, log=true, miniBatchSize=1, shuffle=false, validation}={}) {

        this.miniBatchSize = typeof miniBatchSize=="boolean" && miniBatchSize ? dataSet[0].expected.length : miniBatchSize
        this.validation = validation

        return new Promise((resolve, reject) => {

            if (shuffle) {
                NetUtil.shuffle(dataSet)
            }

            if (log) {
                console.log(`Training started. Epochs: ${epochs} Batch Size: ${this.miniBatchSize}`)
            }

            if (dataSet === undefined || dataSet === null) {
                return void reject("No data provided")
            }

            if (this.state != "initialised") {
                this.initLayers.bind(this, dataSet[0].input.length, dataSet[0].expected.length)()
            }

            this.layers.forEach(layer => layer.state = "training")

            if (this.validation) {
                this.validation.interval = this.validation.interval || dataSet.length // Default to 1 epoch

                if (this.validation.earlyStopping) {
                    switch (this.validation.earlyStopping.type) {
                        case "threshold":
                            this.validation.earlyStopping.threshold = this.validation.earlyStopping.threshold || 0.01
                            break
                        case "patience":
                            this.validation.earlyStopping.patienceCounter = 0
                            this.validation.earlyStopping.bestError = Infinity
                            this.validation.earlyStopping.patience = this.validation.earlyStopping.patience || 20
                            break
                    }
                }
            }

            let iterationIndex = 0
            let epochsCounter = 0
            let elapsed
            const startTime = Date.now()

            const logAndResolve = () => {
                this.layers.forEach(layer => layer.state = "initialised")

                if (this.validation && this.validation.earlyStopping && this.validation.earlyStopping.type == "patience") {
                    for (let l=1; l<this.layers.length; l++) {
                        this.layers[l].restoreValidation()
                    }
                }

                if (log) {
                    console.log(`Training finished. Total time: ${NetUtil.format(elapsed, "time")}  Average iteration time: ${NetUtil.format(elapsed/iterationIndex, "time")}`)
                }
                resolve()
            }

            const doEpoch = () => {
                this.epochs++
                this.error = 0
                this.validationError = 0
                iterationIndex = 0

                if (this.l2Error!=undefined) this.l2Error = 0
                if (this.l1Error!=undefined) this.l1Error = 0

                doIteration()
            }

            const doIteration = async () => {

                if (!dataSet[iterationIndex].hasOwnProperty("input") || !dataSet[iterationIndex].hasOwnProperty("expected")) {
                    return void reject("Data set must be a list of objects with keys: 'input' and 'expected'")
                }

                let trainingError
                let validationError

                const input = dataSet[iterationIndex].input
                const output = this.forward(input)
                const target = dataSet[iterationIndex].expected

                const errors = []
                for (let n=0; n<output.length; n++) {
                    errors[n] = (target[n]==1 ? 1 : 0) - output[n]
                }

                // Do validation
                if (this.validation && iterationIndex && iterationIndex%this.validation.interval==0) {

                    validationError = await this.validate(this.validation.data)

                    if (this.validation.earlyStopping && this.checkEarlyStopping(errors)) {
                        log && console.log("Stopping early")
                        return logAndResolve()
                    }
                }

                this.backward(errors)

                if (++iterationIndex%this.miniBatchSize==0) {
                    this.applyDeltaWeights()
                    this.resetDeltaWeights()
                } else if (iterationIndex >= dataSet.length) {
                    this.applyDeltaWeights()
                }

                trainingError = this.cost(target, output)
                this.error += trainingError
                this.iterations++

                elapsed = Date.now() - startTime

                if (typeof callback=="function") {
                    callback({
                        iterations: this.iterations,
                        validations: this.validations,
                        validationError, trainingError,
                        elapsed, input
                    })
                }

                if (iterationIndex < dataSet.length) {
                    setTimeout(doIteration.bind(this), 0)

                } else {
                    epochsCounter++

                    if (log) {
                        let text = `Epoch: ${this.epochs}\nTraining Error: ${this.error/iterationIndex}`

                        if (validation) {
                            text += `\nValidation Error: ${this.validationError}`
                        }

                        if (this.l2Error!=undefined) {
                            text += `\nL2 Error: ${this.l2Error/iterationIndex}`
                        }

                        text += `\nElapsed: ${NetUtil.format(elapsed, "time")} Average Duration: ${NetUtil.format(elapsed/epochsCounter, "time")}`
                        console.log(text)
                    }

                    if (epochsCounter < epochs) {
                        doEpoch()
                    } else {
                        logAndResolve()
                    }
                }
            }

            this.resetDeltaWeights()
            doEpoch()
        })
    }

    validate (data) {
        return new Promise((resolve, reject) => {
            let validationIndex = 0
            let totalValidationErrors = 0

            const validateItem = (item) => {

                const output = this.forward(data[validationIndex].input)
                const target = data[validationIndex].expected

                this.validations++
                totalValidationErrors += this.cost(target, output)
                // maybe do this only once, as there's no callback anyway
                this.validationError = totalValidationErrors / (validationIndex+1)

                if (++validationIndex<data.length) {
                    setTimeout(() => validateItem(validationIndex), 0)
                } else {
                    this.lastValidationError = totalValidationErrors / data.length
                    resolve(totalValidationErrors / data.length)
                }
            }
            validateItem(validationIndex)
        })
    }

    checkEarlyStopping (errors) {

        let stop = false

        switch (this.validation.earlyStopping.type) {
            case "threshold":
                stop = this.lastValidationError <= this.validation.earlyStopping.threshold

                // Do the last backward pass
                if (stop) {
                    this.backward(errors)
                    this.applyDeltaWeights()
                }

                return stop

            case "patience":

                if (this.lastValidationError<this.validation.earlyStopping.bestError) {
                    this.validation.earlyStopping.patienceCounter = 0
                    this.validation.earlyStopping.bestError = this.lastValidationError

                    for (let l=1; l<this.layers.length; l++) {
                        this.layers[l].backUpValidation()
                    }

                } else {
                    this.validation.earlyStopping.patienceCounter++
                    stop = this.validation.earlyStopping.patienceCounter>=this.validation.earlyStopping.patience
                }
                return stop
        }
    }

    test (testSet, {log=true, callback}={}) {
        return new Promise((resolve, reject) => {

            if (testSet === undefined || testSet === null) {
                reject("No data provided")
            }

            if (log) {
                console.log("Testing started")
            }

            let totalError = 0
            let iterationIndex = 0
            const startTime = Date.now()

            const testInput = () => {

                const input = testSet[iterationIndex].input
                const output = this.forward(input)
                const target = testSet[iterationIndex].expected
                const elapsed = Date.now() - startTime

                const iterationError = this.cost(target, output)
                totalError += iterationError
                iterationIndex++

                if (typeof callback=="function") {
                    callback({
                        iterations: iterationIndex,
                        error: iterationError,
                        elapsed, input
                    })
                }

                if (iterationIndex < testSet.length) {
                    setTimeout(testInput.bind(this), 0)

                } else {

                    if (log) {
                        console.log(`Testing finished. Total time: ${NetUtil.format(elapsed, "time")}  Average iteration time: ${NetUtil.format(elapsed/iterationIndex, "time")}`)
                    }

                    resolve(totalError/testSet.length)
                }
            }
            testInput()
        })
    }

    resetDeltaWeights () {
        this.layers.forEach((layer, li) => li && layer.resetDeltaWeights())
    }

    applyDeltaWeights () {

        this.layers.forEach((layer, li) => li && layer.applyDeltaWeights())

        if (this.maxNorm!=undefined) {
            this.maxNormTotal = Math.sqrt(this.maxNormTotal)
            NetMath.maxNorm.bind(this)()
        }
    }

    toJSON () {
        return {
            layers: this.layers.map(layer => layer.toJSON())
        }
    }

    fromJSON (data) {

        if (data === undefined || data === null) {
            throw new Error("No JSON data given to import.")
        }

        if (data.layers.length != this.layers.length) {
            throw new Error(`Mismatched layers (${data.layers.length} layers in import data, but ${this.layers.length} configured)`)
        }

        this.resetDeltaWeights()
        this.layers.forEach((layer, li) => li && layer.fromJSON(data.layers[li], li))
    }

    static get version () {
        return "3.2.0"
    }
}

/* istanbul ignore next */
typeof window!="undefined" && (window.Network = Network)
exports.Network = Network