"use strict"

class Network {

    constructor ({Module, learningRate, activation="sigmoid", updateFn="vanillasgd", cost="meansquarederror", layers=[],
        momentum=0.9, rmsDecay, rho, lreluSlope, eluAlpha, dropout=1, l2, l1, maxNorm, weightsConfig, channels, conv, pool}) {

        if (!Module) {
            throw new Error("WASM module not provided")
        }

        if (typeof activation == "function" || typeof cost == "function") {
            throw new Error("Custom functions are not (yet) supported with WASM.")
        }

        NetUtil.Module = Module
        this.Module = Module
        this.conv = {}
        this.pool = {}
        this.netInstance = this.Module.ccall("newNetwork", null, null, null)
        this.state = "not-defined"

        // Learning Rate get / set
        Object.defineProperty(this, "learningRate", {
            get: this.Module.cwrap("getLearningRate", null, null).bind(this, this.netInstance),
            set: this.Module.cwrap("setLearningRate", "number", null).bind(this, this.netInstance)
        })

        if (learningRate) this.learningRate = learningRate

        NetUtil.defineProperty(this, "dropout", ["number"], [this.netInstance])
        this.dropout = dropout==false ? 1 : dropout

        if (l2) {
            NetUtil.defineProperty(this, "l2", ["number"], [this.netInstance])
            NetUtil.defineProperty(this, "l2Error", ["number"], [this.netInstance])
            this.l2 = typeof l2=="boolean" ? 0.001 : l2
        }

        if (l1) {
            NetUtil.defineProperty(this, "l1", ["number"], [this.netInstance])
            NetUtil.defineProperty(this, "l1Error", ["number"], [this.netInstance])
            this.l1 = typeof l1=="boolean" ? 0.005 : l1
        }

        if (maxNorm) {
            NetUtil.defineProperty(this, "maxNorm", ["number"], [this.netInstance])
            NetUtil.defineProperty(this, "maxNormTotal", ["number"], [this.netInstance])
            this.maxNorm = typeof maxNorm=="boolean" && maxNorm ? 1000 : maxNorm
        }

        if (channels) {
            NetUtil.defineProperty(this, "channels", ["number"], [this.netInstance])
            this.channels = channels
        }

        if (conv) {
            if (conv.filterSize!=undefined)     this.conv.filterSize = conv.filterSize
            if (conv.zeroPadding!=undefined)    this.conv.zeroPadding = conv.zeroPadding
            if (conv.stride!=undefined)         this.conv.stride = conv.stride
        }

        if (pool) {
            if (pool.size)      this.pool.size = pool.size
            if (pool.stride)    this.pool.stride = pool.stride
        }

        Object.defineProperty(this, "error", {
            get: () => Module.ccall("getError", "number", ["number"], [this.netInstance])
        })
        Object.defineProperty(this, "validationError", {
            get: () => Module.ccall("getValidationError", "number", ["number"], [this.netInstance])
        })
        Object.defineProperty(this, "lastValidationError", {
            get: () => Module.ccall("getLastValidationError", "number", ["number"], [this.netInstance])
        })

        // Activation function get / set
        this.activationName = NetUtil.format(activation)
        Object.defineProperty(this, "activation", {
            get: () => `WASM ${this.activationName}`,
            set: activation => {

                if (NetUtil.activationsIndeces[activation] == undefined) {
                    throw new Error(`The ${activation} activation function does not exist`)
                }
                this.activationName = activation
                this.Module.ccall("setActivation", null, ["number", "number"], [this.netInstance, NetUtil.activationsIndeces[activation]])
            }
        })
        this.activation = this.activationName

        // Cost function get / set
        const costIndeces = {
            meansquarederror: 0,
            crossentropy: 1
        }
        let costFunctionName = NetUtil.format(cost)
        Object.defineProperty(this, "cost", {
            get: () => `WASM ${costFunctionName}`,
            set: cost => {
                if (costIndeces[cost] == undefined) {
                    throw new Error(`The ${cost} function does not exist`)
                }
                costFunctionName = cost
                this.Module.ccall("setCostFunction", null, ["number", "number"], [this.netInstance, costIndeces[cost]])
            }
        })
        this.cost = costFunctionName

        const updateFnIndeces = {
            vanillasgd: 0,
            gain: 1,
            adagrad: 2,
            rmsprop: 3,
            adam: 4,
            adadelta: 5,
            momentum: 6
        }
        NetUtil.defineProperty(this, "updateFn", ["number"], [this.netInstance], {
            getCallback: index => Object.keys(updateFnIndeces).find(key => updateFnIndeces[key]==index),
            setCallback: name => updateFnIndeces[name]
        })
        this.updateFn = NetUtil.format(updateFn)


        // Weights init configs
        const weightsConfigFns = {
            uniform: 0,
            gaussian: 1,
            xavieruniform: 2,
            xaviernormal: 3,
            lecununiform: 4,
            lecunnormal: 5
        }
        this.weightsConfig = {}

        NetUtil.defineProperty(this.weightsConfig, "distribution", ["number"], [this.netInstance], {
            getCallback: index => Object.keys(weightsConfigFns).find(key => weightsConfigFns[key]==Math.round(index)),
            setCallback: name => weightsConfigFns[name]
        })
        NetUtil.defineProperty(this.weightsConfig, "limit", ["number"], [this.netInstance])
        NetUtil.defineProperty(this.weightsConfig, "mean", ["number"], [this.netInstance])
        NetUtil.defineProperty(this.weightsConfig, "stdDeviation", ["number"], [this.netInstance])

        this.weightsConfig.distribution = "xavieruniform"

        if (weightsConfig!=undefined && weightsConfig.distribution) {

            if (typeof weightsConfig.distribution == "function") {
                throw new Error("Custom weights init functions are not (yet) supported with WASM.")
            }

            this.weightsConfig.distribution = NetUtil.format(weightsConfig.distribution)
        }

        this.weightsConfig.limit = weightsConfig && weightsConfig.limit!=undefined ? weightsConfig.limit : 0.1
        this.weightsConfig.mean = weightsConfig && weightsConfig.mean!=undefined ? weightsConfig.mean : 0
        this.weightsConfig.stdDeviation = weightsConfig && weightsConfig.stdDeviation!=undefined ? weightsConfig.stdDeviation : 0.05

        switch (NetUtil.format(updateFn)) {

            case "rmsprop":
                this.learningRate = this.learningRate || 0.001
                break

            case "adam":
                this.learningRate = this.learningRate || 0.01
                break

            case "adadelta":
                NetUtil.defineProperty(this, "rho", ["number"], [this.netInstance])
                this.rho = rho==null ? 0.95 : rho
                break

            case "momentum":
                NetUtil.defineProperty(this, "momentum", ["number"], [this.netInstance])
                this.learningRate = this.learningRate || 0.2
                this.momentum = momentum
                break

            default:

                if (learningRate==undefined) {

                    switch (this.activationName) {
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

        if (this.updateFn=="rmsprop") {
            NetUtil.defineProperty(this, "rmsDecay", ["number"], [this.netInstance])
            this.rmsDecay = rmsDecay===undefined ? 0.99 : rmsDecay
        }

        if (this.activationName=="lrelu") {
            NetUtil.defineProperty(this, "lreluSlope", ["number"], [this.netInstance])
            this.lreluSlope = lreluSlope==undefined ? -0.0005 : lreluSlope
        } else if (this.activationName=="elu") {
            NetUtil.defineProperty(this, "eluAlpha", ["number"], [this.netInstance])
            this.eluAlpha = eluAlpha==undefined ? 1 : eluAlpha
        }

        this.layers = []
        this.epochs = 0

        NetUtil.defineProperty(this, "iterations", ["number"], [this.netInstance])
        NetUtil.defineProperty(this, "validations", ["number"], [this.netInstance])
        NetUtil.defineProperty(this, "validationInterval", ["number"], [this.netInstance])
        NetUtil.defineProperty(this, "trainingLogging", ["number"], [this.netInstance])
        NetUtil.defineProperty(this, "stoppedEarly", ["number"], [this.netInstance])
        NetUtil.defineProperty(this, "earlyStoppingType", ["number"], [this.netInstance])
        NetUtil.defineProperty(this, "earlyStoppingThreshold", ["number"], [this.netInstance])
        NetUtil.defineProperty(this, "earlyStoppingBestError", ["number"], [this.netInstance])
        NetUtil.defineProperty(this, "earlyStoppingPatienceCounter", ["number"], [this.netInstance])
        NetUtil.defineProperty(this, "earlyStoppingPatience", ["number"], [this.netInstance])
        NetUtil.defineProperty(this, "earlyStoppingPercent", ["number"], [this.netInstance])

        this.collectedErrors = {}
        NetUtil.defineArrayProperty(this.collectedErrors, "training", ["number"], [this.netInstance], "auto", {pre: "collected_"})
        NetUtil.defineArrayProperty(this.collectedErrors, "test", ["number"], [this.netInstance], "auto", {pre: "collected_"})
        NetUtil.defineArrayProperty(this.collectedErrors, "validation", ["number"], [this.netInstance], "auto", {pre: "collected_"})

        if (layers.length) {

            this.state = "constructed"

            switch (true) {
                case layers.every(item => Number.isInteger(item)):
                    this.layers = layers.map(size => new FCLayer(size))
                    this.initLayers()
                    break

                case layers.every(layer => layer instanceof FCLayer || layer instanceof ConvLayer || layer instanceof PoolLayer):
                    this.layers = layers
                    this.initLayers()
                    break

                default:
                    throw new Error("There was an error constructing from the layers given.")

            }
        }
    }

    initLayers (input, expected) {

        if (this.state == "initialised") {
            return
        }

        if (this.state == "not-defined") {
            this.layers[0] = new FCLayer(input)
            this.layers[1] = new FCLayer(Math.ceil(input/expected > 5 ? expected + (Math.abs(input-expected))/4
                                                                      : input + expected))
            this.layers[2] = new FCLayer(Math.ceil(expected))
        }

        this.state = "initialised"

        for (let l=0; l<this.layers.length; l++) {

            const layer = this.layers[l]

            switch (true) {
                case layer instanceof FCLayer:
                    this.Module.ccall("addFCLayer", null, ["number", "number"], [this.netInstance, layer.size])

                    if (layer.softmax) {
                        this.Module.ccall("setOutputSoftmax", null, ["number", "number"], [this.netInstance, l])
                    }
                    break

                case layer instanceof ConvLayer:
                    this.Module.ccall("addConvLayer", null, ["number", "number"], [this.netInstance, layer.size])
                    break

                case layer instanceof PoolLayer:
                    this.Module.ccall("addPoolLayer", null, ["number", "number"], [this.netInstance, layer.size])
                    break
            }

            this.joinLayer(layer, l)
        }

        this.Module.ccall("initLayers", null, ["number"], [this.netInstance])
        const outSize = this.layers[this.layers.length-1].size
        const floorFunc = map => map.map(row => row.map(v => Math.floor(v)))

        NetUtil.defineMapProperty(this, "trainingConfusionMatrix", ["number"], [this.netInstance], outSize, outSize, {getCallback: floorFunc})
        NetUtil.defineMapProperty(this, "testConfusionMatrix", ["number"], [this.netInstance], outSize, outSize, {getCallback: floorFunc})
        NetUtil.defineMapProperty(this, "validationConfusionMatrix", ["number"], [this.netInstance], outSize, outSize, {getCallback: floorFunc})
    }

    joinLayer (layer, layerIndex) {

        layer.net = this
        layer.layerIndex = layerIndex

        if (layerIndex) {
            this.layers[layerIndex-1].assignNext(layer)
            layer.assignPrev(this.layers[layerIndex-1], layerIndex)
        }
        layer.init()
    }

    forward (data) {

        if (this.state!="initialised") {
            throw new Error("The network layers have not been initialised.")
        }

        if (data === undefined || data === null) {
            throw new Error("No data passed to Network.forward()")
        }

        // Flatten volume inputs
        if (Array.isArray(data[0])) {
            const flat = []

            for (let c=0; c<data.length; c++) {
                for (let r=0; r<data[0].length; r++) {
                    for (let v=0; v<data[0].length; v++) {
                        flat.push(data[c][r][v])
                    }
                }
            }
            data = flat
        }

        if (data.length != this.layers[0].neurons.length) {
            console.warn("Input data length did not match input layer neurons count.")
        }

        return NetUtil.ccallArrays("forward", "array", ["number", "array"], [this.netInstance, data], {
            heapOut: "HEAPF64",
            returnArraySize: this.layers[this.layers.length-1].neurons.length
        })
    }

    train (data, {epochs=1, callback, callbackInterval=1, collectErrors, miniBatchSize=1, log=true, shuffle=false, validation}={}) {

        miniBatchSize = typeof miniBatchSize=="boolean" && miniBatchSize ? data[0].expected.length : miniBatchSize
        this.Module.ccall("set_miniBatchSize", null, ["number", "number"], [this.netInstance, miniBatchSize])
        this.validation = validation
        this.trainingLogging = log
        this.stoppedEarly = false

        return new Promise((resolve, reject) => {

            if (data === undefined || data === null) {
                return void reject("No data provided")
            }

            if (this.state != "initialised") {
                this.initLayers(data[0].input.length, data[0].expected.length)
            }

            const startTime = Date.now()

            const dimension = this.layers[0].size
            const itemSize = dimension + data[0].expected.length
            const itemsCount = itemSize * data.length

            if (log) {
                console.log(`Training started. Epochs: ${epochs} Batch size: ${miniBatchSize}`)
            }

            // Load training data
            const typedArray = new Float32Array(itemsCount)
            this.loadData(data, typedArray, itemSize, reject)

            const buf = this.Module._malloc(typedArray.length*typedArray.BYTES_PER_ELEMENT)
            this.Module.HEAPF32.set(typedArray, buf >> 2)

            let elapsed

            this.Module.ccall("loadTrainingData", "number", ["number", "number", "number", "number", "number"],
                                                      [this.netInstance, buf, itemsCount, itemSize, dimension])

            if (shuffle) {
                this.Module.ccall("shuffleTrainingData", null, ["number"], [this.netInstance])
            }

            if (collectErrors) {
                this.Module.ccall("collectErrors", null, ["number"], [this.netInstance])
            }

            let validationBuf

            if (this.validation) {

                this.validationInterval = this.validation.interval || data.length // Default to 1 epoch

                if (this.validation.earlyStopping) {
                    switch (this.validation.earlyStopping.type) {
                        case "threshold":
                            this.validation.earlyStopping.threshold = this.validation.earlyStopping.threshold || 0.01
                            this.earlyStoppingThreshold = this.validation.earlyStopping.threshold
                            this.earlyStoppingType = 1
                            break
                        case "patience":
                            this.validation.earlyStopping.patience = this.validation.earlyStopping.patience || 20
                            this.earlyStoppingBestError = Infinity
                            this.earlyStoppingPatienceCounter = 0
                            this.earlyStoppingPatience = this.validation.earlyStopping.patience
                            this.earlyStoppingType = 2
                            break
                        case "divergence":
                            this.validation.earlyStopping.percent = this.validation.earlyStopping.percent || 30
                            this.earlyStoppingBestError = Infinity
                            this.earlyStoppingPercent = this.validation.earlyStopping.percent
                            this.earlyStoppingType = 3
                            break
                    }
                }


                // Load validation data
                if (this.validation.data) {
                    const typedArray = new Float32Array(this.validation.data.length)
                    this.loadData(this.validation.data, typedArray, itemSize , reject)
                    validationBuf = this.Module._malloc(typedArray.length*typedArray.BYTES_PER_ELEMENT)
                    this.Module.HEAPF32.set(typedArray, buf >> 2)

                    this.Module.ccall("loadValidationData", "number", ["number", "number", "number", "number", "number"],
                                                    [this.netInstance, buf, itemsCount, itemSize, dimension])
                }
            }

            const logAndResolve = () => {
                this.Module._free(buf)
                this.Module._free(validationBuf)

                if (this.validation && this.validation.earlyStopping && (this.validation.earlyStopping.type == "patience" || this.validation.earlyStopping.type == "divergence")) {
                    this.Module.ccall("restoreValidation", null, ["number"], [this.netInstance])
                }

                if (log) {
                    console.log(`Training finished. Total time: ${NetUtil.format(elapsed, "time")}`)
                }
                resolve()
            }

            if (callback) {

                let epochIndex = 0
                let iterationIndex = 0

                const doEpoch = () => {

                    if (this.l2) this.l2Error = 0
                    if (this.l1) this.l1Error = 0

                    iterationIndex = 0
                    doIteration()
                }

                const doIteration = () => {

                    this.Module.ccall("train", "number", ["number", "number", "number"], [this.netInstance, miniBatchSize, iterationIndex])

                    if (iterationIndex%callbackInterval == 0 || this.validationError) {
                        callback({
                            iterations: (this.iterations),
                            validations: (this.validations),
                            trainingError: this.error,
                            validationError: this.validationError,
                            elapsed: Date.now() - startTime,
                            input: data[iterationIndex].input
                        })
                    }

                    iterationIndex += miniBatchSize

                    if (iterationIndex < data.length && !this.stoppedEarly) {
                        if (iterationIndex%callbackInterval == 0) {
                            setTimeout(doIteration.bind(this), 0)
                        } else {
                            doIteration()
                        }
                    } else {
                        epochIndex++

                        elapsed = Date.now() - startTime

                        if (log) {
                            let text = `Epoch: ${epochIndex}\nTraining Error: ${this.error}`

                            if (this.validation) {
                                text += `\nValidation Error: ${this.lastValidationError}`
                            }

                            if (this.l2Error!=undefined) {
                                text += `\nL2 Error: ${this.l2Error/iterationIndex}`
                            }

                            text += `\nElapsed: ${NetUtil.format(elapsed, "time")} Average Duration: ${NetUtil.format(elapsed/epochIndex, "time")}`
                            console.log(text)
                        }

                        if (epochIndex < epochs && !this.stoppedEarly) {
                            doEpoch()
                        } else {
                            logAndResolve()
                        }
                    }
                }
                doEpoch()

            } else {
                for (let e=0; e<epochs; e++) {

                    if (this.l2) this.l2Error = 0
                    if (this.l1) this.l1Error = 0

                    this.Module.ccall("train", "number", ["number", "number", "number"], [this.netInstance, -1, 0])
                    elapsed = Date.now() - startTime

                    if (log) {
                        let text = `Epoch: ${e+1}\nTraining Error: ${this.error}`

                        if (validation) {
                            text += `\nValidation Error: ${this.lastValidationError}`
                        }

                        if (this.l2Error!=undefined) {
                            text += `\nL2 Error: ${this.l2Error/data.length}`
                        }

                        text += `\nElapsed: ${NetUtil.format(elapsed, "time")} Average Duration: ${NetUtil.format(elapsed/(e+1), "time")}`
                        console.log(text)
                    }

                    if (this.stoppedEarly) {
                        break
                    }
                }
                logAndResolve()
            }
        })
    }

    loadData (data, typedArray, itemSize, reject) {
        for (let di=0; di<data.length; di++) {

            if (!data[di].hasOwnProperty("input") || !data[di].hasOwnProperty("expected")) {
                return void reject("Data set must be a list of objects with keys: 'input' and 'expected'")
            }

            let index = itemSize * di

            // Volume input
            if (Array.isArray(data[di].input[0])) {
                for (let c=0; c<data[di].input.length; c++) {
                    for (let r=0; r<data[di].input[0].length; r++) {
                        for (let v=0; v<data[di].input[0].length; v++) {
                            typedArray[index] = data[di].input[c][r][v]
                            index++
                        }
                    }
                }
            } else {
                // Flat input
                for (let ii=0; ii<data[di].input.length; ii++) {
                    typedArray[index] = data[di].input[ii]
                    index++
                }
            }

            for (let ei=0; ei<data[di].expected.length; ei++) {
                typedArray[index] = data[di].expected[ei]
                index++
            }
        }
    }

    test (data, {log=true, collectErrors, callback}={}) {
        return new Promise((resolve, reject) => {

            if (data === undefined || data === null) {
                reject("No data provided")
            }

            if (log) {
                console.log("Testing started")
            }

            const startTime = Date.now()
            const dimension = data[0].input.length
            const itemSize = dimension + data[0].expected.length
            const itemsCount = itemSize * data.length
            const typedArray = new Float32Array(itemsCount)

            this.loadData(data, typedArray, itemSize, reject)

            const buf = this.Module._malloc(typedArray.length*typedArray.BYTES_PER_ELEMENT)
            this.Module.HEAPF32.set(typedArray, buf >> 2)

            this.Module.ccall("loadTestingData", "number", ["number", "number", "number", "number", "number"],
                                            [this.netInstance, buf, itemsCount, itemSize, dimension])

            if (collectErrors) {
                this.Module.ccall("collectErrors", null, ["number"], [this.netInstance])
            }

            if (callback) {

                let iterationIndex = 0
                let totalError = 0

                const doIteration = () => {

                    totalError += this.Module.ccall("test", "number", ["number", "number", "number"], [this.netInstance, 1, iterationIndex])

                    callback({
                        iterations: (iterationIndex+1),
                        error: totalError/(iterationIndex+1),
                        elapsed: Date.now() - startTime,
                        input: data[iterationIndex].input
                    })

                    if (++iterationIndex < data.length) {
                        setTimeout(doIteration.bind(this), 0)
                    } else {
                        iterationIndex

                        const elapsed = Date.now() - startTime
                        log && console.log(`Testing finished. Total time: ${NetUtil.format(elapsed, "time")}  Average iteration time: ${NetUtil.format(elapsed/iterationIndex, "time")}`)

                        this.Module._free(buf)
                        resolve(totalError/data.length)
                    }
                }

                doIteration()

            } else {

                const avgError = this.Module.ccall("test", "number", ["number", "number"], [this.netInstance, -1, 0])
                this.Module._free(buf)

                const elapsed = Date.now() - startTime

                if (log) {
                    console.log(`Testing finished. Total time: ${NetUtil.format(elapsed, "time")}  Average iteration time: ${NetUtil.format(elapsed/data.length, "time")}`)
                }

                resolve(avgError)
            }
        })
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

        this.Module.ccall("resetDeltaWeights", null, ["number"], [this.netInstance])
        this.layers.forEach((layer, li) => li && layer.fromJSON(data.layers[li], li))
    }

    toIMG (IMGArrays, opts={}) {

        if (!IMGArrays) {
            throw new Error("The IMGArrays library must be provided. See the documentation for instructions.")
        }

        const data = []

        for (let l=1; l<this.layers.length; l++) {

            const layerData = this.layers[l].toIMG()
            for (let v=0; v<layerData.length; v++) {
                data.push(layerData[v])
            }
        }

        return IMGArrays.toIMG(data, opts)
    }

    fromIMG (rawData, IMGArrays, opts={}) {

        if (!IMGArrays) {
            throw new Error("The IMGArrays library must be provided. See the documentation for instructions.")
        }

        let valI = 0
        const data = IMGArrays.fromIMG(rawData, opts)

        for (let l=1; l<this.layers.length; l++) {

            const dataCount = this.layers[l].getDataSize()
            this.layers[l].fromIMG(data.splice(0, dataCount))
        }
    }

    printConfusionMatrix (type) {
        if (type) {
            NetUtil.printConfusionMatrix(NetUtil.makeConfusionMatrix(this[`${type}ConfusionMatrix`]))
        } else {
            // Total all data
            const data = []

            for (let r=0; r<this.trainingConfusionMatrix.length; r++) {
                const row = []
                for (let c=0; c<this.trainingConfusionMatrix.length; c++) {
                    row.push(this.trainingConfusionMatrix[r][c] + this.testConfusionMatrix[r][c] + this.validationConfusionMatrix[r][c])
                }
                data.push(row)
            }
            NetUtil.printConfusionMatrix(NetUtil.makeConfusionMatrix(data))
        }
    }

    delete () {
        this.Module.ccall("deleteNetwork", "number", ["number"], [this.netInstance])
    }

    static get version () {
        return "3.3.4"
    }
}

/* istanbul ignore next */
typeof window!="undefined" && (window.Network = Network)
exports.Network = Network