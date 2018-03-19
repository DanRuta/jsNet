"use strict"

class NetUtil {

    static format (value, type="string") {
        switch (true) {

            case type=="string" && typeof value=="string":
                value = value.replace(/(_|\s)/g, "").toLowerCase()
                break

            case type=="time" && typeof value=="number":
                const date = new Date(value)
                const formatted = []

                if (value < 1000) {
                    formatted.push(`${date.getMilliseconds()}ms`)

                } else if (value < 60000) {
                    formatted.push(`${date.getSeconds()}.${date.getMilliseconds()}s`)

                } else {

                    if (value >= 3600000) formatted.push(`${date.getHours()}h`)

                    formatted.push(`${date.getMinutes()}m`)
                    formatted.push(`${date.getSeconds()}s`)
                }

                value = formatted.join(" ")
                break
        }

        return value
    }

    static shuffle (arr) {
        for (let i=arr.length; i; i--) {
            const j = Math.floor(Math.random() * i)
            const x = arr[i-1]
            arr[i-1] = arr[j]
            arr[j] = x
        }
    }

    static addZeroPadding (map, zP) {

        const data = []

        for (let row=0; row<map.length; row++) {
            data.push(map[row].slice(0))
        }

        const extraRows = []

        for (let i=0; i<data.length+2*zP; i++) {
            extraRows.push(0)
        }

        for (let col=0; col<data.length; col++) {
            for (let i=0; i<zP; i++) {
                data[col].splice(0, 0, 0)
                data[col].splice(data[col].length+1, data[col].length, 0)
            }
        }

        for (let i=0; i<zP; i++) {
            data.splice(0, 0, extraRows.slice(0))
            data.splice(data.length, data.length-1, extraRows.slice(0))
        }

        return data
    }

    static arrayToMap (arr, size) {
        const map = []

        for (let i=0; i<size; i++) {
            map[i] = []

            for (let j=0; j<size; j++) {
                map[i][j] = arr[i*size+j]
            }
        }

        return map
    }

    static arrayToVolume (arr, channels) {

        const vol = []
        const size = Math.sqrt(arr.length/channels)
        const mapValues = size**2

        for (let d=0; d<Math.floor(arr.length/mapValues); d++) {

            const map = []

            for (let i=0; i<size; i++) {
                map[i] = []

                for (let j=0; j<size; j++) {
                    map[i][j] = arr[d*mapValues  + i*size+j]
                }
            }

            vol[d] = map
        }

        return vol
    }

    static convolve ({input, zeroPadding, weights, channels, stride, bias}) {

        const inputVol = NetUtil.arrayToVolume(input, channels)
        const outputMap = []

        const paddedLength = inputVol[0].length + zeroPadding*2
        const fSSpread = Math.floor(weights[0].length / 2)

        // For each input channel,
        for (let di=0; di<channels; di++) {
            inputVol[di] = NetUtil.addZeroPadding(inputVol[di], zeroPadding)
            // For each inputY without ZP
            for (let inputY=fSSpread; inputY<paddedLength-fSSpread; inputY+=stride) {
                outputMap[(inputY-fSSpread)/stride] = outputMap[(inputY-fSSpread)/stride] || []
                // For each inputX without zP
                for (let inputX=fSSpread; inputX<paddedLength-fSSpread; inputX+=stride) {
                    let sum = 0
                    // For each weightsY on input
                    for (let weightsY=0; weightsY<weights[0].length; weightsY++) {
                        // For each weightsX on input
                        for (let weightsX=0; weightsX<weights[0].length; weightsX++) {
                            sum += inputVol[di][inputY+(weightsY-fSSpread)][inputX+(weightsX-fSSpread)] * weights[di][weightsY][weightsX]
                        }
                    }

                    outputMap[(inputY-fSSpread)/stride][(inputX-fSSpread)/stride] = (outputMap[(inputY-fSSpread)/stride][(inputX-fSSpread)/stride]||0) + sum
                }
            }
        }

        // Then add bias
        for (let outY=0; outY<outputMap.length; outY++) {
            for (let outX=0; outX<outputMap.length; outX++) {
                outputMap[outY][outX] += bias
            }
        }

        return outputMap
    }

    static buildConvErrorMap (nextLayer, errorMap, filterI) {

        // Cache / convenience
        const zeroPadding = nextLayer.zeroPadding
        const paddedLength = errorMap.length + zeroPadding*2
        const fSSpread = Math.floor(nextLayer.filterSize / 2)

        // Zero pad and clear the error map, to allow easy convoling
        const paddedRow = []

        for (let val=0; val<paddedLength; val++) {
            paddedRow.push(0)
        }

        for (let row=0; row<paddedLength; row++) {
            errorMap[row] = paddedRow.slice(0)
        }

        // For each channel in filter in the next layer which corresponds to this filter
        for (let nlFilterI=0; nlFilterI<nextLayer.size; nlFilterI++) {

            const weights = nextLayer.filters[nlFilterI].weights[filterI]
            const errMap = nextLayer.filters[nlFilterI].errorMap

            // Unconvolve their error map using the weights
            for (let inputY=fSSpread; inputY<paddedLength - fSSpread; inputY+=nextLayer.stride) {
                for (let inputX=fSSpread; inputX<paddedLength - fSSpread; inputX+=nextLayer.stride) {

                    for (let weightsY=0; weightsY<nextLayer.filterSize; weightsY++) {
                        for (let weightsX=0; weightsX<nextLayer.filterSize; weightsX++) {
                            errorMap[inputY+(weightsY-fSSpread)][inputX+(weightsX-fSSpread)] += weights[weightsY][weightsX]
                                * errMap[(inputY-fSSpread)/nextLayer.stride][(inputX-fSSpread)/nextLayer.stride]
                        }
                    }
                }
            }
        }

        // Take out the zero padding. Rows:
        errorMap.splice(0, zeroPadding)
        errorMap.splice(errorMap.length-zeroPadding, errorMap.length)

        // Columns:
        for (let emYI=0; emYI<errorMap.length; emYI++) {
            errorMap[emYI] = errorMap[emYI].splice(zeroPadding, errorMap[emYI].length - zeroPadding*2)
        }
    }

    static buildConvDWeights (layer) {

        const weightsCount = layer.filters[0].weights[0].length
        const fSSpread = Math.floor(weightsCount / 2)
        const channelsCount = layer.filters[0].weights.length

        // For each filter
        for (let filterI=0; filterI<layer.filters.length; filterI++) {

            const filter = layer.filters[filterI]

            // Each channel will take the error map and the corresponding inputMap from the input...
            for (let channelI=0; channelI<channelsCount; channelI++) {

                const inputValues = NetUtil.getActivations(layer.prevLayer, channelI, layer.inMapValuesCount)
                const inputMap = NetUtil.addZeroPadding(NetUtil.arrayToMap(inputValues, Math.sqrt(layer.inMapValuesCount)), layer.zeroPadding)

                // ...slide the filter with correct stride across the zero-padded inputMap...
                for (let inputY=fSSpread; inputY<inputMap.length-fSSpread; inputY+=layer.stride) {
                    for (let inputX=fSSpread; inputX<inputMap.length-fSSpread; inputX+=layer.stride) {

                        const error = filter.errorMap[(inputY-fSSpread)/layer.stride][(inputX-fSSpread)/layer.stride]

                        // ...and at each location...
                        for (let weightsY=0; weightsY<weightsCount; weightsY++) {
                            for (let weightsX=0; weightsX<weightsCount; weightsX++) {
                                const activation = inputMap[inputY-fSSpread+weightsY][inputX-fSSpread+weightsX]
                                filter.deltaWeights[channelI][weightsY][weightsX] += activation * error
                            }
                        }
                    }
                }
            }

            // Increment the deltaBias by the sum of all errors in the filter
            for (let eY=0; eY<filter.errorMap.length; eY++) {
                for (let eX=0; eX<filter.errorMap.length; eX++) {
                    filter.deltaBias += filter.errorMap[eY][eX]
                }
            }
        }
    }

    static getActivations (layer, mapStartI, mapSize) {

        const returnArr = []

        if (arguments.length==1) {

            if (layer instanceof FCLayer) {

                for (let ni=0; ni<layer.neurons.length; ni++) {
                    returnArr.push(layer.neurons[ni].activation)
                }

            } else if (layer instanceof ConvLayer) {

                for (let fi=0; fi<layer.filters.length; fi++) {
                    for (let rowI=0; rowI<layer.filters[fi].activationMap.length; rowI++) {
                        for (let colI=0; colI<layer.filters[fi].activationMap[rowI].length; colI++) {
                            returnArr.push(layer.filters[fi].activationMap[rowI][colI])
                        }
                    }
                }

            } else {

                for (let channel=0; channel<layer.activations.length; channel++) {
                    for (let row=0; row<layer.activations[0].length; row++) {
                        for (let col=0; col<layer.activations[0].length; col++) {
                            returnArr.push(layer.activations[channel][row][col])
                        }
                    }
                }
            }

        } else {

            if (layer instanceof FCLayer) {

                for (let i=mapStartI*mapSize; i<(mapStartI+1)*mapSize; i++) {
                    returnArr.push(layer.neurons[i].activation)
                }

            } else if (layer instanceof ConvLayer) {

                for (let row=0; row<layer.filters[mapStartI].activationMap.length; row++) {
                    for (let col=0; col<layer.filters[mapStartI].activationMap[row].length; col++) {
                        returnArr.push(layer.filters[mapStartI].activationMap[row][col])
                    }
                }

            } else {

                for (let row=0; row<layer.activations[mapStartI].length; row++) {
                    for (let col=0; col<layer.activations[mapStartI].length; col++) {
                        returnArr.push(layer.activations[mapStartI][row][col])
                    }
                }
            }
        }

        return returnArr
    }

    static splitData (data, {training=0.7, validation=0.15, test=0.15}={}) {

        const split = {
            training: [],
            validation: [],
            test: []
        }

        // Define here splits, for returning at the end
        for (let i=0; i<data.length; i++) {
            let x = Math.random()

            if (x > 1-training) {
                split.training.push(data[i])
            } else {

                if (x<validation) {
                    split.validation.push(data[i])
                } else {
                    split.test.push(data[i])
                }

            }
        }

        return split
    }

    static normalize (data) {
        let minVal = Infinity
        let maxVal = -Infinity

        for (let i=0; i<data.length; i++) {
            if (data[i] < minVal) {
                minVal = data[i]
            }
            if (data[i] > maxVal) {
                maxVal = data[i]
            }
        }

        if ((-1*minVal + maxVal) != 0) {
            for (let i=0; i<data.length; i++) {
                data[i] = (data[i] + -1*minVal) / (-1*minVal + maxVal)
            }
        } else {
            for (let i=0; i<data.length; i++) {
                data[i] = 0.5
            }
        }

        return {minVal, maxVal}
    }

    static makeConfusionMatrix (originalData) {
        let total = 0
        let totalCorrect = 0
        const data = []

        for (let r=0; r<originalData.length; r++) {
            const row = []
            for (let c=0; c<originalData[r].length; c++) {
                row.push(originalData[r][c])
            }
            data.push(row)
        }


        for (let r=0; r<data.length; r++) {
            for (let c=0; c<data[r].length; c++) {
                total += data[r][c]
            }
        }

        for (let r=0; r<data.length; r++) {

            let rowTotal = 0
            totalCorrect += data[r][r]

            for (let c=0; c<data[r].length; c++) {
                rowTotal += data[r][c]
                data[r][c] = {count: data[r][c], percent: (data[r][c] / total * 100)||0}
            }

            const correctPercent = data[r][r].count / rowTotal * 100

            data[r].total = {
                correct: (correctPercent||0),
                wrong: (100 - correctPercent)||0
            }
        }

        // Collect bottom row percentages
        const bottomRow = []

        for (let c=0; c<data[0].length; c++) {

            let columnTotal = 0

            for (let r=0; r<data.length; r++) {
                columnTotal += data[r][c].count
            }

            const correctPercent = data[c][c].count / columnTotal * 100

            bottomRow.push({
                correct: (correctPercent)||0,
                wrong: (100 - correctPercent)||0
            })
        }

        data.total = bottomRow

        // Calculate final matrix percentage
        data.total.total = {
            correct: (totalCorrect / total * 100)||0,
            wrong: (100 - (totalCorrect / total * 100))||0
        }

        return data
    }

    /* istanbul ignore next */
    static printConfusionMatrix (data) {
        if (typeof window!="undefined") {

            for (let r=0; r<data.length; r++) {
                for (let c=0; c<data[r].length; c++) {
                    data[r][c] = `${data[r][c].count} (${data[r][c].percent.toFixed(1)}%)`
                }
                data[r].total = `${data[r].total.correct.toFixed(1)}% / ${data[r].total.wrong.toFixed(1)}%`
                data.total[r] = `${data.total[r].correct.toFixed(1)}% / ${data.total[r].wrong.toFixed(1)}%`
            }

            data.total.total = `${data.total.total.correct.toFixed(1)}% / ${data.total.total.wrong.toFixed(1)}%`

            console.table(data)
            return
        }


        const padNum = (num, percent) => {
            num = percent ? num.toFixed(1) + "%" : num.toString()
            const leftPad = Math.max(Math.floor((3*2+1 - num.length) / 2), 0)
            const rightPad = Math.max(3*2+1 - (num.length + leftPad), 0)
            return " ".repeat(leftPad)+num+" ".repeat(rightPad)
        }

        let colourText
        let colourBackground

        // Bright
        process.stdout.write("\n\x1b[1m")

        for (let r=0; r<data.length; r++) {

            // Bright white text
            colourText = "\x1b[2m\x1b[37m"

            // Count
            for (let c=0; c<data[r].length; c++) {
                colourBackground =  r==c ? "\x1b[42m" : "\x1b[41m"
                process.stdout.write(`${colourText}${colourBackground}\x1b[1m${padNum(data[r][c].count)}\x1b[22m`)
            }

            // Dim green text on white background
            colourText = "\x1b[2m\x1b[32m"
            colourBackground = "\x1b[47m"
            process.stdout.write(`${colourText}${colourBackground}${padNum(data[r].total.correct, true)}`)

            // Bright white text
            colourText = "\x1b[2m\x1b[37m"
            process.stdout.write(`${colourText}\n`)

            // Percent
            for (let c=0; c<data[r].length; c++) {
                colourBackground =  r==c ? "\x1b[42m" : "\x1b[41m"
                process.stdout.write(`${colourText}${colourBackground}${padNum(data[r][c].percent, true)}`)
            }

            // Dim red
            colourText = "\x1b[2m\x1b[31m"
            colourBackground = "\x1b[47m"
            process.stdout.write(`${colourText}${colourBackground}${padNum(data[r].total.wrong, true)}`)

            // Bright
            process.stdout.write("\x1b[1m\x1b[30m\n")
        }

        // Dim green text
        colourText = "\x1b[22m\x1b[32m"

        // Bottom row correct percentages
        for (const col of data.total) {
            process.stdout.write(`${colourText}${colourBackground}${padNum(col.correct, true)}`)
        }
        // Total correct percentages
        // Blue background
        colourBackground = "\x1b[1m\x1b[44m"
        process.stdout.write(`${colourText}${colourBackground}${padNum(data.total.total.correct, true)}\n`)

        // Dim red on white background
        colourText = "\x1b[22m\x1b[31m"
        colourBackground = "\x1b[47m"

        // Bottom row wrong percentages
        for (const col of data.total) {
            process.stdout.write(`${colourText}${colourBackground}${padNum(col.wrong, true)}`)
        }

        // Bright red on blue background
        colourText = "\x1b[1m\x1b[31m"
        colourBackground = "\x1b[44m"
        process.stdout.write(`${colourText}${colourBackground}${padNum(data.total.total.wrong, true)}\n`)

        // Reset
        process.stdout.write("\x1b[0m\n")
    }
}

/* istanbul ignore next */
typeof window!="undefined" && (window.NetUtil = NetUtil)
exports.NetUtil = NetUtil