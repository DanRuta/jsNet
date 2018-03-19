"use strict"

class NetUtil {

    static ccallArrays (func, returnType, paramTypes, params, {heapIn="HEAPF32", heapOut="HEAPF32", returnArraySize=1}={}) {

        const heapMap = {}
        heapMap.HEAP8 = Int8Array // int8_t
        heapMap.HEAPU8 = Uint8Array // uint8_t
        heapMap.HEAP16 = Int16Array // int16_t
        heapMap.HEAPU16 = Uint16Array // uint16_t
        heapMap.HEAP32 = Int32Array // int32_t
        heapMap.HEAPU32 = Uint32Array // uint32_t
        heapMap.HEAPF32 = Float32Array // float
        heapMap.HEAPF64 = Float64Array // double

        let res
        let error
        paramTypes = paramTypes || []
        const returnTypeParam = returnType=="array" ? "number" : returnType
        const parameters = []
        const parameterTypes = []
        const bufs = []

        try {
            if (params) {
                for (let p=0; p<params.length; p++) {

                    if (paramTypes[p] == "array" || Array.isArray(params[p])) {

                        const typedArray = new heapMap[heapIn](params[p])
                        const buf = NetUtil.Module._malloc(typedArray.length * typedArray.BYTES_PER_ELEMENT)

                        switch (heapIn) {
                            case "HEAP8": case "HEAPU8":
                                NetUtil.Module[heapIn].set(typedArray, buf)
                                break
                            case "HEAP16": case "HEAPU16":
                                NetUtil.Module[heapIn].set(typedArray, buf >> 1)
                                break
                            case "HEAP32": case "HEAPU32": case "HEAPF32":
                                NetUtil.Module[heapIn].set(typedArray, buf >> 2)
                                break
                            case "HEAPF64":
                                NetUtil.Module[heapIn].set(typedArray, buf >> 3)
                                break
                        }

                        bufs.push(buf)
                        parameters.push(buf)
                        parameters.push(params[p].length)
                        parameterTypes.push("number")
                        parameterTypes.push("number")

                    } else {
                        parameters.push(params[p])
                        parameterTypes.push(paramTypes[p]==undefined ? "number" : paramTypes[p])
                    }
                }
            }

            res = NetUtil.Module.ccall(func, returnTypeParam, parameterTypes, parameters)
        } catch (e) {
            error = e
        } finally {
            for (let b=0; b<bufs.length; b++) {
                NetUtil.Module._free(bufs[b])
            }
        }

        if (error) throw error


        if (returnType=="array") {
            const returnData = []

            let v = 0

            if (returnArraySize === "auto") {
                // Use the first value as the returnArraySize value
                v++
                returnArraySize = NetUtil.Module[heapOut][res/heapMap[heapOut].BYTES_PER_ELEMENT] + 1
            }

            for (v; v<returnArraySize; v++) {
                returnData.push(NetUtil.Module[heapOut][res/heapMap[heapOut].BYTES_PER_ELEMENT+v])
            }

            return returnData
        } else {
            return res
        }
    }

    static ccallVolume (func, returnType, paramTypes=[], params=[], {heapIn="HEAPF32", heapOut="HEAPF32", depth=1, rows=1, columns=rows}={}) {

        const totalValues = depth * rows * columns
        const parameters = []
        const parameterTypes = []

        // Loop through parameters, check if they are volumes, flatten them, and send them along with their dimensions
        for (let p=0; p<params.length; p++) {

            let parameter = params[p]
            const isVolume = Array.isArray(parameter) && Array.isArray(parameter[0]) && Array.isArray(parameter[0][0])

            if (paramTypes[p] == "volume" || isVolume) {
                const flat = []

                for (let d=0; d<parameter.length; d++) {
                    for (let r=0; r<parameter[d].length; r++) {
                        for (let c=0; c<parameter[d][r].length; c++) {
                            flat.push(parameter[d][r][c])
                        }
                    }
                }

                parameters.splice(parameters.length, 0, flat, parameter.length, parameter[0].length, parameter[0][0].length)
                parameterTypes.splice(parameterTypes.length, 0, "array", "number", "number", "number")

            } else {
                parameters.push(parameter)
                parameterTypes.push(paramTypes[p])
            }
        }

        const res = NetUtil.ccallArrays(func, returnType=="volume" ? "array" : returnType, parameterTypes, parameters, {heapIn, heapOut, returnArraySize: totalValues})
        const vol = []

        if (returnType == "volume") {
            for (let d=0; d<depth; d++) {
                const map = []

                for (let r=0; r<rows; r++) {
                    const row = []

                    for (let c=0; c<columns; c++) {
                        row.push(res[d * rows * columns + r * columns + c])
                    }
                    map.push(row)
                }
                vol.push(map)
            }
            return vol
        }

        return res
    }

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

    static defineProperty (self, prop, valTypes=[], values=[], {getCallback=x=>x, setCallback=x=>x, pre=""}={}) {
        Object.defineProperty(self, prop, {
            get: () => getCallback(this.Module.ccall(`get_${pre}${prop}`, "number", valTypes, values)),
            set: val => this.Module.ccall(`set_${pre}${prop}`, null, valTypes.concat("number"), values.concat(setCallback(val)))
        })
    }

    static defineArrayProperty (self, prop, valTypes, values, returnSize, {pre=""}={}) {
        Object.defineProperty(self, prop, {
            get: () => NetUtil.ccallArrays(`get_${pre}${prop}`, "array", valTypes, values, {returnArraySize: returnSize, heapOut: "HEAPF64"}),
            set: value => NetUtil.ccallArrays(`set_${pre}${prop}`, null, valTypes.concat("array"), values.concat([value]), {heapIn: "HEAPF64"})
        })
    }

    static defineMapProperty (self, prop, valTypes, values, rows, columns, {getCallback=x=>x, setCallback=x=>x, pre=""}={}) {
        Object.defineProperty(self, prop, {
            get: () => getCallback(NetUtil.ccallVolume(`get_${pre}${prop}`, "volume", valTypes, values, {depth: 1, rows, columns, heapOut: "HEAPF64"})[0]),
            set: value => NetUtil.ccallVolume(`set_${pre}${prop}`, null, valTypes.concat("array"), values.concat([setCallback(value)]), {heapIn: "HEAPF64"})
        })
    }

    static defineVolumeProperty (self, prop, valTypes, values, depth, rows, columns, {getCallback=x=>x, setCallback=x=>x, pre=""}={}) {
        Object.defineProperty(self, prop, {
            get: () => getCallback(NetUtil.ccallVolume(`get_${pre}${prop}`, "volume", valTypes, values, {depth, rows, columns, heapOut: "HEAPF64"})),
            set: value => NetUtil.ccallVolume(`set_${pre}${prop}`, null, valTypes.concat("array"), values.concat([setCallback(value)]), {heapIn: "HEAPF64"})
        })
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

NetUtil.activationsIndeces = {
    noactivation: -1,
    sigmoid: 0,
    tanh: 1,
    lecuntanh: 2,
    relu: 3,
    lrelu: 4,
    rrelu: 5,
    elu: 6
};

/* istanbul ignore next */
typeof window!="undefined" && (window.NetUtil = NetUtil)
exports.NetUtil = NetUtil