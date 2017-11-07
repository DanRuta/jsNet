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

                        const typedArray = new heapMap[heapIn](params[p].length)

                        for (let i=0; i<params[p].length; i++) {
                            typedArray[i] = params[p][i]
                        }

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

            for (let v=0; v<returnArraySize; v++) {
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

    static defineProperty (self, prop, valTypes=[], values=[], {getCallback=x=>x, setCallback=x=>x, pre=""}={}) {
        Object.defineProperty(self, prop, {
            get: () => getCallback(this.Module.ccall(`get_${pre}${prop}`, "number", valTypes, values)),
            set: val => this.Module.ccall(`set_${pre}${prop}`, null, valTypes.concat("number"), values.concat(setCallback(val)))
        })
    }

    static defineArrayProperty (self, prop, valTypes, values, returnSize) {
        Object.defineProperty(self, prop, {
            get: () => NetUtil.ccallArrays(`get_${prop}`, "array", valTypes, values, {returnArraySize: returnSize, heapOut: "HEAPF64"}),
            set: (value) => NetUtil.ccallArrays(`set_${prop}`, null, valTypes.concat("array"), values.concat([value]), {heapIn: "HEAPF64"})
        })
    }

    static defineVolumeProperty (self, prop, valTypes, values, depth, rows, columns, {pre=""}={}) {
        Object.defineProperty(self, prop, {
            get: () => NetUtil.ccallVolume(`get_${pre}${prop}`, "volume", valTypes, values, {depth, rows, columns, heapOut: "HEAPF64"}),
            set: (value) => NetUtil.ccallVolume(`set_${pre}${prop}`, null, valTypes.concat("array"), values.concat([value]), {heapIn: "HEAPF64"})
        })
    }
}

NetUtil.activationsIndeces = {
    sigmoid: 0,
    tanh: 1,
    lecuntanh: 2,
    relu: 3,
    lrelu: 4,
    rrelu: 5,
    elu: 6
}

typeof window=="undefined" && (exports.NetUtil = NetUtil)