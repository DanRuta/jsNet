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

                } else {

                    if (value >= 3600000) formatted.push(`${date.getHours()}h`)
                    if (value >= 60000)   formatted.push(`${date.getMinutes()}m`)

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
        const extraColumns = [...new Array(zP)].map(v => 0)
        map = map.map(row => [...extraColumns, ...row, ...extraColumns])

        const extraRows = [...new Array(zP)].map(r => [...new Array(map.length+zP*2)].map(x => 0))
        return [...extraRows.slice(0), ...map, ...extraRows.slice(0)]
    }

    // 2D Prefix Sum Array
    static build2DPSA (square) {

        const l = square.length
        let map = [...new Array(l+1)].map(row => [...new Array(l+1)].map(v => 0))

        for (let ri=1; ri<=l; ri++) {
            for (let vi=1; vi<=l; vi++) {
                map[ri][vi] = map[ri-1][vi] + square[ri-1][vi-1] 
            }
        }

        for (let ri=1; ri<=l; ri++) {
            for (let vi=1; vi<=l; vi++) {
                map[ri][vi] += map[ri][vi-1]
            }
        }

        return map
    }

    static sum2DPSA (map, zP, fS) {

        const l = map.length
        const sumMap = [...new Array(l-1-(zP*2))].map(row => [...new Array(l-1-(zP*2))].map(v => 0))

        for (let ri=l-zP; ri>zP+1; ri--) {
            for (let v=l-zP; v>zP+1; v--) {

                const l = v - fS
                const t = ri - fS

                sumMap[ri-Math.floor(fS)][v-Math.floor(fS)] = map[ri][v] - map[ri][l] - map[t][v] + map[t][l]   
            }
        }

        return sumMap
    }   
}

typeof window=="undefined" && (exports.NetUtil = NetUtil)