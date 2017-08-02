"use strict"

class NetUtil {
    
    static addZeroPadding (map, zP) {
        const extraColumns = [...new Array(zP)].map(v => 0)
        map = map.map(row => [...extraColumns, ...row, ...extraColumns])

        const extraRows = [...new Array(zP)].map(r => [...new Array(map.length+zP*2)].map(x => 0))
        return [...extraRows.slice(0), ...map, ...extraRows.slice(0)]
    }

    static build2DPrefixSAMap (square) {

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

    static sum2DPSAMap (map, zP, fS) {

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