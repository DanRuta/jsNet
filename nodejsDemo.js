"use strict"

const xor = [
    {input: [0, 0], expected: [0]},
    {input: [0, 1], expected: [1]},
    {input: [1, 0], expected: [1]},
    {input: [1, 1], expected: [0]}
]

// Load the JavaScript only version
const demoJS = () => {
    // npm
    // const {Network, FCLayer} = require("jsNet").js()

    // manual
    const {Network, FCLayer} = require("./dist/jsNet.js").js()

    // manual - js only
    // const {Network, FCLayer} = require("./dist/jsNetJS.min.js")

    console.log("JavaScript version loaded. Starting training...")
    const start = Date.now()

    const net = new Network({layers: [new FCLayer(2), new FCLayer(3), new FCLayer(1)]})

    net.train(xor, {epochs: 1000, log: false}).then(() => {
        console.log("Forward 0,0: ", net.forward(xor[0].input))
        console.log("Forward 0,1: ", net.forward(xor[1].input))
        console.log("Forward 1,0: ", net.forward(xor[2].input))
        console.log("Forward 1,1: ", net.forward(xor[3].input))
        console.log(`\n\nElapsed: ${Date.now()-start}ms`)
    })
}

// Load the WebAsssembly version
const demoWebAssembly = () => {

    // npm
    // const {Module, Network, FCLayer} = require("jsNet").webassembly()

    // manual
    const {Module, Network, FCLayer} = require("./dist/jsNet.js").webassembly("./dist/NetWASM.wasm")

    // manual - webassembly only
    // global.jsNetWASMPath = "./dist/NetWASM.wasm"
    // const {Network, FCLayer} = require("./dist/jsNetWebAssembly.min.js")
    // const Module = require("./dist/NetWASM.js")

    global.onWASMLoaded = () => {
        console.log("WebAsssembly version loaded. Starting training...")
        const start = Date.now()

        const net = new Network({
            Module: Module,
            layers: [new FCLayer(2), new FCLayer(3), new FCLayer(1)]
        })

        net.train(xor, {epochs: 1000, log: false}).then(() => {
            console.log("Forward 0,0: ", net.forward(xor[0].input))
            console.log("Forward 0,1: ", net.forward(xor[1].input))
            console.log("Forward 1,0: ", net.forward(xor[2].input))
            console.log("Forward 1,1: ", net.forward(xor[3].input))
            console.log(`\n\nElapsed: ${Date.now()-start}ms`)
            demoJS()
        })
    }
}

demoWebAssembly()
