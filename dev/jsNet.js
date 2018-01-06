"use strict"

exports.js = () => require("./jsNetJS.min.js")
exports.webassembly = (path="./node_modules/jsnet/dist/NetWASM.wasm") => {
    global.jsNetWASMPath = path
    const jsNet = require("./jsNetWebAssembly.min.js")
    jsNet.Module = require("./NetWASM.js")
    return jsNet
}
