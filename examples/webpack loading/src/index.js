import jsnet from "jsnet"

const { Network, Layer, FCLayer, ConvLayer, PoolLayer, Filter, Neuron, NetMath, NetUtil } = jsnet.webassembly()

window.addEventListener("jsNetWASMLoaded", () => {

    console.log("jsnet", jsnet)
    console.log("Module", Module)

    window.net = new Network({
        Module,
        layers: [2,3,4]
    })
    console.log("net", window.net)
})