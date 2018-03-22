"use strict"

const http = require("http")
const fs = require("fs")
const url = require("url")
const PORT = 1337

http.createServer((request, response) => {

    let path = url.parse(request.url).pathname
    let data

    path = (path=="/"?"/dist/index.html":path)

    console.log(path)

    switch (true) {
        case path.endsWith("/NetWASM.wasm"):
            try {
                console.log("Returning the wasm file", __dirname+"/node_modules/jsnet/dist/NetWASM.wasm")
                data = fs.readFileSync(__dirname+"/node_modules/jsnet/dist/NetWASM.wasm")
            } catch (e) {}
            break
        default:
            try {
                data = fs.readFileSync(__dirname+path)
            } catch (e) {}
    }

    response.end(data)

}).listen(PORT, () => console.log(`Server Listening on port: ${PORT}`))

