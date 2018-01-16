"use strict"

const http = require("http"),
fs = require("fs"),
url = require("url")

http.createServer((request, response) => {

    let path = url.parse(request.url).pathname,
    data

    path = (path=="/"?"/browserDemo.html":path)

    console.log(path)

    switch(path){
        case "/NetWASM.wasm":
            try{
                data = fs.readFileSync(__dirname+"/dist"+path)
            }catch(e){}
            break
        default:
            try{
                data = fs.readFileSync(__dirname+path)
            }catch(e){}
    }

    response.end(data)

}).listen(1337, () => console.log("Server Listening"))

