module.exports = function(grunt){
    grunt.initConfig({
        concat: {
            options: {
                sourceMap: true
            },
            "jsNet": {
                src: ["dev/jsNet.js"],
                dest: "dist/jsNet.js"
            },
            "js-WebAssembly": {
                src: ["dev/js-WebAssembly/*.js", "!dev/js-WebAssembly/NetWASM.js"],
                dest: "dist/jsNetWebAssembly.concat.js"
            },
            "js-noWebAssembly": {
                src: ["dev/js/*.js", "!dev/js/NetAssembly.js"],
                dest: "dist/jsNetJS.concat.js"
            },
            "NetWASM.js": {
                src: ["dist/NetWASM.js", "dev/js-WebAssembly/NetWASM.js"],
                dest: "dist/NetWASM.js"
            }
        },

        uglify: {
            my_target: {
                options: {
                    sourceMap: {
                        includeSources: true,
                    },
                    mangle: false,
                },
                files: {
                    "dist/jsNetWebAssembly.min.js" : ["dist/jsNetWebAssembly.concat.js"],
                    "dist/jsNetJS.min.js" : ["dist/jsNetJS.concat.js"]
                }
            }
        },

        exec: {
            build: "C:/emsdk/emsdk_env.bat & echo Building... & emcc -o ./dist/NetWASM.js ./dev/cpp/emscripten.cpp -O3 -s ALLOW_MEMORY_GROWTH=1 -s WASM=1 -s NO_EXIT_RUNTIME=1 -std=c++14",
            emscriptenTests: "C:/emsdk/emsdk_env.bat & echo Building... & emcc -o ./test/emscriptenTests.js ./test/emscriptenTests.cpp -O3 -s ALLOW_MEMORY_GROWTH=1 -s WASM=1 -s NO_EXIT_RUNTIME=1 -std=c++14"
        },

        watch: {
            jsNet: {
                files: ["dev/jsNet.js"],
                tasks: ["concat:jsNet"]
            },
            cpp: {
                files: ["dev/cpp/*.cpp", "dev/cpp/*.h"],
                tasks: ["exec:build", "concat:NetWASM.js", "concat:js-WebAssembly", "uglify", "replace:emscriptenWASMPath"]
            },
            js: {
                files: ["dev/js/*.js"],
                tasks: ["concat:js-noWebAssembly", "uglify"]
            },
            wa: {
                files: ["dev/js-WebAssembly/*.js"],
                tasks: ["concat:js-WebAssembly", "uglify", "replace:emscriptenWASMPath"]
            },
            emscriptenTests: {
                files: ["test/emscriptenTests.cpp"],
                tasks: ["exec:emscriptenTests", "replace:emscriptenTestsFilePath"]
            }
        },

        replace: {
            emscriptenTestsFilePath: {
                src: ["test/emscriptenTests.js"],
                dest: "test/emscriptenTests.js",
                replacements: [{
                    from: "emscriptenTests.wasm",
                    to: "test/emscriptenTests.wasm"
                }]
            },
            emscriptenWASMPath: {
                src: ["dist/NetWASM.js"],
                dest: ["dist/NetWASM.js"],
                replacements: [{
                    from: `"NetWASM.wasm"`,
                    to: "global.jsNetWASMPath"
                }]
            }
        }
    })

    grunt.loadNpmTasks("grunt-contrib-watch")
    grunt.loadNpmTasks('grunt-contrib-concat')
    grunt.loadNpmTasks('grunt-contrib-uglify-es')
    grunt.loadNpmTasks('grunt-text-replace')
    grunt.loadNpmTasks("grunt-exec")

    grunt.registerTask("default", ["watch"])
}