module.exports = function(grunt){
    grunt.initConfig({
        concat: {
            options: {
                sourceMap: true
            },
            "js-WebAssembly": {
                src: ["dev/js-WebAssembly/*.js"],
                dest: "dist/jsNetWebAssembly.concat.js"
            },
            "js-noWebAssembly": {
                src: ["dev/js/*.js", "!dev/js/NetAssembly.js"],
                dest: "dist/jsNet.concat.js"
            }
        },

        babel: {
            options: {
                presets: ["es2015", "stage-3"]
            },
            dist: {
                files: {
                    "dist/jsNet.min.js": ["dist/jsNet.concat.js"]
                }
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
                    "dist/jsNet.min.js" : ["dist/jsNet.concat.js"]
                }
            }
        },

        exec: {
            build: "C:/emsdk/emsdk_env.bat & echo Building... & emcc -o ./dist/NetWASM.js ./dev/cpp/emscripten.cpp -O3 -s ALLOW_MEMORY_GROWTH=1 -s WASM=1 -s NO_EXIT_RUNTIME=1 -std=c++1z",
            emscriptenTests: "C:/emsdk/emsdk_env.bat & echo Building... & emcc -o ./test/emscriptenTests.js ./test/emscriptenTests.cpp -O3 -s ALLOW_MEMORY_GROWTH=1 -s WASM=1 -s NO_EXIT_RUNTIME=1 -std=c++1z"
        },

        watch: {
            cpp: {
                files: ["dev/cpp/*.cpp", "dev/cpp/*.h"],
                tasks: ["exec:build"]
            },
            js: {
                files: ["dev/js/*.js"],
                tasks: ["concatNoWebAssembly", "babel", "uglify"]
            },
            wa: {
                files: ["dev/js-WebAssembly/*.js"],
                tasks: ["concatWebAssembly", "uglify"]
            },
            emscriptenTests: {
                files: ["test/emscriptenTests.cpp"],
                tasks: ["exec:emscriptenTests", "replace"]
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
            }
        }
    })

    grunt.loadNpmTasks("grunt-babel")
    grunt.loadNpmTasks("grunt-contrib-watch")
    grunt.loadNpmTasks('grunt-contrib-concat')
    grunt.loadNpmTasks('grunt-contrib-uglify')
    grunt.loadNpmTasks('grunt-text-replace')
    grunt.loadNpmTasks("grunt-exec")

    grunt.registerTask("default", ["watch"])
    grunt.registerTask("concatWebAssembly", ["concat:js-WebAssembly"])
    grunt.registerTask("concatNoWebAssembly", ["concat:js-noWebAssembly"])
}