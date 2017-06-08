module.exports = function(grunt){
    grunt.initConfig({
        concat: {
            options: {
                sourceMap: true
            },
            js: {
                src: ["dev/*.js"],
                dest: "dist/Network.concat.js"
            }
        },

        babel: {
            options: {
                presets: ["es2015", "stage-3"]
            },
            dist: {
                options: {
                    sourceMap: true,
                    inputSourceMap: grunt.file.readJSON("./dist/Network.concat.js.map")
                },
                files: {
                    "dist/Network.min.js": ["dist/Network.concat.js"]
                }
            }
        },

        uglify: {
            my_target: {
                options: {
                    sourceMap: {
                        url: "dist/Network.min.js.map",
                        includeSources: true,  
                    },
                    sourceMapIn: "dist/Network.min.js.map"
                },
                files: {
                    "dist/Network.min.js" : ["dist/Network.min.js"]
                }
            }            
        },

        watch: {
            scripts: {
                files: ["dev/*.js"],
                tasks: ["build"]
            }
        }
    })

    grunt.loadNpmTasks("grunt-babel")
    grunt.loadNpmTasks("grunt-contrib-watch")
    grunt.loadNpmTasks('grunt-contrib-concat')
    grunt.loadNpmTasks('grunt-contrib-uglify')

    grunt.registerTask("default", ["watch"])
    grunt.registerTask("build", ["concat", "babel", "uglify"])
}