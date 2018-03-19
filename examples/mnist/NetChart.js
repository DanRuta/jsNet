"use strict"

class NetChart {
    constructor ({container, size = {x: 500, y: 500}, cutOff=0, interval=5, averageOver=10}) {
        const canvas = document.createElement("canvas")
        canvas.width = size.x
        canvas.height = size.y
        canvas.style.maxHeight = size.y

        this.chart = new Chart(canvas.getContext("2d"), {
            type: "line",
            data: {
                datasets: [{
                    label: "Training Error",
                    borderColor: "rgba(0, 0, 0, 0.1)",
                    data: [],
                    pointRadius: 0
                }, {
                    label: "Validation Error",
                    fill: false,
                    data: [],
                    borderColor: "rgba(150, 26, 31, 0.25)",
                    backgroundColor: "rgba(150, 26, 31, 0.25)",
                    pointRadius: 0
                }]
            },
            options: {
                scales: {
                    xAxes: [{
                        type: "linear",
                        position: "bottom"
                    }],
                    yAxes: [{
                        ticks: {
                            beginAtZero: true
                        }
                    }]
                },
                tooltips: {
                    enabled: false
                },
                // maintainAspectRatio: false
                responsive: false
            }
        })
        this.chartX = 0
        this.chartYCount = 0
        this.chartY = 0
        this.chartY2 = 0
        this.chartY2Count = 0
        this.averageOver = averageOver
        this.interval = interval
        this.cutOff = cutOff
        container.appendChild(canvas)
    }

    addTrainingError (err) {

        this.chartY += err
        this.chartYCount++

        if (this.chartYCount==this.averageOver) {

            this.chart.data.datasets[0].data.push({
                x: this.chartX * this.interval,
                y: this.chartY/this.averageOver
            })

            if (this.cutOff && this.chart.data.datasets[0].data.length>this.cutOff/this.averageOver) {
                this.chart.data.datasets[0].data.shift()
            }

            this.chartYCount = 0
            this.chartY = 0
            this.chartX += this.averageOver
            this.chart.update()
        }
    }

    addValidationError (err) {

        this.chart.data.datasets[1].data.push({
            x: this.chartX * this.interval,
            y: err
        })

        this.chartY2Count = 0
        this.chartY2 = 0
    }

    clear () {
        this.chart.data.datasets[0].data = []
        this.chart.data.datasets[1].data = []
        this.chartX = 0
        this.chartYCount = 0
        this.chartY = 0
        this.chartY2Count = 0
        this.chartY2 = 0
        this.chart.update()
    }

    loadAllData ({training, validation, validationRate}) {

        this.clear()

        let chartY = 0
        let chartYCount = 0

        for (let i=0; i<training.length; i++) {

            this.addTrainingError(training[i])

            if (validation && i > validationRate && i%validationRate == 0) {
                this.addValidationError(validation.shift())
            }
        }

        this.chart.update()
    }
}