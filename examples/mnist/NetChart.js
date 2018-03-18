"use strict"

class NetChart {
    constructor ({container, size = {x: 500, y: 500}, cutOff=0, avgSpan=10, validationSplit=avgSpan}) {
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
        this.avgSpan = avgSpan
        this.validationSplit = validationSplit
        this.cutOff = cutOff
        container.appendChild(canvas)
    }

    addTrainingError (err) {

        if (this.chartYCount==this.avgSpan-1) {

            this.chart.data.datasets[0].data.push({
                x: this.chartX,
                y: this.chartY/this.avgSpan
            })

            if (this.cutOff && this.chart.data.datasets[0].data.length>this.cutOff/this.avgSpan) {
                this.chart.data.datasets[0].data.shift()
            }

            this.chartYCount = 0
            this.chartY = 0
            this.chartX += this.avgSpan
            this.chart.update()
        } else {
            this.chartY += err
            this.chartYCount++
        }
    }

    addValidationError (err) {
        this.chart.data.datasets[1].data.push({
            x: this.chartX,
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

}