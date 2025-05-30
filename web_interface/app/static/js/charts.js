class ChartManager {
    constructor() {
        this.charts = new Map();
        this.layout = {
            plot_bgcolor: '#161b22',
            paper_bgcolor: '#161b22',
            font: {
                color: '#e6edf3'
            },
            xaxis: {
                gridcolor: '#30363d',
                linecolor: '#30363d'
            },
            yaxis: {
                gridcolor: '#30363d',
                linecolor: '#30363d'
            }
        };
    }

    createChart(containerId, type = 'candlestick') {
        const container = document.getElementById(containerId);
        if (!container) return;

        const data = [{
            type: type,
            x: [],
            y: [],
            line: {
                color: '#00b4d8'
            }
        }];

        Plotly.newPlot(containerId, data, this.layout);
        this.charts.set(containerId, data);
    }

    updateChart(containerId, newData) {
        if (!this.charts.has(containerId)) {
            this.createChart(containerId);
        }

        Plotly.update(containerId, {
            x: [newData.x],
            y: [newData.y]
        });
    }
}

// Initialisation des graphiques
document.addEventListener('DOMContentLoaded', () => {
    const chartManager = new ChartManager();
    window.chartManager = chartManager; // Pour l'accès global
});
