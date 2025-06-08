class TradingDashboard {
    constructor() {
        this.socket = io();
        this.charts = {};
        this.models = {};
        this.initializeWebSockets();
        this.initializeCharts();
        this.setupEventListeners();
    }

    initializeWebSockets() {
        // Connection WebSocket
        this.socket.on('connect', () => {
            this.updateStatus('Connected', 'success');
        });

        // Mise à jour des données en temps réel
        this.socket.on('market_data', (data) => {
            this.updateMarketData(data);
        });

        // Mise à jour AI
        this.socket.on('ai_update', (data) => {
            this.updateAIDecisions(data);
        });

        // Alertes
        this.socket.on('alert', (data) => {
            this.handleAlert(data);
        });

        // Mise à jour du portfolio
        this.socket.on('portfolio_update', (data) => {
            this.updatePortfolio(data);
        });
    }

    updateMarketData(data) {
        // Mise à jour des WebSockets
        const streamsContainer = document.getElementById('websocket-streams');
        data.streams.forEach(stream => {
            const streamElement = this.createOrUpdateStream(stream);
            this.updateStreamMetrics(streamElement, stream);
        });

        // Mise à jour des performances
        document.getElementById('buffer-latency').textContent = `${data.latency}ms`;
        document.getElementById('compression').textContent = `${data.compression}%`;
    }

    updateAIDecisions(data) {
        // Mise à jour CNN-LSTM
        const confidence = document.querySelector('.confidence-meter');
        confidence.style.setProperty('--value', `${data.confidence * 100}%`);
        confidence.textContent = `${Math.round(data.confidence * 100)}%`;

        // Mise à jour du régime de marché
        const regime = document.getElementById('regime-status');
        regime.textContent = data.regime;
        regime.className = `regime-${data.regime.toLowerCase().replace(' ', '-')}`;
    }

    handleAlert(data) {
        // Création de l'alerte
        const alert = document.createElement('div');
        alert.className = `alert alert-${data.type}`;
        alert.innerHTML = `
            <span class="time">${new Date(data.timestamp).toLocaleTimeString()}</span>
            <span class="message">${data.message}</span>
            <span class="pair">${data.pair}</span>
        `;

        // Ajout au feed
        const feed = document.querySelector('.activity-feed');
        feed.insertBefore(alert, feed.firstChild);

        // Envoi Telegram si activé
        if (data.telegram) {
            this.sendTelegramAlert(data);
        }
    }

    sendTelegramAlert(data) {
        fetch('/api/telegram/send', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
    }

    updatePortfolio(data) {
        // Mise à jour des métriques
        document.getElementById('total-value').textContent = 
            `${data.total_value.toLocaleString()} USDC`;
        document.getElementById('daily-pnl').textContent = 
            `${data.daily_pnl > 0 ? '+' : ''}${data.daily_pnl.toLocaleString()}%`;
        document.getElementById('sharpe-ratio').textContent = 
            data.sharpe_ratio.toFixed(2);
        document.getElementById('win-rate').textContent = 
            `${(data.win_rate * 100).toFixed(1)}%`;
    }

    initializeCharts() {
        // Création des graphiques
        this.charts.liquidity = new LiquidityHeatmap('liquidity-heatmap');
        this.charts.technical = new TechnicalChart('technical-chart');
        this.charts.portfolio = new PortfolioChart('portfolio-chart');
    }

    setupEventListeners() {
        // Écouteurs d'événements pour l'interface utilisateur
        document.querySelectorAll('.timeframe-selector').forEach(button => {
            button.addEventListener('click', (e) => {
                this.changeTimeframe(e.target.dataset.timeframe);
            });
        });
    }
}

// Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new TradingDashboard();
});
EOcat > app/static/js/charts.js << 'EOL'
class LiquidityHeatmap {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.initializeHeatmap();
    }

    initializeHeatmap() {
        // Configuration Plotly pour la heatmap
        const data = [{
            type: 'heatmap',
            z: [],
            colorscale: 'Jet'
        }];

        const layout = {
            title: 'Order Book Liquidity',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {
                color: '#e6edf3'
            }
        };

        Plotly.newPlot(this.container, data, layout);
    }

    update(orderbook) {
        // Mise à jour de la heatmap avec les nouvelles données
        const data = this.processOrderbook(orderbook);
        Plotly.update(this.container, {z: [data]});
    }
}

class TechnicalChart {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.initializeChart();
    }

    initializeChart() {
        // Configuration des graphiques techniques
        const data = [{
            type: 'candlestick',
            x: [],
            open: [],
            high: [],
            low: [],
            close: []
        }];

        const layout = {
            title: 'Price Action',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {
                color: '#e6edf3'
            },
            xaxis: {
                type: 'date',
                rangeslider: {visible: false}
            }
        };

        Plotly.newPlot(this.container, data, layout);
    }
}

class PortfolioChart {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.initializeChart();
    }

    initializeChart() {
        // Configuration du graphique de portfolio
        const data = [{
            type: 'scatter',
            mode: 'lines',
            x: [],
            y: []
        }];

        const layout = {
            title: 'Portfolio Value',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {
                color: '#e6edf3'
            }
        };

        Plotly.newPlot(this.container, data, layout);
    }
}
