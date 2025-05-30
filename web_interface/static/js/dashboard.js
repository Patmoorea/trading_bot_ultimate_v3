// Dashboard JavaScript

// Connexion WebSocket
const socket = io();

// Mise à jour des données en temps réel
socket.on('dashboard_update', function(data) {
    updateDashboardStats(data);
});

// Initialisation du graphique de portfolio
let portfolioChart = null;

document.addEventListener('DOMContentLoaded', function() {
    // Initialise le graphique avec des données vides
    initializePortfolioChart();
    
    // Récupère les données du dashboard
    fetchDashboardStats();
    
    // Met à jour les statistiques toutes les 5 secondes
    setInterval(fetchDashboardStats, 5000);
});

function initializePortfolioChart() {
    const ctx = document.getElementById('portfolio-chart').getContext('2d');
    portfolioChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: generateTimeLabels(24),
            datasets: [{
                label: 'Portfolio Value (USD)',
                data: generateRandomData(24),
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 2,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#f8f9fa'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#f8f9fa'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#f8f9fa'
                    }
                }
            }
        }
    });
}

function fetchDashboardStats() {
    fetch('/api/dashboard/stats')
        .then(response => response.json())
        .then(data => {
            updateDashboardStats(data);
        })
        .catch(error => {
            console.error('Error fetching dashboard stats:', error);
        });
}

function updateDashboardStats(data) {
    // Mise à jour des valeurs sur la page
    document.getElementById('portfolio-value').textContent = `$${data.portfolio_value.toFixed(2)}`;
    
    const dailyChange = document.getElementById('daily-change');
    dailyChange.textContent = `${data.daily_change_percent > 0 ? '+' : ''}${data.daily_change_percent.toFixed(2)}%`;
    dailyChange.className = data.daily_change_percent >= 0 ? 'text-success' : 'text-danger';
    
    document.getElementById('open-positions').textContent = data.open_positions;
    document.getElementById('arbitrage-opps').textContent = data.arbitrage_opportunities;
    document.getElementById('last-update').textContent = data.last_update;
    
    // Mise à jour des performances
    document.getElementById('perf-daily').textContent = `${data.performance.daily > 0 ? '+' : ''}${data.performance.daily.toFixed(2)}%`;
    document.getElementById('perf-daily').className = data.performance.daily >= 0 ? 'text-success' : 'text-danger';
    
    document.getElementById('perf-weekly').textContent = `${data.performance.weekly > 0 ? '+' : ''}${data.performance.weekly.toFixed(2)}%`;
    document.getElementById('perf-weekly').className = data.performance.weekly >= 0 ? 'text-success' : 'text-danger';
    
    document.getElementById('perf-monthly').textContent = `${data.performance.monthly > 0 ? '+' : ''}${data.performance.monthly.toFixed(2)}%`;
    document.getElementById('perf-monthly').className = data.performance.monthly >= 0 ? 'text-success' : 'text-danger';
    
    document.getElementById('perf-yearly').textContent = `${data.performance.yearly > 0 ? '+' : ''}${data.performance.yearly.toFixed(2)}%`;
    document.getElementById('perf-yearly').className = data.performance.yearly >= 0 ? 'text-success' : 'text-danger';
}

// Générateurs de données de test
function generateTimeLabels(count) {
    const labels = [];
    const now = new Date();
    
    for (let i = count - 1; i >= 0; i--) {
        const time = new Date(now);
        time.setHours(now.getHours() - i);
        labels.push(time.getHours() + ':00');
    }
    
    return labels;
}

function generateRandomData(count) {
    const data = [];
    let value = 12000 + Math.random() * 2000;
    
    for (let i = 0; i < count; i++) {
        value += (Math.random() - 0.5) * 200;
        data.push(value);
    }
    
    return data;
}
