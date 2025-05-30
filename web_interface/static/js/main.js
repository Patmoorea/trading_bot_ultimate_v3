/**
 * Fichier JavaScript principal pour l'interface web du bot de trading
 * Created: 2025-05-23 04:50:00
 * @author: Patmoorea
 */

// Initialisation de Socket.IO
const socket = io();

// Variables globales
let refreshInterval = null;
let darkModeEnabled = localStorage.getItem('darkMode') === 'true';

// Fonction pour formater les dates
function formatDate(dateString) {
    return moment(dateString).format('YYYY-MM-DD HH:mm:ss');
}

// Fonction pour formater les pourcentages
function formatPercent(value) {
    return value.toFixed(2) + '%';
}

// Fonction pour formater les nombres avec séparateurs de milliers
function formatNumber(value) {
    return new Intl.NumberFormat().format(value);
}

// Fonction pour formater les prix
function formatPrice(value, decimals = 8) {
    return value.toFixed(decimals);
}

// Gestionnaire de notification
function showNotification(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Ajouter la notification au conteneur
    const container = document.createElement('div');
    container.className = 'position-fixed top-0 end-0 p-3';
    container.style.zIndex = 1050;
    container.appendChild(alertDiv);
    
    document.body.appendChild(container);
    
    // Retirer la notification après 5 secondes
    setTimeout(() => {
        alertDiv.classList.remove('show');
        setTimeout(() => {
            container.remove();
        }, 300);
    }, 5000);
}

// Détecter si la connexion est perdue
socket.on('connect', function() {
    console.log('Connecté au serveur');
    showNotification('Connecté au serveur', 'success');
});

socket.on('disconnect', function() {
    console.log('Déconnecté du serveur');
    showNotification('Connexion au serveur perdue', 'danger');
});

// Recevoir des mises à jour en temps réel
socket.on('new_opportunity', function(data) {
    console.log('Nouvelle opportunité reçue:', data);
    showNotification(`Nouvelle opportunité d'arbitrage détectée: ${data.profit}%`, 'info');
});

socket.on('status_update', function(data) {
    console.log('Mise à jour du statut:', data);
    
    // Mettre à jour l'indicateur de statut dans le navbar
    const statusElement = document.getElementById('bot-status');
    if (statusElement) {
        if (data.active) {
            statusElement.innerHTML = '<i class="fas fa-circle text-success me-1"></i>Bot actif';
        } else {
            statusElement.innerHTML = '<i class="fas fa-circle text-danger me-1"></i>Bot arrêté';
        }
    }
});

// Fonction pour basculer le mode sombre
function toggleDarkMode() {
    darkModeEnabled = !darkModeEnabled;
    localStorage.setItem('darkMode', darkModeEnabled);
    applyDarkMode();
}

// Appliquer le mode sombre si activé
function applyDarkMode() {
    const body = document.body;
    if (darkModeEnabled) {
        body.classList.add('dark-mode');
        // Vous devrez définir les styles CSS appropriés pour le mode sombre
    } else {
        body.classList.remove('dark-mode');
    }
}

// Initialiser au chargement du document
document.addEventListener('DOMContentLoaded', function() {
    // Appliquer le mode sombre si activé
    applyDarkMode();
    
    // Initialiser le temps serveur
    const serverTimeElement = document.getElementById('server-time');
    if (serverTimeElement) {
        setInterval(() => {
            serverTimeElement.textContent = moment().format('YYYY-MM-DD HH:mm:ss');
        }, 1000);
    }
    
    // Ajouter un gestionnaire pour le bouton de mode sombre si présent
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', toggleDarkMode);
    }
    
    console.log('Interface web initialisée');
});
