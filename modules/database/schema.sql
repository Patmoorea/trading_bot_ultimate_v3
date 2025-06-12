-- Schéma de la base de données pour le bot de trading
-- Created: 2025-05-23 04:15:00
-- @author: Patmooreaoui

-- Table des opportunités d'arbitrage
CREATE TABLE IF NOT EXISTS opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,                -- 'triangular', 'inter_exchange'
    detected_at TEXT NOT NULL,         -- Horodatage ISO 8601
    profit REAL NOT NULL,              -- Profit en %
    buy_exchange TEXT,                 -- Exchange d'achat
    sell_exchange TEXT,                -- Exchange de vente
    symbol TEXT,                       -- Symbole principal
    details_json TEXT,                 -- Détails spécifiques au format JSON
    status TEXT DEFAULT 'detected',    -- detected, analyzing, executing, executed, failed, ignored
    execution_result TEXT,             -- Résultat d'exécution au format JSON
    updated_at TEXT                    -- Horodatage de dernière mise à jour
);

-- Index pour accélérer les requêtes
CREATE INDEX IF NOT EXISTS idx_opportunities_type ON opportunities(type);
CREATE INDEX IF NOT EXISTS idx_opportunities_detected_at ON opportunities(detected_at);
CREATE INDEX IF NOT EXISTS idx_opportunities_profit ON opportunities(profit);
CREATE INDEX IF NOT EXISTS idx_opportunities_status ON opportunities(status);
CREATE INDEX IF NOT EXISTS idx_opportunities_exchanges ON opportunities(buy_exchange, sell_exchange);

-- Table des configurations
CREATE TABLE IF NOT EXISTS configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,         -- Nom de la configuration
    value TEXT,                        -- Valeur au format JSON
    created_at TEXT NOT NULL,          -- Date de création
    updated_at TEXT NOT NULL           -- Date de mise à jour
);

-- Table des performances
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_date TEXT NOT NULL,         -- Date de la métrique (YYYY-MM-DD)
    metric_type TEXT NOT NULL,         -- Type de métrique (profit, volume, etc.)
    value REAL NOT NULL,               -- Valeur de la métrique
    details_json TEXT,                 -- Détails additionnels au format JSON
    created_at TEXT NOT NULL           -- Date de création
);

-- Index pour les métriques de performance
CREATE INDEX IF NOT EXISTS idx_performance_metrics_date ON performance_metrics(metric_date);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON performance_metrics(metric_type);

-- Table des logs importants
CREATE TABLE IF NOT EXISTS important_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,           -- Horodatage ISO 8601
    level TEXT NOT NULL,               -- INFO, WARNING, ERROR, CRITICAL
    module TEXT NOT NULL,              -- Module source
    message TEXT NOT NULL,             -- Message du log
    details_json TEXT                  -- Détails additionnels au format JSON
);

-- Index pour les logs
CREATE INDEX IF NOT EXISTS idx_important_logs_timestamp ON important_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_important_logs_level ON important_logs(level);
