"""
Module de gestion des données de marché pour l'analyse de performance
Created: 2025-05-23 05:20:00
@author: Patmoorea
"""

import pandas as pd
import numpy as np
import ccxt
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

class MarketDataProvider:
    """
    Fournisseur de données de marché pour l'analyse des performances
    """
    
    def __init__(self, config_path: str = "config/market_data_config.json"):
        """
        Initialise le fournisseur de données de marché
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = self._load_config()
        self.data_cache = {}
        
    def _load_config(self) -> Dict:
        """
        Charge la configuration depuis le fichier
        
        Returns:
            Configuration du fournisseur de données
        """
        if not os.path.exists(self.config_path):
            self.logger.warning(f"Fichier de configuration '{self.config_path}' introuvable. Utilisation des valeurs par défaut.")
            return {
                "data_sources": ["binance", "kraken", "coinbase"],
                "default_source": "binance",
                "cache_dir": "data/market_data",
                "cache_expiry_hours": 24,
                "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
                "default_timeframe": "1h",
                "symbols": ["BTC/USDT", "ETH/USDT", "ETH/BTC"],
                "market_indicators": {
                    "volatility_window": 20,
                    "trend_window": 50,
                    "volume_window": 10
                }
            }
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {e}")
            raise
    
    def get_market_data(self, 
                        symbol: str, 
                        timeframe: str = None, 
                        start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None, 
                        source: str = None) -> pd.DataFrame:
        """
        Récupère les données de marché pour un symbole
        
        Args:
            symbol: Symbole de la paire (format: 'BTC/USDT')
            timeframe: Intervalle de temps ('1m', '5m', '15m', '1h', '4h', '1d')
            start_date: Date de début
            end_date: Date de fin
            source: Source des données ('binance', 'kraken', etc.)
            
        Returns:
            DataFrame des données OHLCV
        """
        # Utiliser les valeurs par défaut si nécessaire
        timeframe = timeframe or self.config["default_timeframe"]
        source = source or self.config["default_source"]
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=30))
        
        # Vérifier si les données sont en cache
        cache_key = f"{source}_{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # Vérifier si les données sont sur le disque
        cache_dir = self.config["cache_dir"]
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{cache_key}.csv")
        
        if os.path.exists(cache_file):
            # Vérifier si le cache est expiré
            file_modified_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            cache_expiry = timedelta(hours=self.config["cache_expiry_hours"])
            
            if datetime.now() - file_modified_time < cache_expiry:
                self.logger.info(f"Chargement des données depuis le cache: {cache_file}")
                df = pd.read_csv(cache_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                self.data_cache[cache_key] = df
                return df
        
        # Télécharger les données
        self.logger.info(f"Téléchargement des données pour {symbol} depuis {source}")
        df = self._download_market_data(symbol, timeframe, start_date, end_date, source)
        
        # Enregistrer dans le cache
        if not df.empty:
            self.data_cache[cache_key] = df
            df.to_csv(cache_file, index=False)
        
        return df
    
    def _download_market_data(self, 
                             symbol: str, 
                             timeframe: str, 
                             start_date: datetime, 
                             end_date: datetime, 
                             source: str) -> pd.DataFrame:
        """
        Télécharge les données de marché depuis une source
        
        Args:
            symbol: Symbole de la paire
            timeframe: Intervalle de temps
            start_date: Date de début
            end_date: Date de fin
            source: Source des données
            
        Returns:
            DataFrame des données OHLCV
        """
        try:
            # Initialiser l'exchange
            exchange_class = getattr(ccxt, source)
            exchange = exchange_class({
                'enableRateLimit': True
            })
            
            # Convertir les dates en millisecondes
            since = int(start_date.timestamp() * 1000)
            until = int(end_date.timestamp() * 1000)
            
            # Récupérer les données
            all_data = []
            current_since = since
            
            while current_since < until:
                try:
                    data = exchange.fetch_ohlcv(symbol, timeframe, current_since)
                    if not data:
                        break
                    
                    all_data.extend(data)
                    
                    # Obtenir la dernière timestamp
                    last_timestamp = data[-1][0]
                    
                    # Avancer au prochain bloc de données
                    if last_timestamp <= current_since:
                        break
                    
                    current_since = last_timestamp + 1
                except Exception as e:
                    self.logger.error(f"Erreur lors de la récupération des données: {e}")
                    break
            
            # Convertir en DataFrame
            if all_data:
                df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Filtrer par date
                df = df[(df['timestamp'] >= pd.Timestamp(start_date)) & 
                         (df['timestamp'] <= pd.Timestamp(end_date))]
                
                return df
            else:
                return pd.DataFrame()
        
        except Exception as e:
            self.logger.error(f"Erreur lors du téléchargement des données depuis {source}: {e}")
            return pd.DataFrame()
    
    def enrich_trades_with_market_data(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrichit les données de transactions avec les données de marché
        
        Args:
            trades_df: DataFrame des transactions
            
        Returns:
            DataFrame enrichi avec des données de marché
        """
        if trades_df.empty:
            return trades_df
        
        enriched_df = trades_df.copy()
        
        # Grouper par symbole et exchange pour optimiser les requêtes
        grouped = enriched_df.groupby(['symbol', 'exchange'])
        
        for (symbol, exchange), group in grouped:
            # Déterminer les dates
            min_date = group['timestamp'].min() - timedelta(days=1)
            max_date = group['timestamp'].max() + timedelta(days=1)
            
            # Récupérer les données de marché
            market_data = self.get_market_data(
                symbol=symbol,
                start_date=min_date,
                end_date=max_date,
                source=exchange if exchange in self.config['data_sources'] else self.config['default_source']
            )
            
            if market_data.empty:
                continue
            
            # Calculer les indicateurs de marché
            market_data = self._add_market_indicators(market_data)
            
            # Pour chaque transaction, trouver les données de marché correspondantes
            for idx, trade in group.iterrows():
                trade_time = trade['timestamp']
                
                # Trouver la période de marché la plus proche
                closest_time_idx = abs(market_data['timestamp'] - trade_time).idxmin()
                market_row = market_data.loc[closest_time_idx]
                
                # Ajouter les indicateurs de marché à la transaction
                for indicator in ['market_volatility', 'market_trend', 'volume_ratio']:
                    if indicator in market_row:
                        enriched_df.loc[idx, indicator] = market_row[indicator]
                
                # Classifier les conditions de marché
                if 'market_volatility' in market_row:
                    if market_row['market_volatility'] < 0.01:
                        enriched_df.loc[idx, 'market_condition'] = 'Low'
                    elif market_row['market_volatility'] < 0.02:
                        enriched_df.loc[idx, 'market_condition'] = 'Medium'
                    else:
                        enriched_df.loc[idx, 'market_condition'] = 'High'
                
                # Ajouter la tendance du marché
                if 'market_trend' in market_row:
                    if market_row['market_trend'] > 0.02:
                        enriched_df.loc[idx, 'market_direction'] = 'Bullish'
                    elif market_row['market_trend'] < -0.02:
                        enriched_df.loc[idx, 'market_direction'] = 'Bearish'
                    else:
                        enriched_df.loc[idx, 'market_direction'] = 'Sideways'
        
        return enriched_df
    
    def _add_market_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des indicateurs de marché au DataFrame
        
        Args:
            df: DataFrame des données OHLCV
            
        Returns:
            DataFrame avec indicateurs ajoutés
        """
        result = df.copy()
        
        # Configuration des fenêtres
        vol_window = self.config['market_indicators']['volatility_window']
        trend_window = self.config['market_indicators']['trend_window']
        vol_ratio_window = self.config['market_indicators']['volume_window']
        
        # Calculer la volatilité du marché
        if 'close' in result.columns:
            result['returns'] = result['close'].pct_change()
            result['market_volatility'] = result['returns'].rolling(window=vol_window).std()
        
        # Calculer la tendance du marché
        if 'close' in result.columns:
            result['market_trend'] = result['close'].pct_change(periods=trend_window)
        
        # Calculer le ratio de volume
        if 'volume' in result.columns:
            result['volume_sma'] = result['volume'].rolling(window=vol_ratio_window).mean()
            result['volume_ratio'] = result['volume'] / result['volume_sma']
        
        return result
    
    def get_market_phases(self, 
                          symbol: str, 
                          timeframe: str = '1d', 
                          start_date: Optional[datetime] = None, 
                          end_date: Optional[datetime] = None,
                          source: str = None) -> pd.DataFrame:
        """
        Identifie les phases de marché pour une période donnée
        
        Args:
            symbol: Symbole de la paire
            timeframe: Intervalle de temps
            start_date: Date de début
            end_date: Date de fin
            source: Source des données
            
        Returns:
            DataFrame avec les phases de marché
        """
        # Récupérer les données de marché
        df = self.get_market_data(symbol, timeframe, start_date, end_date, source)
        
        if df.empty:
            return pd.DataFrame()
        
        # Ajouter les indicateurs
        df = self._add_market_indicators(df)
        
        # Identifier les phases de marché
        df['phase'] = 'Unknown'
        
        # Phase haussière (bull market)
        bull_condition = (df['market_trend'] > 0.05) & (df['volume_ratio'] > 1.0)
        df.loc[bull_condition, 'phase'] = 'Bull'
        
        # Phase baissière (bear market)
        bear_condition = (df['market_trend'] < -0.05) & (df['volume_ratio'] > 1.0)
        df.loc[bear_condition, 'phase'] = 'Bear'
        
        # Phase d'accumulation
        accum_condition = (df['market_trend'].abs() < 0.03) & (df['market_trend'].shift(1) < -0.05)
        df.loc[accum_condition, 'phase'] = 'Accumulation'
        
        # Phase de distribution
        dist_condition = (df['market_trend'].abs() < 0.03) & (df['market_trend'].shift(1) > 0.05)
        df.loc[dist_condition, 'phase'] = 'Distribution'
        
        # Phase de consolidation
        consol_condition = (df['market_trend'].abs() < 0.03) & (~accum_condition) & (~dist_condition)
        df.loc[consol_condition, 'phase'] = 'Consolidation'
        
        return df
