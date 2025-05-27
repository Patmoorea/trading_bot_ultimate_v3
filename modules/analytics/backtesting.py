#!/usr/bin/env python3
"""
Module de backtesting pour les stratégies d'arbitrage
Créé: 2025-05-23
Auteur: Patmoorea
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal

# Configuration du logging
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Moteur de backtesting pour évaluer les performances des stratégies
    d'arbitrage sur des données historiques.
    """
    
    def __init__(self, data_dir: str = "data/historical"):
        """
        Initialisation du moteur de backtesting
        
        Args:
            data_dir: Répertoire contenant les données historiques
        """
        self.data_dir = data_dir
        self.results = {}
        self.current_test = None
        
        # Créer le répertoire de données s'il n'existe pas
        os.makedirs(data_dir, exist_ok=True)
    
    def load_data(self, exchange: str, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Charge les données historiques pour un exchange et un symbole
        
        Args:
            exchange: Nom de l'exchange
            symbol: Symbole de la paire (ex: 'BTC/USDT')
            start_date: Date de début au format 'YYYY-MM-DD'
            end_date: Date de fin au format 'YYYY-MM-DD'
            
        Returns:
            DataFrame contenant les données historiques
        """
        # Construire le chemin du fichier
        safe_symbol = symbol.replace('/', '_')
        filepath = os.path.join(self.data_dir, f"{exchange}_{safe_symbol}_{start_date}_{end_date}.csv")
        
        # Vérifier si le fichier existe
        if os.path.exists(filepath):
            logger.info(f"Chargement des données depuis {filepath}")
            return pd.read_csv(filepath, parse_dates=['timestamp'])
        
        # Si le fichier n'existe pas, télécharger les données
        logger.info(f"Téléchargement des données pour {exchange} {symbol} du {start_date} au {end_date}")
        data = self._download_historical_data(exchange, symbol, start_date, end_date)
        
        # Sauvegarder les données dans un fichier
        data.to_csv(filepath, index=False)
        logger.info(f"Données sauvegardées dans {filepath}")
        
        return data
    
    def _download_historical_data(self, exchange: str, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Télécharge les données historiques depuis l'API de l'exchange
        
        Args:
            exchange: Nom de l'exchange
            symbol: Symbole de la paire
            start_date: Date de début au format 'YYYY-MM-DD'
            end_date: Date de fin au format 'YYYY-MM-DD'
            
        Returns:
            DataFrame contenant les données historiques
        """
        try:
            import ccxt
            
            # Convertir les dates en timestamps
            start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
            
            # Initialiser l'exchange
            exchange_obj = getattr(ccxt, exchange)({
                'enableRateLimit': True,
            })
            
            # Télécharger les données OHLCV
            data = []
            since = start_timestamp
            
            while since < end_timestamp:
                ohlcv = exchange_obj.fetch_ohlcv(symbol, '1h', since)
                if not ohlcv:
                    break
                
                data.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                # Limite pour éviter les boucles infinies
                if len(data) > 10000:
                    logger.warning(f"Limite de 10000 points de données atteinte")
                    break
            
            # Convertir en DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement des données: {e}")
            # Retourner un DataFrame vide en cas d'erreur
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def load_arbitrage_data(self, exchanges: List[str], symbol: str, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Charge les données historiques pour plusieurs exchanges pour une analyse d'arbitrage
        
        Args:
            exchanges: Liste des exchanges
            symbol: Symbole de la paire
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            Dictionnaire avec les données pour chaque exchange
        """
        data = {}
        
        for exchange in exchanges:
            try:
                df = self.load_data(exchange, symbol, start_date, end_date)
                data[exchange] = df
            except Exception as e:
                logger.error(f"Erreur lors du chargement des données pour {exchange}: {e}")
        
        return data
    
    def run_arbitrage_backtest(self, 
                              exchanges: List[str], 
                              symbol: str, 
                              start_date: str, 
                              end_date: str,
                              threshold: float = 0.5,
                              fees: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Exécute un backtest de stratégie d'arbitrage
        
        Args:
            exchanges: Liste des exchanges à comparer
            symbol: Symbole de la paire
            start_date: Date de début
            end_date: Date de fin
            threshold: Seuil de spread minimal pour l'arbitrage (%)
            fees: Dictionnaire des frais par exchange
            
        Returns:
            Résultats du backtest
        """
        # Configuration par défaut des frais
        if fees is None:
            fees = {exchange: 0.1 for exchange in exchanges}  # 0.1% par défaut
        
        # Charger les données
        data = self.load_arbitrage_data(exchanges, symbol, start_date, end_date)
        
        # Vérifier que nous avons des données pour au moins 2 exchanges
        if len(data) < 2:
            logger.error(f"Pas assez de données pour l'arbitrage. {len(data)} exchanges disponibles.")
            return {
                'status': 'error',
                'message': f"Pas assez de données pour l'arbitrage. {len(data)} exchanges disponibles.",
                'exchanges': list(data.keys())
            }
        
        # Aligner les timestamps
        aligned_data = self._align_timestamps(data)
        
        # Exécuter la simulation
        results = self._simulate_arbitrage(aligned_data, threshold, fees)
        
        # Enregistrer les résultats
        test_id = f"arbitrage_{symbol.replace('/', '_')}_{start_date}_{end_date}_{threshold}"
        self.results[test_id] = results
        self.current_test = test_id
        
        return results
    
    def _align_timestamps(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Aligne les timestamps des différentes sources de données
        
        Args:
            data: Dictionnaire des DataFrames par exchange
            
        Returns:
            Dictionnaire des DataFrames alignés
        """
        # Trouver tous les timestamps uniques
        all_timestamps = set()
        for df in data.values():
            all_timestamps.update(df['timestamp'].tolist())
        
        # Trier les timestamps
        all_timestamps = sorted(all_timestamps)
        
        # Créer un DataFrame d'index
        index_df = pd.DataFrame({'timestamp': all_timestamps})
        
        # Joindre chaque DataFrame sur l'index
        aligned_data = {}
        for exchange, df in data.items():
            merged = pd.merge(index_df, df, on='timestamp', how='left')
            # Remplir les valeurs manquantes avec la dernière valeur connue
            merged = merged.fillna(method='ffill')
            aligned_data[exchange] = merged
        
        return aligned_data
    
    def _simulate_arbitrage(self, data: Dict[str, pd.DataFrame], threshold: float, fees: Dict[str, float]) -> Dict[str, Any]:
        """
        Simule la stratégie d'arbitrage sur les données historiques
        
        Args:
            data: Dictionnaire des DataFrames alignés par exchange
            threshold: Seuil de spread minimal (%)
            fees: Dictionnaire des frais par exchange
            
        Returns:
            Résultats de la simulation
        """
        # Initialiser les variables de résultat
        trades = []
        balance = 1000.0  # Balance initiale en USDT
        position = 0.0    # Position en crypto
        
        # Extraire les timestamps communs
        timestamps = data[list(data.keys())[0]]['timestamp']
        
        # Pour chaque point de données temporel
        for i, timestamp in enumerate(timestamps):
            # Collecter les prix pour chaque exchange
            prices = {}
            for exchange, df in data.items():
                if i < len(df):
                    prices[exchange] = df.iloc[i]['close']
            
            # Trouver le meilleur prix d'achat et de vente
            if len(prices) < 2:
                continue
                
            best_buy = min(prices.items(), key=lambda x: x[1])
            best_sell = max(prices.items(), key=lambda x: x[1])
            
            # Calculer le spread
            spread_pct = (best_sell[1] - best_buy[1]) / best_buy[1] * 100
            
            # Vérifier si le spread dépasse le seuil après frais
            total_fees_pct = fees.get(best_buy[0], 0.1) + fees.get(best_sell[0], 0.1)
            net_spread_pct = spread_pct - total_fees_pct
            
            if net_spread_pct > threshold:
                # Simuler l'arbitrage
                
                # Si nous avons une position, nous vendons
                if position > 0:
                    sell_amount = position
                    sell_price = best_sell[1] * (1 - fees.get(best_sell[0], 0.1) / 100)
                    sell_value = sell_amount * sell_price
                    
                    balance += sell_value
                    position = 0
                    
                    trades.append({
                        'timestamp': timestamp,
                        'type': 'sell',
                        'exchange': best_sell[0],
                        'price': best_sell[1],
                        'amount': sell_amount,
                        'value': sell_value,
                        'balance': balance,
                        'spread_pct': spread_pct,
                        'net_spread_pct': net_spread_pct
                    })
                
                # Si nous avons de la balance, nous achetons
                elif balance > 0:
                    # Utiliser 90% de la balance pour l'achat
                    buy_value = balance * 0.9
                    buy_price = best_buy[1] * (1 + fees.get(best_buy[0], 0.1) / 100)
                    buy_amount = buy_value / buy_price
                    
                    balance -= buy_value
                    position += buy_amount
                    
                    trades.append({
                        'timestamp': timestamp,
                        'type': 'buy',
                        'exchange': best_buy[0],
                        'price': best_buy[1],
                        'amount': buy_amount,
                        'value': buy_value,
                        'balance': balance,
                        'spread_pct': spread_pct,
                        'net_spread_pct': net_spread_pct
                    })
        
        # Calculer les métriques finales
        
        # Convertir la position restante en USDT
        if position > 0:
            last_price = data[list(data.keys())[0]].iloc[-1]['close']
            final_balance = balance + position * last_price
        else:
            final_balance = balance
        
        # Calculer le ROI
        roi = (final_balance / 1000.0 - 1) * 100
        
        # Calculer d'autres métriques
        total_trades = len(trades)
        avg_spread = np.mean([trade['spread_pct'] for trade in trades]) if trades else 0
        avg_net_spread = np.mean([trade['net_spread_pct'] for trade in trades]) if trades else 0
        
        # Préparer les résultats
        results = {
            'status': 'success',
            'initial_balance': 1000.0,
            'final_balance': final_balance,
            'roi': roi,
            'total_trades': total_trades,
            'avg_spread': avg_spread,
            'avg_net_spread': avg_net_spread,
            'trades': trades[:100],  # Limiter à 100 trades pour éviter des objets trop grands
            'trade_count': total_trades,
            'parameters': {
                'threshold': threshold,
                'fees': fees
            }
        }
        
        return results
    
    def get_result(self, test_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère les résultats d'un backtest
        
        Args:
            test_id: Identifiant du test, utilise le test actuel si None
            
        Returns:
            Résultats du backtest
        """
        if test_id is None:
            test_id = self.current_test
        
        if test_id in self.results:
            return self.results[test_id]
        else:
            return {
                'status': 'error',
                'message': f"Test {test_id} non trouvé"
            }
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Sauvegarde les résultats du backtest dans un fichier JSON
        
        Args:
            filename: Nom du fichier, génère un nom basé sur la date si None
            
        Returns:
            Chemin du fichier sauvegardé
        """
        if filename is None:
            # Générer un nom basé sur la date et l'ID du test
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{now}.json"
        
        # Construire le chemin complet
        filepath = os.path.join(self.data_dir, filename)
        
        # Sauvegarder les résultats
        with open(filepath, 'w') as f:
            # Convertir les timestamps en chaînes pour la sérialisation JSON
            results_copy = {}
            for test_id, result in self.results.items():
                result_copy = result.copy()
                if 'trades' in result_copy:
                    for trade in result_copy['trades']:
                        if isinstance(trade['timestamp'], (datetime, pd.Timestamp)):
                            trade['timestamp'] = trade['timestamp'].isoformat()
                results_copy[test_id] = result_copy
            
            json.dump(results_copy, f, indent=2)
        
        logger.info(f"Résultats sauvegardés dans {filepath}")
        return filepath
    
    def plot_results(self, test_id: Optional[str] = None, output_file: Optional[str] = None) -> str:
        """
        Génère un graphique des résultats du backtest
        
        Args:
            test_id: Identifiant du test, utilise le test actuel si None
            output_file: Fichier de sortie pour le graphique
            
        Returns:
            Chemin du fichier de graphique
        """
        try:
            import matplotlib.pyplot as plt
            
            if test_id is None:
                test_id = self.current_test
            
            if test_id not in self.results:
                logger.error(f"Test {test_id} non trouvé")
                return ""
            
            result = self.results[test_id]
            trades = result.get('trades', [])
            
            if not trades:
                logger.warning(f"Pas de trades dans les résultats du test {test_id}")
                return ""
            
            # Préparer les données pour le graphique
            timestamps = [trade['timestamp'] for trade in trades]
            balances = [trade['balance'] for trade in trades]
            spreads = [trade['spread_pct'] for trade in trades]
            
            # Créer le graphique
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Graphique de la balance
            ax1.plot(timestamps, balances, 'b-', label='Balance')
            ax1.set_title(f"Résultats du Backtest d'Arbitrage - ROI: {result['roi']:.2f}%")
            ax1.set_ylabel('Balance (USDT)')
            ax1.legend()
            ax1.grid(True)
            
            # Graphique des spreads
            ax2.plot(timestamps, spreads, 'r-', label='Spread (%)')
            ax2.axhline(y=result['parameters']['threshold'], color='g', linestyle='--', 
                       label=f"Seuil ({result['parameters']['threshold']}%)")
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Spread (%)')
            ax2.legend()
            ax2.grid(True)
            
            # Formater l'axe des dates
            fig.autofmt_xdate()
            
            # Ajuster la mise en page
            plt.tight_layout()
            
            # Sauvegarder le graphique
            if output_file is None:
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(self.data_dir, f"backtest_plot_{now}.png")
            
            plt.savefig(output_file)
            plt.close()
            
            logger.info(f"Graphique sauvegardé dans {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique: {e}")
            return ""

# Test direct du module
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = BacktestEngine()
    
    # Exemple d'utilisation
    results = engine.run_arbitrage_backtest(
        exchanges=['binance', 'gateio', 'okx'],
        symbol='BTC/USDT',
        start_date='2025-01-01',
        end_date='2025-01-07',
        threshold=0.3
    )
    
    print(f"ROI: {results['roi']:.2f}%")
    print(f"Nombre de trades: {results['total_trades']}")
    print(f"Spread moyen: {results['avg_spread']:.2f}%")
    
    # Sauvegarder les résultats
    engine.save_results()
    
    # Générer un graphique
    engine.plot_results()
