"""
Données simulées pour les exemples de l'interface web
Created: 2025-05-23 04:35:00
@author: Patmoorea
"""

import random
import time
from datetime import datetime, timedelta

def get_random_symbols():
    """Génère des symboles aléatoires pour les exemples"""
    base_currencies = ['BTC', 'ETH', 'XRP', 'LTC', 'DOT', 'ADA', 'SOL']
    quote_currencies = ['USDT', 'USDC', 'BTC', 'ETH']
    
    return f"{random.choice(base_currencies)}/{random.choice(quote_currencies)}"

def get_mock_opportunities(arb_type):
    """
    Génère des opportunités d'arbitrage simulées
    
    Args:
        arb_type: Type d'arbitrage ('triangular' ou 'inter_exchange')
        
    Returns:
        Liste d'opportunités simulées
    """
    opportunities = []
    now = datetime.now()
    
    for i in range(10):  # 10 opportunités simulées
        timestamp = now - timedelta(minutes=random.randint(0, 60))
        
        if arb_type == 'triangular':
            # Opportunité d'arbitrage triangulaire
            exchange = random.choice(['binance', 'kraken', 'kucoin'])
            profit = round(random.uniform(0.5, 3.0), 2)
            
            # Créer un cycle d'arbitrage
            currencies = ['USDT', 'BTC', 'ETH', 'XRP', 'LTC', 'DOT', 'ADA', 'SOL']
            random.shuffle(currencies)
            path = currencies[:3] + [currencies[0]]  # Cycle fermé
            
            opportunity = {
                'id': f'tri-{int(time.time())}-{i}',
                'type': 'triangular',
                'exchange': exchange,
                'profit': profit,
                'path': path,
                'timestamp': timestamp.isoformat(),
                'executed': random.random() > 0.8
            }
            
        else:  # inter_exchange
            # Opportunité d'arbitrage inter-exchanges
            exchanges = ['binance', 'kraken', 'kucoin', 'coinbase']
            buy_exchange = random.choice(exchanges)
            
            # Assurer que l'exchange de vente est différent
            exchanges.remove(buy_exchange)
            sell_exchange = random.choice(exchanges)
            
            symbol = get_random_symbols()
            profit = round(random.uniform(0.5, 3.0), 2)
            buy_price = round(random.uniform(100, 50000), 2)
            sell_price = buy_price * (1 + profit / 100)
            
            opportunity = {
                'id': f'inter-{int(time.time())}-{i}',
                'type': 'inter_exchange',
                'buy_exchange': buy_exchange,
                'sell_exchange': sell_exchange,
                'symbol': symbol,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'profit': profit,
                'timestamp': timestamp.isoformat(),
                'executed': random.random() > 0.8
            }
        
        opportunities.append(opportunity)
    
    # Trier par profit décroissant
    opportunities.sort(key=lambda x: x['profit'], reverse=True)
    
    return {
        'opportunities': opportunities,
        'count': len(opportunities),
        'max_profit': max(o['profit'] for o in opportunities),
        'avg_profit': sum(o['profit'] for o in opportunities) / len(opportunities),
        'timestamp': datetime.now().isoformat()
    }

def get_mock_stats():
    """
    Génère des statistiques simulées pour le bot
    
    Returns:
        Dictionnaire de statistiques
    """
    now = datetime.now()
    
    # Générer des données pour les 7 derniers jours
    days = []
    for i in range(7):
        day = now - timedelta(days=i)
        day_str = day.strftime('%Y-%m-%d')
        
        # Statistiques quotidiennes
        tri_opps = random.randint(50, 200)
        tri_executed = random.randint(0, int(tri_opps * 0.2))
        
        inter_opps = random.randint(20, 100)
        inter_executed = random.randint(0, int(inter_opps * 0.2))
        
        days.append({
            'date': day_str,
            'triangular': {
                'opportunities': tri_opps,
                'executed': tri_executed,
                'avg_profit': round(random.uniform(0.8, 2.0), 2),
                'max_profit': round(random.uniform(2.0, 5.0), 2)
            },
            'inter_exchange': {
                'opportunities': inter_opps,
                'executed': inter_executed,
                'avg_profit': round(random.uniform(1.0, 2.5), 2),
                'max_profit': round(random.uniform(2.5, 6.0), 2)
            }
        })
    
    # Statistiques globales
    return {
        'daily_stats': days,
        'total_stats': {
            'opportunities': sum(d['triangular']['opportunities'] + d['inter_exchange']['opportunities'] for d in days),
            'executed': sum(d['triangular']['executed'] + d['inter_exchange']['executed'] for d in days),
            'profit_realized': round(random.uniform(30, 100), 2),
            'runtime_hours': random.randint(100, 500)
        },
        'current_status': {
            'active': True,
            'last_scan': (now - timedelta(seconds=random.randint(0, 300))).isoformat(),
            'scan_interval': 30,
            'memory_usage': random.randint(100, 500)
        }
    }
