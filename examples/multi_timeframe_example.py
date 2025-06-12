"""
Exemple d'utilisation du module d'analyse multi-timeframe
"""
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from src.analysis.multi_timeframe.analyzer import MultiTimeframeAnalyzer

def fetch_ohlcv(symbol="BTC/USDT", timeframes=None):
    """Récupère les données OHLCV pour plusieurs timeframes"""
    exchange = ccxt.binance()
    timeframes = timeframes or ["1m", "5m", "15m", "1h"]
    data = {}
    
    for tf in timeframes:
        ohlcv = exchange.fetch_ohlcv(symbol, tf)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        data[tf] = df
        
    return data

def main():
    # Récupération des données
    print("Récupération des données multi-timeframe...")
    data = fetch_ohlcv(timeframes=["5m", "15m", "1h", "4h"])
    
    # Initialisation de l'analyseur
    analyzer = MultiTimeframeAnalyzer()
    
    # Ajout des données pour chaque timeframe
    for tf, df in data.items():
        analyzer.add_data(tf, df)
    
    # Analyse de la confluence
    results = analyzer.analyze_confluence()
    
    print("\nAnalyse Multi-Timeframe - Résultats de confluence:")
    print("=" * 50)
    
    for indicator, result in results.items():
        direction_emoji = "🟢" if result["direction"] == "bullish" else "🔴" if result["direction"] == "bearish" else "⚪"
        print(f"{direction_emoji} {indicator.upper()}: {result['direction'].capitalize()} (Force: {result['strength']:.2f})")
    
    print("\nRésumé global:")
    directions = [r["direction"] for r in results.values()]
    if all(d == "bullish" for d in directions):
        print("✅ Convergence haussière forte détectée sur tous les timeframes")
    elif all(d == "bearish" for d in directions):
        print("❌ Convergence baissière forte détectée sur tous les timeframes")
    else:
        print("⚠️ Signaux mixtes - pas de convergence claire")

if __name__ == "__main__":
    main()
