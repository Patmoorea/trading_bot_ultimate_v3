import numpy as np
from src.core_merged.engine import TradingEngine

def run_simulation():
    engine = TradingEngine(mode='test')
    data = engine.load_data()
    
    for i in range(len(data)-100):
        print(f"\n--- Fenêtre {i} à {i+100} ---")
        analysis = engine.analyzer.analyze_with_cache(data.iloc[i:i+100])
        
        # Affichage des indicateurs
        print(f"RSI: {analysis['rsi'].iloc[-1]:.2f}")
        bbands = analysis['bbands']
        print(f"BBands - U:{bbands['upper'].iloc[-1]:.2f} M:{bbands['middle'].iloc[-1]:.2f} L:{bbands['lower'].iloc[-1]:.2f}")
        
        time.sleep(0.5)

if __name__ == '__main__':
    run_simulation()
