import pandas as pd
import numpy as np

def deep_clean(input_file, output_file):
    # Chargement avec conversion forcée
    df = pd.read_csv(input_file, dtype={
        'open': np.float64,
        'high': np.float64,
        'low': np.float64,
        'close': np.float64,
        'volume': np.float64
    }, na_values=['null', 'None', ' ', ''])
    
    # Suppression des lignes non valides
    df = df.dropna()
    
    # Conversion finale de sécurité
    num_cols = ['open', 'high', 'low', 'close', 'volume']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    
    df.to_csv(output_file, index=False)
    print(f"Fichier nettoyé : {output_file}")
    print(f"Lignes conservées : {len(df)}/{len(pd.read_csv(input_file))}")

if __name__ == '__main__':
    deep_clean(
        'data/historical/btc_usdt_1h.csv',
        'data/historical/btc_usdt_1h_ultraclean.csv'
    )
