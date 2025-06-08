import pandas as pd

def clean_data(input_file, output_file):
    df = pd.read_csv(input_file)
    keep_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df[keep_cols].to_csv(output_file, index=False)

if __name__ == '__main__':
    clean_data(
        'data/historical/btc_usdt_1h.csv',
        'data/historical/btc_usdt_1h_clean.csv'
    )
    print("Nettoyage terminé. Fichier clean créé.")
