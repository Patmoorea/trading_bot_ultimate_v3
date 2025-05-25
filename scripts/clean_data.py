import pandas as pd

def clean_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    clean_csv(
        'data/historical/btc_usdt_1h.csv',
        'data/historical/btc_usdt_1h_clean.csv'
    )
    print("Nettoyage terminé. Fichier clean créé.")
