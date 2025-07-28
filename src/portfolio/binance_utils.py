def get_avg_entry_price_binance_spot(client, asset, quote="USDC"):
    """
    Calcule le prix d'achat moyen restant sur le portefeuille spot Binance pour un asset.
    Prend en compte tous les achats/ventes de l'historique (API /api/v3/myTrades).
    Retourne None si position close.
    """
    symbol = f"{asset}{quote}"
    try:
        trades = client.get_my_trades(symbol=symbol)
    except Exception as e:
        print(f"Erreur API get_my_trades pour {symbol}: {e}")
        return None

    total_qty = 0
    total_cost = 0
    for trade in trades:
        qty = float(trade["qty"])
        price = float(trade["price"])
        if trade["isBuyer"]:
            total_qty += qty
            total_cost += qty * price
        else:
            total_qty -= qty
            total_cost -= qty * price
    if total_qty > 0:
        return total_cost / total_qty
    else:
        return None  # plus de position ouverte
