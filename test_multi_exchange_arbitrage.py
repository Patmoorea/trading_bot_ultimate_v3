from modules.multi_exchange_arbitrage import MultiExchangeArbitrage

def main():
    arb = MultiExchangeArbitrage()
    print("Recherche d'opportunités cross-exchange...")
    opportunities = arb.check_arbitrage()
    if not opportunities:
        print("Aucune opportunité disponible.")
        return
    best = arb.get_best_spread()
    print("Meilleure opportunité :")
    print(best)
    if best:
        result = arb.execute_arbitrage(best)
        print("Résultat exécution :")
        print(result)

if __name__ == "__main__":
    main()