from typing import List, Dict
from .arbitrage_utils import calculate_profit

class TriangularArbitrage:
    def __init__(self, config: Dict):
        self.config = config
        self.min_profit = config.get('min_profit', 0.5)

    async def find_opportunities(self, pairs: List[str]) -> List[Dict]:
        """Trouve les opportunités d'arbitrage triangulaire"""
        opportunities = []
        # Votre logique existante ici
        return [
            opp for opp in opportunities 
            if opp['profit'] >= self.min_profit
        ]

# Alias pour la rétrocompatibilité
find_triangular_opportunities = TriangularArbitrage(config={}).find_opportunities
