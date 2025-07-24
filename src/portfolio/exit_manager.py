import numpy as np

class ExitManager:
    def __init__(self, tp_levels=None, trailing_pct=0.03):
        """
        tp_levels: Liste de tuples (target_pct, fraction) ex: [(0.03, 0.3), (0.07, 0.3)]
        trailing_pct: Pourcentage du trailing stop (ex: 0.03 = 3%)
        """
        # Par défaut : 30% à +3%, 30% à +7%, 40% trailing
        self.tp_levels = tp_levels or [(0.03, 0.3), (0.07, 0.3)]
        self.trailing_pct = trailing_pct

    def get_tp_targets(self, entry_price):
        return [(entry_price * (1 + pct), fraction) for pct, fraction in self.tp_levels]

    def check_tp_partial(self, entry_price, current_price, filled_targets):
        """Renvoie la fraction à sortir et le nouveau filled_targets"""
        targets = self.get_tp_targets(entry_price)
        to_exit = 0
        new_filled = filled_targets.copy()
        for i, (tp_price, fraction) in enumerate(targets):
            if not filled_targets[i] and current_price >= tp_price:
                to_exit += fraction
                new_filled[i] = True
        return to_exit, new_filled

    def check_trailing(self, entry_price, price_history, trailing_base=None):
        """
        price_history: liste des prix depuis l'entrée
        trailing_base: le plus haut atteint depuis l'entrée (pour le trailing)
        Retourne True si le trailing doit sortir la position
        """
        if not price_history:
            return False, trailing_base
        max_price = max(trailing_base or price_history[0], max(price_history))
        trigger_price = max_price * (1 - self.trailing_pct)
        should_exit = price_history[-1] <= trigger_price
        return should_exit, max_price
