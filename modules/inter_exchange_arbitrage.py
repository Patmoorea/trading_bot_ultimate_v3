"""
Module d'arbitrage inter-exchanges
Détecte et exécute des opportunités d'arbitrage entre différentes plateformes d'échange
Created: 2025-05-23 00:36:44
@author: Patmoorea
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
import ccxt.async_support as ccxt_async
from .arbitrage_utils import calculate_profit, adjust_quantity_async, format_opportunity


class InterExchangeArbitrage:
    """
    Classe gérant l'arbitrage entre différentes plateformes d'échange
    """

    def __init__(self, config: Dict):
        """
        Initialise le module d'arbitrage inter-exchanges

        Args:
            config: Configuration du module avec les paramètres suivants
                - exchanges: Liste des exchanges à utiliser (noms ccxt)
                - min_profit: Profit minimum pour considérer une opportunité (%)
                - quote_currencies: Devises de référence
                - symbols: Liste des symboles à surveiller
                - fees: Dictionnaire des frais par exchange
                - withdrawal_fees: Dictionnaire des frais de retrait par devise et exchange
        """
        self.config = config
        self.min_profit = config.get(
            "min_profit", 1.0
        )  # Par défaut 1% car arbitrage inter-exchanges a plus de frais
        self.quote_currencies = config.get(
            "quote_currencies", ["USDT", "USDC", "BTC", "ETH"]
        )
        self.symbols = config.get("symbols", [])
        self.exchanges = {}
        self.logger = logging.getLogger(__name__)
        self.opportunities = []
        self.last_update = 0

        # Initialiser les connexions aux exchanges
        self._init_exchanges()

    def _init_exchanges(self):
        """Initialise les connexions aux exchanges configurés"""
        exchange_names = self.config.get(
            "exchanges",
            ["binance", "kraken", "kucoin", "gateio", "bingx", "okx", "blofin"],
        )

        for name in exchange_names:
            try:
                exchange_class = getattr(ccxt_async, name)
                api_key = self.config.get(f"{name}_api_key", "")
                api_secret = self.config.get(f"{name}_api_secret", "")
                options = {}
                if name in self.config.get("exchange_options", {}):
                    options = self.config["exchange_options"][name]
                self.exchanges[name] = exchange_class(
                    {
                        "apiKey": api_key,
                        "secret": api_secret,
                        "enableRateLimit": True,
                        "options": options,
                    }
                )
                self.logger.info(f"Exchange {name} initialisé avec succès")
            except Exception as e:
                self.logger.error(
                    f"Erreur lors de l'initialisation de l'exchange {name}: {e}"
                )

    async def close(self):
        """Ferme proprement les connexions aux exchanges"""
        for name, exchange in self.exchanges.items():
            try:
                await exchange.close()
                self.logger.info(f"Connexion à {name} fermée")
            except Exception as e:
                self.logger.error(f"Erreur lors de la fermeture de {name}: {e}")

    async def get_exchange_tickers(self) -> Dict:
        """
        Récupère les tickers des exchanges configurés

        Returns:
            Dictionnaire des tickers par exchange et par symbole
        """
        all_tickers = {}
        symbols = self.symbols.copy() if self.symbols else []
        if not symbols:
            for exchange_name, exchange in self.exchanges.items():
                try:
                    markets = await exchange.load_markets()
                    exchange_symbols = [
                        s
                        for s in markets.keys()
                        if any(
                            s.endswith(f"/{quote}") for quote in self.quote_currencies
                        )
                    ]
                    if exchange_name not in all_tickers:
                        all_tickers[exchange_name] = {}
                    symbols.extend(exchange_symbols)
                except Exception as e:
                    self.logger.error(
                        f"Erreur lors du chargement des marchés pour {exchange_name}: {e}"
                    )
        symbols = list(set(symbols))
        for exchange_name, exchange in self.exchanges.items():
            if exchange_name not in all_tickers:
                all_tickers[exchange_name] = {}
            for symbol in symbols:
                try:
                    if not await self._is_symbol_supported(exchange, symbol):
                        continue
                    ticker = await exchange.fetch_ticker(symbol)
                    all_tickers[exchange_name][symbol] = ticker
                except Exception as e:
                    self.logger.warning(
                        f"Erreur lors de la récupération du ticker {symbol} sur {exchange_name}: {e}"
                    )

        self.last_update = time.time()
        return all_tickers

    async def _is_symbol_supported(self, exchange, symbol: str) -> bool:
        """Vérifie si un symbole est supporté par un exchange"""
        try:
            markets = exchange.markets
            if not markets:
                markets = await exchange.load_markets()
            return symbol in markets and markets[symbol]["active"]
        except Exception:
            return False

    async def find_opportunities(self) -> List[Dict]:
        """
        Recherche des opportunités d'arbitrage entre exchanges

        Returns:
            Liste des opportunités d'arbitrage rentables
        """
        self.opportunities = []
        all_tickers = await self.get_exchange_tickers()
        symbols = set()
        for exchange_tickers in all_tickers.values():
            symbols.update(exchange_tickers.keys())

        for symbol in symbols:
            supporting_exchanges = {}
            for exchange_name, tickers in all_tickers.items():
                if symbol in tickers and isinstance(tickers[symbol], dict):
                    supporting_exchanges[exchange_name] = tickers[symbol]
            if len(supporting_exchanges) < 2:
                continue
            exchange_names = list(supporting_exchanges.keys())
            for i in range(len(exchange_names)):
                for j in range(i + 1, len(exchange_names)):
                    exchange1 = exchange_names[i]
                    exchange2 = exchange_names[j]
                    ticker1 = supporting_exchanges[exchange1]
                    ticker2 = supporting_exchanges[exchange2]
                    await self._check_arbitrage_opportunity(
                        exchange1, exchange2, symbol, ticker1, ticker2
                    )
        self.opportunities.sort(key=lambda x: x["profit"], reverse=True)
        self.logger.info(
            f"Trouvé {len(self.opportunities)} opportunités d'arbitrage inter-exchanges"
        )
        return self.opportunities

    async def _check_arbitrage_opportunity(
        self, exchange1: str, exchange2: str, symbol: str, ticker1: Dict, ticker2: Dict
    ):
        """Vérifie s'il existe une opportunité d'arbitrage entre deux exchanges pour un symbole"""
        fee1 = self.config.get("fees", {}).get(exchange1, 0.1) / 100
        fee2 = self.config.get("fees", {}).get(exchange2, 0.1) / 100
        base, quote = symbol.split("/")
        withdrawal_fee1 = (
            self.config.get("withdrawal_fees", {}).get(exchange1, {}).get(base, 0)
        )
        withdrawal_fee2 = (
            self.config.get("withdrawal_fees", {}).get(exchange2, {}).get(base, 0)
        )
        buy_price1 = ticker1["ask"]
        sell_price1 = ticker1["bid"]
        buy_price2 = ticker2["ask"]
        sell_price2 = ticker2["bid"]

        # Exchange1 -> Exchange2
        if sell_price2 > buy_price1:
            gross_profit = sell_price2 / buy_price1 - 1
            total_fee = fee1 + fee2 + withdrawal_fee1 / buy_price1
            net_profit = (gross_profit - total_fee) * 100
            if net_profit >= self.min_profit:
                opportunity = {
                    "type": "inter_exchange",
                    "buy_exchange": exchange1,
                    "sell_exchange": exchange2,
                    "symbol": symbol,
                    "buy_price": buy_price1,
                    "sell_price": sell_price2,
                    "gross_profit": gross_profit * 100,
                    "fees": total_fee * 100,
                    "profit": net_profit,
                    "direction": f"{exchange1}->{exchange2}",
                    "timestamp": int(time.time() * 1000),
                }
                self.opportunities.append(opportunity)
        # Exchange2 -> Exchange1
        if sell_price1 > buy_price2:
            gross_profit = sell_price1 / buy_price2 - 1
            total_fee = fee1 + fee2 + withdrawal_fee2 / buy_price2
            net_profit = (gross_profit - total_fee) * 100
            if net_profit >= self.min_profit:
                opportunity = {
                    "type": "inter_exchange",
                    "buy_exchange": exchange2,
                    "sell_exchange": exchange1,
                    "symbol": symbol,
                    "buy_price": buy_price2,
                    "sell_price": sell_price1,
                    "gross_profit": gross_profit * 100,
                    "fees": total_fee * 100,
                    "profit": net_profit,
                    "direction": f"{exchange2}->{exchange1}",
                    "timestamp": int(time.time() * 1000),
                }
                self.opportunities.append(opportunity)

    async def execute_arbitrage(self, opportunity: Dict, amount: float) -> Dict:
        """
        Exécute une opportunité d'arbitrage inter-exchanges spot RÉELLE
        Args:
            opportunity: Opportunité d'arbitrage à exécuter
            amount: Montant à investir (en devise quote)
        Returns:
            Résultats de l'exécution
        """
        if not opportunity or "buy_exchange" not in opportunity:
            return {"success": False, "error": "Invalid opportunity format"}
        buy_exchange_name = opportunity["buy_exchange"]
        sell_exchange_name = opportunity["sell_exchange"]
        symbol = opportunity["symbol"]
        base_currency = symbol.split("/")[0]
        quote_currency = symbol.split("/")[1]

        if (
            buy_exchange_name not in self.exchanges
            or sell_exchange_name not in self.exchanges
        ):
            return {"success": False, "error": "Exchange not configured"}

        buy_exchange = self.exchanges[buy_exchange_name]
        sell_exchange = self.exchanges[sell_exchange_name]
        execution = {
            "success": False,
            "buy_order": None,
            "sell_order": None,
            "transfer": None,
            "profit_expected": opportunity["profit"],
            "profit_actual": 0,
            "amount": amount,
            "errors": [],
        }

        try:
            # 1. Achat sur le premier exchange
            self.logger.info(f"[ARBITRAGE] Achat de {symbol} sur {buy_exchange_name}")
            buy_price = opportunity["buy_price"]
            buy_amount = amount / buy_price
            buy_amount = await adjust_quantity_async(buy_exchange, symbol, buy_amount)
            buy_order = await buy_exchange.create_order(
                symbol=symbol, type="market", side="buy", amount=buy_amount
            )
            execution["buy_order"] = {
                "id": buy_order["id"],
                "exchange": buy_exchange_name,
                "amount": buy_amount,
                "price": buy_price,
                "cost": buy_amount * buy_price,
                "status": buy_order["status"],
            }

            # 2. Récupérer l'adresse de dépôt sur le second exchange
            self.logger.info(
                f"[ARBITRAGE] Récupération de l'adresse de dépôt pour {base_currency} sur {sell_exchange_name}"
            )
            deposit_address = await sell_exchange.fetch_deposit_address(base_currency)
            if not deposit_address or not deposit_address.get("address"):
                raise Exception(
                    f"Adresse dépôt introuvable pour {base_currency} sur {sell_exchange_name}"
                )

            # 3. Retrait de l'actif sur le premier exchange vers le second
            self.logger.info(
                f"[ARBITRAGE] Retrait de {buy_amount} {base_currency} depuis {buy_exchange_name} vers {deposit_address['address']} ({sell_exchange_name})"
            )
            withdrawal_fee = (
                self.config.get("withdrawal_fees", {})
                .get(buy_exchange_name, {})
                .get(base_currency, 0)
            )
            transfer_amount = buy_amount - withdrawal_fee
            withdraw_result = await buy_exchange.withdraw(
                code=base_currency,
                amount=transfer_amount,
                address=deposit_address["address"],
            )
            execution["transfer"] = {
                "from_exchange": buy_exchange_name,
                "to_exchange": sell_exchange_name,
                "currency": base_currency,
                "amount": transfer_amount,
                "fee": withdrawal_fee,
                "txid": withdraw_result.get("id", ""),
                "status": withdraw_result.get("status", "pending"),
            }

            # 4. Attendre la confirmation du dépôt sur le second exchange
            self.logger.info(
                f"[ARBITRAGE] Attente de la confirmation du dépôt sur {sell_exchange_name} pour {transfer_amount} {base_currency}"
            )
            max_wait = 1800  # max 30 min d'attente
            poll_interval = 20
            waited = 0
            while waited < max_wait:
                deposits = await sell_exchange.fetch_deposits(code=base_currency)
                if any(
                    d.get("amount", 0) >= transfer_amount and d.get("status") == "ok"
                    for d in deposits
                ):
                    self.logger.info(
                        f"[ARBITRAGE] Dépôt confirmé sur {sell_exchange_name}"
                    )
                    break
                await asyncio.sleep(poll_interval)
                waited += poll_interval
            else:
                raise Exception("Timeout confirmation dépôt")

            # 5. Vente sur le second exchange
            self.logger.info(
                f"[ARBITRAGE] Vente de {transfer_amount} {base_currency} sur {sell_exchange_name}"
            )
            sell_price = opportunity["sell_price"]
            sell_amount = await adjust_quantity_async(
                sell_exchange, symbol, transfer_amount
            )
            sell_order = await sell_exchange.create_order(
                symbol=symbol, type="market", side="sell", amount=sell_amount
            )
            execution["sell_order"] = {
                "id": sell_order["id"],
                "exchange": sell_exchange_name,
                "amount": sell_amount,
                "price": sell_price,
                "cost": sell_amount * sell_price,
                "status": sell_order["status"],
            }

            # 6. Calcul du profit réel
            initial_value = amount
            final_value = sell_amount * sell_price
            execution["profit_actual"] = ((final_value / initial_value) - 1) * 100
            execution["success"] = True
            self.logger.info(
                f"[ARBITRAGE] Arbitrage exécuté avec succès. Profit: {execution['profit_actual']:.2f}%"
            )

        except Exception as e:
            self.logger.error(
                f"[ARBITRAGE] Erreur lors de l'exécution de l'arbitrage: {str(e)}"
            )
            execution["errors"].append(str(e))

        return execution

    def format_opportunities(self) -> List[str]:
        """Formate les opportunités pour l'affichage."""
        formatted = []
        for opp in self.opportunities:
            profit = opp["profit"]
            symbol = opp["symbol"]
            direction = opp["direction"]
            buy_price = opp["buy_price"]
            sell_price = opp["sell_price"]
            formatted.append(
                f"Profit: {profit:.2f}% | {symbol} | {direction} | Buy: {buy_price:.8f} | Sell: {sell_price:.8f}"
            )
        return formatted

    def get_best_opportunity(self) -> Optional[Dict]:
        """Retourne la meilleure opportunité d'arbitrage."""
        return self.opportunities[0] if self.opportunities else None

    def set_notification_manager(self, notification_manager):
        """Définit un gestionnaire de notifications pour les alertes"""
        self.notification_manager = notification_manager
        self.logger.info("Gestionnaire de notifications configuré")

    async def scan_and_notify(self):
        """Recherche des opportunités d'arbitrage et envoie des notifications"""
        opportunities = await self.find_opportunities()
        if hasattr(self, "notification_manager") and opportunities:
            for opportunity in opportunities:
                self.notification_manager.notify_opportunity(opportunity)
        return opportunities
