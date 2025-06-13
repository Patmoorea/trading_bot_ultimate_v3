from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import logging
from decimal import Decimal
import asyncio

class PortfolioMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.portfolio = {
            'assets': {},
            'history': [],
            'trades': [],
            'metrics': {},
            'alerts': []
        }
        self.user = "Patmoorea"
        self.start_time = datetime.now(timezone.utc)

    async def update_portfolio(self, trade_data: Dict) -> Dict:
        try:
            # Vérification que c'est un achat en USDC
            if trade_data['side'] != 'BUY' or not trade_data['symbol'].endswith('USDC'):
                return {
                    'status': 'rejected',
                    'reason': 'Only BUY orders with USDC pairs are allowed'
                }

            # Mise à jour des actifs
            asset = trade_data['symbol'].replace('/USDC', '')
            amount = Decimal(str(trade_data['amount']))
            price = Decimal(str(trade_data['price']))
            
            if asset not in self.portfolio['assets']:
                self.portfolio['assets'][asset] = {
                    'amount': amount,
                    'avg_price': price,
                    'last_update': datetime.now(timezone.utc)
                }
            else:
                current = self.portfolio['assets'][asset]
                total_amount = current['amount'] + amount
                current['avg_price'] = (
                    (current['amount'] * current['avg_price'] + amount * price) / 
                    total_amount
                )
                current['amount'] = total_amount
                current['last_update'] = datetime.now(timezone.utc)

            # Enregistrement du trade
            self.portfolio['trades'].append({
                'timestamp': datetime.now(timezone.utc),
                'type': 'BUY',
                'asset': asset,
                'amount': float(amount),
                'price': float(price),
                'total': float(amount * price),
                'user': self.user
            })

            # Mise à jour des métriques
            await self._update_metrics()

            return {
                'status': 'success',
                'portfolio': self._get_portfolio_summary()
            }

        except Exception as e:
            self.logger.error(f"Portfolio update error: {e}")
            return {'status': 'error', 'reason': str(e)}

    async def _update_metrics(self):
        try:
            # Calcul des métriques de performance
            trades_df = pd.DataFrame(self.portfolio['trades'])
            if len(trades_df) > 0:
                trades_df['pnl'] = trades_df.apply(
                    lambda x: self._calculate_trade_pnl(x), axis=1
                )
                
                self.portfolio['metrics'] = {
                    'total_trades': len(trades_df),
                    'total_volume': float(trades_df['total'].sum()),
                    'avg_trade_size': float(trades_df['total'].mean()),
                    'best_trade': float(trades_df['pnl'].max()),
                    'worst_trade': float(trades_df['pnl'].min()),
                    'total_pnl': float(trades_df['pnl'].sum()),
                    'win_rate': float(len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)),
                    'last_update': datetime.now(timezone.utc)
                }

                # Calcul des alertes si nécessaire
                await self._check_alerts()

        except Exception as e:
            self.logger.error(f"Metrics update error: {e}")

    def _calculate_trade_pnl(self, trade) -> float:
        try:
            asset = trade['asset']
            if asset in self.portfolio['assets']:
                current_price = self._get_current_price(asset)  # À implémenter avec l'API
                return (current_price - trade['price']) * trade['amount']
            return 0.0
        except Exception as e:
            self.logger.error(f"PnL calculation error: {e}")
            return 0.0

    async def _check_alerts(self):
        try:
            current_time = datetime.now(timezone.utc)
            metrics = self.portfolio['metrics']

            # Vérification des conditions d'alerte
            if metrics['win_rate'] < 0.5:
                await self._add_alert(
                    'warning',
                    f"Win rate below 50%: {metrics['win_rate']:.1%}"
                )

            if metrics['total_pnl'] < 0:
                await self._add_alert(
                    'danger',
                    f"Negative total PnL: {metrics['total_pnl']:.2f} USDC"
                )

            # Alerte de performance quotidienne
            daily_trades = [t for t in self.portfolio['trades'] 
                          if (current_time - t['timestamp']).days < 1]
            if daily_trades:
                daily_pnl = sum(t['pnl'] for t in daily_trades)
                if daily_pnl < -1000:  # Seuil d'alerte à -1000 USDC
                    await self._add_alert(
                        'danger',
                        f"High daily loss: {daily_pnl:.2f} USDC"
                    )

        except Exception as e:
            self.logger.error(f"Alert check error: {e}")

    async def _add_alert(self, level: str, message: str):
        try:
            alert = {
                'timestamp': datetime.now(timezone.utc),
                'level': level,
                'message': message,
                'user': self.user
            }
            self.portfolio['alerts'].append(alert)

            # Envoi de l'alerte via Telegram
            await self._send_telegram_alert(alert)

        except Exception as e:
            self.logger.error(f"Alert creation error: {e}")

    async def _send_telegram_alert(self, alert: Dict):
        try:
            from ..services.telegram_service import TelegramService
            telegram = TelegramService()
            
            emoji = "🔴" if alert['level'] == 'danger' else "⚠️"
            message = f"{emoji} *Portfolio Alert*\n\n"
            message += f"*Level:* {alert['level']}\n"
            message += f"*Message:* {alert['message']}\n"
            message += f"*Time:* {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            message += f"*User:* {alert['user']}"

            await telegram.send_alert(message, importance="high" if alert['level'] == 'danger' else "normal")

        except Exception as e:
            self.logger.error(f"Telegram alert error: {e}")

    def _get_portfolio_summary(self) -> Dict:
        try:
            total_value = Decimal('0')
            assets_summary = {}

            for asset, details in self.portfolio['assets'].items():
                current_price = self._get_current_price(asset)
                value = details['amount'] * Decimal(str(current_price))
                total_value += value

                assets_summary[asset] = {
                    'amount': float(details['amount']),
                    'avg_price': float(details['avg_price']),
                    'current_price': current_price,
                    'value_usdc': float(value),
                    'pnl_percent': float((current_price / details['avg_price'] - 1) * 100)
                }

            return {
                'total_value_usdc': float(total_value),
                'assets': assets_summary,
                'metrics': self.portfolio['metrics'],
                'last_update': datetime.now(timezone.utc),
                'user': self.user
            }

        except Exception as e:
            self.logger.error(f"Portfolio summary error: {e}")
            return {}

    def _get_current_price(self, asset: str) -> float:
        # À implémenter avec l'API de prix en temps réel
        # Pour l'exemple, on retourne un prix fictif
        return 100.0

