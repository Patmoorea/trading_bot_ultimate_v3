from modules.utils.telegram_logger import TelegramLogger


def alert_strategy_signal(signal):
    """Notifie les signaux de trading via Telegram"""
    logger = TelegramLogger()
    message = (
        f"📈 Signal: {signal.get('pair', 'N/A')} | "
        f"Action: {signal.get('action', 'N/A')} | "
        f"Confiance: {signal.get('confidence', 0):.2f}"
    )
    return logger.log(message)


EXAMPLE_SIGNAL = {
    "pair": "BTC/USDT",
    "action": "BUY",
    "confidence": 0.87,
    "timestamp": 1630000000,
}


def generate_signal(pair, action, confidence, indicators=None):
    """Génère un signal enrichi"""
    signal = {
        "pair": pair,
        "action": action,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat(),
        "indicators": indicators or {},
    }
    return signal
