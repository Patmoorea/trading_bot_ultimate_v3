import logging
from datetime import datetime


def setup_logger():
    logger = logging.getLogger("trading_bot")
    logger.setLevel(logging.DEBUG)

    # Format personnalisé
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Fichier de logs
    file_handler = logging.FileHandler(
        f'logs/bot_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def check_module_health():
    """Nouveau contrôle d'intégrité"""
    required_methods = {
        "TechnicalAnalyzer": ["analyze"],
        "ArbitrageEngine": ["find_usdc_arbitrage"],
    }
    for module, methods in required_methods.items():
        for method in methods:
            assert hasattr(
                globals()[module](), method), f"{module}.{method} manquant"
