import asyncio
import logging
from datetime import datetime, timezone
import websockets
import json


class WebSocketManager:
    def __init__(self, bot):
        self.bot = bot
        self.streams = {}
        self.running = False
        self.lock = asyncio.Lock()
        self.advanced_indicators = {}
        # Correction des valeurs par défaut
        self.pairs = bot.config.get("TRADING", {}).get(
            "pairs", ["BTC/USDT", "ETH/USDT"]
        )
        self.timeframes = bot.config.get("TRADING", {}).get(
            "timeframes", ["1m", "5m", "15m", "1h", "4h", "1d"]
        )
        self.retry_count = 0
        self.max_retries = WEBSOCKET_CONFIG["MAX_RETRIES"]
        self.retry_delay = WEBSOCKET_CONFIG["RETRY_DELAY"]

    async def start(self):
        await self._initialize_analyzers()
        """Démarre les WebSockets"""
        async with self.lock:
            if self.running:
                return True

            try:
                # Initialisation du client Binance
                self.bot.binance_ws = await AsyncClient.create(
                    api_key=os.getenv("BINANCE_API_KEY"),
                    api_secret=os.getenv("BINANCE_API_SECRET"),
                )

                # Initialisation du socket manager
                self.bot.socket_manager = BinanceSocketManager(self.bot.binance_ws)

                # Configuration des streams
                if not await self._setup_streams():
                    raise Exception("Failed to setup streams")

                self.running = True
                return True

            except Exception as e:
                logger.error(f"WebSocket start error: {e}")
                await self.cleanup()
                return False

    async def _setup_streams(self):
        """Configure les streams"""
        try:
            for pair in self.pairs:
                # Stream de trades
                ts = self.bot.socket_manager.trade_socket(pair)
                self.streams[f"{pair}_trades"] = asyncio.create_task(
                    self._handle_stream(ts, "trade", pair)
                )

                # Stream d'orderbook
                ds = self.bot.socket_manager.depth_socket(pair)
                self.streams[f"{pair}_depth"] = asyncio.create_task(
                    self._handle_stream(ds, "depth", pair)
                )

                # Stream de klines
                for tf in self.timeframes:
                    ks = self.bot.socket_manager.kline_socket(pair, tf)
                    self.streams[f"{pair}_kline_{tf}"] = asyncio.create_task(
                        self._handle_stream(ks, "kline", pair, tf)
                    )

            return True

        except Exception as e:
            logger.error(f"Stream setup error: {e}")
            return False

    async def _handle_stream(self, socket, stream_type, pair, timeframe=None):
        """Gère un stream WebSocket"""
        while self.running:
            try:
                async with socket as sock:
                    msg = await sock.recv()
                    if msg:
                        # Traitement selon le type
                        if stream_type == "trade":
                            await self.bot._handle_trade(msg)
                        elif stream_type == "depth":
                            await self.bot._handle_orderbook(msg)
                        elif stream_type == "kline":
                            await self.bot._handle_kline(msg)

            except Exception as e:
                if "shutdown" not in str(e).lower() and "closed" not in str(e).lower():
                    logger.error(f"Stream error ({stream_type}-{pair}): {e}")
                    if self.running:
                        await asyncio.sleep(self.retry_delay)
                        continue
                return


async def initialize_websocket(bot):
    """
    Initialise la connexion WebSocket avec gestion améliorée des erreurs et des reconnexions.
    """
    try:
        # Vérification du statut d'initialisation
        if getattr(bot, "_ws_initializing", False):
            logger.warning("⚠️ Initialisation WebSocket déjà en cours")
            return False

        bot._ws_initializing = True

        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║         INITIALISATION WEBSOCKET                ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {os.getenv('USER', 'Patmoorea')}
╚═════════════════════════════════════════════════╝
        """
        )

        # 1. Vérification des credentials
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")

        if not api_key or not api_secret:
            logger.error(
                """
╔═════════════════════════════════════════════════╗
║         ERREUR CREDENTIALS                      ║
╠═════════════════════════════════════════════════╣
║ API Key ou Secret manquants                    ║
╚═════════════════════════════════════════════════╝
            """
            )
            return False

        # 2. Nettoyage des connexions existantes si nécessaire
        if hasattr(bot, "binance_ws") and bot.binance_ws:
            try:
                await bot.binance_ws.close_connection()
                bot.binance_ws = None
            except Exception as cleanup_error:
                logger.warning(
                    f"⚠️ Erreur nettoyage connexion existante: {cleanup_error}"
                )

        # 3. Création du client avec timeout et retry
        try:
            bot.binance_ws = await AsyncClient.create(
                api_key=api_key, api_secret=api_secret, tld="com"
            )
            logger.info("✅ Client Binance initialisé")
        except Exception as client_error:
            logger.error(f"❌ Erreur création client: {client_error}")
            return False

        # 4. Configuration du socket manager avec paramètres optimisés
        try:
            bot.socket_manager = BinanceSocketManager(
                bot.binance_ws,
            )
            logger.info("✅ Socket Manager configuré")
        except Exception as manager_error:
            logger.error(f"❌ Erreur configuration socket manager: {manager_error}")
            return False

        # 5. Configuration des streams avec gestion d'erreur
        try:
            # Définition des streams
            streams = [
                "btcusdt@trade",  # Stream de trades
                "btcusdt@depth",  # Stream d'orderbook
                "btcusdt@kline_1m",  # Stream de klines 1m
            ]

            # Réinitialisation des tâches
            bot.ws_tasks = []

            # Création du socket multiplexé avec retry
            multiplex_socket = bot.socket_manager.multiplex_socket(streams)

            # Création de la tâche principale avec gestion d'erreur
            main_task = asyncio.create_task(
                handle_socket_message(bot, multiplex_socket, "market_data")
            )
            main_task.set_name("main_market_data_stream")
            bot.ws_tasks.append(main_task)

            # Ajout d'un heartbeat pour maintenir la connexion
            heartbeat_task = asyncio.create_task(websocket_heartbeat(bot))
            heartbeat_task.set_name("websocket_heartbeat")
            bot.ws_tasks.append(heartbeat_task)

            logger.info("✅ Streams configurés")

        except Exception as stream_error:
            logger.error(f"❌ Erreur configuration streams: {stream_error}")
            return False

        # 6. Mise à jour du statut de connexion
        bot.ws_connection = {
            "enabled": True,
            "status": "connected",
            "tasks": bot.ws_tasks,
            "last_heartbeat": datetime.now(timezone.utc),
            "reconnect_count": 0,
            "max_reconnects": 3,
            "start_time": datetime.now(timezone.utc),
        }

        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║         WEBSOCKET INITIALISÉ                    ║
╠═════════════════════════════════════════════════╣
║ Status: Connected
║ Streams: {len(streams)}
║ Tasks: {len(bot.ws_tasks)}
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
╚═════════════════════════════════════════════════╝
        """
        )

        return True

    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║         ERREUR INITIALISATION                   ║
╠═════════════════════════════════════════════════╣
║ Error: {str(e)}
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
╚═════════════════════════════════════════════════╝
        """
        )

        # Nettoyage en cas d'erreur
        try:
            if hasattr(bot, "binance_ws") and bot.binance_ws:
                await bot.binance_ws.close_connection()
            if hasattr(bot, "socket_manager"):
                bot.socket_manager = None
        except:
            pass

        return False

    finally:
        bot._ws_initializing = False
        # Vérification finale de la connexion
        if not bot.ws_connection.get("enabled", False):
            logger.warning("⚠️ WebSocket non initialisé correctement")


async def setup_websocket_streams(bot):
    """Configure les streams WebSocket"""
    try:
        tasks = []

        # Configuration des streams par paire
        for pair in bot.config["TRADING"]["pairs"]:
            # Stream de trades en temps réel
            trade_socket = bot.socket_manager.trade_socket(pair)
            tasks.append(
                asyncio.create_task(handle_socket_message(bot, trade_socket, "trade"))
            )

            # Stream d'orderbook
            depth_socket = bot.socket_manager.depth_socket(pair)
            tasks.append(
                asyncio.create_task(handle_socket_message(bot, depth_socket, "depth"))
            )

            # Stream de klines pour chaque timeframe
            for timeframe in bot.config["TRADING"]["timeframes"]:
                kline_socket = bot.socket_manager.kline_socket(pair, timeframe)
                tasks.append(
                    asyncio.create_task(
                        handle_socket_message(bot, kline_socket, "kline")
                    )
                )

        # Mise à jour du statut de connexion
        bot.ws_connection.update(
            {
                "enabled": True,
                "status": "connected",
                "tasks": tasks,
                "start_time": time.time(),
            }
        )

        # Attendre que tous les streams soient initialisés
        await asyncio.gather(*[asyncio.shield(task) for task in tasks])

        return True

    except Exception as e:
        logger.error(f"❌ Stream setup error: {e}")
        return False


async def websocket_heartbeat(bot):
    """Maintient la connexion WebSocket active"""
    while True:
        try:
            if not bot.ws_connection["enabled"]:
                break

            # Update heartbeat timestamp
            bot.ws_connection["last_heartbeat"] = datetime.now(timezone.utc)

            await asyncio.sleep(30)  # Heartbeat toutes les 30 secondes

        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            await asyncio.sleep(5)


async def handle_socket_message(bot, socket, stream_name):
    """Gestion des messages avec meilleure gestion des erreurs"""
    async with socket as tscm:
        while True:
            try:
                msg = await asyncio.wait_for(
                    tscm.recv(), timeout=60  # Timeout plus long pour la réception
                )

                if msg:
                    # Mise à jour des données
                    if "data" not in bot.latest_data:
                        bot.latest_data["data"] = {}

                    bot.latest_data["data"][stream_name] = msg

                    # Mise à jour du timestamp
                    bot.ws_connection["last_message"] = datetime.now(timezone.utc)

            except asyncio.TimeoutError:
                # Au lieu de se déconnecter, on continue
                continue

            except Exception as e:
                logger.error(f"Socket error ({stream_name}): {e}")
                await asyncio.sleep(1)
                continue


async def cleanup_websocket(bot):
    """Clean WebSocket resources"""
    try:
        logger.info("🔄 Closing WebSocket...")

        if hasattr(bot, "ws_tasks"):
            for task in bot.ws_tasks:
                task.cancel()
            bot.ws_tasks = []

        if hasattr(bot, "socket_manager"):
            await bot.socket_manager.close()

        if hasattr(bot, "binance_ws"):
            await bot.binance_ws.close_connection()

        bot.ws_connection = {"enabled": False, "status": "disconnected", "tasks": []}

        logger.info("✅ WebSocket closed successfully")

    except Exception as e:
        logger.error(f"❌ WebSocket cleanup error: {e}")
