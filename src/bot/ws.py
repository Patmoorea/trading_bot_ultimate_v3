import asyncio
import logging
from datetime import datetime, timezone
import websockets
import json
from .utils import WEBSOCKET_CONFIG


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
        """await self._initialize_analyzers()"""
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
                self.logger.error(f"WebSocket start error: {e}")
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
            self.logger.error(f"Stream setup error: {e}")
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
                    self.logger.error(f"Stream error ({stream_type}-{pair}): {e}")
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
            self.logger.warning("⚠️ Initialisation WebSocket déjà en cours")
            return False

        bot._ws_initializing = True

        self.logger.info(
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
            self.logger.error(
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
                self.logger.warning(
                    f"⚠️ Erreur nettoyage connexion existante: {cleanup_error}"
                )

        # 3. Création du client avec timeout et retry
        try:
            bot.binance_ws = await AsyncClient.create(
                api_key=api_key, api_secret=api_secret, tld="com"
            )
            self.logger.info("✅ Client Binance initialisé")
        except Exception as client_error:
            self.logger.error(f"❌ Erreur création client: {client_error}")
            return False

        # 4. Configuration du socket manager avec paramètres optimisés
        try:
            bot.socket_manager = BinanceSocketManager(
                bot.binance_ws,
            )
            self.logger.info("✅ Socket Manager configuré")
        except Exception as manager_error:
            self.logger.error(f"❌ Erreur configuration socket manager: {manager_error}")
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

            self.logger.info("✅ Streams configurés")

        except Exception as stream_error:
            self.logger.error(f"❌ Erreur configuration streams: {stream_error}")
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

        self.logger.info(
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
        self.logger.error(
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
            self.logger.warning("⚠️ WebSocket non initialisé correctement")


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
        self.logger.error(f"❌ Stream setup error: {e}")
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
            self.logger.error(f"Heartbeat error: {e}")
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
                self.logger.error(f"Socket error ({stream_name}): {e}")
                await asyncio.sleep(1)
                continue


async def cleanup_websocket(bot):
    """Clean WebSocket resources"""
    try:
        self.logger.info("🔄 Closing WebSocket...")

        if hasattr(bot, "ws_tasks"):
            for task in bot.ws_tasks:
                task.cancel()
            bot.ws_tasks = []

        if hasattr(bot, "socket_manager"):
            await bot.socket_manager.close()

        if hasattr(bot, "binance_ws"):
            await bot.binance_ws.close_connection()

        bot.ws_connection = {"enabled": False, "status": "disconnected", "tasks": []}

        self.logger.info("✅ WebSocket closed successfully")

    except Exception as e:
        self.logger.error(f"❌ WebSocket cleanup error: {e}")


async def setup_streams(bot):
    """Configure and setup WebSocket streams"""
    try:
        tasks = []

        async def setup_single_stream(
            stream_type, setup_func, symbol="BTCUSDT", interval="1m"
        ):
            retry_count = 0
            while retry_count < WEBSOCKET_CONFIG["MAX_RETRIES"]:
                try:
                    self.logger.info(
                        f"Setting up {stream_type} stream (attempt {retry_count + 1}/{WEBSOCKET_CONFIG['MAX_RETRIES']})..."
                    )

                    # Création du socket avec le bon symbole
                    socket = None
                    if stream_type == "ticker":
                        socket = bot.socket_manager.trade_socket(symbol)
                    elif stream_type == "depth":
                        socket = bot.socket_manager.depth_socket(symbol)
                    elif stream_type == "kline":
                        socket = bot.socket_manager.kline_socket(symbol, interval)

                    if socket:
                        task = asyncio.create_task(
                            handle_socket_message(bot, socket, stream_type)
                        )
                        task.set_name(f"{stream_type}_stream_{symbol}")
                        return task

                    retry_count += 1
                    await asyncio.sleep(WEBSOCKET_CONFIG["RETRY_DELAY"])

                except Exception as e:
                    retry_count += 1
                    self.logger.error(f"Error setting up {stream_type} stream: {e}")
                    if retry_count < WEBSOCKET_CONFIG["MAX_RETRIES"]:
                        await asyncio.sleep(WEBSOCKET_CONFIG["RETRY_DELAY"])
                    else:
                        self.logger.error(
                            f"Failed to setup {stream_type} stream after {WEBSOCKET_CONFIG['MAX_RETRIES']} attempts"
                        )
                        return None

        # Configuration des streams avec le bon ordre
        ticker_task = await setup_single_stream(
            "ticker", bot.socket_manager.trade_socket
        )
        depth_task = await setup_single_stream("depth", bot.socket_manager.depth_socket)
        kline_task = await setup_single_stream("kline", bot.socket_manager.kline_socket)

        # Collecte des tâches réussies
        tasks = [t for t in [ticker_task, depth_task, kline_task] if t is not None]

        if len(tasks) > 0:
            self.logger.info(
                f"✅ Successfully setup {len(tasks)}/{len(WEBSOCKET_CONFIG['STREAM_TYPES'])} streams"
            )
            return tasks
        else:
            self.logger.error("❌ Failed to setup any streams")
            return None

    except Exception as e:
        self.logger.error(f"❌ Stream setup error: {e}")
        return None


async def cleanup_existing_connections(bot):
    """Nettoie les connexions WebSocket existantes"""
    try:
        # Fermeture du socket manager
        if hasattr(bot, "socket_manager") and bot.socket_manager:
            try:
                # Fermeture des connexions WebSocket individuelles
                for socket_name in dir(bot.socket_manager):
                    if socket_name.startswith("_socket_"):
                        socket = getattr(bot.socket_manager, socket_name)
                        if hasattr(socket, "close"):
                            await socket.close()

                # Fermeture du socket manager lui-même
                if hasattr(bot.socket_manager, "close_connection"):
                    await bot.socket_manager.close_connection()

            except Exception as e:
                self.logger.warning(f"⚠️ Error closing socket manager: {e}")
            finally:
                bot.socket_manager = None

        # Fermeture du client WebSocket
        if hasattr(bot, "binance_ws") and bot.binance_ws:
            try:
                await bot.binance_ws.close_connection()
            except Exception as e:
                self.logger.warning(f"⚠️ Error closing Binance client: {e}")
            finally:
                bot.binance_ws = None

        return True

    except Exception as e:
        self.logger.error(f"❌ Error during cleanup: {e}")
        return False


async def create_binance_client(bot):
    """
    Crée une nouvelle instance du client Binance

    Args:
        bot: Instance du bot de trading
    """
    try:
        # Création du client avec les credentials
        bot.binance_ws = await AsyncClient.create(
            api_key=os.getenv("BINANCE_API_KEY"),
            api_secret=os.getenv("BINANCE_API_SECRET"),
        )

        # Création du socket manager
        bot.socket_manager = BinanceSocketManager(bot.binance_ws)

        return True

    except Exception as e:
        self.logger.error(f"❌ Error creating Binance client: {e}")
        return False


async def cleanup_resources(bot):
    """
    Nettoyage sécurisé des ressources avec protection de session et logging détaillé.

    Args:
        bot: Instance du bot de trading à nettoyer

    Returns:
        bool: True si le nettoyage a réussi, False sinon
    """
    current_time = datetime.now(timezone.utc)

    # Log de début de tentative de nettoyage
    self.logger.info(
        f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP ATTEMPT STARTED                ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {os.getenv('USER', 'Patmoorea')}
║ Bot Status: {'Running' if st.session_state.get('bot_running') else 'Stopped'}
╚═════════════════════════════════════════════════╝
    """
    )

    # Vérification des conditions de protection
    protection_conditions = {
        "prevent_cleanup": st.session_state.get("prevent_cleanup", True),
        "keep_alive": st.session_state.get("keep_alive", True),
        "bot_running": st.session_state.get("bot_running", False),
        "ws_initializing": getattr(bot, "_ws_initializing", False),
        "bot_initialized": getattr(bot, "_initialized", False),
        "cleanup_in_progress": getattr(bot, "cleanup_in_progress", False),
        "force_cleanup": not st.session_state.get("force_cleanup", False),
        "cleanup_allowed": not st.session_state.get("cleanup_allowed", False),
    }

    # Si une condition de protection est active
    if any(protection_conditions.values()):
        # Log détaillé des conditions qui empêchent le nettoyage
        active_protections = [k for k, v in protection_conditions.items() if v]
       self. logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP PREVENTED                      ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Active Protections: {', '.join(active_protections)}
║ Session ID: {st.session_state.get('session_id', 'Unknown')}
╚═════════════════════════════════════════════════╝
        """
        )

        # Renforcer la protection
        session_manager.protect_session()
        return False

    try:
        # Marquer le début du nettoyage
        bot.cleanup_in_progress = True
        self.logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP STARTED                        ║
╠═════════════════════════════════════════════════╣
║ Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC
║ WebSocket Status: {bot.ws_connection.get('status', 'unknown')}
╚═════════════════════════════════════════════════╝
        """
        )

        # Fermeture du WebSocket
        await close_websocket(bot)

        # Log de succès
        self.logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP SUCCESSFUL                     ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Resources Cleaned: WebSocket, Buffer, Data
╚═════════════════════════════════════════════════╝
        """
        )
        return True

    except Exception as e:
        # Log d'erreur détaillé
        self.logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP ERROR                          ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(e)}
║ Type: {type(e).__name__}
╚═════════════════════════════════════════════════╝
        """
        )
        return False

    finally:
        # Nettoyage final et restauration de la protection
        try:
            bot.cleanup_in_progress = False
            session_manager.protect_session()

            # Log final
            self.logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║           CLEANUP FINALIZED                      ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Protection Restored: True
║ Session Status: Protected
╚═════════════════════════════════════════════════╝
            """
            )

        except Exception as final_error:
            self.logger.error(f"Final cleanup error: {final_error}")


async def check_websocket_health(bot):
    """Vérifie l'état du WebSocket et le réinitialise si nécessaire"""
    try:
        # Vérifier si les streams sont actifs
        if not bot.ws_connection.get("tasks"):
            return await reset_websocket(bot)

        # Vérifier l'état des tâches
        active_tasks = [t for t in bot.ws_connection["tasks"] if not t.done()]
        if not active_tasks:
            return await reset_websocket(bot)

        # Vérifier si on reçoit des données
        if not bot.latest_data:
            return await reset_websocket(bot)

        return True

    except Exception as e:
        self.logger.error(f"❌ WebSocket health check error: {e}")
        await reset_websocket(bot)
        return False


async def close_websocket(bot):
    """Ferme proprement la connexion WebSocket"""
    try:
        self.logger.info("🔄 Closing WebSocket...")

        # Fermeture des tâches
        if bot.ws_connection and bot.ws_connection.get("tasks"):
            for task in bot.ws_connection["tasks"]:
                try:
                    if not task.done():
                        task.cancel()
                        try:
                            await asyncio.wait_for(task, timeout=5.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass
                except:
                    pass

        # Fermeture du socket manager
        if hasattr(bot, "socket_manager") and bot.socket_manager:
            try:
                await asyncio.wait_for(bot.socket_manager.close(), timeout=5.0)
            except:
                pass
            finally:
                bot.socket_manager = None

        # Fermeture du client websocket
        if hasattr(bot, "binance_ws") and bot.binance_ws:
            try:
                await asyncio.wait_for(bot.binance_ws.close_connection(), timeout=5.0)
            except:
                pass
            finally:
                bot.binance_ws = None

        # Fermeture explicite de la session client
        if hasattr(bot, "client_session") and bot.client_session:
            if not bot.client_session.closed:
                await bot.client_session.close()
                await asyncio.sleep(0.1)  # Petit délai pour assurer la fermeture
            bot.client_session = None

        # Réinitialisation de l'état
        bot.ws_connection = {"enabled": False, "status": "disconnected", "tasks": []}

        self.logger.info("✅ WebSocket closed successfully")
        return True

    except Exception as e:
        self.logger.error(f"❌ WebSocket close error: {e}")
        return False


async def process_ws_message(bot, msg):
    """Process WebSocket messages"""
    try:
        if not msg:
            self.logger.warning("Empty message received")
            return

        if "e" not in msg:
            self.logger.warning(f"Invalid message format: {msg}")
            return

        if msg["e"] == "ticker":
            # Mise à jour du prix
            bot.latest_data["price"] = float(msg["c"])
            bot.latest_data["volume"] = float(msg["v"])
            self.logger.debug(f"💰 Price updated: {bot.latest_data['price']}")

        elif msg["e"] == "depth":
            # Mise à jour de l'orderbook
            bot.latest_data["orderbook"] = {"bids": msg["b"][:5], "asks": msg["a"][:5]}
            self.logger.debug("📚 Orderbook updated")

        elif msg["e"] == "kline":
            # Mise à jour des klines
            k = msg["k"]
            bot.latest_data["klines"] = {
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
            }
            self.logger.debug("📊 Klines updated")

        # Mise à jour du timestamp
        bot.latest_data["timestamp"] = msg.get("E", int(time.time() * 1000))
        bot.ws_connection["last_message"] = time.time()

    except Exception as e:
        self.logger.error(f"❌ Message processing error: {e}")


async def handle_ticker_message(bot, msg):
    """Gestion des messages de ticker"""
    try:
        if "s" in msg and "p" in msg:
            symbol = msg["s"]
            price = float(msg["p"])

            # Mise à jour des données
            if not hasattr(bot, "latest_prices"):
                bot.latest_prices = {}
            bot.latest_prices[symbol] = price

            # Mise à jour du timestamp
            bot.ws_connection["last_message"] = time.time()

    except Exception as e:
        self.logger.error(f"❌ Ticker message error: {e}")


async def handle_kline_message(bot, msg):
    """Gestion des messages de klines"""
    try:
        if "k" in msg:
            kline = msg["k"]
            if all(k in kline for k in ["t", "o", "h", "l", "c", "v"]):
                candle = {
                    "timestamp": kline["t"],
                    "open": float(kline["o"]),
                    "high": float(kline["h"]),
                    "low": float(kline["l"]),
                    "close": float(kline["c"]),
                    "volume": float(kline["v"]),
                }

                if not hasattr(bot, "latest_klines"):
                    bot.latest_klines = []
                bot.latest_klines.append(candle)

                if len(bot.latest_klines) > 1000:
                    bot.latest_klines.pop(0)

    except Exception as e:
        self.logger.error(f"❌ Kline message error: {e}")


async def handle_depth_message(bot, msg):
    """Gestion des messages d'orderbook"""
    try:
        if "a" in msg and "b" in msg:
            orderbook = {
                "asks": [[float(price), float(qty)] for price, qty in msg["a"]],
                "bids": [[float(price), float(qty)] for price, qty in msg["b"]],
                "timestamp": time.time(),
            }

            if not hasattr(bot, "latest_orderbook"):
                bot.latest_orderbook = {}
            bot.latest_orderbook = orderbook

    except Exception as e:
        self.logger.error(f"❌ Depth message error: {e}")


async def handle_kline_message(bot, msg):
    """Gestion des messages de klines"""
    try:
        if "k" in msg:
            kline = msg["k"]
            if all(k in kline for k in ["t", "o", "h", "l", "c", "v"]):
                candle = {
                    "timestamp": kline["t"],
                    "open": float(kline["o"]),
                    "high": float(kline["h"]),
                    "low": float(kline["l"]),
                    "close": float(kline["c"]),
                    "volume": float(kline["v"]),
                }

                if not hasattr(bot, "latest_klines"):
                    bot.latest_klines = []
                bot.latest_klines.append(candle)

                if len(bot.latest_klines) > 1000:
                    bot.latest_klines.pop(0)

    except Exception as e:
        self.logger.error(f"❌ Kline message error: {e}")


async def handle_depth_message(bot, msg):
    """Gestion des messages d'orderbook"""
    try:
        if "a" in msg and "b" in msg:
            orderbook = {
                "asks": [[float(price), float(qty)] for price, qty in msg["a"]],
                "bids": [[float(price), float(qty)] for price, qty in msg["b"]],
                "timestamp": time.time(),
            }

            if not hasattr(bot, "latest_orderbook"):
                bot.latest_orderbook = {}
            bot.latest_orderbook = orderbook

    except Exception as e:
        self.logger.error(f"❌ Depth message error: {e}")


async def cleanup_session(bot):
    """Nettoyage d'une session avec verrou et cooldown"""
    global cleanup_in_progress, last_cleanup_time

    try:
        # Vérification du cooldown
        current_time = time.time()
        if current_time - last_cleanup_time < CLEANUP_COOLDOWN:
            return

        # Utilisation d'un verrou pour éviter les nettoyages simultanés
        async with cleanup_lock:
            if cleanup_in_progress:
                return

            cleanup_in_progress = True
            last_cleanup_time = current_time

            try:
                # Nettoyage des ressources
                await cleanup_resources(bot)

                # Un seul message de log
                self.logger.info("✅ Session cleaned successfully")
                self.logger.info(
                    """
╔═════════════════════════════════════════════════╗
║              CLEANUP COMPLETED                   ║
╠═════════════════════════════════════════════════╣
║ All resources cleaned successfully              ║
╚═════════════════════════════════════════════════╝
                """
                )

            finally:
                cleanup_in_progress = False

    except Exception as e:
        self.logger.error(f"❌ Cleanup error: {e}")


async def cleanup(self):
    """Nettoie les ressources WebSocket"""
    try:
        self.running = False

        # Annulation des tâches
        for stream in self.streams.values():
            if not stream.done():
                stream.cancel()
                try:
                    await stream
                except asyncio.CancelledError:
                    pass

        self.streams.clear()

        # Fermeture du socket manager
        if hasattr(self.bot, "socket_manager") and self.bot.socket_manager:
            try:
                # Modification pour utiliser stop_socket
                for socket in self.bot.socket_manager.sockets:
                    await self.bot.socket_manager.stop_socket(socket)
                self.bot.socket_manager = None
            except Exception as e:
                self.logger.warning(f"Error closing socket manager: {e}")

        # Fermeture du client Binance
        if hasattr(self.bot, "binance_ws") and self.bot.binance_ws:
            try:
                await self.bot.binance_ws.close_connection()
            except Exception as e:
                self.logger.warning(f"Error closing Binance client: {e}")
            self.bot.binance_ws = None
    except Exception as e:
        self.logger.error(f"Cleanup error: {e}")


async def cleanup_session(bot):
    """Nettoyage d'une session avec verrou et cooldown"""
    global cleanup_in_progress, last_cleanup_time

    try:
        # Vérification du cooldown
        current_time = time.time()
        if current_time - last_cleanup_time < CLEANUP_COOLDOWN:
            return

        # Utilisation d'un verrou pour éviter les nettoyages simultanés
        async with cleanup_lock:
            if cleanup_in_progress:
                return

            cleanup_in_progress = True
            last_cleanup_time = current_time

            try:
                # Nettoyage des ressources
                await cleanup_resources(bot)

                # Un seul message de log
                self.logger.info("✅ Session cleaned successfully")
                self.logger.info(
                    """
╔═════════════════════════════════════════════════╗
║              CLEANUP COMPLETED                   ║
╠═════════════════════════════════════════════════╣
║ All resources cleaned successfully              ║
╚═════════════════════════════════════════════════╝
                """
                )

            finally:
                cleanup_in_progress = False

    except Exception as e:
        self.logger.error(f"❌ Cleanup error: {e}")


async def shutdown():
    """Arrêt propre de l'application"""
    try:
        # Récupération des tâches en cours
        tasks = [
            t
            for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]

        if tasks:
            # Annulation des tâches
            for task in tasks:
                task.cancel()

            # Attente de la fin des tâches avec timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=5.0
                )
            except asyncio.TimeoutError:
                self.logger.warning("Timeout during tasks cancellation")

        # Nettoyage via le gestionnaire de sessions
        await session_manager.cleanup()

        # Nettoyage des ressources du bot
        if "bot_instance" in st.session_state:
            bot = st.session_state.bot_instance
            await cleanup_resources(bot)

        self.logger.info(
            """
╔═════════════════════════════════════════════════╗
║              SHUTDOWN COMPLETED                  ║
╠═════════════════════════════════════════════════╣
║ All resources cleaned and sessions closed       ║
╚═════════════════════════════════════════════════╝
        """
        )

    except Exception as e:
        self.logger.error(f"Shutdown error: {e}")
