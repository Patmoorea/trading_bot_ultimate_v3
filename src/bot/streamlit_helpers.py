import streamlit as st
import asyncio
import nest_asyncio
import os
from datetime import datetime, timezone

from bot.core import TradingBotM4

# Ajoute les autres imports nécessaires (logger, session_manager, etc.)


@st.cache_resource(ttl=None)
def get_bot():
    """Create or get the bot instance with lifecycle protection"""
    if "bot_instance" in st.session_state and st.session_state.bot_instance is not None:
        return st.session_state.bot_instance

    try:
        session_manager.protect_session()  # Protection explicite
        logger.info("Creating new bot instance...")
        bot = TradingBotM4()
        st.session_state.bot_instance = bot
        return bot
    except Exception as e:
        logger.error(f"Bot creation error: {e}")
        return None

        logger.info(
            f"""
╔═════════════════════════════════════════════════╗
║             CREATING BOT INSTANCE                ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ User: {os.getenv('USER', 'Patmoorea')}
╚═════════════════════════════════════════════════╝
        """
        )

        # Création du bot
        bot = TradingBotM4()

        # Configuration de la boucle d'événements
        if not st.session_state.get("loop"):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                nest_asyncio.apply()
                st.session_state.loop = loop
                logger.info("✅ Event loop configured successfully")
            except Exception as loop_error:
                logger.error(
                    f"""
╔═════════════════════════════════════════════════╗
║             EVENT LOOP ERROR                     ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(loop_error)}
╚═════════════════════════════════════════════════╝
                """
                )
                raise

        # Initialisation du bot
        async def initialize_bot():
            try:
                if not await bot.start():
                    raise Exception("Bot initialization failed")
                bot._initialized = True
                logger.info("✅ Bot initialization successful")
                return bot
            except Exception as init_error:
                logger.error(
                    f"""
╔═════════════════════════════════════════════════╗
║             INITIALIZATION ERROR                 ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(init_error)}
╚═════════════════════════════════════════════════╝
                """
                )
                raise

        try:
            # Initialisation avec gestion des erreurs de boucle
            try:
                bot = st.session_state.loop.run_until_complete(initialize_bot())
            except RuntimeError as e:
                if "This event loop is already running" in str(e):
                    logger.warning(
                        "⚠️ Event loop already running, applying nest_asyncio"
                    )
                    nest_asyncio.apply()
                    bot = st.session_state.loop.run_until_complete(initialize_bot())
                else:
                    raise

            if not bot or not getattr(bot, "_initialized", False):
                raise Exception("Bot initialization incomplete")

            # Sauvegarde dans la session state
            st.session_state.bot_instance = bot

            logger.info(
                f"""
╔═════════════════════════════════════════════════╗
║             BOT INSTANCE READY                   ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Status: {bot.ws_connection.get('status', 'initializing')}
║ Trading Mode: {getattr(bot, 'trading_mode', 'production')}
║ User: {os.getenv('USER', 'Patmoorea')}
╚═════════════════════════════════════════════════╝
            """
            )

            return bot

        except Exception as run_error:
            logger.error(
                f"""
╔═════════════════════════════════════════════════╗
║             RUNTIME ERROR                        ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(run_error)}
╚═════════════════════════════════════════════════╝
            """
            )
            # Nettoyage sécurisé
            if hasattr(bot, "_cleanup"):
                try:
                    st.session_state.loop.run_until_complete(bot._cleanup())
                except:
                    pass
            raise

    except Exception as e:
        logger.error(
            f"""
╔═════════════════════════════════════════════════╗
║             BOT CREATION ERROR                   ║
╠═════════════════════════════════════════════════╣
║ Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
║ Error: {str(e)}
║ User: {os.getenv('USER', 'Patmoorea')}
╚═════════════════════════════════════════════════╝
        """
        )

        # Nettoyage de la session
        if "bot_instance" in st.session_state:
            del st.session_state.bot_instance
        if "loop" in st.session_state:
            del st.session_state.loop

        return None
