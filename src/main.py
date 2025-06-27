import streamlit as st
import json
import os

st.set_page_config(
    page_title="Trading Bot Ultimate v4 - Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATUS_FILE = "bot_status.json"


def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            try:
                return json.load(f)
            except Exception as e:
                st.error(f"Erreur de lecture du status : {e}")
                return {}
    return {}


st.title("Trading Bot Ultimate v4 - Dashboard")

status = load_status()

if not status:
    st.warning(
        "Aucun status du bot trouvé. Le bot tourne-t-il ? (python bot_runner.py)"
    )
elif "error" in status:
    st.error(f"[BOT ERROR] {status['error']}")
else:
    st.success(f"Cycle : {status.get('cycle', '?')}")
    st.markdown(f"**Régime détecté :** {status.get('regime', '?')}")
    st.markdown(f"**Stratégie actuelle :** {status.get('strategy', '?')}")
    st.markdown(f"**Date/Heure :** {status.get('datetime', '?')}")
    st.markdown("**Signaux :**")
    st.json(status.get("signals", {}))

st.divider()
st.info(
    "Ce dashboard ne pilote pas le bot : il affiche uniquement le status en temps réel généré par le process autonome.\n\n"
    "Pour démarrer le bot : `python bot_runner.py`\n"
    "Pour surveiller, rafraîchez cette page."
)

if st.button("🔄 Rafraîchir le status"):
    st.rerun()
