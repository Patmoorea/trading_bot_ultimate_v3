#!/bin/bash
# Bot Control Script
# Updated: 2025-05-27 16:32:27 UTC
# Author: Patmoorea

# Couleurs pour les logs
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Vérification de l'environnement virtuel
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}Error: Virtual environment not activated${NC}"
    echo "Please run: source .venv/bin/activate"
    exit 1
fi

case "$1" in
    "start")
        echo -e "${GREEN}Starting Trading Bot...${NC}"
        python start_bot.py &
        echo $! > bot.pid
        ;;
    "stop")
        if [ -f bot.pid ]; then
            echo -e "${YELLOW}Stopping Trading Bot...${NC}"
            kill $(cat bot.pid)
            rm bot.pid
        else
            echo -e "${RED}Bot is not running${NC}"
        fi
        ;;
    "restart")
        $0 stop
        sleep 2
        $0 start
        ;;
    "status")
        if [ -f bot.pid ] && ps -p $(cat bot.pid) > /dev/null; then
            echo -e "${GREEN}Bot is running${NC}"
        else
            echo -e "${RED}Bot is not running${NC}"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac

exit 0
