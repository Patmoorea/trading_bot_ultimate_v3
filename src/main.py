from visualization.dashboard import TradingDashboard
import asyncio
import logging
from datetime import datetime

async def main():
    # Configuration logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('TradingBot')
    
    try:
        # Initialize dashboard
        logger.info(f"Starting Trading Dashboard - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        dashboard = TradingDashboard(update_interval=1000)  # 1 second updates
        
        # Start market data stream in background
        asyncio.create_task(dashboard.start_data_stream())
        
        # Run dashboard
        dashboard.run(host='0.0.0.0', port=8050, debug=False)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
