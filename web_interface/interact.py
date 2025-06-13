import asyncio
import argparse
from app.services.orchestrator import TradingOrchestrator
from app.config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main(command: str, **kwargs):
    orchestrator = TradingOrchestrator()
    
    if command == "start":
        await orchestrator.start()
    
    elif command == "status":
        status = orchestrator.get_status()
        print("\n=== Trading Bot Status ===")
        print(f"User: {Config.CURRENT_USER}")
        print(f"Mode: {Config.TRADING_MODE}")
        print(f"Active Since: {Config.START_TIME}")
        print(f"Current State: {status['state']}")
        print("\n=== Portfolio ===")
        print(f"Total Value: {status['portfolio']['total_value_usdc']} USDC")
        print(f"Today's PnL: {status['portfolio']['daily_pnl']} USDC")
        print(f"Open Positions: {len(status['portfolio']['positions'])}")
    
    elif command == "configure":
        param = kwargs.get('param')
        value = kwargs.get('value')
        if param and value:
            await orchestrator.configure(param, value)
            print(f"Updated {param} to {value}")
    
    elif command == "analyze":
        pair = kwargs.get('pair', 'BTC/USDC')
        analysis = await orchestrator.get_analysis(pair)
        print(f"\n=== Analysis for {pair} ===")
        print(f"AI Confidence: {analysis['ai_confidence']:.2%}")
        print(f"Technical Score: {analysis['technical_score']:.2f}")
        print(f"Risk Level: {analysis['risk_level']}")
        print(f"Recommendation: {analysis['recommendation']}")
    
    elif command == "stop":
        await orchestrator.stop()
        print("Trading bot stopped successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trading Bot Controller')
    parser.add_argument('command', choices=['start', 'stop', 'status', 'configure', 'analyze'])
    parser.add_argument('--param', help='Parameter to configure')
    parser.add_argument('--value', help='Value to set')
    parser.add_argument('--pair', help='Trading pair to analyze')
    
    args = parser.parse_args()
    asyncio.run(main(args.command, param=args.param, value=args.value, pair=args.pair))
