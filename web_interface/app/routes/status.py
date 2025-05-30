from flask import Blueprint, jsonify
from datetime import datetime, timezone
from ..config import Config

status_bp = Blueprint('status', __name__)

@status_bp.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'config': {
            'websockets': Config.MAX_WEBSOCKETS,
            'timeframes': Config.TIMEFRAMES,
            'news_update': f"Every {Config.NEWS_UPDATE_INTERVAL} seconds"
        }
    })

@status_bp.route('/system')
def system_status():
    import psutil
    
    return jsonify({
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })
