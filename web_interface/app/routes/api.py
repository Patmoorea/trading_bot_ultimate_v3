from flask import Blueprint, jsonify
from datetime import datetime, timezone

api_bp = Blueprint('api', __name__)

@api_bp.route('/status')
def status():
    return jsonify({
        'status': 'active',
        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'user': 'Patmoorea'
    })
