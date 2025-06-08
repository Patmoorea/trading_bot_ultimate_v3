from flask import Blueprint
from .. import socketio

ws_bp = Blueprint('ws', __name__)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
