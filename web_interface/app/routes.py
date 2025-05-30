from flask import Blueprint, render_template
from .config import Config

main_bp = Blueprint('main_bp', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html', config=Config)

@main_bp.route('/status')
def status():
    return {'status': 'running'}
