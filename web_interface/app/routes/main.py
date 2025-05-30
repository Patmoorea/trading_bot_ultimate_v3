from flask import Blueprint, render_template
from datetime import datetime, timezone

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('dashboard.html', 
                         last_update=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                         user="Patmoorea")
