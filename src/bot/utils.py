import logging
from datetime import datetime, timezone
import os


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default
