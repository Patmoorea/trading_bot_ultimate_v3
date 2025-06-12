import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from ai.models.hybrid_engine import TechnicalCNN_LSTM
    print("Import OK")
except Exception as e:
    print("Import failed:", e)
