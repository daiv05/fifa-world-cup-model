import sys
from pathlib import Path

# Agrega repository/ al path para que 'from src.X import ...' funcione en pytest
sys.path.insert(0, str(Path(__file__).parent))
