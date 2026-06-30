import sys
from pathlib import Path

# Permite `import ga_core, ga_metrics` desde los tests sin instalar el paquete.
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
