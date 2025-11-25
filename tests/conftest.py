import os
import sys
from pathlib import Path

os.environ.setdefault("TEST_MODE", "true")
os.environ.setdefault("USE_MEMORY_STORE", "true")
os.environ.setdefault("ALLOW_REDIS_FALLBACK_DEV", "true")


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
