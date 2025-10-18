import json
from pathlib import Path
from typing import Any, Dict

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.json"

# 只读一次，以后到处用
config: Dict[str, Any] = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))