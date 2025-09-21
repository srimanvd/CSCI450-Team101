from __future__ import annotations
import json, sys
from typing import Iterable, Dict, Any

def emit(records: Iterable[Dict[str, Any]]) -> None:
    for r in records:
        sys.stdout.write(json.dumps(r, ensure_ascii=False) + "\n")
