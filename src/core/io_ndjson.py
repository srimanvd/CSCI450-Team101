from __future__ import annotations
from typing import Any, Dict, Iterable, TextIO
import json, sys

def _coerce_ms(v: Any) -> int:
    try: return int(round(float(v)))
    except Exception: return 0

def write_rows(rows: Iterable[Dict[str, Any]], out: TextIO = sys.stdout) -> None:
    for r in rows:
        for k in list(r.keys()):
            if k.endswith("_latency"):
                r[k] = _coerce_ms(r[k])
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
        out.flush()
