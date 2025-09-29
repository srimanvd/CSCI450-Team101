from __future__ import annotations

import json
import sys
from typing import Any, Dict, Iterable, TextIO


def _coerce_ms(v: Any) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return 0


def _round_floats(obj: Any) -> Any:
    """
    Recursively round all float values to 2 decimal places.
    Preserves ints and non-numeric types. Works on nested dicts/lists.
    """
    if isinstance(obj, float):
        return round(obj, 2)
    if isinstance(obj, list):
        return [_round_floats(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    return obj


def write_rows(rows: Iterable[Dict[str, Any]], out: TextIO = sys.stdout) -> None:
    for r in rows:
        # Ensure latency fields are ints
        for k in list(r.keys()):
            if k.endswith("_latency"):
                r[k] = _coerce_ms(r[k])

        # Round all floats (including nested structures) to 2 decimals
        r = _round_floats(r)

        out.write(json.dumps(r, ensure_ascii=False) + "\n")
        out.flush()
