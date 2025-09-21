from __future__ import annotations
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from core.metrics import compute_metrics
from ndjsonio.ndjson import emit

def main() -> None:
    urls: List[str] = [ln.strip() for ln in sys.stdin if ln.strip()]
    if not urls:
        return

    results: List[Dict[str, Any]] = []
    workers = min(8, max(2, len(urls)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(compute_metrics, u): u for u in urls}
        for f in as_completed(futs):
            try:
                r = f.result()
                if r:
                    results.append(r)
            except Exception as e:
                sys.stderr.write(f"Error processing {futs[f]}: {e}\n")

    emit(results)

if __name__ == "__main__":
    main()
