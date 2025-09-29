from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, List


def run_parallel(funcs: Iterable[Callable[[], Any]], max_workers: int | None = None) -> List[Any]:
    n = max_workers or min(32, (os.cpu_count() or 4))
    results: List[Any] = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(f) for f in funcs]
        for f in as_completed(futs):
            results.append(f.result())
    return results
