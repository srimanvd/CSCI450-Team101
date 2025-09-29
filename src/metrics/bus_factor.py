from __future__ import annotations

import datetime as dt
import shutil
import time
from typing import Any, Dict, Iterable

from .base import MetricResult
from .code_quality import _safe_clone  # reuse

LOOKBACK_DAYS = 180


def _author_stats(root: str, since: dt.datetime) -> Dict[str, int]:
    try:
        import git

        repo = git.Repo(root)
        counts: Dict[str, int] = {}
        for c in repo.iter_commits(since=since.isoformat()):
            name = (c.author.name or "unknown").strip()
            counts[name] = counts.get(name, 0) + 1
        return counts
    except Exception:
        return {}


def _gini(values: Iterable[int]) -> float:
    v = sorted([float(x) for x in values if x > 0.0])
    if not v:
        return 1.0
    n = len(v)
    cum = 0.0
    s = sum(v)
    for i, x in enumerate(v, 1):
        cum += i * x
    g = (2 * cum) / (n * s) - (n + 1) / n
    return max(0.0, min(1.0, g))


class BusFactorMetric:
    name = "bus_factor"

    def compute(self, ctx: Dict[str, Any]) -> MetricResult:
        t0 = time.perf_counter()
        repo_url = None
        for u in ctx.get("code") or []:
            if isinstance(u, str) and "github.com" in u:
                repo_url = u
                break

        if not repo_url:
            return MetricResult(
                score=0.0,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                extras={"reason": "no_repo"},
            )

        root = _safe_clone(repo_url)
        if not root:
            return MetricResult(
                score=0.0,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                extras={"reason": "clone_timeout"},
            )

        try:
            since = dt.datetime.utcnow() - dt.timedelta(days=LOOKBACK_DAYS)
            counts = _author_stats(root, since)
            contrib = len(counts)
            commits = sum(counts.values())
            gini = _gini(counts.values()) if counts else 1.0
            s_contrib = min(1.0, contrib / 10.0)
            s_conc = 1.0 - gini
            score = 0.6 * s_contrib + 0.4 * s_conc
            return MetricResult(
                score=score,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                extras={"contributors": contrib, "commits": commits, "gini": gini},
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)
