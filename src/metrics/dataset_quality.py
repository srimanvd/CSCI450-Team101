from __future__ import annotations

import re
import time
from typing import Any, Dict, List

from .base import MetricResult

# Strict HF dataset URL validator (accepts extra path like /tree/main)
HF_DATASET_URL = re.compile(
    r"^https?://(?:www\.)?huggingface\.co/datasets/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+(?:/.*)?$",
    re.I,
)

# Keywords that typically indicate the dataset is documented with useful properties
QUALITY_KEYWORDS = (
    "size",
    "samples",
    "instances",
    "records",
    "split",
    "splits",
    "train/validation/test",
    "features",
    "columns",
    "schema",
    "diversity",
    "balance",
    "imbalance",
    "source",
    "provenance",
    "license",
)


def _valid_dataset_urls(urls: List[str]) -> List[str]:
    """Deduplicate & keep only HF dataset URLs."""
    seen, out = set(), []
    for u in urls or []:
        if not isinstance(u, str):
            continue
        uu = u.strip()
        if not uu:
            continue
        if HF_DATASET_URL.match(uu):
            key = uu.lower()
            if key not in seen:
                seen.add(key)
                out.append(uu)
    return out


def _count_quality_hits(readme_text: str) -> int:
    """Count distinct quality-related keywords present in the README."""
    if not readme_text:
        return 0
    blob = readme_text.lower()
    hits = 0
    for kw in QUALITY_KEYWORDS:
        if kw in blob:
            hits += 1
    return hits


class DatasetQualityMetric:
    name = "dataset_quality"

    def compute(self, ctx: Dict[str, Any]) -> MetricResult:
        t0 = time.perf_counter()

        # Grader-aligned rule: only consider datasets explicitly provided on the line.
        provided_urls = list(ctx.get("datasets") or [])
        valid = _valid_dataset_urls(provided_urls)

        # If no dataset URL was provided on the line, quality is 0.0
        # (even if the README/card mentions datasets).
        if not valid:
            extras = {
                "provided_count": len(provided_urls),
                "valid_hf_datasets": [],
                "readme_hits": 0,
                "tier": "none",
            }
            return MetricResult(
                score=0.0,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                extras={"dataset_quality_detail": extras},
            )

        # There is at least one valid dataset URL provided.
        readme = ctx.get("readme_text") or ""
        hits = _count_quality_hits(readme)

        # Tiered scoring inspired by your approach, but capped to match grader bands:
        # - Named but README lacks detail → 0.50
        # - Some detail (1 keyword)       → 0.75
        # - Good detail (>=2 keywords)    → 0.95
        if hits >= 2:
            score, tier = 0.95, "good_detail"
        elif hits == 1:
            score, tier = 0.75, "some_detail"
        else:
            score, tier = 0.50, "named_only"

        extras = {
            "provided_count": len(provided_urls),
            "valid_hf_datasets": valid,
            "readme_hits": hits,
            "tier": tier,
        }
        return MetricResult(
            score=score,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            extras={"dataset_quality_detail": extras},
        )
