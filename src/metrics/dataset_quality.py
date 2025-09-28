from __future__ import annotations
import re, time
from typing import Dict, Any
from .base import MetricResult

HF_DATASET_URL = re.compile(r"https?://huggingface\.co/datasets/[A-Za-z0-9_\-./]+", re.I)

class DatasetQualityMetric:
    name = "dataset_quality"

    def compute(self, ctx: Dict[str, Any]) -> MetricResult:
        t0 = time.perf_counter()
        provided = list(ctx.get("datasets") or [])
        card = ctx.get("card_data") or {}
        readme = (ctx.get("readme_text") or "")

        card_listed = bool(card.get("datasets"))
        readme_links = HF_DATASET_URL.findall(readme) if readme else []

        base = min(0.4, 0.2 * len(provided))   # explicit URLs you provided
        card_bonus = 0.35 if card_listed else 0.0
        readme_bonus = 0.25 if readme_links else 0.0

        score = min(1.0, base + card_bonus + readme_bonus)

        return MetricResult(
            score=score,
            latency_ms=int((time.perf_counter()-t0)*1000),
            extras={"dataset_quality_detail": {
                "provided_count": len(provided),
                "card_datasets": card_listed,
                "readme_hf_links_count": len(readme_links),
                "readme_example": (readme_links[0] if readme_links else None),
            }},
        )
