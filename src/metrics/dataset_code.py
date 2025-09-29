from __future__ import annotations

import time
from typing import Any, Dict

from .base import MetricResult


class DatasetCodePresenceMetric:
    name = "dataset_and_code_score"

    def compute(self, ctx: Dict[str, Any]) -> MetricResult:
        t0 = time.perf_counter()
        provided_ds = list(ctx.get("datasets") or [])
        provided_code = list(ctx.get("code") or [])

        dataset = bool(provided_ds)
        code = bool(provided_code)

        score = 0.5 * float(dataset) + 0.5 * float(code)
        detail = {"dataset": {"provided": dataset}, "code": {"provided": code}}
        return MetricResult(
            score=score, latency_ms=int((time.perf_counter() - t0) * 1000), extras={"dataset_and_code_detail": detail}
        )
