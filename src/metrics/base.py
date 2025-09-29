from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass
class MetricResult:
    score: float
    latency_ms: int
    extras: Dict[str, Any] | None = None


class Metric(Protocol):
    name: str

    def compute(self, ctx: Dict[str, Any]) -> MetricResult: ...
