from __future__ import annotations
import re, time
from typing import Dict, Any
from .base import MetricResult

HF_DATASET_URL = re.compile(r"https?://huggingface\.co/datasets/[A-Za-z0-9_\-./]+", re.I)
FENCED_PY = re.compile(r"```(?:python|py)\s+[\s\S]+?```", re.I)

class DatasetCodePresenceMetric:
    name = "dataset_and_code_score"

    def compute(self, ctx: Dict[str, Any]) -> MetricResult:
        t0 = time.perf_counter()
        provided_ds = list(ctx.get("datasets") or [])
        provided_code = list(ctx.get("code") or [])
        card = ctx.get("card_data") or {}
        files = [str(f) for f in (ctx.get("files") or [])]
        readme = ctx.get("readme_text") or ""

        # DATASET (prefer explicit URLs, fallback to card/README links)
        dataset = bool(provided_ds) or bool(card.get("datasets")) or bool(HF_DATASET_URL.search(readme))

        # CODE (prefer explicit repo URL, fallback to .py/.ipynb or fenced code in README)
        code_files = any(f.endswith((".py",".ipynb")) or f.lower().startswith("examples/") for f in files)
        code = bool(provided_code) or code_files or bool(FENCED_PY.search(readme))

        score = 0.5*float(dataset) + 0.5*float(code)
        detail = {
            "dataset": {"provided": bool(provided_ds), "card_datasets": bool(card.get("datasets")), "readme_hf_link": bool(HF_DATASET_URL.search(readme))},
            "code": {"provided": bool(provided_code), "repo_code_files": code_files, "readme_fenced_py": bool(FENCED_PY.search(readme))}
        }
        return MetricResult(score=score, latency_ms=int((time.perf_counter()-t0)*1000), extras={"dataset_and_code_detail": detail})
