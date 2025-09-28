from __future__ import annotations
import json, re, time
from typing import Dict, Any
from .base import MetricResult

MODEL_INDEX_NAMES = {"model_index.json","model-index.json"}
METRIC_HINTS = ("accuracy","f1","exact match","em","bleu","rouge","mmlu","hellaswag","gsm8k","arc","big-bench","winogrande")

RESULTS_HEAD = re.compile(r"^\s{0,3}#{1,3}\s*(results?|evaluation|benchmarks?)\b", re.I | re.M)
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)

def _has_model_index(files: list[str]) -> bool:
    return any(f.lower() in MODEL_INDEX_NAMES for f in files)

def _parse_model_index(ctx: Dict[str, Any]) -> int:
    files = [str(f) for f in (ctx.get("files") or [])]
    name = next((f for f in files if f.lower() in MODEL_INDEX_NAMES), None)
    if not name:
        return 0
    try:
        from huggingface_hub import hf_hub_download
        repo_id = ctx.get("repo_id") or ctx.get("name") or ""
        path = hf_hub_download(repo_id=repo_id, filename=name, repo_type="model")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        # count distinct metrics entries
        tasks = data.get("model-index", data.get("model_index", []))
        if isinstance(tasks, dict):
            tasks = [tasks]
        claims = 0
        for t in tasks or []:
            res = (t.get("results") or [])
            claims += len(res)
        return claims
    except Exception:
        return 0

class PerformanceClaimsMetric:
    name = "performance_claims"

    def compute(self, ctx: Dict[str, Any]) -> MetricResult:
        t0 = time.perf_counter()
        files = [str(f) for f in (ctx.get("files") or [])]
        readme = ctx.get("readme_text") or ""

        model_index_claims = _parse_model_index(ctx) if _has_model_index(files) else 0

        # README: results section with a Markdown table + known metric hints
        has_results_head = bool(RESULTS_HEAD.search(readme))
        has_table = bool(TABLE_ROW.search(readme))
        has_metric_kw = any(h in readme.lower() for h in METRIC_HINTS)

        # rubric: strong if model-index present; else recognize structured README claims
        if model_index_claims > 0:
            score = min(1.0, 0.6 + 0.1*min(model_index_claims,4))  # 0.6..1.0
        else:
            score = 0.0
            if has_results_head: score += 0.3
            if has_table:        score += 0.3
            if has_metric_kw:    score += 0.4
            score = min(1.0, score)

        extras = {
            "model_index_claims": model_index_claims,
            "readme_results_heading": has_results_head,
            "readme_results_table": has_table,
            "readme_metric_keywords": has_metric_kw,
        }
        return MetricResult(score=score, latency_ms=int((time.perf_counter()-t0)*1000), extras=extras)
