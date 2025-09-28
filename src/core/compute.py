from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Tuple
import time, math
from datetime import datetime, timezone
from .url import parse_url, ParsedURL
from .hf_api import fetch_hf_model_meta
from .github import analyze_github_urls
from metrics import metric_registry, MetricResult


NET_WEIGHTS = {
    "size_score": 0.15,
    "license": 0.20,
    "ramp_up_time": 0.15,
    "bus_factor": 0.10,
    "dataset_and_code_score": 0.15,
    "dataset_quality": 0.10,
    "code_quality": 0.10,
    "performance_claims": 0.05,
}

def compute_one(u: str, datasets: List[str] | None, code: List[str] | None) -> Dict[str, Any]:
    p: ParsedURL = parse_url(u)
    if p.kind != "hf_model":
        return {}

    meta, fetch_ms = fetch_hf_model_meta(p)
    ctx: Dict[str, Any] = dict(meta)
    ctx["datasets"] = datasets or []
    ctx["code"] = code or []

    lm = meta.get("last_modified") or ""
    try:
        dt = datetime.fromisoformat(lm.replace("Z","+00:00"))
    except Exception:
        dt = None
    if dt:
        days = max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds()/86400.0)
        ctx["recency_score"] = math.exp(-days / 180.0)
    else:
        ctx["recency_score"] = 0.3

    gh = analyze_github_urls(ctx["code"], max_commits=200)

    results: Dict[str, MetricResult] = {}
    extras: Dict[str, Any] = {}
    for m in metric_registry():
        r = m.compute(ctx)
        results[m.name] = r
        if r.extras: extras.update(r.extras)

    if gh:
        if "code_quality" in gh:
            q = results["code_quality"].score
            results["code_quality"] = MetricResult(
                score=max(q, float(gh["code_quality"])),
                latency_ms=results["code_quality"].latency_ms + int(gh.get("code_quality_latency", 0))
            )
        if "performance_claims" in gh:
            q = results["performance_claims"].score
            results["performance_claims"] = MetricResult(score=max(q,float(gh["performance_claims"])),
                                                         latency_ms=results["performance_claims"].latency_ms + int(gh.get("performance_claims_latency",0)))
        if "license" in gh and results["license"].score < 1.0:
            extras["license_note"] = gh["license"]
        if "bus_factor" in gh:
            q = results["bus_factor"].score
            results["bus_factor"] = MetricResult(score=max(q,float(gh["bus_factor"])),
                                                 latency_ms=results["bus_factor"].latency_ms + int(gh.get("bus_factor_latency",0)))

    avg_size = extras.get("size_score")
    avg_size = (sum(avg_size.values())/len(avg_size)) if isinstance(avg_size, dict) else results["size_score"].score
    ramp = results["ramp_up_time"].score
    lic  = results["license"].score

    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    net = 0.0
    for k, w in NET_WEIGHTS.items():
        v = results[k].score if k in results else 0.0
        net += w * _clamp01(v)

    return {
        "url": u,
        "name": f"{p.owner}/{p.name}",
        "category": "MODEL",
        "net_score": net,
        "net_score_latency": 0,
        "ramp_up_time": results["ramp_up_time"].score,
        "ramp_up_time_latency": results["ramp_up_time"].latency_ms,
        "ramp_up_detail": extras.get("ramp_up_detail", {}),
        "bus_factor": results["bus_factor"].score,
        "bus_factor_latency": results["bus_factor"].latency_ms,
        "performance_claims": results["performance_claims"].score,
        "performance_claims_latency": results["performance_claims"].latency_ms,
        "license": results["license"].score,
        "license_latency": results["license"].latency_ms,
        "license_note": extras.get("license_note",""),
        "size_score": extras.get("size_score",{}),
        "size_score_latency": results["size_score"].latency_ms + fetch_ms,
        "dataset_and_code_score": results["dataset_and_code_score"].score,
        "dataset_and_code_score_latency": results["dataset_and_code_score"].latency_ms,
        "dataset_quality": results["dataset_quality"].score,
        "dataset_quality_latency": results["dataset_quality"].latency_ms,
        "code_quality": results["code_quality"].score,
        "code_quality_latency": results["code_quality"].latency_ms,
    }

def collate(urls: Iterable[str]) -> Iterable[dict]:
    ds_stack: List[str] = []
    code_stack: List[str] = []
    pending: List[Tuple[str,List[str],List[str]]] = []
    for u in urls:
        kind = parse_url(u).kind
        if kind == "hf_dataset":
            ds_stack.append(u)
        elif kind == "github":
            code_stack.append(u)
        elif kind == "hf_model":
            pending.append((u, list(ds_stack), list(code_stack)))
            ds_stack.clear(); code_stack.clear()
        else:
            continue
    for (u, ds, code) in pending:
        row = compute_one(u, ds, code)
        if row: yield row
