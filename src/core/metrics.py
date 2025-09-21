from __future__ import annotations
from typing import Any, Dict
import time

from core.url_kind import parse_url, ParsedURL
from core.hf_api import fetch_hf_model_meta

def _elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)

def _size_score_from_files(files: list[str]) -> dict[str, float]:
    est_gb = 0.4 * sum(1 for f in files if f.endswith((".bin", ".safetensors", ".onnx", ".tflite")))
    if est_gb == 0:
        est_gb = 0.2
    def s(gb: float, cap: float) -> float:
        return max(0.0, min(1.0, 1.0 - (gb / cap)))
    return {
        "edge_8gb": s(est_gb, 8.0),
        "workstation_24gb": s(est_gb, 24.0),
        "server_80gb": s(est_gb, 80.0),
    }

_COMPAT = {"lgpl-2.1", "lgpl-2.1-only", "lgpl-2.1-or-later", "mit", "bsd-2-clause", "bsd-3-clause", "apache-2.0", "isc"}
_INCOMPAT_HINTS = {"gpl-3", "cc-by-nc", "noncommercial", "proprietary"}

def _norm(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalnum() or c in ".-+")

def _license_score(card_data: dict[str, Any] | None) -> tuple[float, str]:
    if not card_data:
        return 0.2, "undetected"
    lic_raw = str(card_data.get("license", "")).strip()
    if not lic_raw:
        return 0.2, "undetected"
    lic = _norm(lic_raw)
    if lic in _COMPAT:
        return 1.0, lic
    if any(h in lic for h in _INCOMPAT_HINTS):
        return 0.0, lic
    return 0.2, lic

def _availability(files: list[str], card_data: dict[str, Any]) -> tuple[float, dict[str, bool]]:
    have = {
        "readme": bool(card_data),
        "license": bool(card_data.get("license")) if card_data else False,
        "config": any(f.endswith(("config.json","config.yaml","config.yml")) for f in files),
        "tokenizer": any(("tokenizer" in f) and f.endswith((".json",".model")) for f in files),
        "weights": any(f.endswith((".bin",".safetensors",".onnx",".tflite")) for f in files),
    }
    return sum(have.values()) / len(have), have

def compute_metrics(u: str) -> Dict[str, Any]:
    p: ParsedURL = parse_url(u)
    if p.kind != "hf_model":
        return {}

    overall_t0 = time.perf_counter()
    meta, fetch_ms = fetch_hf_model_meta(p)

    # size_score
    t0 = time.perf_counter()
    size_profiles = _size_score_from_files(meta.get("files", []))
    size_ms = _elapsed_ms(t0) + fetch_ms  # include fetch time

    # license_score
    t0 = time.perf_counter()
    lic_s, lic_note = _license_score(meta.get("card_data"))
    lic_ms = _elapsed_ms(t0)

    # availability_score
    t0 = time.perf_counter()
    avail_s, avail_detail = _availability(meta.get("files", []), meta.get("card_data", {}))
    avail_ms = _elapsed_ms(t0)

    return {
        "url": u,
        "name": f"{p.owner}/{p.name}",
        "category": "model",
        "size_score": size_profiles,
        "size_latency_ms": size_ms,
        "license_score": lic_s,
        "license_latency_ms": lic_ms,
        "license_note": lic_note,
        "availability_score": avail_s,
        "availability_latency_ms": avail_ms,
        "availability_detail": avail_detail,
        "overall_latency_ms": _elapsed_ms(overall_t0),
    }
