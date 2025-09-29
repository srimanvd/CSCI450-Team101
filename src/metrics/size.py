from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, Iterable, Tuple

from .base import MetricResult

WEIGHT_EXTS = (
    ".safetensors",
    ".bin",
    ".pt",
    ".ckpt",
    ".onnx",
    ".tflite",
    ".gguf",
    ".ggml",
    ".npz",
    ".h5",
)
INDEX_FILES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)

CAPS_GIB: Dict[str, float] = {
    "raspberry_pi": 1.0,
    "jetson_nano": 4.0,
    "desktop_pc": 32.0,
    "aws_server": 96.0,
}

SAFE_OCCUPANCY = 0.70
TAU = 0.15


def _logistic(util: float, tau: float = TAU) -> float:
    return 1.0 / (1.0 + math.exp((util - 1.0) / tau))


def _is_weight_file(path: str) -> bool:
    p = path.lower()
    return p.endswith(WEIGHT_EXTS)


def _sum_files_meta_bytes(files_meta: Iterable[dict]) -> Tuple[int, Dict[str, int]]:
    total = 0
    by_ext: Dict[str, int] = {}
    for fm in files_meta:
        name = str(fm.get("rfilename", ""))
        if not _is_weight_file(name):
            continue
        size = int(fm.get("size", 0) or 0)
        if size <= 0:
            continue
        total += size
        ext = name.split(".")[-1].lower()
        by_ext[ext] = by_ext.get(ext, 0) + size
    return total, by_ext


def _parse_index_total_size(ctx: Dict[str, Any]) -> int:
    files = ctx.get("files") or []
    index_name = next((f for f in files if f in INDEX_FILES), None)
    if not index_name:
        return 0
    try:
        from huggingface_hub import hf_hub_download

        repo_id = ctx.get("repo_id") or ctx.get("name") or ""
        path = hf_hub_download(repo_id=repo_id, filename=index_name, repo_type="model")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        meta = data.get("metadata") or {}
        ts = int(meta.get("total_size", 0) or 0)
        return ts if ts > 0 else 0
    except Exception:
        return 0


class SizeMetric:
    name = "size_score"

    def compute(self, ctx: Dict[str, Any]) -> MetricResult:
        t0 = time.perf_counter()

        files_meta = ctx.get("files_meta") or []
        total_bytes, per_ext = _sum_files_meta_bytes(files_meta)

        method_used = "files_meta"
        if total_bytes == 0:
            idx_total = _parse_index_total_size(ctx)
            if idx_total > 0:
                total_bytes = idx_total
                method_used = "index_json"

        if total_bytes == 0:
            total_bytes = 200 * 1024**2
            method_used = "heuristic"

        total_gib = total_bytes / float(1024**3)

        profiles: Dict[str, float] = {}
        for dev, cap_gib in CAPS_GIB.items():
            util = (total_gib / (cap_gib * SAFE_OCCUPANCY)) if cap_gib > 0 else 10.0
            profiles[dev] = _logistic(util)

        overall = sum(profiles.values()) / max(1, len(profiles))

        return MetricResult(
            score=overall,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            extras={
                "size_score": profiles,
                "overall_size_score": overall,
                "total_bytes": total_bytes,
                "total_gib": total_gib,
                "per_ext_bytes": per_ext,
                "method": method_used,
            },
        )
