from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple

from .base import MetricResult

# Extensions that indicate model weight files
WEIGHT_EXTS: Tuple[str, ...] = (
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

# Known HF index files that can contain aggregate size information
INDEX_FILES: Tuple[str, ...] = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)

# Device capacity (GiB) used to compute a deployability score
CAPS_GIB: Dict[str, float] = {
    "raspberry_pi": 1.0,
    "jetson_nano": 4.0,
    "desktop_pc": 32.0,
    "aws_server": 96.0,
}

SAFE_OCCUPANCY: float = 0.70
TAU: float = 0.15


def _logistic(util: float, tau: float = TAU) -> float:
    """Logistic mapping where util ~= 1 means at the safe-occupancy threshold."""
    return 1.0 / (1.0 + math.exp((util - 1.0) / tau))


def _is_weight_file(path: str) -> bool:
    """Return True if the path looks like a model weight artifact."""
    p: str = path.lower()
    return p.endswith(WEIGHT_EXTS)


def _sum_files_meta_bytes(files_meta: Iterable[Mapping[str, Any]]) -> Tuple[int, Dict[str, int]]:
    """
    Sum sizes for weight-like files given HF 'files_metadata'.

    Returns:
        total_bytes: Sum of sizes in bytes.
        by_ext: Mapping from extension (e.g., 'bin', 'pt') to total bytes.
    """
    total: int = 0
    by_ext: Dict[str, int] = {}
    for fm in files_meta:
        name: str = str(fm.get("rfilename", ""))
        if not _is_weight_file(name):
            continue
        size: int = int(fm.get("size", 0) or 0)
        if size <= 0:
            continue
        total += size
        ext: str = name.split(".")[-1].lower()
        by_ext[ext] = by_ext.get(ext, 0) + size
    return total, by_ext


def _parse_index_total_size(ctx: Mapping[str, Any]) -> int:
    """
    If an index JSON is present, parse it and extract metadata.total_size (bytes).

    Args:
        ctx: Metric context; expects 'files' and optionally 'repo_id' or 'name'.

    Returns:
        Total model bytes if available, else 0.
    """
    files_obj: Any = ctx.get("files")
    files: List[str] = list(files_obj) if isinstance(files_obj, Iterable) else []  # type: ignore[typeddict-item]
    index_name: str | None = next((f for f in files if f in INDEX_FILES), None)
    if not index_name:
        return 0
    try:
        from huggingface_hub import hf_hub_download

        repo_id: str = str(ctx.get("repo_id") or ctx.get("name") or "")
        path: str = hf_hub_download(repo_id=repo_id, filename=index_name, repo_type="model")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data: Dict[str, Any] = json.load(f)
        meta_obj: Any = data.get("metadata") or {}
        if isinstance(meta_obj, Mapping):
            ts = int(meta_obj.get("total_size", 0) or 0)
            return ts if ts > 0 else 0
        return 0
    except Exception:
        # Network / parsing failures are non-fatal; fall back to other methods.
        return 0


class SizeMetric:
    name: str = "size_score"

    def compute(self, ctx: Mapping[str, Any]) -> MetricResult:
        """
        Estimate deployability based on total model bytes vs several device capacity profiles.
        Higher score => easier to deploy across a range of hardware.

        Strategy:
          1) Sum sizes from files_meta for weight-like files.
          2) If absent/zero, parse index JSON (if present).
          3) If still zero, fall back to a small heuristic size (for stability offline).
        """
        t0: float = time.perf_counter()

        files_meta_obj: Any = ctx.get("files_meta") or []
        files_meta: Iterable[Mapping[str, Any]] = (
            files_meta_obj if isinstance(files_meta_obj, Iterable) else []
        )

        total_bytes, per_ext = _sum_files_meta_bytes(files_meta)

        method_used: str = "files_meta"
        if total_bytes == 0:
            idx_total: int = _parse_index_total_size(ctx)
            if idx_total > 0:
                total_bytes = idx_total
                method_used = "index_json"

        if total_bytes == 0:
            # Conservative default (200 MiB) for offline/edge cases.
            total_bytes = 200 * 1024**2
            method_used = "heuristic"

        total_gib: float = total_bytes / float(1024**3)

        profiles: Dict[str, float] = {}
        for dev, cap_gib in CAPS_GIB.items():
            util: float = (total_gib / (cap_gib * SAFE_OCCUPANCY)) if cap_gib > 0 else 10.0
            profiles[dev] = _logistic(util)

        overall: float = sum(profiles.values()) / max(1, len(profiles))

        extras: MutableMapping[str, Any] = {
            "size_score": profiles,
            "overall_size_score": overall,
            "total_bytes": total_bytes,
            "total_gib": total_gib,
            "per_ext_bytes": per_ext,
            "method": method_used,
        }

        latency_ms: int = int((time.perf_counter() - t0) * 1000)
        return MetricResult(score=overall, latency_ms=latency_ms, extras=dict(extras))
