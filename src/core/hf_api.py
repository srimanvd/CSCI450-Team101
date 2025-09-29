from __future__ import annotations

import io
import time
from typing import Any, Callable, Dict, Tuple

from huggingface_hub import HfApi, hf_hub_download

from .url import ParsedURL

_api = HfApi()


def _timer() -> Tuple[float, Callable[[], int]]:
    start = time.perf_counter()
    return start, lambda: int((time.perf_counter() - start) * 1000)


def _extract_hf_license(info: Any) -> str | None:
    def ok(v: str | None) -> str | None:
        if not v:
            return None
        vv = str(v).strip()
        return vv if vv and vv.lower() not in {"other", "unknown"} else None

    v = ok(getattr(info, "license", None))
    if v:
        return v

    for t in getattr(info, "tags", []) or []:
        if isinstance(t, str) and t.lower().startswith("license:"):
            vv = ok(t.split(":", 1)[1])
            if vv:
                return vv

    cd = getattr(info, "cardData", None)
    if cd:
        d = getattr(cd, "to_dict", None)
        data = d() if callable(d) else getattr(cd, "data", None)
        if isinstance(data, dict):
            v = ok(data.get("license_name")) or ok(data.get("license"))
            if v:
                return v
        if isinstance(cd, dict):
            v = ok(cd.get("license_name")) or ok(cd.get("license"))
            if v:
                return v
    return None


def _readme_text(repo_id: str) -> str:
    """Best-effort fetch of README.md, no crash if missing."""
    try:
        path = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="model")
        with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def fetch_hf_model_meta(p: ParsedURL) -> Tuple[Dict[str, Any], int]:
    start, end = _timer()
    info = _api.model_info(f"{p.owner}/{p.name}", files_metadata=True)
    latency_ms = end()
    siblings = list(info.siblings or [])
    files = [sib.rfilename for sib in (info.siblings or [])]
    files_meta = [{"rfilename": sib.rfilename, "size": getattr(sib, "size", 0)} for sib in siblings]
    card_data = info.cardData or {}
    data: Dict[str, Any] = {
        "files": files,
        "files_meta": files_meta,
        "card_data": card_data,
        "hf_license": _extract_hf_license(info),
        "last_modified": str(getattr(info, "lastModified", "")),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
        "readme_text": _readme_text(info.id),
        "repo_id": info.id,
    }
    return data, latency_ms
