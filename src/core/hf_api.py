from __future__ import annotations
from typing import Any, Dict, Tuple
import time
from huggingface_hub import HfApi
from core.url_kind import ParsedURL

_api = HfApi()

def _timer() -> tuple[float, callable]:
    start = time.perf_counter()
    return start, lambda: int((time.perf_counter() - start) * 1000)

def fetch_hf_model_meta(p: ParsedURL) -> tuple[Dict[str, Any], int]:
    start, end = _timer()
    info = _api.model_info(f"{p.owner}/{p.name}", files_metadata=True)
    latency_ms = end()

    files = [sib.rfilename for sib in (info.siblings or [])]
    card_data = info.cardData or {}
    data = {
        "files": files,
        "card_data": card_data,
        "last_modified": str(getattr(info, "lastModified", "")),
        "downloads": getattr(info, "downloads", None),
        "likes": getattr(info, "likes", None),
    }
    return data, latency_ms
