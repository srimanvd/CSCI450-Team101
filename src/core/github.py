from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, Iterator, List, Optional

from git import Repo
from github import Github


def analyze_github_urls(urls: List[str], max_commits: int = 200) -> Dict[str, Any]:
    t0 = time.perf_counter()
    gh = Github()
    result: Dict[str, Any] = {}
    try:
        repo_url = next((u for u in urls if "github.com" in u.lower()), None)
        if not repo_url:
            return result
        parts = [x for x in repo_url.split("/") if x][-2:]
        if len(parts) != 2:
            return result
        owner, name = parts
        r = gh.get_repo(f"{owner}/{name}")

        contribs = list(r.get_contributors()[:50])
        stars = r.stargazers_count or 0
        bus = min(1.0, (len(contribs) / 10.0) + (stars / 5000.0) * 0.2)
        result["bus_factor"] = bus
        result["bus_factor_latency"] = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            Repo.clone_from(repo_url, tmp, depth=1)
            has_tests = any("test" in f.lower() for f in _walk(tmp, (".py", ".ipynb")))
            has_ci = any((".github/workflows/" in f.replace("\\", "/")) for f in _walk(tmp))
            has_type = any(f.endswith(".pyi") for f in _walk(tmp))
            code_quality = min(1.0, 0.5 * int(has_tests) + 0.3 * int(has_ci) + 0.2 * int(has_type))
            has_eval = any(
                ("eval" in os.path.basename(f).lower() or "benchmark" in os.path.basename(f).lower())
                for f in _walk(tmp, (".py", ".ipynb", ".md"))
            )
            perf_claims = 1.0 if has_eval else 0.0

        result["code_quality"] = code_quality
        result["code_quality_latency"] = int((time.perf_counter() - t1) * 1000)
        result["performance_claims"] = perf_claims
        result["performance_claims_latency"] = 0
        lic = getattr(r.get_license().license, "spdx_id", None) if hasattr(r, "get_license") else None
        if lic:
            result["license"] = (lic or "").lower()
            result["license_latency"] = 0
        return result
    except Exception:
        return result


def _walk(root: str, exts: Optional[tuple[str, ...]] = None) -> Iterator[str]:
    for d, _, files in os.walk(root):
        for f in files:
            if not exts or f.lower().endswith(exts):
                yield os.path.join(d, f)
