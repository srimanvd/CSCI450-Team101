from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from typing import Any, Dict, Iterable

from .base import MetricResult

TRY_FILES = (
    "pytest.ini",
    "conftest.py",
    "tox.ini",
    ".github/workflows",
    "tests/",
    "test/",
    ".circleci/",
    "azure-pipelines.yml",
)
TYPE_CFG = ("mypy.ini", "pyproject.toml", "setup.cfg")
LINT_CFG = ("pyproject.toml", "setup.cfg", ".flake8", "ruff.toml", ".pylintrc")
DOC_HINTS = re.compile(r"```(?:python|py)\s+[\s\S]+?```", re.I)


def _has_any(root: str, names: Iterable[str]) -> bool:
    for n in names:
        p = os.path.join(root, n)
        if os.path.isdir(p) or os.path.isfile(p):
            return True
    return False


def _pyproject_has(section: str, key_sub: str, root: str) -> bool:
    pp = os.path.join(root, "pyproject.toml")
    if not os.path.isfile(pp):
        return False
    try:
        import tomllib

        with open(pp, "rb") as f:
            data = tomllib.load(f)
        sect = data.get(section, {})
        # very light check for deps keys
        blob = str(sect)
        return key_sub in blob
    except Exception:
        return False


def _safe_clone(url: str, max_seconds: int = 5) -> str | None:
    try:
        import threading

        import git

        d = tempfile.mkdtemp(prefix="repo_")
        ok: Dict[str, Any] = {"done": False, "path": d, "err": None}

        def _do() -> None:
            try:
                git.Repo.clone_from(url, d, depth=1, no_single_branch=True)
                ok["done"] = True
            except Exception as e:
                ok["err"] = e

        th = threading.Thread(target=_do, daemon=True)
        th.start()
        th.join(timeout=max_seconds)
        if not ok["done"]:
            shutil.rmtree(d, ignore_errors=True)
            return None
        return d
    except Exception:
        return None


class CodeQualityMetric:
    name = "code_quality"

    def compute(self, ctx: Dict[str, Any]) -> MetricResult:
        t0 = time.perf_counter()
        repo_url = None
        for u in ctx.get("code") or []:
            if isinstance(u, str) and "github.com" in u:
                repo_url = u
                break
        readme = ctx.get("readme_text") or ""

        base = 0.0
        extras: Dict[str, Any] = {"repo_used": repo_url, "checks": {}}

        if not repo_url:
            # fallback: score small points from README evidence
            extras["checks"]["readme_code_blocks"] = bool(DOC_HINTS.search(readme))
            base = 0.2 if extras["checks"]["readme_code_blocks"] else 0.0
            return MetricResult(score=base, latency_ms=int((time.perf_counter() - t0) * 1000), extras=extras)

        root = _safe_clone(repo_url)  # may return None on timeout
        if not root:
            extras["clone_timeout"] = True
            return MetricResult(score=0.0, latency_ms=int((time.perf_counter() - t0) * 1000), extras=extras)

        try:
            has_tests = _has_any(root, TRY_FILES)
            has_ci = _has_any(root, [".github/workflows", ".circleci/", "azure-pipelines.yml"])
            has_type = _has_any(root, TYPE_CFG) or _pyproject_has("tool.mypy", "plugins", root)
            has_lint = _has_any(root, LINT_CFG) or _pyproject_has("tool.ruff", "select", root)
            readme_blocks = bool(DOC_HINTS.search(readme))

            # Weighted rubric
            score = (
                0.25 * float(has_tests)
                + 0.25 * float(has_ci)
                + 0.15 * float(has_type)
                + 0.15 * float(has_lint)
                + 0.10 * float(readme_blocks)
                + 0.10 * float(_pyproject_has("project", "dependencies", root))
            )
            score = min(1.0, score)

            extras["checks"] = {
                "tests": has_tests,
                "ci": has_ci,
                "types": has_type,
                "lint": has_lint,
                "readme_code_blocks": readme_blocks,
                "pyproject_deps": _pyproject_has("project", "dependencies", root),
            }
            return MetricResult(score=score, latency_ms=int((time.perf_counter() - t0) * 1000), extras=extras)
        finally:
            shutil.rmtree(root, ignore_errors=True)
