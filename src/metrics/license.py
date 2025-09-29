from __future__ import annotations

import re
import time
from typing import Any, Dict

from .base import MetricResult

# SPDX-ish normalization
_COMPAT_1_0 = {
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "apache-2.0",
    "isc",
    "lgpl-2.1",
    "lgpl-2.1-only",
    "lgpl-2.1-or-later",
    "mpl-2.0",
    "cddl-1.0",
    "cddl-1.1",
}
_INCOMPAT_0_0 = {"gpl-3.0", "gpl-3", "agpl-3.0", "agpl-3", "cc-by-nc", "cc-by-nc-4.0", "proprietary", "noncommercial"}
_ALIASES = {
    "apache2": "apache-2.0",
    "bsd3": "bsd-3-clause",
    "bsd2": "bsd-2-clause",
    "gpl3": "gpl-3.0",
    "lgpl2.1": "lgpl-2.1",
    "lgpl-2.1+": "lgpl-2.1-or-later",
    "cc-by-nc-4": "cc-by-nc-4.0",
}

LICENSE_SEC = re.compile(r"^\s{0,3}#{1,3}\s*license\b.*?$([\s\S]*?)(^\s{0,3}#{1,3}\s|\Z)", re.I | re.M)


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("license:", "").strip()
    s = s.replace("licence", "license")
    s = re.sub(r"[^a-z0-9.+-]", "", s)
    return _ALIASES.get(s, s)


def _score(token: str) -> float:
    if token in _INCOMPAT_0_0:
        return 0.0
    if token in _COMPAT_1_0:
        return 1.0
    if token in {"mpl-2.0", "cddl-1.0", "cddl-1.1"}:
        return 0.7
    if token in {"other", "unknown", ""}:
        return 0.2
    return 0.2


class LicenseMetric:
    name = "license"

    def compute(self, ctx: Dict[str, Any]) -> MetricResult:
        t0 = time.perf_counter()
        card = ctx.get("card_data") or {}
        readme = ctx.get("readme_text") or ""
        hf_lic = _norm(str(ctx.get("hf_license") or card.get("license_name") or card.get("license") or ""))

        # try README section if HF is unknown/other
        readme_lic = ""
        m = LICENSE_SEC.search(readme)
        if m:
            blob = m.group(1) or ""
            # pick shortest token that looks like a license
            candidates = re.findall(
                r"(apache[-\s]?2\.0|mit|bsd[\s-]?(?:2|3)|lgpl[-\s]?2\.1(?:-or-later)?|gpl[-\s]?3|agpl[-\s]?3|"
                r"mpl[-\s]?2\.0|cddl[-\s]?1\.?1?|cc[-\s]?by[-\s]?nc)",
                blob,
                re.I,
            )
            if candidates:
                readme_lic = _norm(min((c.strip() for c in candidates), key=len))

        sources, token = [], ""
        if hf_lic and hf_lic not in {"other", "unknown"}:
            token = hf_lic
            sources.append("hf")
        if (not token) and readme_lic:
            token = readme_lic
            sources.append("readme")

        s = _score(token or (hf_lic or "other"))
        note = token or (hf_lic or "undetected")
        conf = 1.0 if "hf" in sources else (0.7 if "readme" in sources else 0.3)

        return MetricResult(
            score=s,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            extras={"license_note": note, "license_sources": sources, "license_confidence": conf},
        )
