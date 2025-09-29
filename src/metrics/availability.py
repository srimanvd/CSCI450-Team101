from __future__ import annotations

import os
import re
import time
from typing import Any, Dict

from .base import MetricResult

# Optional provider import (note: providers plural)
try:
    from providers.purdue_genai import (  # type: ignore
        PurdueGenAIError,
        score_ramp_up_with_llm,
    )
except Exception:  # pragma: no cover
    score_ramp_up_with_llm = None  # type: ignore

FENCED_CODE = re.compile(r"```[a-zA-Z0-9_-]*\s+[\s\S]+?```", re.I)
QUICKSTART_HINT = re.compile(r"\b(quick\s*start|getting\s*started|usage|example[s]?)\b", re.I)
INSTALL_HINT = re.compile(
    r"(?:^|\n)\s*(?:pip(?:3)?|conda|poetry)\s+install[^\n]*|requirements\.txt|pyproject\.toml|environment\.yml|setup\.py",
    re.I,
)
CODE_EXAMPLE_HINT = re.compile(
    r"(?:from\s+transformers\s+import|AutoModel|AutoTokenizer|pipeline\(|\.generate\(|\.forward\()",
    re.I,
)

MODEL_INDEX_NAMES = {"model_index.json", "model-index.json"}
CONFIG_ENDINGS = ("config.json", "config.yaml", "config.yml")
WEIGHT_EXTS = (".bin", ".safetensors", ".onnx", ".tflite", ".pt", ".ckpt")
TOKENIZER_EXTS = (".json", ".model")


def _has_any_env_key() -> bool:
    return bool(os.getenv("GEN_AI_STUDIO_API_KEY") or os.getenv("PURDUE_GENAISTUDIO_API_KEY"))


class AvailabilityMetric:
    name = "ramp_up_time"

    def compute(self, ctx: Dict[str, Any]) -> MetricResult:
        t0 = time.perf_counter()
        files = [str(f) for f in (ctx.get("files") or [])]
        readme = ctx.get("readme_text") or ""
        card = ctx.get("card_data") or {}

        files_lower = [f.lower() for f in files]
        basenames = {f.split("/")[-1].split("\\")[-1] for f in files_lower}

        have_readme = bool(readme or card)
        have_license = bool(card.get("license") or card.get("license_name") or ctx.get("hf_license"))
        have_config = any(f.endswith(CONFIG_ENDINGS) for f in files_lower)
        have_tokenizer = any(
            ("tokenizer" in f and f.endswith(TOKENIZER_EXTS)) or "/tokenizer" in f or "\\tokenizer" in f
            for f in files_lower
        )
        have_weights = any(f.endswith(WEIGHT_EXTS) for f in files_lower)
        have_model_index = any(name in basenames for name in MODEL_INDEX_NAMES)

        has_fenced_code = bool(FENCED_CODE.search(readme))
        has_qs_heading = bool(QUICKSTART_HINT.search(readme))
        has_install = bool(INSTALL_HINT.search(readme))
        has_code_example = bool(CODE_EXAMPLE_HINT.search(readme))

        quickstart_strong = has_install or has_code_example or has_fenced_code

        score = (
            0.05 * float(have_readme)
            + 0.05 * float(have_license)
            + 0.10 * float(have_config)
            + 0.05 * float(have_tokenizer)
            + 0.10 * float(have_weights)
            + 0.40 * float(quickstart_strong)
            + 0.10 * float(has_qs_heading)
            + 0.15 * float(have_model_index)
        )
        score = max(0.0, min(1.0, score))

        detail: Dict[str, Any] = {
            "readme": have_readme,
            "license": have_license,
            "config": have_config,
            "tokenizer": have_tokenizer,
            "weights": have_weights,
            "model_index": have_model_index,
            "has_install": has_install,
            "has_code_example": has_code_example,
            "has_fenced_code": has_fenced_code,
            "has_qs_heading": has_qs_heading,
            "quickstart_strong": quickstart_strong,
            "method": "heuristic",
        }

        final = score
        if _has_any_env_key() and (score_ramp_up_with_llm is not None) and readme.strip():
            try:
                llm_score, llm_detail = score_ramp_up_with_llm(
                    readme_text=readme,
                    meta={
                        "files": files[:40],
                        "card": {k: card.get(k) for k in ("license", "license_name", "tags", "datasets") if k in card},
                        "have_model_index": have_model_index,
                        "have_weights": have_weights,
                        "quickstart_strong": quickstart_strong,
                    },
                )
                final = max(final, float(llm_score))
                detail["method"] = "heuristic+llm"
                detail["llm"] = llm_detail
            except Exception as e:  # provider errors are non-fatal
                detail["llm_error"] = f"{e.__class__.__name__}: {str(e)[:200]}"

        latency_ms = int((time.perf_counter() - t0) * 1000)
        return MetricResult(score=final, latency_ms=latency_ms, extras=detail)
