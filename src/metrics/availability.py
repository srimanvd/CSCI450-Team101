from __future__ import annotations
import re, time
from typing import Dict, Any
from .base import MetricResult

FENCED_CODE = re.compile(r"```(?:python|py)\s+[\s\S]+?```", re.I)
MODEL_INDEX_NAMES = {"model_index.json","model-index.json"}
CONFIG_ENDINGS = ("config.json","config.yaml","config.yml")
TOKENIZER_HINT = ("tokenizer",)
WEIGHT_EXTS = (".bin",".safetensors",".onnx",".tflite",".pt",".ckpt")

class AvailabilityMetric:
    name = "ramp_up_time"

    def compute(self, ctx: Dict[str, Any]) -> MetricResult:
        t0 = time.perf_counter()
        files = [str(f) for f in (ctx.get("files") or [])]
        readme = ctx.get("readme_text") or ""
        card = ctx.get("card_data") or {}

        have_readme = bool(readme or card)
        have_license = bool(card.get("license") or ctx.get("hf_license"))
        have_config = any(f.endswith(CONFIG_ENDINGS) for f in files)
        have_tokenizer = any(("tokenizer" in f.lower()) and f.lower().endswith((".json",".model")) for f in files)
        have_weights = any(f.lower().endswith(WEIGHT_EXTS) for f in files)
        have_model_index = any(n in {f.lower() for f in files} for n in MODEL_INDEX_NAMES)
        have_quickstart = bool(FENCED_CODE.search(readme))

        # Weighted rubric
        score = (
            0.20*float(have_readme) +
            0.10*float(have_license) +
            0.15*float(have_config) +
            0.10*float(have_tokenizer) +
            0.10*float(have_weights) +
            0.20*float(have_quickstart) +
            0.15*float(have_model_index)
        )
        score = min(1.0, score)

        detail = {
            "readme": have_readme, "license": have_license, "config": have_config,
            "tokenizer": have_tokenizer, "weights": have_weights,
            "quickstart": have_quickstart, "model_index": have_model_index
        }
        return MetricResult(score=score, latency_ms=int((time.perf_counter()-t0)*1000), extras=detail)
