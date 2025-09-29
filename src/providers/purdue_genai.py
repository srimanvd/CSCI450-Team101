from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple, cast

import requests

GENAI_BASE_URL = os.getenv("GENAI_BASE_URL", "https://genai.rcac.purdue.edu/api/chat/completions")


class PurdueGenAIError(RuntimeError):
    pass


def _get_api_key() -> Optional[str]:
    # Support both names
    return os.getenv("GEN_AI_STUDIO_API_KEY") or os.getenv("PURDUE_GENAISTUDIO_API_KEY")


def _post_chat_completion(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    stream: bool = False,
    timeout: int = 30,
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body: Dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    resp = requests.post(GENAI_BASE_URL, headers=headers, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise PurdueGenAIError(f"GenAI HTTP {resp.status_code}: {resp.text[:500]}")
    try:
        return cast(Dict[str, Any], resp.json())
    except Exception as e:
        raise PurdueGenAIError(f"GenAI response is not valid JSON: {e}") from e


def score_ramp_up_with_llm(readme_text: str, meta: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Uses Purdue GenAI Studio to judge ramp-up qualities from README + metadata.
    Returns (score_0_to_1, raw_details).
    """
    api_key = _get_api_key()
    model = os.getenv("GENAI_MODEL", "llama3.1:latest")
    if not api_key:
        raise PurdueGenAIError("Missing GEN_AI_STUDIO_API_KEY (or PURDUE_GENAISTUDIO_API_KEY)")

    system = (
        "You are evaluating how quickly an engineer can get started with a model repository. "
        "ONLY return compact JSON with these exact keys:\n"
        "{"
        "\"has_install\": true|false,"
        "\"has_quickstart\": true|false,"
        "\"has_examples\": true|false,"
        "\"has_requirements\": true|false,"
        "\"has_license\": true|false,"
        "\"clarity_0_1\": number_between_0_and_1"
        "}\n"
        "No prose, no markdown."
    )
    user = (
        "README:\n---\n" + (readme_text[:120000]) + "\n---\n"
        "Metadata (optional):\n" + json.dumps(meta or {}, ensure_ascii=False) + "\n"
        "Evaluate based on presence and clarity of install/quickstart/examples/dependencies/license."
    )

    t0 = time.perf_counter()
    data = _post_chat_completion(
        api_key=api_key,
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        stream=False,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    content = data["choices"][0]["message"]["content"]
    try:
        obj = json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(content[start:end + 1])  # no whitespace before colon, satisfies flake8 E203
        else:
            raise PurdueGenAIError("GenAI returned non-JSON content.")

    weights = {
        "has_install": 0.20,
        "has_quickstart": 0.20,
        "has_examples": 0.20,
        "has_requirements": 0.15,
        "has_license": 0.10,
        "clarity_0_1": 0.15,
    }
    score = (
        (1.0 if obj.get("has_install") else 0.0) * weights["has_install"]
        + (1.0 if obj.get("has_quickstart") else 0.0) * weights["has_quickstart"]
        + (1.0 if obj.get("has_examples") else 0.0) * weights["has_examples"]
        + (1.0 if obj.get("has_requirements") else 0.0) * weights["has_requirements"]
        + (1.0 if obj.get("has_license") else 0.0) * weights["has_license"]
        + float(obj.get("clarity_0_1", 0.0)) * weights["clarity_0_1"]
    )
    score = max(0.0, min(1.0, score))

    details: Dict[str, Any] = {"provider": "purdue_genai", "model": model, "raw": obj, "latency_ms": latency_ms}
    return score, details
