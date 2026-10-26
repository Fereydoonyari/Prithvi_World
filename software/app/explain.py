"""Hybrid Explain assistant: canned FAQ + optional OpenAI chat with UI context."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"

CANNED: dict[str, dict[str, str]] = {
    "what_is_ndvi": {
        "title": "What is NDVI?",
        "body": (
            "NDVI (Normalized Difference Vegetation Index) measures green vegetation from "
            "red (B04) and near-infrared (B8A) reflectance: (NIR − Red) / (NIR + Red). "
            "Values near −1 to 0 are bare soil / water / sparse cover; higher positive values "
            "(e.g. 0.4–0.8) indicate denser green vegetation. In Prediction, decode means "
            "use Prithvi-reconstructed bands so forecast vs real is comparable; HLS means "
            "are satellite ground truth and can sit on a slightly different absolute scale."
        ),
    },
    "decode_vs_hls": {
        "title": "Decode vs HLS RGB / NDVI",
        "body": (
            "HLS panels show the real satellite patch on disk. Decode panels run the frozen "
            "Prithvi decoder on token embeddings (zₜ, real zₜ₊Δ, or forecast ẑₜ₊Δ). "
            "Decoded RGB/NDVI are useful for comparing the dynamics model in token space, "
            "but they are not pixel-perfect copies of HLS. Prefer decode means when judging "
            "forecast quality; use HLS means for true surface greening/browning."
        ),
    },
    "change_map": {
        "title": "How to read the change map",
        "body": (
            "The Overlay change map is a 14×14 grid of Prithvi patch tokens (~480 m cells). "
            "Each cell shows how much the token embedding changed between two dates "
            "(1 − cosine similarity). Darker / hotter cells mean larger latent change "
            "(land-cover shift, phenology, residual clouds, etc.). Vegetation heatmap is "
            "different: it uses signed NDVI change on 30 m HLS pixels (loss → gain)."
        ),
    },
    "persistence": {
        "title": "What does persistence mean?",
        "body": (
            "Persistence is the naive baseline: assume tomorrow looks like today by decoding "
            "the last tokens zₜ again at the target date. If your forecast looks worse than "
            "persistence on NDVI or RGB, the dynamics model is not yet beating ‘copy last scene’. "
            "Good models should improve on persistence especially when vegetation actually changes."
        ),
    },
    "how_prediction_works": {
        "title": "How prediction works",
        "body": (
            "Pipeline: HLS scenes → frozen Prithvi encoder → token sequence → residual dynamics "
            "model (Linear / Token LSTM / S4D / ConvLSTM) trained to forecast tokens ~min Δ days "
            "ahead (often ~21) → frozen Prithvi decoder → RGB and NDVI. The Explain chat can "
            "interpret your latest decode means and NDVI error when you send a question with "
            "current context attached."
        ),
    },
    "reading_metrics": {
        "title": "Reading train metrics",
        "body": (
            "Train/val/test metrics summarize token-forecast error for each architecture and seed "
            "(lower is better for losses such as MSE/RMSE). Compare models on the same experiment "
            "and AOI. Metrics are in embedding space, not NDVI units — use Prediction NDVI err "
            "(forecast − real decode mean) for a vegetation-facing check on one sample window."
        ),
    },
    "architecture": {
        "title": "Workbench architecture",
        "body": (
            "Left rail: AOIs, jobs, scene lists. Center: map of the study pin and overlays. "
            "Right tasks: AOI → Overlay → Train → Predict → Explain. Auth keeps private AOIs "
            "per user; shared demo sites are visible to everyone. Heavy work (ingest, encode, "
            "train) runs as background jobs so the UI stays responsive."
        ),
    },
}


SYSTEM_PROMPT = """You are the Explain assistant for PrithviWORLD Workbench, a scientific UI for
HLS satellite patches, frozen IBM-NASA Prithvi EO tokens, residual dynamics forecasting, overlays,
and NDVI. Be concise, accurate, and exam-friendly.

Rules:
- Use ONLY the supplied JSON context plus general remote-sensing knowledge.
- If a number or date is missing from context, say it is not available — never invent metrics,
  coordinates, NDVI means, or experiment results.
- Prefer plain language. Use short paragraphs or bullets.
- Do not claim the dynamics model is production-ready; treat results as research/demo outputs.
- Do not ask the user for API keys or run training/jobs yourself.
"""


def openai_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def openai_model() -> str:
    return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def list_topics() -> list[dict[str, str]]:
    return [{"id": k, "title": v["title"]} for k, v in CANNED.items()]


def canned_answer(topic_id: str) -> dict[str, str]:
    item = CANNED.get(topic_id)
    if item is None:
        raise KeyError(topic_id)
    return {"id": topic_id, "title": item["title"], "reply": item["body"], "source": "canned"}


def _compact_context(context: dict[str, Any] | None) -> dict[str, Any]:
    if not context:
        return {}
    # Keep payload small and serializable.
    raw = json.loads(json.dumps(context, default=str))
    text = json.dumps(raw, ensure_ascii=False)
    if len(text) > 6000:
        return {"_truncated": True, "note": "Context was too large; ask about a specific panel."}
    return raw


def chat_with_openai(
    message: str,
    context: dict[str, Any] | None = None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OpenAI is not configured. Set OPENAI_API_KEY in software/.env (or the environment)."
        )
    user_msg = (message or "").strip()
    if not user_msg:
        raise ValueError("Message is empty")
    if len(user_msg) > 2000:
        raise ValueError("Message too long (max 2000 characters)")

    ctx = _compact_context(context)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": "Current workbench context (JSON):\n" + json.dumps(ctx, ensure_ascii=False, indent=2),
        },
    ]
    for turn in (history or [])[-8:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content[:2000]})
    messages.append({"role": "user", "content": user_msg})

    body = {
        "model": openai_model(),
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 700,
    }
    req = urllib.request.Request(
        OPENAI_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "PrithviWORLD-Workbench/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI request failed: {exc.reason}") from exc

    try:
        reply = payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise RuntimeError("Unexpected OpenAI response shape") from exc

    return {
        "reply": reply,
        "source": "openai",
        "model": body["model"],
    }
