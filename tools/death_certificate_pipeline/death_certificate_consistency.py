from __future__ import annotations

import base64
import importlib
import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

from tools.fake_image_detector.file_formats import gemini_payload
from tools.fake_image_detector.google_clients import (
    gemini_client,
    request_timeout,
    retry_wait,
    retryable,
)
from tools.fake_image_detector.gemini_settings import thinking_config, today


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_CONSISTENCY_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "certificate": {
            "type": "object",
            "properties": {
                "full_name":           {"type": "string", "nullable": True},
                "date_of_death":       {"type": "string", "nullable": True},
                "place_of_death":      {"type": "string", "nullable": True},
                "age_at_death":        {"type": "integer", "nullable": True},
                "cause_of_death":      {"type": "string", "nullable": True},
                "certificate_number":  {"type": "string", "nullable": True},
                "issuing_authority":   {"type": "string", "nullable": True},
                "registration_date":   {"type": "string", "nullable": True},
                "other_visible_details": {"type": "object"},
            },
            "required": [
                "full_name", "date_of_death", "place_of_death", "age_at_death",
                "cause_of_death", "certificate_number", "issuing_authority",
                "registration_date", "other_visible_details",
            ],
        },
        "claimant_account_present": {"type": "boolean"},
        "claimant": {
            "type": "object",
            "properties": {
                "relationship_to_deceased": {"type": "string", "nullable": True},
                "dependants": {"type": "array", "items": {"type": "string"}},
                "other_details": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["relationship_to_deceased", "dependants", "other_details"],
        },
        "case_note": {"type": "string"},
        "consistency_score":  {"type": "number"},
        "consistency_label":  {"type": "string", "enum": ["high", "moderate", "low"]},
        "confidence":         {"type": "number"},
        "matches":            {"type": "array", "items": {"type": "string"}},
        "mismatches":         {"type": "array", "items": {"type": "string"}},
        "uncertain_points":   {"type": "array", "items": {"type": "string"}},
        "summary":            {"type": "string"},
    },
    "required": [
        "certificate", "claimant_account_present", "claimant", "case_note", "consistency_score", "consistency_label", "confidence",
        "matches", "mismatches", "uncertain_points", "summary",
    ],
}

_CONSISTENCY_PROMPT = """You are comparing a chat history against a death certificate image.

Use the chat history and the image together, but make only one model call.

Your tasks:
1. Extract the visible facts from the death certificate image.
2. Decide whether the claimant has given their own account of the death. Look only at the claimant's messages (lines starting "user:"), not the assistant's. An account says at least who passed away and roughly when or where. Greetings, requests for help, or messages that only send a document are not an account. Set claimant_account_present accordingly.
3. If there is an account, compare the certificate facts against it and produce a narrative consistency score where 1.0 means the account and certificate are highly consistent, and 0.0 means they strongly conflict.
4. If there is no account, there is nothing to compare: still extract the certificate fully, set consistency_score to 0, consistency_label to "low", and add "no claimant account" to uncertain_points. Do not infer an account from the assistant's messages or from the certificate itself.
5. From the claimant's own messages only, record in "claimant": who they are to the deceased (relationship_to_deceased, e.g. "sister"; null if they did not say), any children or other dependants they mention (dependants, e.g. "two children of the deceased, now in the claimant's care"), and any other circumstances relevant to an aid request (other_details). Record only what they said; do not infer.
6. Write case_note: 2 to 4 plain sentences for a GiveLight caseworker covering who is asking and how they are related, any dependants mentioned, what they said happened, and whether that matches the certificate. Use only what the claimant said and what the certificate shows. Do not judge eligibility or authenticity.

Rules:
- Use only information visible in the image and explicitly present in the chat history.
- Do not invent missing facts.
- If a field is unreadable, use null.
- Treat the chat history as the source of narrative claims and the image as the source of certificate facts.

Respond ONLY with structured data matching this schema:
{
  "certificate": {
    "full_name": string|null,
    "date_of_death": string|null,
    "place_of_death": string|null,
    "age_at_death": integer|null,
    "cause_of_death": string|null,
    "certificate_number": string|null,
    "issuing_authority": string|null,
    "registration_date": string|null,
    "other_visible_details": object
  },
  "claimant_account_present": boolean,
  "claimant": {
    "relationship_to_deceased": string|null,
    "dependants": [string],
    "other_details": [string]
  },
  "case_note": string,
  "consistency_score": number,
  "consistency_label": "high"|"moderate"|"low",
  "confidence": number,
  "matches": [string],
  "mismatches": [string],
  "uncertain_points": [string],
  "summary": string
}

Do not include any text outside the structured response."""


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _render_chat_history(chat_history: str | Sequence[Any]) -> str:
    if isinstance(chat_history, str):
        return chat_history.strip()

    lines: list[str] = []
    for item in chat_history:
        if isinstance(item, Mapping):
            role = str(item.get("role", "message")).strip() or "message"
            content = item.get("content", "")
            if isinstance(content, (list, tuple)):
                content = " ".join(str(part) for part in content)
            lines.append(f"{role}: {content}")
        else:
            lines.append(str(item))

    return "\n".join(lines).strip()


def _load_gemini_client() -> tuple[Any, Any]:
    try:
        genai    = importlib.import_module("google.genai")
        gentypes = importlib.import_module("google.genai.types")
    except ImportError as exc:
        raise ImportError(
            "google-genai is required for death certificate consistency analysis."
        ) from exc
    return genai, gentypes


def _parse_json_response(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        match = _JSON_RE.search(raw)
        if not match:
            raise ValueError("no JSON object in model response")
        raw = match.group()

    match = _JSON_RE.search(raw)
    if match:
        match_text = match.group()
    elif raw.startswith("{") and raw.endswith("}"):
        match_text = raw
    else:
        raise ValueError("no JSON object in model response")

    data = json.loads(match_text)
    if not isinstance(data, dict):
        raise ValueError("model response JSON must be an object")
    return data


def _normalize_structured_response(response: Any) -> dict[str, Any]:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            parsed = parsed.model_dump()
        if isinstance(parsed, dict):
            return parsed

    raw_text = getattr(response, "text", "") or ""
    return _parse_json_response(raw_text)


def analyze_death_certificate_consistency(
    chat_history: str | Sequence[Any],
    image_bytes: bytes,
    *,
    api_key: str | None = None,
    project: str | None = None,
    location: str | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
    thinking_level: str | None = None,
    attempts: int = 1,
) -> dict[str, Any]:
    """Extract death certificate facts and score narrative consistency in one Gemini call.

    Supports both Vertex AI (project/location) and Gemini API key authentication.
    Vertex AI takes precedence when GOOGLE_CLOUD_PROJECT is set.
    """
    transcript = _render_chat_history(chat_history)
    if not transcript:
        raise ValueError("chat_history must not be empty")
    if not image_bytes:
        raise ValueError("image_bytes must not be empty")

    vertex_project  = project  or os.environ.get("GOOGLE_CLOUD_PROJECT")
    vertex_location = location or os.environ.get("VERTEX_LOCATION", "us-central1")
    gemini_api_key  = api_key  or os.environ.get("GEMINI_API_KEY")

    if not vertex_project and not gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY is required for Gemini calls "
            "(or set GOOGLE_CLOUD_PROJECT for Vertex AI)"
        )

    # Required by tests/unit/tools/death_certificate_pipeline/test_death_certificate_consistency.py
    # so the default model used by live OCR/vision checks is explicit and asserted.
    if model is None:
        # Imported here: the settings loader imports the models module, and this
        # module is imported early by the package.
        from tools.death_certificate_pipeline.config_loader import default_scoring_config

        model = default_scoring_config().consistency_model  # scoring.yaml
    gemini_model = model

    genai, gentypes = _load_gemini_client()
    # Shared per process (see google_clients); Vertex takes precedence.
    if vertex_project:
        client = gemini_client(genai, project=vertex_project, location=vertex_location)
    else:
        client = gemini_client(genai, api_key=gemini_api_key)

    gemini_bytes, gemini_mime = gemini_payload(bytes(image_bytes))
    image_part = gentypes.Part.from_bytes(data=gemini_bytes, mime_type=gemini_mime)

    prompt = (
        f"{_CONSISTENCY_PROMPT}\n\n"
        f"Today's date is {today()}. Resolve relative dates in the chat history "
        f"(\"last month\", \"two weeks ago\") against it, and treat any date on or "
        f"before today as past, not future.\n\n"
        f"Chat history:\n{transcript}\n"
    )

    config = gentypes.GenerateContentConfig(
        responseMimeType="application/json",
        responseSchema=_CONSISTENCY_RESPONSE_SCHEMA,
        # Default temperature: Google advises against lowering it on Gemini 3,
        # which can loop or degrade below 1.0.
        thinking_config=thinking_config(gentypes, thinking_level),
        # Ends each attempt itself on timeout, so a stalled call is retried rather
        # than waited out, and the caller's thread is released when it gives up.
        http_options=request_timeout(gentypes, timeout_seconds),
    )

    # Client, image and prompt are prepared once; only the request is retried.
    for attempt in range(max(1, attempts)):
        try:
            response = client.models.generate_content(
                model=gemini_model,
                contents=[image_part, prompt],
                config=config,
            )
            break
        except Exception as exc:
            if attempt >= attempts - 1 or not retryable(exc):
                raise
            time.sleep(retry_wait(attempt))

    parsed = _normalize_structured_response(response)

    certificate = parsed.get("certificate") or {}
    if not isinstance(certificate, dict):
        certificate = {}

    return {
        "certificate":       certificate,
        # Defaults to True only so a response lacking the field is scored as
        # before; the schema marks it required.
        "claimant_account_present": bool(parsed.get("claimant_account_present", True)),
        "claimant": parsed.get("claimant") if isinstance(parsed.get("claimant"), dict) else {},
        "case_note": str(parsed.get("case_note", "")),
        "consistency_score": round(_clamp01(parsed.get("consistency_score")), 3),
        "consistency_label": str(parsed.get("consistency_label", "moderate")),
        "confidence":        round(_clamp01(parsed.get("confidence")), 3),
        "matches":           [str(i) for i in parsed.get("matches", [])],
        "mismatches":        [str(i) for i in parsed.get("mismatches", [])],
        "uncertain_points":  [str(i) for i in parsed.get("uncertain_points", [])],
        "summary":           str(parsed.get("summary", "")),
        "model":             gemini_model,
    }


def analyze_death_certificate_consistency_base64(
    chat_history: str | Sequence[Any],
    image_b64: str,
    *,
    api_key: str | None = None,
    project: str | None = None,
    location: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper — accepts base64-encoded image instead of raw bytes."""
    return analyze_death_certificate_consistency(
        chat_history,
        base64.b64decode(image_b64),
        api_key=api_key,
        project=project,
        location=location,
        model=model,
    )
