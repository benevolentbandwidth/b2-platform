from __future__ import annotations

import asyncio
import json
import os
import re

from tools.fake_image_detector.checks.base_check import BaseCheck
from tools.fake_image_detector.models import (
    CheckResult,
    GL9_FLAG_EDITING_ARTIFACTS,
    GL9_FLAG_FOUND_ONLINE,
    GL9_FLAG_POSSIBLE_STOCK,
    GL9_HARD_ESCALATION_FLAGS,
    NormalizedSignals,
)

_PHOTO_PROMPT = """You are a fraud-detection assistant. Analyze this image and assess whether it is a genuine, original photograph submitted by a real person, or whether it is deceptive in any of the following ways:

1. AI-GENERATED or SYNTHETIC — produced by a GAN, diffusion model, or other generative system.
2. DIGITALLY MANIPULATED — a real photo that has been edited, composited, or had elements added/removed.
3. STAGED or STOCK — a professional shoot, stock image, or heavily posed photo unlikely to be a genuine personal submission.
4. BACKGROUND INCONSISTENCY — background does not match the subject (composited, greenscreen, mismatched lighting).
5. LIGHTING or SHADOW INCONSISTENCY — light sources or shadows are inconsistent across the image.
6. AGE INCONSISTENCY — the estimated age of the subject does not match the age that would be expected for a genuine personal submission (e.g. the photo appears to show a much older or younger person than the context suggests).

Do NOT make a final verdict. Report only the signals you observe.

Respond ONLY with valid JSON matching this schema:
{
  "is_deceptive": <bool, true if the image is AI-generated, manipulated, staged, or otherwise not a genuine personal photo>,
  "fake_likelihood": <float 0.0-1.0, probability the image is deceptive>,
  "confidence": <float 0.0-1.0, how certain you are about your assessment>,
  "estimated_age": <integer or null, estimated age of the primary subject in years, null if no person is visible>,
  "signals": [<list of short specific observed-indicator strings, e.g. "skin texture too smooth", "background composited", "subject appears 40+ years old">],
  "flags": [<zero or more from: GAN_ARTIFACTS, DIFFUSION_ARTIFACTS, EDITING_ARTIFACTS,
             INCONSISTENT_LIGHTING, UNNATURAL_TEXTURE, BACKGROUND_INCONSISTENCY,
             POSSIBLE_STOCK, STAGING_ARTIFACTS, METADATA_MISMATCH, AGE_INCONSISTENCY, CLEAN>]
}
Do not include any text outside the JSON object."""

_DOCUMENT_PROMPT_TEMPLATE = """\
You are a fraud-detection assistant. This image has been identified as a {doc_type}{country_clause}.
{extracted_section}
Assess whether this appears to be a GENUINE, AUTHENTIC document or whether it is deceptive in any of the following ways:

1. FORGED or FABRICATED — a printed template, photoshop creation, or entirely made-up document.
2. PHOTO OF A PHOTO — a photograph taken of another physical document or screen.
3. DIGITALLY MANIPULATED — an authentic document with altered fields (name, date, number).
4. TEMPLATE DETECTED — produced from an online template without official security features.
5. INCONSISTENT SECURITY FEATURES — missing expected holograms, watermarks, or official markings.
6. LANGUAGE INCONSISTENCY — the language or script used in the document does not match what is expected for the claimed country or document type (e.g. an English-only passport from a non-English-issuing country, mismatched official seals or text).

Do NOT make a final verdict. Report only the signals you observe.

Respond ONLY with valid JSON matching this schema:
{{
  "is_deceptive": <bool, true if the document appears forged, altered, or not genuine>,
  "fake_likelihood": <float 0.0-1.0, probability the document is not genuine>,
  "confidence": <float 0.0-1.0, how certain you are about your assessment>,
  "signals": [<list of short specific observed-indicator strings, e.g. "no visible hologram", "font inconsistency on expiry date", "document language does not match issuing country">],
  "flags": [<zero or more from: FORGED_DOCUMENT, PHOTO_OF_PHOTO, EDITING_ARTIFACTS,
             TEMPLATE_DETECTED, INCONSISTENT_SECURITY_FEATURES, LANGUAGE_INCONSISTENCY, CLEAN>]
}}
Do not include any text outside the JSON object.\
"""

# Strip optional markdown code fences Gemini sometimes wraps around JSON
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_SAFE_TOKEN_RE = re.compile(r"[^a-zA-Z0-9 _-]+")

_FLAG_ALIASES = {
    "EDITING_DETECTED": GL9_FLAG_EDITING_ARTIFACTS,
    "STOCK_PHOTO_INDICATORS": GL9_FLAG_POSSIBLE_STOCK,
    "STOCK_PHOTO_REUSE": GL9_FLAG_POSSIBLE_STOCK,
    "FOUND_ONLINE_REUSE": GL9_FLAG_FOUND_ONLINE,
}

_EDITING_SOURCE_FLAGS = {
    "GAN_ARTIFACTS",
    "DIFFUSION_ARTIFACTS",
    "INCONSISTENT_LIGHTING",
    "UNNATURAL_TEXTURE",
    "BACKGROUND_INCONSISTENCY",
    "METADATA_MISMATCH",
}


def _sanitize_doc_type(value: object) -> str:
    if value is None:
        return "document"
    cleaned = _SAFE_TOKEN_RE.sub("", str(value)).strip().replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:64] if cleaned else "document"


def _sanitize_country(value: object) -> str:
    if value is None:
        return ""
    cleaned = _SAFE_TOKEN_RE.sub("", str(value)).strip().upper()
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
    return cleaned[:3]


def _sniff_mime(image_bytes: bytes) -> str:
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if image_bytes[:4] == b"\x89PNG":
        return "image/png"
    if len(image_bytes) >= 12 and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _normalize_flags(raw_flags: list[str]) -> list[str]:
    normalized: list[str] = []
    for raw_flag in raw_flags:
        canonical = _FLAG_ALIASES.get(raw_flag, raw_flag)
        if canonical not in normalized:
            normalized.append(canonical)

    if any(flag in _EDITING_SOURCE_FLAGS for flag in normalized):
        if GL9_FLAG_EDITING_ARTIFACTS not in normalized:
            normalized.append(GL9_FLAG_EDITING_ARTIFACTS)

    # AGE_INCONSISTENCY is already canonical; keep it if present and avoid duplicate insertions.

    return normalized


class GeminiVisionCheck(BaseCheck):
    check_id = "gemini_vision"

    def __init__(
        self,
        project: str | None = None,
        location: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ):
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("VERTEX_LOCATION", "us-central1")
        self._timeout_seconds = timeout_seconds
        self._max_retries = max(1, max_retries)

    async def run(self, image_bytes: bytes, context: dict) -> CheckResult:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._run_sync, image_bytes, context),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            return self._error_result("Gemini vision check timed out", "CHECK_TIMEOUT")

    def _error_result(self, error: str, flag: str = "CHECK_RUNTIME_ERROR") -> CheckResult:
        return CheckResult(
            check=self.check_id,
            passed=False,
            fake_score=1.0,
            confidence=1.0,
            flags=[flag],
            signals={"error": error},
            skipped=False,
            error=error,
        )

    def _run_sync(self, image_bytes: bytes, context: dict) -> CheckResult:
        try:
            from google import genai
            from google.genai import types as gentypes
        except ImportError as e:
            return self._error_result(str(e))

        if not self._project:
            return self._error_result("GOOGLE_CLOUD_PROJECT not set")

        doc_type = context.get("doc_type")
        country = context.get("country")

        if doc_type:
            safe_doc_type = _sanitize_doc_type(doc_type)
            safe_country = _sanitize_country(country)
            country_clause = f" from {safe_country}" if safe_country else ""
            extracted = context.get("extracted_fields", {})
            if extracted:
                lines = "\n".join(f"  - {k}: {v}" for k, v in extracted.items())
                extracted_section = f"\nExtracted fields:\n{lines}\n"
            else:
                extracted_section = ""
            prompt = _DOCUMENT_PROMPT_TEMPLATE.format(
                doc_type=safe_doc_type,
                country_clause=country_clause,
                extracted_section=extracted_section,
            )
        else:
            safe_doc_type = None
            safe_country = ""
            prompt = _PHOTO_PROMPT

        model = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")

        raw = ""
        for attempt in range(self._max_retries):
            try:
                client = genai.Client(
                    vertexai=True, project=self._project, location=self._location
                )
                image_part = gentypes.Part.from_bytes(
                    data=image_bytes, mime_type=_sniff_mime(image_bytes)
                )
                response = client.models.generate_content(
                    model=model, contents=[image_part, prompt]
                )
                raw = response.text
                break
            except Exception as e:
                is_last_attempt = attempt == self._max_retries - 1
                if is_last_attempt:
                    return self._error_result(str(e))
                continue

        try:
            match = _JSON_RE.search(raw)
            if not match:
                raise ValueError("no JSON object in response")
            data = json.loads(match.group())
        except Exception as e:
            raw_snippet = raw[:120] if raw else ""
            return self._error_result(
                f"JSON parse error: {e}; raw_snippet={raw_snippet!r}",
                "GEMINI_PARSE_ERROR",
            )

        fake_likelihood = max(0.0, min(1.0, float(data.get("fake_likelihood", 0.0))))
        gemini_confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        is_deceptive = bool(data.get("is_deceptive", data.get("is_synthetic", False)))
        flags = _normalize_flags([str(f) for f in data.get("flags", [])])
        raw_age = data.get("estimated_age")
        estimated_age = int(raw_age) if raw_age is not None else None
        escalation_reasons = [
            f"{flag} detected by gemini_vision check"
            for flag in flags
            if flag in GL9_HARD_ESCALATION_FLAGS
        ]
        signals = {
            "doc_type": safe_doc_type if doc_type else None,
            "country": safe_country if doc_type else "",
            "is_deceptive": is_deceptive,
            "estimated_age": estimated_age,
            "signals": data.get("signals", []),
        }

        return CheckResult(
            check=self.check_id,
            passed=not is_deceptive,
            fake_score=round(fake_likelihood, 3),
            confidence=round(gemini_confidence, 3),
            flags=flags,
            signals=signals,
            normalized_signals=NormalizedSignals(
                category="document_authenticity" if doc_type else "synthetic",
                confidence=round(gemini_confidence, 3),
                indicators=flags or ["CLEAN"],
                document_type=safe_doc_type if doc_type else None,
                country_code=safe_country if doc_type else None,
                synthetic_score=round(fake_likelihood, 3) if not doc_type else None,
                manipulation_score=round(fake_likelihood, 3),
                staging_score=round(fake_likelihood, 3) if any(
                    f in {"STAGING_ARTIFACTS", GL9_FLAG_POSSIBLE_STOCK} for f in flags
                ) else None,
            ),
            human_escalate=bool(escalation_reasons),
            escalation_reasons=escalation_reasons,
        )
