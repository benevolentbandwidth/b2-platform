from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import time

from tools.fake_image_detector.checks.base_check import BaseCheck
from tools.fake_image_detector.file_formats import gemini_payload
from tools.fake_image_detector.gemini_settings import thinking_config, today
from tools.fake_image_detector.google_clients import (
    attempts_budget_seconds,
    gemini_client,
    request_timeout,
    retry_wait,
    retryable,
)
from tools.fake_image_detector.models import (
    CheckResult,
    GL9_FLAG_EDITING_ARTIFACTS,
    GL9_FLAG_FOUND_ONLINE,
    GL9_FLAG_POSSIBLE_STOCK,
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
Today's date is {today}. Any date on or before today is in the past, not the future; judge dates against today, not against what you know.

Assess whether this appears to be a GENUINE, AUTHENTIC document or whether it is deceptive in any of the following ways:

1. FORGED or FABRICATED — a printed template, photoshop creation, or entirely made-up document.
2. PHOTO OF A PHOTO — a photograph taken of another physical document or screen.
3. DIGITALLY MANIPULATED — an authentic document with altered fields (name, date, number).
4. TEMPLATE DETECTED — produced from an online template without official security features.
5. INCONSISTENT SECURITY FEATURES — missing expected holograms, watermarks, or official markings.
6. LANGUAGE INCONSISTENCY — the language or script used in the document does not match what is expected for the claimed country or document type (e.g. an English-only passport from a non-English-issuing country, mismatched official seals or text).
7. INTERNAL INCONSISTENCY — details on the document contradict each other: a registration or issue date before the date of death, an age that does not match the dates of birth and death, or a date that cannot exist. Documents may show dates in more than one calendar (e.g. Hijri alongside Gregorian on Moroccan certificates) or in local formats; convert before comparing, and never report a difference that is only a calendar or format conversion. Report this only when you can name the two details that contradict each other, and name them in "signals".

Rule for INTERNAL INCONSISTENCY: compare only dates, ages and places that are written out on the document in words or as dates. Treat every identity or registration number (an Indonesian NIK, a certificate number, any ID number) as opaque: never decode one for any purpose — not a date, not a region, not a sex — and never compare anything derived from one against the rest of the document. Examples of what is NOT an internal inconsistency and NOT evidence of manipulation: the birth date or the region code encoded in an NIK not matching the written birth date or place. People move, and NIKs are often issued with different details.

Ordinary capture and scan artifacts are NOT evidence of deception on their own. Do not treat a photograph of a physical document, perspective distortion, glare or reflections, table or background surroundings, shadows, cropping, uneven illumination, compression, or scanner/CamScanner cleanup as evidence of forgery or alteration.

Set "is_deceptive" to true, or include PHOTO_OF_PHOTO or EDITING_ARTIFACTS, only when there is visible document-content or compositing inconsistency that supports the finding. Do not use those findings for capture or scan quality alone.

Do NOT make a final verdict. Report only the signals you observe.

Respond ONLY with valid JSON matching this schema:
{{
  "is_deceptive": <bool, true if the document appears forged, altered, or not genuine>,
  "fake_likelihood": <float 0.0-1.0, probability the document is not genuine>,
  "confidence": <float 0.0-1.0, how certain you are about your assessment>,
  "signals": [<list of short specific observed-indicator strings, e.g. "no visible hologram", "font inconsistency on expiry date", "document language does not match issuing country">],
  "flags": [<zero or more from: FORGED_DOCUMENT, PHOTO_OF_PHOTO, EDITING_ARTIFACTS,
             TEMPLATE_DETECTED, INCONSISTENT_SECURITY_FEATURES, LANGUAGE_INCONSISTENCY,
             INTERNAL_INCONSISTENCY, CLEAN>]
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



_COUNTRY_NAMES = {
    "ID": "Indonesia",
    "MA": "Morocco",
    "DE": "Germany",
    "KE": "Kenya",
    "NG": "Nigeria",
}


def _sanitize_country(value: object) -> str:
    if value is None:
        return ""
    cleaned = _SAFE_TOKEN_RE.sub("", str(value)).strip().upper()
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
    return cleaned[:3]


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
        attempts: int = 3,
        model: str = "gemini-2.5-flash",
        thinking_level: str | None = None,
    ):
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("VERTEX_LOCATION", "us-central1")
        self._timeout_seconds = timeout_seconds
        self._model = model
        self._thinking_level = thinking_level
        self._attempts = max(1, attempts)

    async def run(self, image_bytes: bytes, context: dict) -> CheckResult:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._run_sync, image_bytes, context),
                # Backstop only, in case the client fails to enforce its own timeout.
                # timeout_seconds limits each attempt, so a stalled call is retried.
                timeout=attempts_budget_seconds(self._attempts, self._timeout_seconds),
            )
        except TimeoutError:
            return self._error_result("Gemini vision check timed out", "CHECK_TIMEOUT")

    def _error_result(self, error: str, flag: str = "CHECK_RUNTIME_ERROR") -> CheckResult:
        """An unreachable detector is not evidence of forgery.

        Routed to human review via human_escalate rather than scored as a
        certain fake. Scoring it 1.0 meant a timeout or a malformed model
        response auto-rejected a genuine certificate.
        """
        return CheckResult(
            check=self.check_id,
            passed=False,
            fake_score=0.0,
            confidence=0.0,
            flags=[flag],
            signals={"error": error},
            skipped=False,
            human_escalate=True,
            error=error,
        )

    def _run_sync(self, image_bytes: bytes, context: dict) -> CheckResult:
        try:
            # importlib (not `from google import genai`) so the lookup goes through
            # sys.modules. Attribute-style import resolves off the already-imported
            # `google` package and silently bypasses test stubs.
            genai = importlib.import_module("google.genai")
            gentypes = importlib.import_module("google.genai.types")
        except ImportError as e:
            return self._error_result(str(e))

        if not self._project:
            return self._error_result("GOOGLE_CLOUD_PROJECT not set")

        doc_type = context.get("doc_type")
        country = context.get("country")

        if doc_type:
            safe_doc_type = _sanitize_doc_type(doc_type)
            safe_country = _sanitize_country(country)
            # The sanitised code is mapped to a fixed name, so the guard against
            # prompt injection still holds; a bare "ID" reads as "identity".
            country_name = _COUNTRY_NAMES.get(safe_country, safe_country)
            country_clause = f" from {country_name}" if safe_country else ""
            # The extraction check's readings are deliberately NOT passed in. This
            # check reads the image itself; given a second set of readings it hunted
            # for differences and invented them (a genuine certificate scored 0.90
            # "forged" with them, 0.10 without, 3/3 each). Those readings also vary
            # run to run, so a different genuine certificate was flagged each time,
            # and they carried unfiltered text from the document into this prompt.
            prompt = _DOCUMENT_PROMPT_TEMPLATE.format(
                doc_type=safe_doc_type,
                country_clause=country_clause,
                today=today(),
            )
        else:
            safe_doc_type = None
            safe_country = ""
            prompt = _PHOTO_PROMPT

        model = self._model  # from gemini.model in pipeline.yaml

        try:
            # Shared per process (see google_clients). The payload is prepared
            # once, not per attempt: it decodes and may resize a large image.
            client = gemini_client(genai, project=self._project, location=self._location)
            gemini_bytes, gemini_mime = gemini_payload(image_bytes)
            image_part = gentypes.Part.from_bytes(data=gemini_bytes, mime_type=gemini_mime)
        except Exception as e:
            return self._error_result(str(e))

        raw = ""
        for attempt in range(self._attempts):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[image_part, prompt],
                    # Default temperature: Google advises against lowering it on
                    # Gemini 3, which can loop or degrade below 1.0.
                    config=gentypes.GenerateContentConfig(
                        # JSON output, not JSON asked for in prose: at the default
                        # temperature a reply wrapped in text or cut short would
                        # otherwise send a genuine case to review.
                        response_mime_type="application/json",
                        thinking_config=thinking_config(gentypes, self._thinking_level),
                        # Ends this attempt itself, so a stalled call is retried.
                        http_options=request_timeout(gentypes, self._timeout_seconds),
                    ),
                )
                raw = response.text
                break
            except Exception as e:
                if attempt == self._attempts - 1 or not retryable(e):
                    return self._error_result(str(e))
                time.sleep(retry_wait(attempt))

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
            # Flags only: whether a flag forces human review is decided by the
            # pipeline from hard_escalation_flags in pipeline.yaml, so that list
            # is the single source of truth.
        )
