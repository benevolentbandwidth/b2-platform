from __future__ import annotations

import asyncio
import importlib
import json
import os
import re

from tools.fake_image_detector.checks.base_check import BaseCheck
from tools.fake_image_detector.checks.checksum_check import checksum_fields
from tools.fake_image_detector.file_formats import gemini_payload
from tools.fake_image_detector.google_clients import gemini_client, request_timeout
from tools.fake_image_detector.gemini_settings import thinking_config, thinking_level
from tools.fake_image_detector.models import CheckResult

# Fields to extract per document type, based on what each document typically contains.
_EXTRACT_FIELDS: dict[str, list[str]] = {
    "passport": ["full_name", "document_number", "date_of_birth", "expiry_date", "nationality"],
    "national_id": ["full_name", "id_number", "date_of_birth", "expiry_date"],
    "birth_certificate": ["full_name", "date_of_birth", "place_of_birth", "signed_by"],
    "death_certificate": ["full_name", "date_of_death", "place_of_death", "signed_by"],
    "driving_license": ["full_name", "licence_number", "date_of_birth", "expiry_date"],
    "bank_statement": ["account_holder", "iban", "account_number"],
}

_EXTRACT_PROMPT_TEMPLATE = """\
This image contains a {doc_type}. Extract the following fields exactly as they appear in the document:

{field_list}

Respond ONLY with a valid JSON object. Use null for any field that is not visible or not present.
Example: {{"full_name": "Max Mustermann", "date_of_birth": "1985-03-15"}}
Do not include any text outside the JSON object.\
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class GeminiExtractCheck(BaseCheck):
    check_id = "gemini_extract"

    def __init__(self, params: dict | None = None, project: str | None = None, location: str | None = None):
        params = params or {}
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = (
            location or params.get("location") or os.environ.get("VERTEX_LOCATION", "us-central1")
        )
        self._timeout_seconds = float(params.get("timeout_seconds", 60.0))
        # From this check's params in pipeline.yaml, where it is required; the
        # default only serves direct construction.
        self._model = str(params.get("model") or "gemini-2.5-flash")
        self._thinking_level = thinking_level(params.get("thinking_level"), "gemini_extract params")

    async def run(self, image_bytes: bytes, context: dict) -> CheckResult:
        # Without a limit a hung Gemini call stalled the whole verification.
        # Extraction only gathers information, so a timeout just skips it.
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._run_sync, image_bytes, context),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            return CheckResult(
                check=self.check_id,
                passed=True,
                confidence=0.0,
                skipped=True,
                signals={"reason": f"timed out after {self._timeout_seconds:.0f}s"},
            )

    def _run_sync(self, image_bytes: bytes, context: dict) -> CheckResult:
        doc_type = context.get("doc_type")
        if not doc_type:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True)

        fields = _EXTRACT_FIELDS.get(doc_type)
        if not fields:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True)

        # The readings feed only the check-digit check. Without check-digit fields
        # (death certificates have none) they are thrown away, so the call is
        # skipped: it cost a paid Gemini call and held up the fraud check behind it.
        if not checksum_fields(doc_type, context.get("country")):
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True)

        try:
            # importlib (not `from google import genai`) so the lookup goes through
            # sys.modules. Attribute-style import resolves off the already-imported
            # `google` package and silently bypasses test stubs.
            genai = importlib.import_module("google.genai")
            gentypes = importlib.import_module("google.genai.types")
        except ImportError as e:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True, error=str(e))

        if not self._project:
            return CheckResult(
                check=self.check_id, passed=True, confidence=0.0,
                skipped=True, error="GOOGLE_CLOUD_PROJECT not set",
            )

        field_list = "\n".join(f"- {f}" for f in fields)
        prompt = _EXTRACT_PROMPT_TEMPLATE.format(
            doc_type=doc_type.replace("_", " "),
            field_list=field_list,
        )
        model = self._model

        try:
            client = gemini_client(genai, project=self._project, location=self._location)
            gemini_bytes, gemini_mime = gemini_payload(image_bytes)
            image_part = gentypes.Part.from_bytes(data=gemini_bytes, mime_type=gemini_mime)
            response = client.models.generate_content(
                model=model,
                contents=[image_part, prompt],
                # Default temperature: Google advises against lowering it on Gemini 3.
                config=gentypes.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=thinking_config(gentypes, self._thinking_level),
                    # The request ends itself on timeout, so a timed-out call does
                    # not keep holding a worker thread after run() has given up.
                    http_options=request_timeout(gentypes, self._timeout_seconds),
                ),
            )
            raw = response.text
        except Exception as e:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True, error=str(e))

        try:
            match = _JSON_RE.search(raw)
            if not match:
                raise ValueError("no JSON object in response")
            extracted = json.loads(match.group())
        except Exception as e:
            return CheckResult(
                check=self.check_id, passed=True, confidence=0.0,
                skipped=True, error=f"JSON parse error: {e}",
            )

        # Drop null values; store the rest for downstream checks and the auth prompt
        extracted_fields = {k: v for k, v in extracted.items() if v is not None}
        context["extracted_fields"] = extracted_fields

        return CheckResult(
            check=self.check_id,
            passed=True,
            confidence=0.0,
            skipped=True,
            signals={"doc_type": doc_type, "extracted": extracted_fields},
        )
