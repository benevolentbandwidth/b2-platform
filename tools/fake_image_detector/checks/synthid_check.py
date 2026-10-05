"""SynthID check: does the image carry the watermark Google's Imagen adds?

Calls Google's Imagen watermark verification model, a Vertex AI publisher model
(imageverification@001). It only recognises Imagen's watermark, so a negative
result says nothing about images made by other generators.

Two facts established against the live API:
- The model only exists in us-central1, whatever region the rest of the
  platform uses.
- Google restricts access to it: a project without access gets HTTP 403. Until
  access is granted the check skips, with that reason, on every image.

Anything that stops the check running — no access, a timeout, an unexpected
response — skips it. It never escalates: an unavailable optional check is not
evidence about the document.
"""

from __future__ import annotations

import asyncio
import base64
import os

import httpx

from tools.fake_image_detector.checks.base_check import BaseCheck
from tools.fake_image_detector.file_formats import PDF, gemini_payload, sniff
from tools.fake_image_detector.google_clients import access_token
from tools.fake_image_detector.models import (
    CheckResult,
    GL9_FLAG_LIKELY_AI_GENERATED,
    NormalizedSignals,
)

_MODEL = "imageverification@001"
_DEFAULT_LOCATION = "us-central1"
_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_FLAG_DETECTED = "SYNTHID_WATERMARK_DETECTED"

# Google's SDK passes `decision` straight through without documenting its
# values, and access is restricted, so these could not be confirmed live. Any
# other value is treated as "did not run" rather than guessed at, because a
# detection forces human review.
_DETECTED = "ACCEPT"
_NOT_DETECTED = "REJECT"


def _access_token() -> str:
    # Shared credentials, refreshed only on expiry (see google_clients).
    return access_token(_SCOPE)


class VertexSynthIDCheck(BaseCheck):
    check_id = "synthid"

    def __init__(
        self,
        params: dict | None = None,
        *,
        project: str | None = None,
        location: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        params = params or {}
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or params.get("location") or _DEFAULT_LOCATION
        self._timeout_seconds = float(params.get("timeout_seconds", timeout_seconds))

    async def run(self, image_bytes: bytes, context: dict) -> CheckResult:
        if not self._project:
            return self._skip("GOOGLE_CLOUD_PROJECT not set")
        fmt = sniff(image_bytes)
        if fmt is None or fmt is PDF:
            return self._skip("not an image this model can check")
        # Full size: SynthID needs no help reading the image, and shrinking
        # could only weaken the watermark signal. TIFF still becomes PNG.
        data, _mime = gemini_payload(image_bytes, max_edge=None)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._verify, data), timeout=self._timeout_seconds
            )
        except TimeoutError:
            return self._skip(f"timed out after {self._timeout_seconds:.0f}s")

    def _verify(self, data: bytes) -> CheckResult:
        url = (
            f"https://{self._location}-aiplatform.googleapis.com/v1/projects/{self._project}"
            f"/locations/{self._location}/publishers/google/models/{_MODEL}:predict"
        )
        body = {
            "instances": [{"image": {"bytesBase64Encoded": base64.b64encode(data).decode("ascii")}}],
            "parameters": {},
        }
        try:
            response = httpx.post(
                url,
                json=body,
                headers={"Authorization": f"Bearer {_access_token()}"},
                timeout=self._timeout_seconds,
            )
        except Exception as exc:  # network, DNS, credentials
            return self._skip(f"request failed: {type(exc).__name__}")

        if response.status_code in (401, 403, 404):
            return self._skip(
                f"no access to {_MODEL} (HTTP {response.status_code}); Google restricts this model"
            )
        if response.status_code >= 400:
            return self._skip(f"HTTP {response.status_code}")

        try:
            decision = response.json()["predictions"][0]["decision"]
        except (ValueError, KeyError, IndexError, TypeError):
            return self._skip("unexpected response shape")

        if decision == _DETECTED:
            flags = [_FLAG_DETECTED, GL9_FLAG_LIKELY_AI_GENERATED]
            return CheckResult(
                check=self.check_id,
                passed=False,
                fake_score=1.0,
                confidence=0.95,
                flags=flags,
                signals={"decision": decision, "model": _MODEL},
                normalized_signals=NormalizedSignals(
                    category="synthetic",
                    confidence=0.95,
                    indicators=flags,
                    synthetic_score=1.0,
                ),
            )
        if decision == _NOT_DETECTED:
            # Ran, found no Imagen watermark. That is no evidence the document is
            # genuine, so it carries no weight: reported like the other
            # information-only checks, and it cannot pull the risk score down.
            return CheckResult(
                check=self.check_id,
                passed=True,
                confidence=0.0,
                skipped=True,
                signals={"decision": decision, "model": _MODEL},
            )
        return self._skip(f"unrecognised decision {decision!r}")

    def _skip(self, reason: str) -> CheckResult:
        return CheckResult(
            check=self.check_id,
            passed=True,
            confidence=0.0,
            skipped=True,
            signals={"reason": reason},
        )
