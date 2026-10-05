import asyncio
import io

from tools.fake_image_detector.checks.base_check import BaseCheck
from tools.fake_image_detector.models import CheckResult, NormalizedSignals


# Machine-readable zones (ICAO 9303) appear on passports and ID cards only.
_MRZ_DOCUMENT_TYPES = {"passport", "national_id"}


class MRZCheck(BaseCheck):
    check_id = "mrz"

    async def run(self, image_bytes: bytes, context: dict) -> CheckResult:
        doc_type = context.get("doc_type")
        if doc_type and doc_type not in _MRZ_DOCUMENT_TYPES:
            # Not applicable. Running anyway mis-read death-certificate text as a
            # partial MRZ, reported it as a check that "ran", and cost time.
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True)
        return await asyncio.to_thread(self._run_sync, image_bytes)

    def _run_sync(self, image_bytes: bytes) -> CheckResult:
        try:
            from passporteye import read_mrz
        except ImportError as e:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True, error=str(e))

        try:
            mrz = read_mrz(io.BytesIO(image_bytes))
        except Exception as e:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True, error=str(e))

        if mrz is None:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True)

        data = mrz.to_dict()
        valid_score = data.get("valid_score", 0)

        if valid_score == 100:
            return CheckResult(
                check=self.check_id,
                passed=True,
                fake_score=0.0,
                confidence=1.0,
                signals={"type": data.get("type"), "country": data.get("country")},
                normalized_signals=NormalizedSignals(
                    category="document_authenticity",
                    confidence=1.0,
                    indicators=["MRZ_VALID"],
                    document_type=data.get("type"),
                    country_code=data.get("country"),
                ),
            )

        # Partial read: phone-photo OCR errors cause checksum failures on real passports.
        # Skip scoring and let Gemini validate — report the score for observability.
        return CheckResult(
            check=self.check_id,
            passed=True,
            confidence=0.0,
            skipped=True,
            signals={"valid_score": valid_score, "type": data.get("type")},
            normalized_signals=NormalizedSignals(
                category="document_authenticity",
                confidence=0.0,
                indicators=["MRZ_PARTIAL_READ"],
                document_type=data.get("type"),
            ),
        )
