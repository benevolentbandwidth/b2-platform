import asyncio
import io
import re

from tools.fake_image_detector.checks.base_check import BaseCheck, CheckContext
from tools.fake_image_detector.checks.checksums import validate_iban, validate_luhn, validate_mrz_digit
from tools.fake_image_detector.config_loader import load_document_schemas
from tools.fake_image_detector.models import CheckResult, NormalizedSignals

_ALGORITHMS = {
    "mrz": validate_mrz_digit,
    "luhn": validate_luhn,
    "iban": validate_iban,
}

_SCHEMAS = None


def _get_schemas() -> dict:
    global _SCHEMAS
    if _SCHEMAS is None:
        _SCHEMAS = load_document_schemas()
    return _SCHEMAS


def _ocr_text(image_bytes: bytes) -> str | None:
    try:
        import pytesseract
        from PIL import Image
        return pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)))
    except Exception:
        return None


def checksum_fields(doc_type: str | None, country: str | None) -> list[dict]:
    """The check-digit fields document_schemas.yaml defines for this document."""
    if not doc_type:
        return []
    type_schemas = _get_schemas().get("schemas", {}).get(doc_type) or {}
    schema = type_schemas.get(country) or type_schemas.get("base") or {}
    return [f for f in schema.get("required_fields", []) if f.get("checksum")]


class ChecksumCheck(BaseCheck):
    check_id = "checksum"

    async def run(self, image_bytes: bytes, context: CheckContext) -> CheckResult:
        return await asyncio.to_thread(self._run_sync, image_bytes, context)

    def _run_sync(self, image_bytes: bytes, context: CheckContext) -> CheckResult:
        doc_type = context.get("doc_type")
        country = context.get("country")
        fields = checksum_fields(doc_type, country)
        if not fields:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True)

        # Gemini-extracted fields (set by GeminiExtractCheck) are more reliable than OCR regex.
        # Fall back to OCR only when extraction didn't run or didn't find the value.
        extracted = context.get("extracted_fields", {})
        ocr_text: str | None = None

        # A field we could not read is not a failed check digit. CHECKSUM_FAIL
        # forces human review, so it must mean "read the number, digit wrong",
        # never "could not find the number" (blurry photo, or a document
        # misclassified as a type that carries a check digit).
        failures: list[str] = []
        unread: list[str] = []
        validated: list[str] = []
        for field in fields:
            algorithm = field["checksum"]
            fn = _ALGORITHMS.get(algorithm)
            if fn is None:
                continue  # unsupported algorithm: nothing was checked

            # Prefer extracted value keyed by algorithm name (e.g. "iban" → extracted["iban"])
            value = extracted.get(algorithm)

            if value is None:
                # Fall back to OCR regex
                regex = field.get("regex")
                if regex:
                    if ocr_text is None:
                        ocr_text = _ocr_text(image_bytes)
                    if ocr_text:
                        m = re.search(regex, ocr_text)
                        value = m.group(0) if m else None

            if value is None:
                unread.append(field["name"])
                continue

            if fn(str(value)):
                validated.append(field["name"])
            else:
                failures.append(field["name"])

        if not failures and not validated:
            return CheckResult(
                check=self.check_id,
                passed=True,
                confidence=0.0,
                skipped=True,
                signals={"reason": "check-digit fields not found", "unread_fields": unread},
            )

        if failures:
            return CheckResult(
                check=self.check_id,
                passed=False,
                fake_score=1.0,
                confidence=0.9,
                flags=["CHECKSUM_FAIL"],
                signals={"failed_fields": failures, "unread_fields": unread},
                normalized_signals=NormalizedSignals(
                    category="document_authenticity",
                    confidence=0.9,
                    indicators=["CHECKSUM_FAIL"],
                    document_type=doc_type,
                    country_code=country,
                    manipulation_score=1.0,
                ),
            )
        return CheckResult(
            check=self.check_id,
            passed=True,
            fake_score=0.0,
            confidence=0.9,
            normalized_signals=NormalizedSignals(
                category="document_authenticity",
                confidence=0.9,
                indicators=["CHECKSUM_VALID"],
                document_type=doc_type,
                country_code=country,
                manipulation_score=0.0,
            ),
        )
