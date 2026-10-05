import asyncio
import io

from tools.fake_image_detector.checks.base_check import BaseCheck
from tools.fake_image_detector.config_loader import load_document_schemas
from tools.fake_image_detector.models import CheckResult, NormalizedSignals

_SCHEMAS = None


def _get_schemas() -> dict:
    global _SCHEMAS
    if _SCHEMAS is None:
        _SCHEMAS = load_document_schemas()
    return _SCHEMAS


class OCRDocumentCheck(BaseCheck):
    check_id = "ocr_document"

    async def run(self, image_bytes: bytes, context: dict) -> CheckResult:
        return await asyncio.to_thread(self._run_sync, image_bytes, context)

    def _run_sync(self, image_bytes: bytes, context: dict) -> CheckResult:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as e:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True, error=str(e))

        try:
            image = Image.open(io.BytesIO(image_bytes))
            text = pytesseract.image_to_string(image)
        except Exception as e:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True, error=str(e))

        text_lower = text.lower()
        schemas = _get_schemas()

        # A caller that already knows the document type (the death-certificate
        # pipeline) sets it; keyword detection only fills it in when missing and
        # must not overwrite it. Country detection runs either way.
        doc_type = context.get("doc_type") or self._detect_doc_type(
            text_lower, schemas.get("detection_keywords", {})
        )

        if doc_type is None:
            return CheckResult(check=self.check_id, passed=True, confidence=0.0, skipped=True)

        context["doc_type"] = doc_type

        if not context.get("country"):
            detected = self._detect_country(text_lower)
            if detected:
                context["country"] = detected

        # Detection complete — Gemini validates the document content
        return CheckResult(
            check=self.check_id,
            passed=True,
            confidence=0.0,
            skipped=True,
            signals={"doc_type": doc_type, "country": context.get("country")},
            normalized_signals=NormalizedSignals(
                category="document_authenticity",
                confidence=0.0,
                indicators=["DOCUMENT_DETECTED"],
                document_type=doc_type,
                country_code=context.get("country"),
            ),
        )

    def _detect_doc_type(self, text_lower: str, keywords: dict) -> str | None:
        for doc_type, kws in keywords.items():
            if any(kw.lower() in text_lower for kw in kws):
                return doc_type
        return None

    def _detect_country(self, text_lower: str) -> str | None:
        _COUNTRY_SIGNALS = {
            "DE": ["bundesrepublik deutschland", "germany", "deutschland"],
            "KE": ["republic of kenya", "kenya"],
            "NG": ["federal republic of nigeria", "nigeria"],
            "ID": ["republik indonesia", "indonesia"],
            # Moroccan civil documents carry French alongside Arabic; Tesseract
            # reads the French with its default language.
            "MA": ["royaume du maroc", "المملكة المغربية", "maroc", "morocco"],
        }
        for country, signals in _COUNTRY_SIGNALS.items():
            if any(s in text_lower for s in signals):
                return country
        return None
