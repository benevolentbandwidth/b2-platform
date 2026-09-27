import asyncio
import io
import sys
from unittest.mock import MagicMock

from PIL import Image

from tools.fake_image_detector.checks.ocr_document_check import OCRDocumentCheck


def run(coro):
    return asyncio.run(coro)


def _jpeg_bytes(width=64, height=64) -> bytes:
    img = Image.new("RGB", (width, height), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _install_mock_tesseract(extracted_text: str) -> None:
    mock_pytesseract = MagicMock()
    mock_pytesseract.image_to_string.return_value = extracted_text
    sys.modules["pytesseract"] = mock_pytesseract


def _remove_mock_tesseract() -> None:
    sys.modules.pop("pytesseract", None)


class TestOCRDocumentCheckSkip:
    def test_check_id(self):
        assert OCRDocumentCheck.check_id == "ocr_document"

    def test_skips_when_pytesseract_not_installed(self):
        sys.modules["pytesseract"] = None  # type: ignore[assignment]
        try:
            result = run(OCRDocumentCheck().run(_jpeg_bytes(), {}))
        finally:
            sys.modules.pop("pytesseract", None)

        assert result.skipped is True
        assert result.passed is True
        assert result.confidence == 0.0
        assert result.error is not None

    def test_skips_when_no_document_keywords_found(self):
        _install_mock_tesseract("hello world this is a random image")
        try:
            result = run(OCRDocumentCheck().run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_tesseract()

        assert result.skipped is True
        assert result.passed is True

    def test_skips_on_ocr_error(self):
        mock_pytesseract = MagicMock()
        mock_pytesseract.image_to_string.side_effect = RuntimeError("tesseract not found")
        sys.modules["pytesseract"] = mock_pytesseract
        try:
            result = run(OCRDocumentCheck().run(_jpeg_bytes(), {}))
        finally:
            _remove_mock_tesseract()

        assert result.skipped is True
        assert result.passed is True
        assert result.error is not None


class TestOCRDocumentCheckDetection:
    def test_detects_passport_from_keyword(self):
        _install_mock_tesseract("REPUBLIC OF KENYA\nPASSPORT\nName: John Doe")
        context: dict = {}
        try:
            result = run(OCRDocumentCheck().run(_jpeg_bytes(), context))
        finally:
            _remove_mock_tesseract()

        assert result.skipped is True
        assert result.passed is True
        assert context["doc_type"] == "passport"
        assert result.signals["doc_type"] == "passport"
        assert result.normalized_signals is not None
        assert result.normalized_signals.document_type == "passport"
        assert "DOCUMENT_DETECTED" in result.normalized_signals.indicators

    def test_detects_national_id_from_keyword(self):
        _install_mock_tesseract("IDENTITY CARD\nNational ID Number: 12345678")
        context: dict = {}
        try:
            run(OCRDocumentCheck().run(_jpeg_bytes(), context))
        finally:
            _remove_mock_tesseract()

        assert context["doc_type"] == "national_id"

    def test_detects_bank_statement_from_iban_keyword(self):
        _install_mock_tesseract("Bank Statement\nIBAN: DE89370400440532013000")
        context: dict = {}
        try:
            run(OCRDocumentCheck().run(_jpeg_bytes(), context))
        finally:
            _remove_mock_tesseract()

        assert context["doc_type"] == "bank_statement"

    def test_detects_germany_country_from_text(self):
        _install_mock_tesseract("REISEPASS\nBundesrepublik Deutschland\nName: Müller")
        context: dict = {}
        try:
            run(OCRDocumentCheck().run(_jpeg_bytes(), context))
        finally:
            _remove_mock_tesseract()

        assert context.get("doc_type") == "passport"
        assert context.get("country") == "DE"

    def test_detects_kenya_country_from_text(self):
        _install_mock_tesseract("PASSPORT\nRepublic of Kenya\nName: Kamau")
        context: dict = {}
        try:
            run(OCRDocumentCheck().run(_jpeg_bytes(), context))
        finally:
            _remove_mock_tesseract()

        assert context.get("country") == "KE"

    def test_does_not_overwrite_existing_country_in_context(self):
        _install_mock_tesseract("PASSPORT\nRepublic of Kenya\nName: Kamau")
        context: dict = {"country": "NG"}
        try:
            run(OCRDocumentCheck().run(_jpeg_bytes(), context))
        finally:
            _remove_mock_tesseract()

        assert context["country"] == "NG"

    def test_sets_doc_type_in_context_for_downstream_checks(self):
        _install_mock_tesseract("Birth Certificate\nName: Anna Schmidt\nDate of Birth: 01.01.2000")
        context: dict = {}
        try:
            run(OCRDocumentCheck().run(_jpeg_bytes(), context))
        finally:
            _remove_mock_tesseract()

        assert context["doc_type"] == "birth_certificate"
