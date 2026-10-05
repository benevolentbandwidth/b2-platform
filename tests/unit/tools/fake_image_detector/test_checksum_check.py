import sys
from unittest.mock import MagicMock, patch

import pytest

from tools.fake_image_detector.checks.checksum_check import ChecksumCheck


def _make_schemas(checksum_value="iban"):
    return {
        "schemas": {
            "bank_statement": {
                "base": {
                    "required_fields": [
                        {
                            "name": "IBAN",
                            "regex": "[A-Z]{2}[0-9]{2}[A-Z0-9]{4,30}",
                            "checksum": checksum_value,
                        }
                    ]
                }
            }
        }
    }


def _make_ocr_mock(return_text: str):
    """Return a context manager that injects mock pytesseract + PIL into sys.modules."""
    mock_pytesseract = MagicMock()
    mock_pytesseract.image_to_string.return_value = return_text

    mock_image = MagicMock()
    mock_pil = MagicMock()
    mock_pil.Image.open.return_value = mock_image

    return patch.dict(
        sys.modules,
        {
            "pytesseract": mock_pytesseract,
            "PIL": mock_pil,
            "PIL.Image": mock_pil.Image,
        },
    )


@pytest.fixture()
def check():
    return ChecksumCheck()


class TestChecksumCheckSkip:
    def test_skips_when_no_doc_type_in_context(self, check):
        import asyncio
        result = asyncio.run(check.run(b"", {}))
        assert result.skipped is True

    def test_skips_when_schema_not_found(self, check):
        import asyncio
        with patch(
            "tools.fake_image_detector.checks.checksum_check._get_schemas",
            return_value={"schemas": {}},
        ):
            result = asyncio.run(
                check.run(b"", {"doc_type": "bank_statement"})
            )
        assert result.skipped is True

    def test_skips_when_no_checksum_fields(self, check):
        import asyncio
        schemas = {
            "schemas": {
                "passport": {
                    "base": {
                        "required_fields": [
                            {"name": "Document Number", "regex": "[A-Z0-9]{6,9}", "checksum": None}
                        ]
                    }
                }
            }
        }
        with patch(
            "tools.fake_image_detector.checks.checksum_check._get_schemas",
            return_value=schemas,
        ):
            result = asyncio.run(
                check.run(b"", {"doc_type": "passport"})
            )
        assert result.skipped is True


class TestChecksumCheckIban:
    def _run(self, check, ocr_text, doc_type="bank_statement"):
        import asyncio
        with patch(
            "tools.fake_image_detector.checks.checksum_check._get_schemas",
            return_value=_make_schemas(),
        ), _make_ocr_mock(ocr_text):
            return asyncio.run(
                check.run(b"fake", {"doc_type": doc_type})
            )

    def test_passes_valid_iban(self, check):
        result = self._run(check, "IBAN: DE89370400440532013000")
        assert result.passed is True
        assert result.fake_score == 0.0
        assert result.skipped is False
        assert result.normalized_signals is not None
        assert result.normalized_signals.category == "document_authenticity"
        assert result.normalized_signals.indicators == ["CHECKSUM_VALID"]

    def test_fails_invalid_iban(self, check):
        result = self._run(check, "IBAN: DE89370400440532013001")
        assert result.passed is False
        assert result.fake_score == 1.0
        assert "CHECKSUM_FAIL" in result.flags
        assert "IBAN" in result.signals["failed_fields"]
        assert result.normalized_signals is not None
        assert result.normalized_signals.category == "document_authenticity"
        assert result.normalized_signals.indicators == ["CHECKSUM_FAIL"]

    def test_skips_when_iban_not_found_in_text(self, check):
        """Not finding the number is not a failed check digit.

        CHECKSUM_FAIL forces human review, so an unreadable or absent field
        must skip rather than fail.
        """
        result = self._run(check, "No IBAN present here")
        assert result.skipped is True
        assert "CHECKSUM_FAIL" not in result.flags
        assert result.signals["unread_fields"] == ["IBAN"]
