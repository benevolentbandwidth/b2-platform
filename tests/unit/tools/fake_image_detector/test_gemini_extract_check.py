import asyncio
import io
import json
import os
import sys
from unittest.mock import MagicMock

import pytest
from PIL import Image

from tools.fake_image_detector.checks import checksum_check
from tools.fake_image_detector.checks import gemini_extract_check as extract_module
from tools.fake_image_detector.checks.gemini_extract_check import GeminiExtractCheck


@pytest.fixture(autouse=True)
def _a_check_digit_field_needs_the_readings(monkeypatch):
    """Extraction only runs when a check-digit field will use its readings. The
    shipped schemas define one only for bank statements, so these tests stand
    one in for every document type."""
    monkeypatch.setattr(
        extract_module, "checksum_fields", lambda doc_type, country: [{"name": "N", "checksum": "luhn"}]
    )


def run(coro):
    return asyncio.run(coro)


def _jpeg_bytes(width=64, height=64) -> bytes:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _install_mock_genai(response_text: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.text = response_text

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    mock_genai = MagicMock()
    mock_genai.Client.return_value = mock_client

    sys.modules["google.genai"] = mock_genai
    sys.modules["google.genai.types"] = MagicMock()
    return mock_client


def _remove_mock_genai() -> None:
    sys.modules.pop("google.genai", None)
    sys.modules.pop("google.genai.types", None)


class TestGeminiExtractCheckSkip:
    def test_check_id(self):
        assert GeminiExtractCheck.check_id == "gemini_extract"

    def test_skips_when_no_doc_type(self):
        result = run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), {}))
        assert result.skipped is True
        assert result.passed is True
        assert result.confidence == 0.0

    def test_skips_when_doc_type_has_no_field_list(self):
        result = run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), {"doc_type": "unknown_type"}))
        assert result.skipped is True

    def test_skips_when_project_not_set(self):
        env_backup = os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
        sys.modules["google.genai"] = MagicMock()
        sys.modules["google.genai.types"] = MagicMock()
        try:
            result = run(GeminiExtractCheck(project=None).run(_jpeg_bytes(), {"doc_type": "passport"}))
        finally:
            _remove_mock_genai()
            if env_backup is not None:
                os.environ["GOOGLE_CLOUD_PROJECT"] = env_backup
        assert result.skipped is True
        assert "GOOGLE_CLOUD_PROJECT" in result.error

    def test_skips_on_api_error(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("quota exceeded")
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client
        sys.modules["google.genai"] = mock_genai
        sys.modules["google.genai.types"] = MagicMock()
        try:
            result = run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), {"doc_type": "passport"}))
        finally:
            _remove_mock_genai()
        assert result.skipped is True
        assert "quota exceeded" in result.error

    def test_skips_on_json_parse_error(self):
        _install_mock_genai("Sorry, I cannot extract fields from this image.")
        try:
            result = run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), {"doc_type": "passport"}))
        finally:
            _remove_mock_genai()
        assert result.skipped is True
        assert "JSON parse error" in result.error


class TestGeminiExtractCheckExtraction:
    def test_extracts_passport_fields_and_stores_in_context(self):
        payload = json.dumps({
            "full_name": "Max Mustermann",
            "document_number": "C01X00T47",
            "date_of_birth": "1985-03-15",
            "expiry_date": "2030-03-14",
            "nationality": "DEU",
        })
        _install_mock_genai(payload)
        context = {"doc_type": "passport"}
        try:
            result = run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), context))
        finally:
            _remove_mock_genai()

        assert result.skipped is True
        assert result.passed is True
        extracted = context.get("extracted_fields", {})
        assert extracted["full_name"] == "Max Mustermann"
        assert extracted["document_number"] == "C01X00T47"
        assert result.signals["extracted"]["date_of_birth"] == "1985-03-15"

    def test_null_values_excluded_from_extracted_fields(self):
        payload = json.dumps({
            "full_name": "Anna Schmidt",
            "document_number": None,
            "date_of_birth": "1990-07-22",
            "expiry_date": None,
            "nationality": "DEU",
        })
        _install_mock_genai(payload)
        context = {"doc_type": "passport"}
        try:
            run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), context))
        finally:
            _remove_mock_genai()

        extracted = context.get("extracted_fields", {})
        assert "document_number" not in extracted
        assert "expiry_date" not in extracted
        assert extracted["full_name"] == "Anna Schmidt"

    def test_extracts_bank_statement_iban(self):
        payload = json.dumps({
            "account_holder": "Max Mustermann",
            "iban": "DE89370400440532013000",
            "account_number": "0532013000",
        })
        _install_mock_genai(payload)
        context = {"doc_type": "bank_statement"}
        try:
            run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), context))
        finally:
            _remove_mock_genai()

        assert context["extracted_fields"]["iban"] == "DE89370400440532013000"

    def test_extracts_birth_certificate_fields(self):
        payload = json.dumps({
            "full_name": "Emma Müller",
            "date_of_birth": "2000-01-10",
            "place_of_birth": "Berlin",
            "signed_by": "Standesamt Berlin",
        })
        _install_mock_genai(payload)
        context = {"doc_type": "birth_certificate"}
        try:
            result = run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), context))
        finally:
            _remove_mock_genai()

        extracted = context.get("extracted_fields", {})
        assert extracted["signed_by"] == "Standesamt Berlin"
        assert result.signals["doc_type"] == "birth_certificate"

    def test_markdown_wrapped_json_is_parsed(self):
        payload = json.dumps({"full_name": "Test User", "date_of_death": "2024-05-01", "place_of_death": "Hamburg", "signed_by": "Amt"})
        _install_mock_genai(f"```json\n{payload}\n```")
        context = {"doc_type": "death_certificate"}
        try:
            result = run(GeminiExtractCheck(project="test").run(_jpeg_bytes(), context))
        finally:
            _remove_mock_genai()

        assert result.skipped is True
        assert context["extracted_fields"]["full_name"] == "Test User"


def test_a_hung_extraction_times_out_and_skips(monkeypatch):
    """No limit used to mean a hung Gemini call stalled the whole verification."""
    import time

    from tools.fake_image_detector.checks.gemini_extract_check import GeminiExtractCheck

    check = GeminiExtractCheck(params={"timeout_seconds": 0.05}, project="p")
    monkeypatch.setattr(check, "_run_sync", lambda image_bytes, context: time.sleep(0.5))

    import asyncio

    result = asyncio.run(check.run(b"img", {"doc_type": "death_certificate"}))

    assert result.skipped is True
    assert "timed out" in result.signals["reason"]


def test_death_certificates_are_not_extracted_since_nothing_uses_the_readings(monkeypatch):
    """Their readings fed only the check-digit check, which has no fields for a
    death certificate: a paid call whose answer was thrown away, ahead of the
    fraud check."""
    monkeypatch.setattr(extract_module, "checksum_fields", checksum_check.checksum_fields)
    mock_client = _install_mock_genai("{}")
    try:
        result = run(GeminiExtractCheck(project="test").run(
            _jpeg_bytes(), {"doc_type": "death_certificate", "country": "ID"}
        ))
    finally:
        _remove_mock_genai()

    assert result.skipped is True
    mock_client.models.generate_content.assert_not_called()
