import asyncio
import sys
from unittest.mock import MagicMock

from tools.fake_image_detector.checks.mrz_check import MRZCheck


def run(coro):
    return asyncio.run(coro)


def _install_mock_passporteye(mrz_dict: dict | None) -> None:
    mock_mrz = None
    if mrz_dict is not None:
        mock_mrz = MagicMock()
        mock_mrz.to_dict.return_value = mrz_dict

    mock_module = MagicMock()
    mock_module.read_mrz.return_value = mock_mrz
    sys.modules["passporteye"] = mock_module


def _remove_mock_passporteye() -> None:
    sys.modules.pop("passporteye", None)


class TestMRZCheck:
    def test_check_id(self):
        assert MRZCheck.check_id == "mrz"

    def test_skips_when_passporteye_not_installed(self):
        sys.modules["passporteye"] = None  # type: ignore[assignment]
        try:
            result = run(MRZCheck().run(b"img", {}))
        finally:
            sys.modules.pop("passporteye", None)

        assert result.skipped is True
        assert result.passed is True
        assert result.confidence == 0.0
        assert result.error is not None

    def test_skips_when_no_mrz_detected(self):
        _install_mock_passporteye(None)
        try:
            result = run(MRZCheck().run(b"img", {}))
        finally:
            _remove_mock_passporteye()

        assert result.skipped is True
        assert result.passed is True
        assert result.confidence == 0.0

    def test_passes_with_full_valid_score(self):
        _install_mock_passporteye({
            "valid_score": 100,
            "type": "P",
            "country": "DEU",
        })
        try:
            result = run(MRZCheck().run(b"img", {}))
        finally:
            _remove_mock_passporteye()

        assert result.passed is True
        assert result.skipped is False
        assert result.fake_score == 0.0
        assert result.confidence == 1.0
        assert result.signals["type"] == "P"
        assert result.signals["country"] == "DEU"
        assert result.normalized_signals is not None
        assert result.normalized_signals.category == "document_authenticity"
        assert "MRZ_VALID" in result.normalized_signals.indicators
        assert result.normalized_signals.document_type == "P"
        assert result.normalized_signals.country_code == "DEU"

    def test_skips_on_partial_mrz_read(self):
        _install_mock_passporteye({
            "valid_score": 60,
            "type": "P",
            "country": "KEN",
        })
        try:
            result = run(MRZCheck().run(b"img", {}))
        finally:
            _remove_mock_passporteye()

        assert result.skipped is True
        assert result.passed is True
        assert result.confidence == 0.0
        assert result.signals["valid_score"] == 60
        assert result.signals["type"] == "P"
        assert result.normalized_signals is not None
        assert "MRZ_PARTIAL_READ" in result.normalized_signals.indicators

    def test_skips_when_read_mrz_raises(self):
        mock_module = MagicMock()
        mock_module.read_mrz.side_effect = RuntimeError("corrupt scan")
        sys.modules["passporteye"] = mock_module
        try:
            result = run(MRZCheck().run(b"bad", {}))
        finally:
            _remove_mock_passporteye()

        assert result.skipped is True
        assert result.passed is True
        assert result.confidence == 0.0
        assert result.error is not None


def test_not_run_on_documents_without_an_mrz():
    """Death certificates have no MRZ; the reader used to mis-read their text as one."""
    called = []
    sys.modules["passporteye"] = type("P", (), {"read_mrz": lambda *a, **k: called.append(1)})
    try:
        result = run(MRZCheck().run(b"img", {"doc_type": "death_certificate"}))
    finally:
        _remove_mock_passporteye()

    assert result.skipped is True
    assert result.signals == {}
    assert called == []
