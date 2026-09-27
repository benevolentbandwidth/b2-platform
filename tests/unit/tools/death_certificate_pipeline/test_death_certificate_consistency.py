import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tools.death_certificate_pipeline.death_certificate_consistency import analyze_death_certificate_consistency

CHAT_HISTORY = [
    {"role": "user", "content": "The person died on May 1, 2024 in Seattle."},
    {"role": "assistant", "content": "The family mentioned a cardiac event."},
]

FAKE_IMAGE = b"\x89PNG\r\n\x1a\n"

MOCK_PARSED = {
    "certificate": {
        "full_name": "Jane Doe",
        "date_of_death": "2024-05-01",
        "place_of_death": "Seattle",
        "age_at_death": 72,
        "cause_of_death": "cardiac arrest",
        "certificate_number": "DC-12345",
        "issuing_authority": "King County",
        "registration_date": "2024-05-03",
        "other_visible_details": {"marital_status": "married"},
    },
    "consistency_score": 0.91,
    "consistency_label": "high",
    "confidence": 0.88,
    "matches": ["date of death aligns with chat history"],
    "mismatches": [],
    "uncertain_points": ["cause of death is only partially visible"],
    "summary": "The chat history is consistent with the certificate.",
}


@pytest.fixture()
def mock_gemini(monkeypatch):
    response = MagicMock()
    response.parsed = MOCK_PARSED

    client = MagicMock()
    client.models.generate_content.return_value = response

    genai = MagicMock()
    genai.Client.return_value = client

    types = MagicMock()
    types.GenerateContentConfig.side_effect = lambda **kw: SimpleNamespace(**kw)

    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", types)
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")

    return client


def test_returns_expected_shape(mock_gemini):
    result = analyze_death_certificate_consistency(CHAT_HISTORY, FAKE_IMAGE)

    assert isinstance(result, dict)
    assert result.keys() >= {
        "certificate", "consistency_score", "consistency_label",
        "confidence", "matches", "mismatches", "uncertain_points",
        "summary", "model",
    }
    cert = result["certificate"]
    assert isinstance(cert, dict)
    assert cert.keys() >= {
        "full_name", "date_of_death", "place_of_death", "age_at_death",
        "cause_of_death", "certificate_number", "issuing_authority",
        "registration_date", "other_visible_details",
    }


def test_scores_clamped_between_0_and_1(mock_gemini):
    result = analyze_death_certificate_consistency(CHAT_HISTORY, FAKE_IMAGE)
    assert 0.0 <= result["consistency_score"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0


def test_uses_ga_gemini_default_model(mock_gemini):
    result = analyze_death_certificate_consistency(CHAT_HISTORY, FAKE_IMAGE)

    assert result["model"] == "gemini-2.5-flash"
    mock_gemini.models.generate_content.assert_called_once()
    assert mock_gemini.models.generate_content.call_args.kwargs["model"] == "gemini-2.5-flash"


def test_empty_chat_history_raises(mock_gemini):
    with pytest.raises(ValueError, match="chat_history"):
        analyze_death_certificate_consistency([], FAKE_IMAGE)


def test_empty_image_raises(mock_gemini):
    with pytest.raises(ValueError, match="image_bytes"):
        analyze_death_certificate_consistency(CHAT_HISTORY, b"")


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        analyze_death_certificate_consistency(CHAT_HISTORY, FAKE_IMAGE)
