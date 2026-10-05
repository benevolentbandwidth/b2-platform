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


def test_uses_the_model_from_the_scoring_settings(mock_gemini):
    from tools.death_certificate_pipeline.config_loader import default_scoring_config

    expected = default_scoring_config().consistency_model
    result = analyze_death_certificate_consistency(CHAT_HISTORY, FAKE_IMAGE)

    assert result["model"] == expected
    mock_gemini.models.generate_content.assert_called_once()
    assert mock_gemini.models.generate_content.call_args.kwargs["model"] == expected


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


def test_pdf_is_sent_to_gemini_as_pdf(mock_gemini):
    """A PDF certificate must reach Gemini labelled as a PDF, not as a JPEG."""
    analyze_death_certificate_consistency(CHAT_HISTORY, b"%PDF-1.7\n rest of pdf")

    types = sys.modules["google.genai.types"]
    assert types.Part.from_bytes.call_args.kwargs["mime_type"] == "application/pdf"


def test_tiff_is_sent_to_gemini_as_png(mock_gemini):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buf, format="TIFF")
    analyze_death_certificate_consistency(CHAT_HISTORY, buf.getvalue())

    kwargs = sys.modules["google.genai.types"].Part.from_bytes.call_args.kwargs
    assert kwargs["mime_type"] == "image/png"
    assert kwargs["data"].startswith(b"\x89PNG")


def test_gemini_is_asked_whether_the_claimant_gave_an_account(mock_gemini):
    """Chat history always contains the assistant's own lines, so 'no story' can't
    be detected by emptiness; Gemini judges the claimant's messages instead."""
    analyze_death_certificate_consistency(CHAT_HISTORY, FAKE_IMAGE)

    call = mock_gemini.models.generate_content.call_args.kwargs
    prompt = call["contents"][1]
    assert "claimant_account_present" in prompt
    assert 'lines starting "user:"' in prompt
    assert "claimant_account_present" in call["config"].responseSchema["required"]


def test_no_claimant_account_is_reported(mock_gemini):
    mock_gemini.models.generate_content.return_value.parsed = {
        **MOCK_PARSED, "claimant_account_present": False, "consistency_score": 0.0,
    }
    result = analyze_death_certificate_consistency(CHAT_HISTORY, FAKE_IMAGE)

    assert result["claimant_account_present"] is False
    assert result["certificate"]["full_name"] == "Jane Doe"  # still extracted


def test_prompt_states_todays_date(mock_gemini, monkeypatch):
    """"My father died last month" can only be checked if Gemini knows when now is."""
    from tools.death_certificate_pipeline import death_certificate_consistency as module

    monkeypatch.setattr(module, "today", lambda: "2026-10-04")
    analyze_death_certificate_consistency(CHAT_HISTORY, FAKE_IMAGE)

    prompt = mock_gemini.models.generate_content.call_args.kwargs["contents"][1]
    assert "Today's date is 2026-10-04" in prompt


def test_claimant_details_and_case_note_come_back_from_the_same_call(mock_gemini):
    """No extra Gemini call: the story check already reads the whole chat."""
    mock_gemini.models.generate_content.return_value.parsed = {
        **MOCK_PARSED,
        "claimant": {"relationship_to_deceased": "sister", "dependants": ["two children"], "other_details": []},
        "case_note": "The deceased's sister is asking for help for his two children.",
    }
    result = analyze_death_certificate_consistency(CHAT_HISTORY, FAKE_IMAGE)

    assert result["claimant"]["relationship_to_deceased"] == "sister"
    assert result["case_note"].startswith("The deceased's sister")
    call = mock_gemini.models.generate_content.call_args.kwargs
    assert {"claimant", "case_note"} <= set(call["config"].responseSchema["required"])
    assert "Do not judge eligibility or authenticity" in call["contents"][1]
    assert mock_gemini.models.generate_content.call_count == 1


def test_a_stalled_story_call_is_retried(mock_gemini, monkeypatch):
    """Four story calls stalled past their limit in one run and sent cases to
    review; a stalled attempt is now retried within the stage's budget."""
    from tools.fake_image_detector import google_clients

    monkeypatch.setattr(google_clients, "RETRY_BACKOFF_SECONDS", (0.0,))
    ok = mock_gemini.models.generate_content.return_value
    mock_gemini.models.generate_content.side_effect = [TimeoutError("stalled"), ok]

    result = analyze_death_certificate_consistency(CHAT_HISTORY, FAKE_IMAGE, attempts=2)

    assert mock_gemini.models.generate_content.call_count == 2
    assert 0.0 <= result["consistency_score"] <= 1.0


def test_the_last_failed_attempt_raises(mock_gemini, monkeypatch):
    import pytest
    from tools.fake_image_detector import google_clients

    monkeypatch.setattr(google_clients, "RETRY_BACKOFF_SECONDS", (0.0,))
    mock_gemini.models.generate_content.side_effect = TimeoutError("stalled")

    with pytest.raises(TimeoutError):
        analyze_death_certificate_consistency(CHAT_HISTORY, FAKE_IMAGE, attempts=2)
    assert mock_gemini.models.generate_content.call_count == 2
