from __future__ import annotations

from src import summary_response


def test_summary_payload_uses_verification_summary_without_tool_framing() -> None:
    payload = summary_response.summary_payload(
        "The request was accepted.",
        [
            {
                "tool": "death_certificate_verification",
                "accepted": True,
                "handed_off": True,
                "summary": "Verification passed.",
                "flags": [],
                "authenticity": {
                    "verdict": "PASS",
                    "risk_score": 0.0,
                    "checks": [],
                },
            }
        ],
    )

    assert payload == {
        "assistant_response": "The request was accepted.",
        "death_certificate_verification": {
            "status": None,
            "accepted": True,
            "handed_off": True,
            "summary": "Verification passed.",
            "flags": [],
        },
    }


def test_summary_prompt_excludes_internal_e2e_language() -> None:
    prompt = summary_response.summary_prompt(
        "The request was accepted.",
        [
            {
                "tool": "death_certificate_verification",
                "accepted": True,
                "summary": "Verification passed.",
                "flags": [],
            }
        ],
    )

    assert "WhatsApp-ready" in prompt
    assert "Do not mention internal tools" in prompt
    assert "e2e test case result" not in prompt
    assert "debug output" in prompt


def test_generate_summary_tool_response_returns_none_on_error(monkeypatch) -> None:
    async def fake_call(prompt):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(summary_response, "call_summary_model", fake_call)

    result = summary_response.generate_summary_tool_response(
        final_response="The request was accepted.",
        tool_events=[],
    )

    import asyncio

    assert asyncio.run(result) is None


class _RecordingAgent:
    """Stands in for the Gemini agent; keeps the prompt it was given."""

    prompts: list[str] = []

    def __init__(self, model, model_settings=None):
        pass

    async def run(self, prompt):
        _RecordingAgent.prompts.append(prompt)

        class Result:
            output = "summary text"

        return Result()


def _summary(monkeypatch, tool_events):
    import asyncio

    _RecordingAgent.prompts = []
    monkeypatch.setattr(summary_response, "Agent", _RecordingAgent)
    monkeypatch.setattr(summary_response, "_summary_model", lambda: "test-model")
    result = asyncio.run(
        summary_response.generate_summary_tool_response(
            final_response="Your certificate was verified.", tool_events=tool_events
        )
    )
    return result, _RecordingAgent.prompts[0]


def test_summary_is_produced_end_to_end(monkeypatch) -> None:
    """Regression: an argument mismatch made every summary crash, and the crash
    was swallowed, so no claimant received one."""
    result, _ = _summary(monkeypatch, [])
    assert result == "summary text"


def test_summary_includes_case_reference_from_the_verification_event(monkeypatch) -> None:
    events = [{"tool": "death_certificate_verification", "case_reference": "DC-7F3A-9C21-B4E8"}]
    _, prompt = _summary(monkeypatch, events)
    assert "DC-7F3A-9C21-B4E8" in prompt


def test_summary_without_a_case_has_no_reference_line(monkeypatch) -> None:
    _, prompt = _summary(monkeypatch, [{"tool": "death_certificate_verification"}])
    assert "case reference" not in prompt.lower()



def test_the_message_prompt_reaches_gemini_directly(monkeypatch) -> None:
    """Regression: routed through the conversation summariser, its "summarise this
    conversation" instruction won and claimants got a summary of the instructions."""
    _, prompt = _summary(monkeypatch, [])
    assert prompt.startswith("Write a WhatsApp-ready message")
    assert "Summarize the following conversation" not in prompt



def test_summary_model_uses_vertex_credentials_not_an_api_key(monkeypatch) -> None:
    """Regression: with GEMINI_API_KEY set, the shortcut model string sent the key
    to Vertex, which rejects keys, and every summary silently came back empty."""
    built = {}

    class FakeProvider:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setenv("GEMINI_API_KEY", "a-key-that-must-be-ignored")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "b2-platform")
    monkeypatch.setattr(summary_response, "GoogleProvider", FakeProvider)
    monkeypatch.setattr(summary_response, "GoogleModel", lambda name, provider: (name, provider))

    summary_response._summary_model.cache_clear()
    try:
        summary_response._summary_model()
    finally:
        summary_response._summary_model.cache_clear()

    assert built["vertexai"] is True
    assert built["project"] == "b2-platform"
    assert "api_key" not in built


def test_dev_summary_helper_never_uses_the_session_id_as_a_reference(monkeypatch) -> None:
    """src/loop.py passed the session id (a phone number) positionally, which
    bound it as the case reference and put it into the summary prompt."""
    import asyncio
    import sys
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools"))
    import summary_tool

    seen = {}

    class FakeTool:
        def __init__(self, **kwargs):
            pass

        async def run(self, messages, case_reference=None):
            seen["case_reference"] = case_reference
            return "ok"

    monkeypatch.setattr(summary_tool, "ConversationSummaryTool", FakeTool)
    monkeypatch.setattr(summary_tool, "Agent", lambda model: None)
    monkeypatch.setattr(summary_tool, "PiiScrubber", lambda: None)
    monkeypatch.setattr(summary_tool, "PiiAuditStore", lambda: None)

    asyncio.run(summary_tool.handle([]))
    assert seen["case_reference"] is None
    asyncio.run(summary_tool.handle([], case_reference="DC-1111-2222-3333"))
    assert seen["case_reference"] == "DC-1111-2222-3333"
    sys.modules.pop("summary_tool", None)



def test_summary_model_comes_from_the_whatsapp_settings(monkeypatch, tmp_path) -> None:
    import pytest

    settings = tmp_path / "whatsapp.yaml"
    settings.write_text(
        "version: 1\nsummary:\n  model: some-newer-model\n  location: global\n  thinking_level: low\n"
    )
    monkeypatch.setattr(summary_response, "_SETTINGS_PATH", settings)
    summary_response.summary_settings.cache_clear()
    try:
        assert summary_response.summary_settings() == summary_response.SummarySettings(
            model="some-newer-model", location="global", thinking_level="LOW"
        )
        assert summary_response._summary_model_settings() == {
            "google_thinking_config": {"thinking_level": "LOW"}
        }

        settings.write_text("version: 1\nsummary:\n  model: m\n")
        summary_response.summary_settings.cache_clear()
        # Unset: $VERTEX_LOCATION and the model's own default thinking.
        assert summary_response.summary_settings().location is None
        assert summary_response._summary_model_settings() == {}

        settings.write_text("version: 1\nsummary:\n  model: m\n  thinking_level: none\n")
        summary_response.summary_settings.cache_clear()
        with pytest.raises(ValueError, match="thinking_level"):
            summary_response.summary_settings()

        settings.write_text("version: 1\nsummary: {}\n")
        summary_response.summary_settings.cache_clear()
        with pytest.raises(ValueError, match="summary.model"):
            summary_response.summary_settings()
    finally:
        summary_response.summary_settings.cache_clear()
