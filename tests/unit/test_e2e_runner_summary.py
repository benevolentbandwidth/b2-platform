from __future__ import annotations

from pathlib import Path

from e2e import runner
from e2e.cases import Case


def test_run_case_adds_summary_tool_response_without_replacing_tool_summary(monkeypatch, tmp_path: Path) -> None:
    case = Case(
        country="example",
        kind="real",
        case_id="case_001",
        path=tmp_path / "case_001",
        narrative="Please verify this certificate.",
        certificate_path=None,
        expected={"expected_outcome": "accept"},
    )
    captured = {}
    tool_result = {
        "tool": "death_certificate_verification",
        "accepted": True,
        "summary": "Original verification summary.",
    }

    def fake_post_turn(client, base_url, payload, headers, label):
        return {
            "label": label,
            "status_code": 200,
            "latency_seconds": 0.1,
            "response_json": {
                "response": "verification passed",
                "debug": {"tool_results": [tool_result]},
            },
            "response_text": "verification passed",
        }

    def fake_summarize(result):
        captured["result"] = dict(result)
        return "summary from summary tool"

    monkeypatch.setattr(runner, "post_turn", fake_post_turn)
    monkeypatch.setattr(runner, "summarize_case_result", fake_summarize)

    result = runner.run_case(
        object(),
        case,
        base_url="http://testserver",
        secret="",
        local_mode=True,
        debug_tools=False,
    )

    assert result["passed"] is True
    assert result["classification"] == "TP"
    assert result["tool_result"]["summary"] == "Original verification summary."
    assert result["summary_tool_response"] == "summary from summary tool"
    assert result["final_response"] == "verification passed"
    assert captured["result"]["tool_result"]["summary"] == "Original verification summary."
    assert "summary_tool_response" not in captured["result"]


def test_local_store_refuses_what_production_refuses() -> None:
    """The e2e harness must not accept files production would drop."""
    from e2e.local_app import InMemoryStore
    from src.session_store import MAX_MEDIA_BYTES

    store = InMemoryStore()
    assert store.save_media("s", b"x" * MAX_MEDIA_BYTES, mime_type="image/png") is True
    assert store.save_media("s", b"x" * (MAX_MEDIA_BYTES + 1), mime_type="image/png") is False
