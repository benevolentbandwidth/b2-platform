from __future__ import annotations

import json

from e2e.reports import render_markdown, write_reports


def test_render_markdown_includes_summary_tool_response_column() -> None:
    markdown = render_markdown(
        {
            "base_url": "http://testserver",
            "total": 1,
            "passed": 0,
            "failed": 1,
            "stats": {
                "overall": {
                    "total": 1,
                    "classified": 1,
                    "accuracy": 0.0,
                    "false_positives": 0,
                    "false_negatives": 1,
                    "unknown": 0,
                },
                "by_country": {},
            },
            "results": [
                {
                    "case": "example/real/case_001",
                    "passed": False,
                    "expected_outcome": "accept",
                    "actual_outcome": "reject",
                    "classification": "FN",
                    "final_response": "Rejected.",
                    "tool_result": {
                        "authenticity": {
                            "verdict": "FLAG",
                            "risk_score": 0.2,
                            "escalation": "HUMAN_REVIEW",
                            "early_exit": True,
                            "early_exit_reason": "ocr_document runtime error | unavailable",
                        }
                    },
                    "summary_tool_response": "Line one\nLine | two",
                    "errors": ["actual outcome 'reject' != expected 'accept'"],
                }
            ],
        }
    )

    assert (
        "| Case | Result | Expected | Actual | Class | Authenticity Verdict | Risk Score | "
        "Escalation | Early Exit | Early Exit Reason | Final Response Chars | "
        "Summary Tool Response | Errors |"
    ) in markdown
    assert "| Scope | Total | Classified | Accuracy | FP | FN |" in markdown
    assert "| --- | ---: | ---: | ---: | ---: | ---: |" in markdown
    assert "| `overall` | 1 | 1 | 0.0% | 0 | 1 |" in markdown
    assert "| Scope | Total | Classified | Accuracy | FP | FN | Unknown |" not in markdown
    assert (
        "| `example/real/case_001` | FAIL | accept | reject | FN | FLAG | 0.2 | "
        "HUMAN_REVIEW | Yes | ocr_document runtime error \\| unavailable | 9 | "
        "Line one<br>Line \\| two | actual outcome 'reject' != expected 'accept' |"
    ) in markdown


def test_render_markdown_handles_missing_summary_tool_response() -> None:
    markdown = render_markdown(
        {
            "base_url": "http://testserver",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "stats": {
                "overall": {
                    "total": 1,
                    "classified": 1,
                    "accuracy": 1.0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "unknown": 0,
                },
                "by_country": {},
            },
            "results": [
                {
                    "case": "example/real/case_001",
                    "passed": True,
                    "expected_outcome": "accept",
                    "actual_outcome": "accept",
                    "classification": "TP",
                    "final_response": "Accepted.",
                    "errors": [],
                }
            ],
        }
    )

    assert (
        "| `example/real/case_001` | PASS | accept | accept | TP | n/a | n/a | "
        "n/a | n/a | n/a | 9 |  |  |"
    ) in markdown


def test_write_reports_preserves_authenticity_diagnostics(tmp_path) -> None:
    authenticity = {
        "verdict": "FLAG",
        "risk_score": 0.2,
        "escalation": "HUMAN_REVIEW",
        "early_exit": True,
        "early_exit_reason": "ocr_document runtime error",
        "checks": [],
    }
    result = {
        "case": "example/real/case_001",
        "country": "example",
        "passed": False,
        "expected_outcome": "accept",
        "actual_outcome": "reject",
        "classification": "FN",
        "final_response": "Rejected.",
        "tool_result": {"authenticity": authenticity},
        "errors": ["rejected"],
    }

    json_path, _, _ = write_reports(tmp_path, [result], "http://testserver")
    written = json.loads(json_path.read_text(encoding="utf-8"))

    assert written["results"][0]["tool_result"]["authenticity"] == authenticity


def test_render_markdown_preserves_zero_risk_score() -> None:
    markdown = render_markdown(
        {
            "base_url": "http://testserver",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "stats": {
                "overall": {
                    "total": 1,
                    "classified": 1,
                    "accuracy": 1.0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "unknown": 0,
                },
                "by_country": {},
            },
            "results": [
                {
                    "case": "example/real/case_001",
                    "passed": True,
                    "expected_outcome": "accept",
                    "actual_outcome": "accept",
                    "classification": "TP",
                    "final_response": "Accepted.",
                    "tool_result": {
                        "authenticity": {
                            "verdict": "PASS",
                            "risk_score": 0.0,
                            "escalation": "AUTO_ACCEPT",
                            "early_exit": False,
                            "early_exit_reason": None,
                        }
                    },
                    "errors": [],
                }
            ],
        }
    )

    assert "| PASS | 0.0 | AUTO_ACCEPT | No |  |" in markdown
