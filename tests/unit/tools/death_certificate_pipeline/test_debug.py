from __future__ import annotations

from tools.death_certificate_pipeline.debug import build_verification_debug_event
from tools.fake_image_detector.models import CheckResult, Escalation, ToolResult, Verdict


def test_build_verification_debug_event_includes_bounded_authenticity_details() -> None:
    authenticity = ToolResult(
        verdict=Verdict.FLAG,
        risk_score=0.2,
        escalation=Escalation.HUMAN_REVIEW,
        early_exit=True,
        early_exit_reason="ocr_document runtime error",
        checks=[
            CheckResult(
                check="ocr_document",
                passed=False,
                skipped=False,
                fake_score=0.2,
                confidence=1.0,
                flags=["CHECK_RUNTIME_ERROR"],
                human_escalate=False,
                escalation_reasons=[],
                error="provider unavailable",
                signals={"raw_document_text": "must not leak"},
            )
        ],
    )

    event = build_verification_debug_event(
        {
            "status": "verified",
            "score": 92,
            "band": "escalate",
            "handed_off": False,
            "flags": ["HARD_ESCALATION"],
            "extracted_fields": {"full_name": "Jane Doe"},
            "summary": "Needs review.",
        },
        accepted=False,
        authenticity=authenticity,
    )

    assert event["authenticity"] == {
        "verdict": "FLAG",
        "risk_score": 0.2,
        "escalation": "HUMAN_REVIEW",
        "early_exit": True,
        "early_exit_reason": "ocr_document runtime error",
        "checks": [
            {
                "check": "ocr_document",
                "status": "error: provider unavailable",
                "passed": False,
                "skipped": False,
                "fake_score": 0.2,
                "confidence": 1.0,
                "flags": ["CHECK_RUNTIME_ERROR"],
                "human_escalate": False,
                "escalation_reasons": [],
                "error": "provider unavailable",
            }
        ],
    }
    assert "signals" not in event["authenticity"]["checks"][0]
    assert "normalized_signals" not in event["authenticity"]["checks"][0]


def test_build_verification_debug_event_omits_authenticity_when_unavailable() -> None:
    event = build_verification_debug_event(
        {"status": "no_document", "handed_off": False, "summary": "Send a document."},
        accepted=None,
        authenticity=None,
    )

    assert "authenticity" not in event


def test_each_check_says_plainly_whether_it_ran():
    """Six identical "skipped: true" lines used to hide three different situations."""
    from tools.death_certificate_pipeline.debug import check_status
    from tools.fake_image_detector.models import CheckResult

    def status(**kw):
        return check_status(CheckResult(check="c", **kw))

    assert status(passed=True, fake_score=0.1, confidence=0.9) == "passed"
    assert status(passed=False, fake_score=0.8, confidence=0.9) == "flagged"
    assert status(passed=True, skipped=True, signals={"doc_type": "death_certificate"}) == (
        "ran (information only, not scored)"
    )
    assert status(passed=True, skipped=True) == "not applicable to this document"
    assert status(passed=True, skipped=True, signals={"reason": "no access"}) == "unavailable: no access"
    # An unavailable check reports its own error, not a guess at the cause.
    timed_out = status(passed=True, skipped=True, flags=["REVERSE_SEARCH_UNAVAILABLE", "CHECK_TIMEOUT"],
                       signals={"error": "Google Vision reverse search timed out"})
    assert timed_out == "unavailable: Google Vision reverse search timed out"
    switched_off = status(passed=True, skipped=True, flags=["REVERSE_SEARCH_UNAVAILABLE"],
                          signals={"errors": ["503 unavailable", "403 Vision API has not been used"]})
    assert switched_off == "unavailable: 403 Vision API has not been used"
    assert status(passed=True, skipped=True, flags=["REVERSE_SEARCH_UNAVAILABLE"]) == "unavailable"
