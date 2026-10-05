"""Unit tests for the context-aware death-certificate verification tool."""

from types import SimpleNamespace

from tools.death_certificate_pipeline import verify as verify_module
from tools.death_certificate_pipeline.models import Band, ReliabilityResult
from tools.death_certificate_pipeline.pipeline import PipelineExecution
from tools.death_certificate_pipeline.verify import verify_death_certificate
from tools.fake_image_detector.models import Escalation, ToolResult, Verdict


class FakeStore:
    def __init__(self, media):
        self._media = media

    def load_latest_media(self, session_id):
        return self._media


def _ctx(store, history_text="my mother Jane Doe passed away", session_id="wa-123", debug_events=None):
    return SimpleNamespace(
        deps=SimpleNamespace(
            session_id=session_id,
            store=store,
            history_text=history_text,
            debug_events=debug_events,
        )
    )


def _result(band, flags=None):
    return ReliabilityResult(
        score=80 if band in (Band.HIGH, Band.MEDIUM) else 30,
        band=band,
        sub_scores={},
        weights={},
        flags=flags or [],
        justification="ok",
        extracted_fields={"full_name": "Jane Doe"},
    )


def _execution(band, flags=None):
    return PipelineExecution(
        result=_result(band, flags),
        authenticity=ToolResult(
            verdict=Verdict.PASS,
            risk_score=0.1,
            escalation=Escalation.AUTO_ACCEPT,
            checks=[],
        ),
    )


async def test_verify_accepts_and_hands_off(monkeypatch):
    seen = {}

    async def fake_pipeline(submission):
        seen["narrative"] = submission.narrative
        return _execution(Band.HIGH)

    async def fake_deliver(payload, image_bytes, mime_type):
        seen["payload"] = payload
        seen["handoff_image"] = image_bytes
        seen["handoff_mime"] = mime_type
        return True

    monkeypatch.setattr(verify_module, "run_pipeline_with_diagnostics", fake_pipeline)
    monkeypatch.setattr(verify_module, "deliver_to_gl", fake_deliver)

    result = await verify_death_certificate(_ctx(FakeStore((b"\xff\xd8jpeg", "image/jpeg"))))

    assert result["status"] == "verified"
    assert result["band"] == "high"
    assert result["handed_off"] is True
    # narrative came from the conversation history, not the model
    assert "Jane Doe" in seen["narrative"]
    # handoff payload carries a non-reversible contact id and the score
    assert seen["payload"]["score"] == 80
    assert seen["payload"]["contact_identifier"] and len(seen["payload"]["contact_identifier"]) == 64
    assert seen["handoff_image"] == b"\xff\xd8jpeg"
    assert seen["handoff_mime"] == "image/jpeg"


async def test_verify_sends_escalated_cases_to_givelight_for_review(monkeypatch):
    """There is no review queue here: GiveLight's reviewers are the human review,
    so an escalated case must reach them, marked as needing review."""
    delivered = {}

    async def fake_pipeline(submission):
        return _execution(Band.ESCALATE, flags=["HARD_ESCALATION"])

    async def fake_deliver(payload, image_bytes, mime_type):
        delivered["payload"] = payload
        return True

    monkeypatch.setattr(verify_module, "run_pipeline_with_diagnostics", fake_pipeline)
    monkeypatch.setattr(verify_module, "deliver_to_gl", fake_deliver)

    result = await verify_death_certificate(_ctx(FakeStore((b"\xff\xd8jpeg", "image/jpeg"))))

    sent = delivered["payload"]
    assert sent["decision"] == "needs_review"
    assert sent["review_reasons"], "a reviewer needs to know why"
    assert sent["case_reference"] == result["case_reference"]
    assert result["accepted"] is False
    assert result["handed_off"] is True
    assert result["case_reference"] in result["summary"]


async def test_verify_reports_no_document_when_store_empty(monkeypatch):
    async def fake_pipeline(submission):  # pragma: no cover - must not run
        raise AssertionError("pipeline should not run without media")

    monkeypatch.setattr(verify_module, "run_pipeline_with_diagnostics", fake_pipeline)

    result = await verify_death_certificate(_ctx(FakeStore(None), history_text=""))

    assert result["status"] == "no_document"
    assert result["handed_off"] is False


async def test_verify_appends_debug_event(monkeypatch):
    async def fake_pipeline(submission):
        return _execution(Band.MEDIUM)

    async def fake_deliver(payload, image_bytes, mime_type):
        return False

    monkeypatch.setattr(verify_module, "run_pipeline_with_diagnostics", fake_pipeline)
    monkeypatch.setattr(verify_module, "deliver_to_gl", fake_deliver)

    debug_events = []
    result = await verify_death_certificate(
        _ctx(FakeStore((b"\xff\xd8jpeg", "image/jpeg")), debug_events=debug_events)
    )

    assert result["status"] == "verified"
    assert debug_events == [
        {
            "tool": "death_certificate_verification",
            "status": "verified",
            "score": 80,
            "band": "medium",
            "config_versions": {},
            "decision": "accepted",
            "case_reference": None,
            "accepted": True,
            "handed_off": False,
            "flags": [],
            "extracted_fields": {"full_name": "Jane Doe"},
            "summary": result["summary"],
            "authenticity": {
                "verdict": "PASS",
                "risk_score": 0.1,
                "escalation": "AUTO_ACCEPT",
                "early_exit": False,
                "early_exit_reason": None,
                "checks": [],
            },
        }
    ]


async def test_handoff_carries_phone_but_logs_do_not(monkeypatch, caplog):
    """GiveLight gets the claimant's number to contact the family; logs get a hash.

    Anyone with log access on the project can read log lines, so the raw
    WhatsApp ID (a phone number) must never appear there.
    """
    phone = "6281234567890"
    seen = {}

    async def fake_pipeline(submission):
        return _execution(Band.HIGH)

    async def fake_deliver(payload, image_bytes, mime_type):
        seen["payload"] = payload
        return True

    monkeypatch.setattr(verify_module, "run_pipeline_with_diagnostics", fake_pipeline)
    monkeypatch.setattr(verify_module, "deliver_to_gl", fake_deliver)

    caplog.set_level("INFO", logger=verify_module.__name__)
    await verify_death_certificate(_ctx(FakeStore((b"img", "image/jpeg")), session_id=phone))

    assert seen["payload"]["contact_phone"] == phone
    assert phone[:8] not in caplog.text


async def test_failed_upload_asks_to_resend_and_gives_no_reference(monkeypatch):
    """GiveLight is the only place a case can land; if the upload fails, nothing
    else will pick it up, so the claimant must not be promised a handover."""

    async def fake_pipeline(submission):
        return _execution(Band.HIGH)

    async def fake_deliver(payload, image_bytes, mime_type):
        return False

    monkeypatch.setattr(verify_module, "run_pipeline_with_diagnostics", fake_pipeline)
    monkeypatch.setattr(verify_module, "deliver_to_gl", fake_deliver)

    result = await verify_death_certificate(_ctx(FakeStore((b"\xff\xd8jpeg", "image/jpeg"))))

    assert result["case_reference"] is None
    assert "send it again" in result["summary"]
    assert "manually" not in result["summary"]


async def test_delivered_case_reference_reaches_the_debug_event(monkeypatch):
    """The WhatsApp summary reads the reference from this event."""

    async def fake_pipeline(submission):
        return _execution(Band.HIGH)

    async def fake_deliver(payload, image_bytes, mime_type):
        return True

    monkeypatch.setattr(verify_module, "run_pipeline_with_diagnostics", fake_pipeline)
    monkeypatch.setattr(verify_module, "deliver_to_gl", fake_deliver)

    events = []
    result = await verify_death_certificate(
        _ctx(FakeStore((b"\xff\xd8jpeg", "image/jpeg")), debug_events=events)
    )

    assert result["case_reference"].startswith("DC-")
    assert events[0]["case_reference"] == result["case_reference"]


def _reasons(flags, consistency=0.5, authenticity=None):
    from tools.death_certificate_pipeline.config_loader import default_scoring_config

    result = ReliabilityResult(
        score=60, band=Band.ESCALATE, sub_scores={"consistency": consistency},
        weights={}, flags=flags, justification="j",
    )
    return verify_module._review_reasons(result, authenticity, default_scoring_config())


def test_review_reasons_explain_a_story_mismatch():
    (reason,) = _reasons(["CONSISTENCY_BELOW_MINIMUM"], consistency=0.18)
    assert "disagree" in reason and "0.18" in reason


def test_review_reasons_explain_an_unavailable_comparison():
    (reason,) = _reasons(["CONSISTENCY_UNAVAILABLE"])
    assert "could not be compared" in reason


def test_review_reasons_name_the_check_that_errored():
    """An outage must read as an outage, not as a suspicious document."""
    from tools.fake_image_detector.models import CheckResult

    auth = ToolResult(
        verdict=Verdict.FLAG, risk_score=0.2, escalation=Escalation.HUMAN_REVIEW, checks=[
            CheckResult(check="gemini_vision", passed=False, human_escalate=True,
                        flags=["CHECK_TIMEOUT"], error="Gemini vision check timed out"),
        ],
        early_exit=True, early_exit_reason="gemini_vision triggered hard escalation: []",
    )
    headline, detail = _reasons(["HARD_ESCALATION"], authenticity=auth)
    # An outage must not read as suspicion, starting with the headline.
    assert headline == "fraud check could not be completed, so the document was not fully checked"
    assert "flagged" not in headline
    assert "could not run (Gemini vision check timed out)" in detail
    assert "not fully checked" in detail


def test_review_reasons_explain_a_missing_account():
    (reason,) = _reasons(["NO_CLAIMANT_ACCOUNT"])
    assert "has not described what happened" in reason



def test_review_reasons_quote_what_gemini_objected_to():
    """A reviewer needs the claim itself to confirm or dismiss it in seconds,
    not an internal flag name."""
    from tools.fake_image_detector.models import CheckResult

    claim = "The issue date (14 August 2025) is in the future"
    auth = ToolResult(
        verdict=Verdict.FLAG, risk_score=0.7, escalation=Escalation.HUMAN_REVIEW, checks=[
            CheckResult(check="exif", passed=True, fake_score=0.0, confidence=1.0),
            CheckResult(check="gemini_vision", passed=False, fake_score=0.7, confidence=0.9,
                        flags=["INTERNAL_INCONSISTENCY"], signals={"signals": [claim]}),
        ],
        early_exit=True,
        early_exit_reason="gemini_vision triggered hard escalation: ['INTERNAL_INCONSISTENCY']",
    )

    reasons = _reasons(["HARD_ESCALATION"], authenticity=auth)

    assert reasons[0] == (
        "fraud check flagged it (risk 0.70): details on the certificate contradict each other"
    )
    assert f"gemini_vision noted: {claim}" in reasons
    assert not any("triggered hard escalation" in r for r in reasons)
    assert not any(r.startswith("exif") for r in reasons)  # passing checks stay out


async def test_givelight_gets_the_case_note_claimant_words_and_story_evidence(monkeypatch):
    seen = {}
    result = ReliabilityResult(
        score=95, band=Band.HIGH, sub_scores={}, weights={}, flags=[], justification="j",
        matches=["name matches"], mismatches=[], uncertain_points=[],
        story_summary="Fully consistent.",
        claimant={"relationship_to_deceased": "sister", "dependants": ["two children"], "other_details": []},
        case_note="His sister is caring for his two children.",
    )

    async def fake_pipeline(submission):
        return PipelineExecution(result=result, authenticity=ToolResult(
            verdict=Verdict.PASS, risk_score=0.1, escalation=Escalation.AUTO_ACCEPT, checks=[]))

    async def fake_deliver(payload, image_bytes, mime_type):
        seen["payload"] = payload
        return True

    monkeypatch.setattr(verify_module, "run_pipeline_with_diagnostics", fake_pipeline)
    monkeypatch.setattr(verify_module, "deliver_to_gl", fake_deliver)
    ctx = _ctx(FakeStore((b"\xff\xd8jpeg", "image/jpeg")))
    ctx.deps.claimant_messages = ["My brother died.", "I am his sister."]

    await verify_death_certificate(ctx)

    payload = seen["payload"]
    assert payload["case_note"] == "His sister is caring for his two children."
    assert payload["claimant"]["relationship_to_deceased"] == "sister"
    assert payload["claimant_messages"] == ["My brother died.", "I am his sister."]
    assert payload["story_check"] == {
        "summary": "Fully consistent.", "matches": ["name matches"],
        "mismatches": [], "uncertain_points": [],
    }



def test_review_reasons_say_where_the_image_was_found():
    from tools.fake_image_detector.models import CheckResult

    auth = ToolResult(
        verdict=Verdict.FLAG, risk_score=0.75, escalation=Escalation.HUMAN_REVIEW, checks=[
            CheckResult(check="reverse_image", passed=False, fake_score=0.75, confidence=0.7,
                        flags=["FOUND_ONLINE"], signals={"domains": ["example.com", "blog.test"]}),
        ],
        early_exit=True, early_exit_reason="reverse_image triggered hard escalation: ['FOUND_ONLINE']",
    )
    reasons = _reasons(["HARD_ESCALATION"], authenticity=auth)

    assert reasons[0].endswith("the document was found published online")
    assert "reverse_image found it on: example.com, blog.test" in reasons
