"""Stage-3 consistency: off-loop execution and bounded duration."""

import asyncio
import dataclasses
import time

import tools.death_certificate_pipeline.pipeline as pipeline_module
from tools.fake_image_detector import google_clients
from tools.death_certificate_pipeline.config_loader import default_scoring_config
from tools.death_certificate_pipeline.models import Submission

_JPEG = b"\xff\xd8\xff\xe0 fake jpeg body"


def _ok_result(**_kwargs):
    return {
        "consistency_score": 1.0,
        "consistency_label": "high",
        "confidence": 1.0,
        "certificate": {},
        "matches": [],
        "mismatches": [],
        "uncertain_points": [],
        "summary": "",
    }


async def test_consistency_call_does_not_block_the_event_loop(monkeypatch):
    """The analyzer is synchronous; it must not run on the event loop.

    Other in-flight cases on the same instance would otherwise stall for the
    full duration of every Gemini call.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def slow(**_kwargs):
        time.sleep(0.3)
        return _ok_result()

    monkeypatch.setattr(pipeline_module, "analyze_death_certificate_consistency", slow)

    ticks = 0

    async def heartbeat(stop: asyncio.Event) -> None:
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    stop = asyncio.Event()
    beat = asyncio.create_task(heartbeat(stop))
    signal = await pipeline_module._stage_consistency(
        Submission(image=_JPEG, narrative="claimant narrative")
    )
    stop.set()
    await beat

    assert signal.available is True
    assert ticks > 5, f"event loop only advanced {ticks} times — call ran on the loop"


async def test_consistency_timeout_marks_the_stage_unavailable(monkeypatch):
    """A hung call must read as 'no evidence', never as a score of 0.0."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    short = dataclasses.replace(
        default_scoring_config(), consistency_timeout_seconds=0.05, consistency_attempts=1
    )
    monkeypatch.setattr(google_clients, "TIMEOUT_GRACE_SECONDS", 0.0)

    def hangs(**_kwargs):
        # Only needs to outlast the timeout; the thread is not cancellable and
        # pytest waits on it at teardown, so keep it short.
        time.sleep(0.5)
        return _ok_result()

    monkeypatch.setattr(pipeline_module, "analyze_death_certificate_consistency", hangs)

    signal = await pipeline_module._stage_consistency(
        Submission(image=_JPEG, narrative="claimant narrative"), short
    )

    assert signal.available is False
    assert signal.consistency_score == 0.0
    assert "timed out" in signal.summary


async def test_no_claimant_account_keeps_fields_but_skips_the_comparison(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def no_account(**_kwargs):
        return {**_ok_result(), "claimant_account_present": False,
                "certificate": {"full_name": "Jane Doe"}, "consistency_score": 0.0}

    monkeypatch.setattr(pipeline_module, "analyze_death_certificate_consistency", no_account)

    signal = await pipeline_module._stage_consistency(
        Submission(image=_JPEG, narrative="user: hello\nassistant: please send the certificate")
    )

    assert signal.claimant_account_present is False
    assert signal.available is False
    assert signal.extracted_fields == {"full_name": "Jane Doe"}


async def test_authenticity_stage_states_the_document_type(monkeypatch):
    """Stated, not guessed: keyword guessing misses non-English certificates."""
    seen = {}

    class FakeDetector:
        config = type("C", (), {"version": 1})()

        async def run(self, image_bytes, context):
            seen.update(context)
            from tools.fake_image_detector.models import Escalation, ToolResult, Verdict

            return ToolResult(verdict=Verdict.PASS, risk_score=0.0,
                              escalation=Escalation.AUTO_ACCEPT, checks=[])

    monkeypatch.setattr(pipeline_module, "_build_authenticity_pipeline", FakeDetector)
    await pipeline_module._stage_authenticity(Submission(image=_JPEG, narrative="n"))

    assert seen == {"input_type": "document", "doc_type": "death_certificate"}



async def test_the_timeout_reaches_the_gemini_call_itself(monkeypatch):
    """A timeout only around the thread left the call running and holding a
    worker; the HTTP request must be given the limit so it actually ends."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    seen = {}

    def record(**kwargs):
        seen.update(kwargs)
        return _ok_result()

    monkeypatch.setattr(pipeline_module, "analyze_death_certificate_consistency", record)
    config = dataclasses.replace(
        default_scoring_config(), consistency_timeout_seconds=42, consistency_attempts=3
    )

    await pipeline_module._stage_consistency(Submission(image=_JPEG, narrative="n"), config)

    # Per attempt, so a stalled call is retried rather than waited out.
    assert seen["timeout_seconds"] == 42
    assert seen["attempts"] == 3


def test_attempts_budget_covers_every_attempt():
    """Regression: the outer limit equalled one attempt's timeout, so it fired
    before any retry and the retries ran unseen in the background."""
    budget = google_clients.attempts_budget_seconds(2, 30)
    assert budget == 2 * 30 + google_clients.retry_wait(0) + google_clients.TIMEOUT_GRACE_SECONDS


async def test_fraud_and_story_checks_run_at_the_same_time(monkeypatch):
    """Each waits on Gemini (~10-15 s); sequentially the claimant waited for both."""
    from tools.death_certificate_pipeline.models import AuthenticitySignal, ConsistencySignal
    from tools.fake_image_detector.models import Escalation, ToolResult, Verdict

    async def slow_auth(submission):
        await asyncio.sleep(0.3)
        return AuthenticitySignal(result=ToolResult(
            verdict=Verdict.PASS, risk_score=0.0, escalation=Escalation.AUTO_ACCEPT, checks=[]))

    async def slow_consistency(submission, config=None):
        await asyncio.sleep(0.3)
        return ConsistencySignal(consistency_score=1.0)

    monkeypatch.setattr(pipeline_module, "_stage_authenticity", slow_auth)
    monkeypatch.setattr(pipeline_module, "_stage_consistency", slow_consistency)

    started = time.monotonic()
    await pipeline_module.run_pipeline_with_diagnostics(Submission(image=_JPEG, narrative="n"))

    assert time.monotonic() - started < 0.5  # ~0.3 together, ~0.6 one after the other



async def test_story_check_model_comes_from_scoring_settings(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    seen = {}

    def record(**kwargs):
        seen.update(kwargs)
        return _ok_result()

    monkeypatch.setattr(pipeline_module, "analyze_death_certificate_consistency", record)
    config = dataclasses.replace(default_scoring_config(), consistency_model="some-newer-model")

    await pipeline_module._stage_consistency(Submission(image=_JPEG, narrative="n"), config)

    assert seen["model"] == "some-newer-model"
