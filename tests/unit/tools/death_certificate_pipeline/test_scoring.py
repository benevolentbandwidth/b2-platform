import dataclasses

from tools.death_certificate_pipeline.config_loader import default_scoring_config
from tools.death_certificate_pipeline.models import (
    AuthenticitySignal,
    Band,
    ConsistencySignal,
    DocumentSignal,
)
from tools.death_certificate_pipeline.pipeline import _stage_score
from tools.fake_image_detector.models import Escalation, ToolResult, Verdict

# Band-cutoff tests opt out of the consistency minimum so they isolate the
# weighted-score arithmetic; the minimum has its own tests.
_NO_MINIMUM = dataclasses.replace(default_scoring_config(), consistency_min_score=0.0)


def _auth(risk_score: float, escalation: Escalation = Escalation.AUTO_ACCEPT) -> AuthenticitySignal:
    verdict = Verdict.PASS
    if escalation == Escalation.HUMAN_REVIEW:
        verdict = Verdict.FLAG
    elif escalation == Escalation.AUTO_REJECT:
        verdict = Verdict.REJECT

    return AuthenticitySignal(
        result=ToolResult(
            verdict=verdict,
            risk_score=risk_score,
            escalation=escalation,
            checks=[],
        )
    )


async def test_score_high_band_from_strong_stage_scores():
    result = await _stage_score(
        DocumentSignal(legible=True, document_type="death_certificate"),
        _auth(0.05),
        ConsistencySignal(consistency_score=0.9, consistency_label="high"),
    )

    assert result.score == 92
    assert result.band == Band.HIGH
    assert result.sub_scores == {"authenticity": 0.95, "consistency": 0.9}


async def test_score_medium_and_low_bands_from_weighted_scores():
    medium = await _stage_score(
        DocumentSignal(legible=True, document_type="death_certificate"),
        _auth(0.3),
        ConsistencySignal(consistency_score=0.4, consistency_label="moderate"),
        _NO_MINIMUM,
    )
    low = await _stage_score(
        DocumentSignal(legible=True, document_type="death_certificate"),
        _auth(0.6),
        ConsistencySignal(consistency_score=0.3, consistency_label="low"),
        _NO_MINIMUM,
    )

    # 100 x (0.5 x authenticity + 0.5 x consistency)
    assert medium.score == 55
    assert medium.band == Band.MEDIUM
    assert low.score == 35
    assert low.band == Band.LOW


async def test_hard_authenticity_escalation_overrides_numeric_high_score():
    result = await _stage_score(
        DocumentSignal(legible=True, document_type="death_certificate"),
        _auth(0.05, Escalation.HUMAN_REVIEW),
        ConsistencySignal(consistency_score=0.9, consistency_label="high"),
    )

    assert result.score == 92
    assert result.band == Band.ESCALATE
    assert result.flags == ["HARD_ESCALATION"]


async def test_unavailable_consistency_escalates_instead_of_auto_accepting():
    """A consistency stage that never ran must not read as a passing case.

    Its 0.0 is absence of evidence, not a contradiction. Scored as one it still
    reached 60 — inside MEDIUM — and auto-forwarded the case to GiveLight on
    any Gemini outage or credentials failure.
    """
    result = await _stage_score(
        DocumentSignal(legible=True, document_type="death_certificate"),
        _auth(0.0),
        ConsistencySignal(consistency_score=0.0, available=False),
    )

    assert result.band is Band.ESCALATE
    assert "CONSISTENCY_UNAVAILABLE" in result.flags


async def test_consistency_that_ran_is_scored_normally():
    """The guard must key on availability, not on the score being 0.0."""
    result = await _stage_score(
        DocumentSignal(legible=True, document_type="death_certificate"),
        _auth(0.0),
        ConsistencySignal(consistency_score=0.0, available=True),
    )

    assert "CONSISTENCY_UNAVAILABLE" not in result.flags


async def test_consistency_below_minimum_escalates():
    """A strong overall score must not carry a story the certificate contradicts."""
    config = dataclasses.replace(default_scoring_config(), consistency_min_score=0.3)
    result = await _stage_score(
        DocumentSignal(legible=True, document_type="death_certificate"),
        _auth(0.0),
        ConsistencySignal(consistency_score=0.29),
        config,
    )

    assert result.score >= config.band_medium  # would otherwise auto-accept
    assert result.band is Band.ESCALATE
    assert "CONSISTENCY_BELOW_MINIMUM" in result.flags


async def test_consistency_at_minimum_is_scored_normally():
    config = dataclasses.replace(default_scoring_config(), consistency_min_score=0.3)
    result = await _stage_score(
        DocumentSignal(legible=True, document_type="death_certificate"),
        _auth(0.0),
        ConsistencySignal(consistency_score=0.3),
        config,
    )

    assert "CONSISTENCY_BELOW_MINIMUM" not in result.flags
    assert result.band is not Band.ESCALATE


async def test_verdict_records_both_settings_versions():
    """A verdict must trace to every settings file that shaped it."""
    config = dataclasses.replace(default_scoring_config(), version=42)
    auth = _auth(0.1)
    auth.config_version = 9
    result = await _stage_score(
        DocumentSignal(legible=True, document_type="death_certificate"),
        auth,
        ConsistencySignal(consistency_score=0.9),
        config,
    )

    assert result.config_versions == {"scoring": 42, "detector": 9}


async def test_alternative_configs_can_be_compared_in_process():
    """The point of the settings file: same signals, different settings, different verdict."""
    signals = (
        DocumentSignal(legible=True, document_type="death_certificate"),
        _auth(0.2),
        ConsistencySignal(consistency_score=0.5),
    )
    baseline = default_scoring_config()
    authenticity_heavy = dataclasses.replace(
        baseline, stage_weights={"authenticity": 0.8, "consistency": 0.2}
    )

    a = await _stage_score(*signals, baseline)
    b = await _stage_score(*signals, authenticity_heavy)

    assert a.score != b.score
    assert a.weights == baseline.stage_weights
    assert b.weights == authenticity_heavy.stage_weights


async def test_document_stage_recognises_heic():
    """iPhone HEIC photos used to be scored as unreadable."""
    from tools.death_certificate_pipeline.models import Submission
    from tools.death_certificate_pipeline.pipeline import _stage_document

    signal = await _stage_document(
        Submission(image=b"\x00\x00\x00\x18ftypheic rest", narrative="n")
    )
    assert signal.legible is True


async def test_every_escalation_reason_is_recorded():
    """A reviewer should see all the problems, not just the first one found."""
    result = await _stage_score(
        DocumentSignal(legible=True, document_type="death_certificate"),
        _auth(0.2, Escalation.HUMAN_REVIEW),
        ConsistencySignal(consistency_score=0.0, available=False),
    )

    assert result.band is Band.ESCALATE
    assert set(result.flags) == {"HARD_ESCALATION", "CONSISTENCY_UNAVAILABLE"}


async def test_unreadable_document_goes_to_review_and_earns_no_points():
    """Recognising the file format is a gate, not evidence: no free points."""
    signals = (_auth(0.1), ConsistencySignal(consistency_score=0.8))
    readable = await _stage_score(DocumentSignal(legible=True), *signals)
    unreadable = await _stage_score(DocumentSignal(legible=False), *signals)

    assert "document" not in readable.sub_scores
    assert unreadable.band is Band.ESCALATE
    # The only reason: the other stages are not run on a file we cannot read.
    assert unreadable.flags == ["UNREADABLE_DOCUMENT"]
    assert unreadable.sub_scores == {}


async def test_missing_claimant_account_goes_to_review_with_its_own_reason():
    result = await _stage_score(
        DocumentSignal(legible=True),
        _auth(0.05),
        ConsistencySignal(consistency_score=0.0, available=False, claimant_account_present=False),
    )

    assert result.band is Band.ESCALATE
    assert "NO_CLAIMANT_ACCOUNT" in result.flags
    assert "CONSISTENCY_UNAVAILABLE" not in result.flags  # one reason, the precise one


async def test_an_unreadable_file_is_never_sent_to_gemini(monkeypatch):
    """Gemini rejects it, so both checks failed and the reviewer saw 'could not
    be completed' reasons on top of the one real problem."""
    import tools.death_certificate_pipeline.pipeline as pipeline_module
    from tools.death_certificate_pipeline.models import Submission

    async def must_not_run(*_args, **_kwargs):
        raise AssertionError("stage should not run on an unreadable file")

    monkeypatch.setattr(pipeline_module, "_stage_authenticity", must_not_run)
    monkeypatch.setattr(pipeline_module, "_stage_consistency", must_not_run)

    execution = await pipeline_module.run_pipeline_with_diagnostics(
        Submission(image=b"not an image at all", narrative="user: my father died")
    )

    assert execution.result.flags == ["UNREADABLE_DOCUMENT"]
    assert execution.result.band is Band.ESCALATE
    assert "not run" in (execution.authenticity.early_exit_reason or "")
