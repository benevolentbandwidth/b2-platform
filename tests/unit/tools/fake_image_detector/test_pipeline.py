import asyncio

from tools.fake_image_detector.config_loader import CheckConfig, PipelineConfig
from tools.fake_image_detector.models import CheckResult, Escalation, Verdict
from tools.fake_image_detector.pipeline import FakeImageDetectorPipeline


def run(coro):
    return asyncio.run(coro)


class _StubCheck:
    def __init__(self, result: CheckResult):
        self._result = result
        self.calls = 0

    async def run(self, image_bytes: bytes, context: dict) -> CheckResult:
        self.calls += 1
        return self._result


def _cfg(check_id: str, early_exit_on_fail: bool = False) -> CheckConfig:
    return CheckConfig(id=check_id, enabled=True, early_exit_on_fail=early_exit_on_fail)


def _pipeline(
    checks: list[tuple[CheckConfig, _StubCheck]],
    gemini_check: _StubCheck | None = None,
) -> FakeImageDetectorPipeline:
    config = PipelineConfig(clear_fail=0.8, clear_pass=0.2, checks=[cfg for cfg, _ in checks])
    return FakeImageDetectorPipeline(config=config, checks=checks, gemini_check=gemini_check)


# helpers for common check shapes
def _passing(check_id: str) -> CheckResult:
    return CheckResult(check=check_id, passed=True, fake_score=0.0, confidence=0.8)


def _ambiguous(check_id: str) -> CheckResult:
    # score = 0.5*0.8 / 0.8 = 0.5 → ambiguous zone [0.2, 0.8)
    return CheckResult(check=check_id, passed=False, fake_score=0.5, confidence=0.8)


def _failing(check_id: str) -> CheckResult:
    # score = 1.0*0.9 / 0.9 = 1.0 → clear fail ≥ 0.8
    return CheckResult(check=check_id, passed=False, fake_score=1.0, confidence=0.9)


def test_unknown_input_type_flags_immediately():
    async def classify_unknown(image_bytes):
        return "unknown"

    p = _pipeline([(_cfg("exif"), _StubCheck(_passing("exif")))])
    p._classify_input_type = classify_unknown
    result = run(p.run(b"img", {"input_type": "unknown"}))
    assert result.verdict == Verdict.FLAG
    assert result.escalation == Escalation.HUMAN_REVIEW
    assert result.early_exit is True
    assert result.early_exit_reason == "UNKNOWN_INPUT_TYPE"
    assert result.checks == []


def test_document_route_runs_only_document_checks():
    exif = _StubCheck(_passing("exif"))
    ocr = _StubCheck(_passing("ocr_document"))
    ela = _StubCheck(_passing("ela"))
    p = _pipeline([(_cfg("exif"), exif), (_cfg("ocr_document"), ocr), (_cfg("ela"), ela)])
    result = run(p.run(b"img", {"input_type": "document"}))
    assert result.verdict == Verdict.PASS
    assert [r.check for r in result.checks] == ["exif", "ocr_document"]
    assert ela.calls == 0


def test_no_active_checks_for_route_flags():
    ela = _StubCheck(_passing("ela"))
    p = _pipeline([(_cfg("ela"), ela)])
    result = run(p.run(b"img", {"input_type": "document"}))
    assert result.verdict == Verdict.FLAG
    assert result.early_exit_reason == "NO_ACTIVE_CHECKS_FOR_DOCUMENT"
    assert ela.calls == 0


def test_skipped_check_error_is_ignored():
    exif = _StubCheck(CheckResult(check="exif", passed=True, skipped=True, error="boom"))
    ela = _StubCheck(_passing("ela"))
    p = _pipeline([(_cfg("exif"), exif), (_cfg("ela"), ela)])
    result = run(p.run(b"img", {"input_type": "face"}))

    assert result.verdict == Verdict.PASS
    assert result.escalation == Escalation.AUTO_ACCEPT
    assert result.early_exit is False
    assert result.checks[0].skipped is True
    assert result.checks[0].error == "boom"
    assert ela.calls == 1


def test_unskipped_check_runtime_error_flags_immediately():
    exif = _StubCheck(CheckResult(check="exif", passed=False, skipped=False, error="boom"))
    p = _pipeline([(_cfg("exif"), exif)])
    result = run(p.run(b"img", {"input_type": "face"}))

    assert result.verdict == Verdict.FLAG
    assert result.escalation == Escalation.HUMAN_REVIEW
    assert result.early_exit is True
    assert "CHECK_RUNTIME_ERROR" in result.checks[0].flags


def test_all_skipped_checks_force_human_review_at_clear_pass_threshold():
    exif = _StubCheck(CheckResult(check="exif", passed=True, skipped=True))
    p = _pipeline([(_cfg("exif"), exif)])
    result = run(p.run(b"img", {"input_type": "face"}))

    assert result.risk_score == 0.2
    assert result.verdict == Verdict.FLAG
    assert result.escalation == Escalation.HUMAN_REVIEW
    assert result.early_exit is False


def test_hard_escalation_flag_forces_human_review_before_score_classification():
    reverse_image = _StubCheck(
        CheckResult(
            check="reverse_image",
            passed=False,
            fake_score=0.95,
            confidence=0.95,
            flags=["FOUND_ONLINE"],
            human_escalate=True,
            escalation_reasons=["FOUND_ONLINE detected by reverse_image check"],
        )
    )
    p = _pipeline([(_cfg("reverse_image"), reverse_image)])
    result = run(p.run(b"img", {"input_type": "face"}))
    assert result.early_exit is True
    assert result.verdict == Verdict.FLAG
    assert result.escalation == Escalation.HUMAN_REVIEW
    assert "hard escalation" in (result.early_exit_reason or "")


def test_early_exit_on_fail_uses_fake_score():
    mrz = _StubCheck(CheckResult(check="mrz", passed=False, fake_score=1.0, confidence=1.0, flags=["MRZ_CHECKSUM_FAIL"]))
    p = _pipeline([(_cfg("mrz", early_exit_on_fail=True), mrz)])
    result = run(p.run(b"img", {"input_type": "document"}))
    assert result.early_exit is True
    assert result.verdict == Verdict.REJECT
    assert result.escalation == Escalation.AUTO_REJECT
    assert result.risk_score == 1.0


def test_gemini_called_in_ambiguous_zone():
    ela = _StubCheck(_ambiguous("ela"))
    gemini = _StubCheck(CheckResult(check="gemini_vision", passed=False, fake_score=0.6, confidence=0.85, flags=["GAN_ARTIFACTS"]))
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=gemini)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert gemini.calls == 1
    assert any(r.check == "gemini_vision" for r in result.checks)


def test_gemini_not_called_on_clear_pass():
    ela = _StubCheck(_passing("ela"))
    gemini = _StubCheck(CheckResult(check="gemini_vision", passed=False, fake_score=0.9, confidence=0.9))
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=gemini)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert gemini.calls == 0
    assert result.verdict == Verdict.PASS


def test_gemini_not_called_on_clear_fail():
    ela = _StubCheck(_failing("ela"))
    gemini = _StubCheck(CheckResult(check="gemini_vision", passed=False, fake_score=0.95, confidence=0.95))
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=gemini)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert gemini.calls == 0
    assert result.verdict == Verdict.REJECT


def test_gemini_skipped_returns_stage1_with_skip_record():
    ela = _StubCheck(_ambiguous("ela"))
    gemini = _StubCheck(CheckResult(check="gemini_vision", passed=True, skipped=True))
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=gemini)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert result.verdict == Verdict.FLAG
    assert result.risk_score == 0.5
    assert any(r.check == "gemini_vision" and r.skipped for r in result.checks)


def test_gemini_hard_escalation_overrides_clear_reject():
    ela = _StubCheck(_ambiguous("ela"))
    gemini = _StubCheck(
        CheckResult(
            check="gemini_vision",
            passed=False,
            fake_score=0.95,
            confidence=0.95,
            flags=["POSSIBLE_STOCK"],
            human_escalate=True,
            escalation_reasons=["POSSIBLE_STOCK detected by gemini_vision check"],
        )
    )
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=gemini)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert result.verdict == Verdict.FLAG
    assert result.escalation == Escalation.HUMAN_REVIEW
    assert result.early_exit is True


def test_gemini_real_verdict_overrides_ambiguous_stage1():
    ela = _StubCheck(_ambiguous("ela"))
    gemini = _StubCheck(CheckResult(check="gemini_vision", passed=True, fake_score=0.05, confidence=0.9))
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=gemini)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert result.verdict == Verdict.PASS
    assert result.risk_score == 0.05


def test_gemini_none_returns_stage1():
    ela = _StubCheck(_ambiguous("ela"))
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=None)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert result.verdict == Verdict.FLAG
    assert result.risk_score == 0.5


def test_gemini_always_called_for_documents():
    # Documents always go to Gemini regardless of stage-1 score.
    ocr = _StubCheck(_passing("ocr_document"))
    gemini = _StubCheck(CheckResult(check="gemini_vision", passed=True, fake_score=0.1, confidence=0.9))
    p = _pipeline([(_cfg("ocr_document"), ocr)], gemini_check=gemini)
    result = run(p.run(b"img", {"input_type": "document"}))
    assert gemini.calls == 1
    assert any(r.check == "gemini_vision" for r in result.checks)
    assert result.verdict == Verdict.PASS
    assert result.risk_score == 0.1


def test_gemini_not_called_for_face_on_clear_fail():
    # For faces, a clear REJECT bypasses Gemini (no need to spend API calls).
    ela = _StubCheck(_failing("ela"))
    gemini = _StubCheck(CheckResult(check="gemini_vision", passed=False, fake_score=0.95, confidence=0.95))
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=gemini)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert gemini.calls == 0
    assert result.verdict == Verdict.REJECT


def test_zero_confidence_stage1_results_flag_for_review():
    ela = _StubCheck(CheckResult(check="ela", passed=True, fake_score=0.0, confidence=0.0))
    p = _pipeline([(_cfg("ela"), ela)], gemini_check=None)
    result = run(p.run(b"img", {"input_type": "face"}))
    assert result.risk_score == 0.21
    assert result.verdict == Verdict.FLAG
    assert result.escalation == Escalation.HUMAN_REVIEW


def _synthid_hit() -> CheckResult:
    return CheckResult(
        check="synthid",
        passed=False,
        fake_score=0.9,
        confidence=0.9,
        flags=["SYNTHID_WATERMARK_DETECTED"],
    )


def test_hard_escalation_flags_come_from_config():
    """Adding a flag to hard_escalation_flags makes that signal decisive.

    Built from explicit configs so it holds whatever the shipped list says:
    without the flag a watermark hit alone does not force review; with it, it
    does, with no code change.
    """
    def build(flags):
        synthid = _StubCheck(_synthid_hit())
        config = PipelineConfig(
            clear_fail=0.8,
            clear_pass=0.2,
            checks=[_cfg("synthid")],
            hard_escalation_flags=frozenset(flags),
        )
        return FakeImageDetectorPipeline(config=config, checks=[(_cfg("synthid"), synthid)], gemini_check=None)

    default = run(build({"FOUND_ONLINE", "POSSIBLE_STOCK"}).run(b"img", {"input_type": "document"}))
    decisive = run(
        build({"FOUND_ONLINE", "POSSIBLE_STOCK", "SYNTHID_WATERMARK_DETECTED"}).run(
            b"img", {"input_type": "document"}
        )
    )

    assert "hard escalation" not in (default.early_exit_reason or "")
    assert decisive.escalation == Escalation.HUMAN_REVIEW
    assert "hard escalation" in (decisive.early_exit_reason or "")


def test_shipped_config_lists_hard_escalation_flags():
    from tools.fake_image_detector.config_loader import load_pipeline_config

    assert load_pipeline_config().hard_escalation_flags == frozenset(
        {"FOUND_ONLINE", "POSSIBLE_STOCK", "SYNTHID_WATERMARK_DETECTED", "CHECKSUM_FAIL",
         "INTERNAL_INCONSISTENCY"}
    )


def test_shipped_config_has_a_version():
    from tools.fake_image_detector.config_loader import load_pipeline_config

    assert isinstance(load_pipeline_config().version, int)


def test_config_without_version_is_refused(tmp_path):
    import pytest

    from tools.fake_image_detector.config_loader import load_pipeline_config

    path = tmp_path / "pipeline.yaml"
    path.write_text(
        "thresholds:\n  clear_fail: 0.8\n  clear_pass: 0.2\n"
        "hard_escalation_flags: [FOUND_ONLINE]\nchecks: []\n"
    )
    with pytest.raises(ValueError, match="version"):
        load_pipeline_config(path)


def test_config_without_hard_escalation_flags_is_refused(tmp_path):
    import pytest

    from tools.fake_image_detector.config_loader import load_pipeline_config

    path = tmp_path / "pipeline.yaml"
    path.write_text("version: 1\nthresholds:\n  clear_fail: 0.8\n  clear_pass: 0.2\nchecks: []\n")
    with pytest.raises(ValueError, match="hard_escalation_flags"):
        load_pipeline_config(path)


def test_internal_inconsistency_from_gemini_forces_review_with_shipped_config():
    """A certificate whose own details contradict each other goes to a person."""
    from tools.fake_image_detector.config_loader import load_pipeline_config

    gemini = _StubCheck(
        CheckResult(
            check="gemini_vision",
            passed=False,
            fake_score=0.4,
            confidence=0.8,
            flags=["INTERNAL_INCONSISTENCY"],
        )
    )
    config = load_pipeline_config()
    exif = _StubCheck(_passing("exif"))
    pipeline = FakeImageDetectorPipeline(
        config=PipelineConfig(
            clear_fail=config.clear_fail,
            clear_pass=config.clear_pass,
            checks=[_cfg("exif")],
            hard_escalation_flags=config.hard_escalation_flags,
        ),
        checks=[(_cfg("exif"), exif)],
        gemini_check=gemini,
    )

    result = run(pipeline.run(b"img", {"input_type": "document"}))

    assert result.escalation == Escalation.HUMAN_REVIEW
    assert "INTERNAL_INCONSISTENCY" in (result.early_exit_reason or "")


def test_gemini_timeouts_come_from_settings():
    from tools.fake_image_detector.config_loader import load_pipeline_config

    config = load_pipeline_config()
    extract = next(c for c in config.checks if c.id == "gemini_extract")
    # Per attempt for the fraud check, which retries a stalled call.
    assert config.gemini.timeout_seconds == 30
    assert config.gemini.attempts == 2
    assert extract.params["timeout_seconds"] == 60



def test_the_settings_list_decides_escalation_in_both_directions():
    """Checks used to set human_escalate from a hardcoded list, which bypassed
    the settings: a flag removed from pipeline.yaml still forced review."""
    found_online = CheckResult(  # shape the real reverse_image check returns
        check="reverse_image", passed=False, fake_score=0.75, confidence=0.71,
        flags=["FOUND_ONLINE"],
    )

    def build(flags):
        config = PipelineConfig(
            clear_fail=0.8, clear_pass=0.2, checks=[_cfg("reverse_image")],
            hard_escalation_flags=frozenset(flags),
        )
        return FakeImageDetectorPipeline(
            config=config, checks=[(_cfg("reverse_image"), _StubCheck(found_online))]
        )

    listed = run(build({"FOUND_ONLINE"}).run(b"img", {"input_type": "document"}))
    removed = run(build(set()).run(b"img", {"input_type": "document"}))

    assert "hard escalation" in (listed.early_exit_reason or "")
    assert "hard escalation" not in (removed.early_exit_reason or "")



def test_gemini_models_come_from_settings(tmp_path):
    """Models used to come from VERTEX_MODEL or a hardcoded default."""
    import pytest

    from tools.fake_image_detector.config_loader import load_pipeline_config
    from tools.fake_image_detector.pipeline import build_pipeline

    pipeline = build_pipeline()
    extract = next(check for cfg, check in pipeline._checks if cfg.id == "gemini_extract")
    config = load_pipeline_config()
    assert pipeline._gemini_check._model == config.gemini.model
    assert extract._model == next(c for c in config.checks if c.id == "gemini_extract").params["model"]

    path = tmp_path / "pipeline.yaml"
    path.write_text(
        "version: 1\nthresholds: {clear_fail: 0.8, clear_pass: 0.2}\n"
        "hard_escalation_flags: [FOUND_ONLINE]\ngemini: {enabled: true}\nchecks: []\n"
    )
    with pytest.raises(ValueError, match="gemini.model"):
        load_pipeline_config(path)
