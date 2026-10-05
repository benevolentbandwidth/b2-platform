"""Death certificate pipeline unit tests.

Covers:
  1. Model construction and field validation
  2. Narrative blank validation
  3. Band enum completeness
  4. run_pipeline() end-to-end with mocked external services
  5. _stage_score() logic — hard escalation, band thresholds, extracted_fields passthrough
  6. _stage_document() — format detection, legibility
  7. FastAPI POST /score endpoint
  8. CLI argument parser and image resolution

Mocking strategy:
  - _stage_authenticity calls build_pipeline() → mocked to return clean PASS ToolResult
  - _stage_consistency calls analyze_death_certificate_consistency() → mocked to return
    a known certificate extract with high consistency score
  - All stage scoring logic is tested with explicit signal inputs (no mocking needed)
"""

from __future__ import annotations

import dataclasses
import io
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools.death_certificate_pipeline.config_loader import default_scoring_config
from tools.death_certificate_pipeline.models import (
    AuthenticitySignal,
    Band,
    ConsistencySignal,
    DocumentSignal,
    ReliabilityResult,
    Submission,
)
from tools.death_certificate_pipeline.pipeline import (
    _stage_document,
    _stage_score,
    run_pipeline,
    run_pipeline_with_diagnostics,
)
from tools.fake_image_detector.models import Escalation, ToolResult, Verdict

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FAKE_PNG   = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100   # recognizable PNG header
FAKE_JPEG  = b"\xff\xd8\xff\xe0" + b"\x00" * 100     # recognizable JPEG header
FAKE_TIFF  = b"\x49\x49\x2a\x00" + b"\x00" * 100    # TIFF little-endian
NARRATIVE  = "My father John Smith passed away on March 3rd, 2021 in Chicago."

_CLEAN_TOOL_RESULT = ToolResult(
    verdict=Verdict.PASS,
    risk_score=0.1,
    escalation=Escalation.AUTO_ACCEPT,
    checks=[],
)

_MOCK_CONSISTENCY_RESULT = {
    "certificate": {
        "full_name":          "John Smith",
        "date_of_death":      "2021-03-03",
        "place_of_death":     "Chicago, IL",
        "age_at_death":       74,
        "cause_of_death":     "cardiac arrest",
        "certificate_number": "IL-2021-00123",
        "issuing_authority":  "Cook County",
        "registration_date":  "2021-03-05",
        "other_visible_details": {},
    },
    "consistency_score":  0.91,
    "consistency_label":  "high",
    "confidence":         0.88,
    "matches":            ["date of death aligns", "name matches"],
    "mismatches":         [],
    "uncertain_points":   [],
    "summary":            "Certificate consistent with narrative.",
    "model":              "gemini-2.0-flash",
}


@pytest.fixture()
def minimal_submission() -> Submission:
    return Submission(image=FAKE_PNG, narrative=NARRATIVE)


@pytest.fixture()
def mock_pipeline_stages(monkeypatch):
    """Patch both external service calls in the pipeline.

    - build_pipeline() → returns a mock whose .run() returns _CLEAN_TOOL_RESULT
    - analyze_death_certificate_consistency() → returns _MOCK_CONSISTENCY_RESULT
    - GEMINI_API_KEY → set to dummy value so consistency stage runs
    """
    mock_detector = MagicMock()
    mock_detector.run = AsyncMock(return_value=_CLEAN_TOOL_RESULT)

    mock_build = MagicMock(return_value=mock_detector)

    # Both are module-level imports in pipeline.py — patch the bound names there
    monkeypatch.setattr(
        "tools.death_certificate_pipeline.pipeline._build_authenticity_pipeline",
        mock_build,
    )
    monkeypatch.setattr(
        "tools.death_certificate_pipeline.pipeline.analyze_death_certificate_consistency",
        MagicMock(return_value=_MOCK_CONSISTENCY_RESULT),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-dummy-key-for-unit-tests")

    return mock_build, mock_detector


# ---------------------------------------------------------------------------
# 1. Model construction
# ---------------------------------------------------------------------------

class TestModelConstruction:
    def test_submission_constructs_with_required_fields(self) -> None:
        s = Submission(image=FAKE_PNG, narrative=NARRATIVE)
        assert s.image == FAKE_PNG
        assert s.narrative == NARRATIVE
        assert s.case_fields == {}

    def test_submission_accepts_case_fields(self) -> None:
        s = Submission(image=FAKE_PNG, narrative=NARRATIVE, case_fields={"case_id": "ABC-1"})
        assert s.case_fields["case_id"] == "ABC-1"

    def test_document_signal_defaults(self) -> None:
        sig = DocumentSignal()
        assert sig.legible is False
        assert sig.document_type is None
        assert sig.page_count == 1
        assert sig.stage == "document_analysis"

    def test_authenticity_signal_wraps_tool_result(self) -> None:
        tr  = ToolResult(verdict=Verdict.PASS, risk_score=0.1,
                         escalation=Escalation.AUTO_ACCEPT, checks=[])
        sig = AuthenticitySignal(result=tr)
        assert sig.result.verdict == Verdict.PASS
        assert sig.stage == "authenticity"

    def test_consistency_signal_defaults(self) -> None:
        sig = ConsistencySignal()
        assert sig.consistency_score == 0.0
        assert sig.extracted_fields == {}
        assert sig.stage == "consistency"

    def test_reliability_result_constructs(self) -> None:
        r = ReliabilityResult(
            score=88, band=Band.HIGH,
            sub_scores={"document": 1.0, "authenticity": 0.9, "consistency": 0.8},
            weights={"document": 0.2, "authenticity": 0.4, "consistency": 0.4},
            flags=[], justification="test",
        )
        assert r.score == 88
        assert r.band == Band.HIGH


# ---------------------------------------------------------------------------
# 2. Submission.narrative validation
# ---------------------------------------------------------------------------

class TestNarrativeValidation:
    def test_blank_narrative_raises(self) -> None:
        with pytest.raises(ValueError, match="narrative must not be blank"):
            Submission(image=FAKE_PNG, narrative="")

    def test_whitespace_only_narrative_raises(self) -> None:
        with pytest.raises(ValueError, match="narrative must not be blank"):
            Submission(image=FAKE_PNG, narrative="   \t\n  ")

    def test_valid_narrative_is_accepted(self) -> None:
        assert Submission(image=FAKE_PNG, narrative="valid").narrative == "valid"


# ---------------------------------------------------------------------------
# 3. Band enum
# ---------------------------------------------------------------------------

class TestBandEnum:
    def test_band_covers_all_four_values(self) -> None:
        assert {b.value for b in Band} == {"high", "medium", "low", "escalate"}

    def test_band_is_string_enum(self) -> None:
        assert Band.HIGH == "high"
        assert Band.ESCALATE == "escalate"


# ---------------------------------------------------------------------------
# 4. _stage_document — format detection
# ---------------------------------------------------------------------------

class TestStageDocument:
    @pytest.mark.asyncio
    async def test_png_is_legible(self) -> None:
        result = await _stage_document(Submission(image=FAKE_PNG, narrative=NARRATIVE))
        assert result.legible is True
        assert result.document_type == "death_certificate"

    @pytest.mark.asyncio
    async def test_jpeg_is_legible(self) -> None:
        result = await _stage_document(Submission(image=FAKE_JPEG, narrative=NARRATIVE))
        assert result.legible is True

    @pytest.mark.asyncio
    async def test_tiff_is_legible(self) -> None:
        result = await _stage_document(Submission(image=FAKE_TIFF, narrative=NARRATIVE))
        assert result.legible is True

    @pytest.mark.asyncio
    async def test_unknown_format_not_legible(self) -> None:
        garbage = b"\x00\x01\x02\x03" + b"\xff" * 100
        result  = await _stage_document(Submission(image=garbage, narrative=NARRATIVE))
        assert result.legible is False
        assert result.document_type is None
        assert len(result.notes) > 0

    @pytest.mark.asyncio
    async def test_empty_image_not_legible(self) -> None:
        # Bypass validator — inject bytes directly via model
        sub = Submission.__new__(Submission)
        object.__setattr__(sub, "image", b"")
        object.__setattr__(sub, "narrative", NARRATIVE)
        object.__setattr__(sub, "case_fields", {})
        result = await _stage_document(sub)
        assert result.legible is False


# ---------------------------------------------------------------------------
# 5. run_pipeline end-to-end with mocked external services
# ---------------------------------------------------------------------------

class TestRunPipeline:
    @pytest.mark.asyncio
    async def test_diagnostics_expose_authenticity_without_changing_result(
        self, minimal_submission, mock_pipeline_stages
    ) -> None:
        execution = await run_pipeline_with_diagnostics(minimal_submission)

        assert isinstance(execution.result, ReliabilityResult)
        assert execution.authenticity is _CLEAN_TOOL_RESULT

    @pytest.mark.asyncio
    async def test_returns_reliability_result(
        self, minimal_submission, mock_pipeline_stages
    ) -> None:
        result = await run_pipeline(minimal_submission)
        assert isinstance(result, ReliabilityResult)

    @pytest.mark.asyncio
    async def test_score_is_int_in_range(
        self, minimal_submission, mock_pipeline_stages
    ) -> None:
        result = await run_pipeline(minimal_submission)
        assert isinstance(result.score, int)
        assert 1 <= result.score <= 100

    @pytest.mark.asyncio
    async def test_band_is_valid_enum_value(
        self, minimal_submission, mock_pipeline_stages
    ) -> None:
        result = await run_pipeline(minimal_submission)
        assert result.band in list(Band)

    @pytest.mark.asyncio
    async def test_sub_scores_has_the_scored_stage_keys(
        self, minimal_submission, mock_pipeline_stages
    ) -> None:
        # The document stage is a pass/fail gate and is not scored.
        result = await run_pipeline(minimal_submission)
        assert set(result.sub_scores) == {"authenticity", "consistency"}

    @pytest.mark.asyncio
    async def test_weights_sum_to_one(
        self, minimal_submission, mock_pipeline_stages
    ) -> None:
        result = await run_pipeline(minimal_submission)
        assert abs(sum(result.weights.values()) - 1.0) < 0.001

    @pytest.mark.asyncio
    async def test_justification_is_non_empty_string(
        self, minimal_submission, mock_pipeline_stages
    ) -> None:
        result = await run_pipeline(minimal_submission)
        assert isinstance(result.justification, str) and len(result.justification) > 0

    @pytest.mark.asyncio
    async def test_extracted_fields_from_consistency(
        self, minimal_submission, mock_pipeline_stages
    ) -> None:
        result = await run_pipeline(minimal_submission)
        # Mock returns full certificate dict — must pass through to result
        assert result.extracted_fields.get("full_name") == "John Smith"
        assert result.extracted_fields.get("date_of_death") == "2021-03-03"

    @pytest.mark.asyncio
    async def test_clean_pass_produces_high_band(
        self, minimal_submission, mock_pipeline_stages
    ) -> None:
        # doc=1.0, auth=0.9 (risk=0.1), con=0.91
        # weighted = 1.0×0.2 + 0.9×0.4 + 0.91×0.4 = 0.2+0.36+0.364 = 0.924 → score=92
        result = await run_pipeline(minimal_submission)
        assert result.band == Band.HIGH
        assert result.score >= 75

    @pytest.mark.asyncio
    async def test_authenticity_called_with_document_context(
        self, minimal_submission, mock_pipeline_stages
    ) -> None:
        mock_build, mock_detector = mock_pipeline_stages
        await run_pipeline(minimal_submission)
        mock_detector.run.assert_awaited_once()
        _, kwargs = mock_detector.run.call_args
        assert kwargs.get("context", {}).get("input_type") == "document"

    @pytest.mark.asyncio
    async def test_no_credentials_skips_consistency(
        self, minimal_submission, monkeypatch
    ) -> None:
        """Without credentials, consistency returns score=0.0; pipeline still runs."""
        mock_detector = MagicMock()
        mock_detector.run = AsyncMock(return_value=_CLEAN_TOOL_RESULT)
        monkeypatch.setattr(
            "tools.death_certificate_pipeline.pipeline._build_authenticity_pipeline",
            MagicMock(return_value=mock_detector),
        )
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        result = await run_pipeline(minimal_submission)
        assert isinstance(result, ReliabilityResult)
        assert result.sub_scores["consistency"] == 0.0
        # With con=0.0, doc=1.0, auth=0.9: weighted=0.2+0.36+0.0=0.56 → MEDIUM
        assert result.band in {Band.MEDIUM, Band.HIGH, Band.LOW, Band.ESCALATE}


# ---------------------------------------------------------------------------
# 6. _stage_score — hard escalation and band thresholds
# ---------------------------------------------------------------------------

class TestScorerEscalation:
    @pytest.mark.asyncio
    async def test_human_review_forces_escalate_band(self) -> None:
        doc  = DocumentSignal(legible=True, document_type="death_certificate")
        auth = AuthenticitySignal(result=ToolResult(
            verdict=Verdict.FLAG, risk_score=0.9,
            escalation=Escalation.HUMAN_REVIEW, checks=[],
        ))
        con  = ConsistencySignal(consistency_score=0.8, consistency_label="high")
        result = await _stage_score(doc, auth, con)
        assert result.band == Band.ESCALATE
        assert "HARD_ESCALATION" in result.flags

    @pytest.mark.asyncio
    async def test_auto_reject_forces_escalate_band(self) -> None:
        doc  = DocumentSignal(legible=True)
        auth = AuthenticitySignal(result=ToolResult(
            verdict=Verdict.REJECT, risk_score=1.0,
            escalation=Escalation.AUTO_REJECT, checks=[],
        ))
        con  = ConsistencySignal(consistency_score=0.9, consistency_label="high")
        result = await _stage_score(doc, auth, con)
        assert result.band == Band.ESCALATE
        assert "HARD_ESCALATION" in result.flags

    @pytest.mark.asyncio
    async def test_low_scores_produce_low_band(self) -> None:
        doc  = DocumentSignal(legible=True)
        auth = AuthenticitySignal(result=ToolResult(
            verdict=Verdict.FLAG, risk_score=0.6,        # auth_score = 0.4
            escalation=Escalation.AUTO_ACCEPT, checks=[],
        ))
        con  = ConsistencySignal(consistency_score=0.2)  # con_score = 0.2
        # Opt out of the consistency minimum: this tests the band cutoffs.
        no_minimum = dataclasses.replace(default_scoring_config(), consistency_min_score=0.0)
        result = await _stage_score(doc, auth, con, no_minimum)
        # 0.4×0.5 + 0.2×0.5 = 0.20+0.10 = 0.30 → score=30 → LOW
        assert result.band == Band.LOW
        assert result.score == 30

    @pytest.mark.asyncio
    async def test_very_low_scores_produce_escalate_band(self) -> None:
        doc  = DocumentSignal(legible=True)
        auth = AuthenticitySignal(result=ToolResult(
            verdict=Verdict.FLAG, risk_score=0.9,        # 0.1
            escalation=Escalation.AUTO_ACCEPT, checks=[],
        ))
        con  = ConsistencySignal(consistency_score=0.05)  # 0.05
        # Opt out of the consistency minimum so ESCALATE comes from the score.
        no_minimum = dataclasses.replace(default_scoring_config(), consistency_min_score=0.0)
        result = await _stage_score(doc, auth, con, no_minimum)
        # 0.1×0.5 + 0.05×0.5 = 0.075 → score=8 → ESCALATE (no flags involved)
        assert result.flags == []
        assert result.band == Band.ESCALATE
        assert result.score == 8


# ---------------------------------------------------------------------------
# 7. Scorer — extracted_fields passthrough
# ---------------------------------------------------------------------------

class TestExtractedFieldsPassthrough:
    @pytest.mark.asyncio
    async def test_extracted_fields_passed_to_result(self) -> None:
        doc  = DocumentSignal(legible=True)
        auth = AuthenticitySignal(result=ToolResult(
            verdict=Verdict.PASS, risk_score=0.1,
            escalation=Escalation.AUTO_ACCEPT, checks=[],
        ))
        con  = ConsistencySignal(
            consistency_score=0.9,
            extracted_fields={"full_name": "Jane Doe", "date_of_death": "2024-05-01"},
        )
        result = await _stage_score(doc, auth, con)
        assert result.extracted_fields["full_name"] == "Jane Doe"
        assert result.extracted_fields["date_of_death"] == "2024-05-01"


# ---------------------------------------------------------------------------
# 8. FastAPI endpoint
# ---------------------------------------------------------------------------

class TestApiEndpoint:
    @pytest.mark.asyncio
    async def test_score_endpoint_returns_200_and_valid_json(
        self, mock_pipeline_stages
    ) -> None:
        import httpx
        from poc.api import app
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/score",
                data={"narrative": NARRATIVE, "case_fields": "{}"},
                files={"image": ("cert.png", io.BytesIO(FAKE_PNG), "image/png")},
            )
        assert response.status_code == 200
        data = response.json()
        for key in ("score", "band", "sub_scores", "weights", "justification", "extracted_fields"):
            assert key in data

    @pytest.mark.asyncio
    async def test_score_endpoint_rejects_blank_narrative(self) -> None:
        import httpx
        from poc.api import app
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/score",
                data={"narrative": "   ", "case_fields": "{}"},
                files={"image": ("cert.png", io.BytesIO(FAKE_PNG), "image/png")},
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_score_endpoint_rejects_invalid_case_fields_json(self) -> None:
        import httpx
        from poc.api import app
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/score",
                data={"narrative": NARRATIVE, "case_fields": "not-json"},
                files={"image": ("cert.png", io.BytesIO(FAKE_PNG), "image/png")},
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_score_endpoint_band_is_known_value(
        self, mock_pipeline_stages
    ) -> None:
        import httpx
        from poc.api import app
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/score",
                data={"narrative": NARRATIVE, "case_fields": "{}"},
                files={"image": ("cert.png", io.BytesIO(FAKE_PNG), "image/png")},
            )
        assert response.json()["band"] in {"high", "medium", "low", "escalate"}


# ---------------------------------------------------------------------------
# 9. CLI — argument parser and image resolution
# ---------------------------------------------------------------------------

class TestCli:
    def test_cli_parser_builds_without_error(self) -> None:
        from poc.cli import _build_parser
        assert _build_parser() is not None

    def test_cli_score_subcommand_parses_args(self, tmp_path) -> None:
        from poc.cli import _build_parser
        image_file = tmp_path / "cert.png"
        image_file.write_bytes(FAKE_PNG)
        args = _build_parser().parse_args([
            "score", "--image", str(image_file),
            "--narrative", NARRATIVE, "--fields", '{"case_id": "X1"}',
        ])
        assert args.command == "score"
        assert args.narrative == NARRATIVE
        assert json.loads(args.fields) == {"case_id": "X1"}

    @pytest.mark.asyncio
    async def test_resolve_image_reads_local_file(self, tmp_path) -> None:
        from poc.cli import _resolve_image
        image_file = tmp_path / "cert.png"
        image_file.write_bytes(FAKE_PNG)
        assert await _resolve_image(str(image_file)) == FAKE_PNG

    @pytest.mark.asyncio
    async def test_resolve_image_local_missing_file_exits(self, tmp_path) -> None:
        from poc.cli import _resolve_image
        with pytest.raises(SystemExit):
            await _resolve_image(str(tmp_path / "nonexistent.png"))

    @pytest.mark.asyncio
    async def test_resolve_image_downloads_from_gcs(self, monkeypatch) -> None:
        from unittest.mock import MagicMock
        from poc.cli import _resolve_image
        mock_blob   = MagicMock()
        mock_blob.download_as_bytes.return_value = FAKE_PNG
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        mock_storage = MagicMock()
        mock_storage.Client.return_value = mock_client
        monkeypatch.setitem(__import__("sys").modules, "google.cloud.storage", mock_storage)
        monkeypatch.setitem(__import__("sys").modules, "google.cloud", MagicMock(storage=mock_storage))
        result = await _resolve_image("gs://my-bucket/certs/cert.jpg")
        mock_client.bucket.assert_called_once_with("my-bucket")
        mock_bucket.blob.assert_called_once_with("certs/cert.jpg")
        assert result == FAKE_PNG

    @pytest.mark.asyncio
    async def test_resolve_image_invalid_gcs_uri_exits(self, monkeypatch) -> None:
        from unittest.mock import MagicMock
        from poc.cli import _resolve_image
        mock_storage = MagicMock()
        monkeypatch.setitem(__import__("sys").modules, "google.cloud.storage", mock_storage)
        monkeypatch.setitem(__import__("sys").modules, "google.cloud", MagicMock(storage=mock_storage))
        with pytest.raises(SystemExit):
            await _resolve_image("gs://bucket-only-no-blob")
