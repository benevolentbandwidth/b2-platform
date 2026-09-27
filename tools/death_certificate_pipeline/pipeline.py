"""Death certificate reliability pipeline.

Workflow: Submission → document → authenticity → consistency → ReliabilityResult

Tools used:
  tools.fake_image_detector.pipeline   — Stage 2: image authenticity (EXIF, checksum,
                                         Gemini extract, reverse image search, SynthID)
  tools.death_certificate_pipeline     — Stage 3: Gemini extracts certificate facts and
  .death_certificate_consistency         scores them against the claimant narrative

Weights: document=0.2, authenticity=0.4, consistency=0.4
Hard escalation (HUMAN_REVIEW or AUTO_REJECT from authenticity) overrides the band.

Environment:
  GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT  — required for authenticity (Gemini checks)
                                            and consistency stage
  Without credentials, Gemini checks are skipped inside fake_image_detector and
  consistency returns score=0.0 — pipeline still completes with honest partial data.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os

from tools.fake_image_detector.models import Escalation, ToolResult
from tools.fake_image_detector.pipeline import build_pipeline as _build_authenticity_pipeline
from tools.death_certificate_pipeline.death_certificate_consistency import (
    analyze_death_certificate_consistency,
)
from tools.death_certificate_pipeline.models import (
    AuthenticitySignal,
    Band,
    ConsistencySignal,
    DocumentSignal,
    ReliabilityResult,
    Submission,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineExecution:
    """Final reliability result plus bounded-stage inputs needed for diagnostics."""

    result: ReliabilityResult
    authenticity: ToolResult


_STAGE_WEIGHTS: dict[str, float] = {
    "document":     0.2,
    "authenticity": 0.4,
    "consistency":  0.4,
}

# Supported image formats detected from magic bytes (no PIL needed)
_IMAGE_MAGIC: list[tuple[bytes, int, str]] = [
    (b"\x49\x49\x2a\x00", 4, "tiff"),   # TIFF little-endian (common for scanned docs)
    (b"\x4d\x4d\x00\x2a", 4, "tiff"),   # TIFF big-endian
    (b"\xff\xd8",          2, "jpeg"),
    (b"\x89PNG",           4, "png"),
    (b"%PDF",              4, "pdf"),
]


# Stage 1: document — validate image format, determine legibility

async def _stage_document(submission: Submission) -> DocumentSignal:
    """Validate the image and detect its format from magic bytes.

    Runs without any external dependencies (PIL, Tesseract, Gemini).
    WEBP requires checking offset 8–12 so it is handled separately.
    """
    image = submission.image

    if not image:
        return DocumentSignal(legible=False, notes=["image is empty"])

    fmt = "unknown"
    for magic, length, name in _IMAGE_MAGIC:
        if image[:length] == magic:
            fmt = name
            break

    if fmt == "unknown" and len(image) >= 12 and image[8:12] == b"WEBP":
        fmt = "webp"

    legible = fmt != "unknown"
    notes   = [] if legible else [
        f"unrecognized image format — first 4 bytes: {image[:4].hex()}"
    ]

    logger.info("stage.document fmt=%s legible=%s bytes=%d", fmt, legible, len(image))

    return DocumentSignal(
        legible=legible,
        document_type="death_certificate" if legible else None,
        notes=notes,
    )


# Stage 2: authenticity — fake image detector

async def _stage_authenticity(submission: Submission) -> AuthenticitySignal:
    """Run the fake image detector against the certificate image.

    Uses _build_authenticity_pipeline() which loads the check configuration
    from tools/fake_image_detector/config/pipeline.yaml. Checks with missing
    optional dependencies (PIL, Tesseract, Google Vision) are skipped
    automatically — the pipeline still runs the remaining checks.

    context={"input_type": "document"} bypasses Tesseract-based auto-
    classification so the document check suite always runs regardless of
    whether Tesseract is installed.
    """
    pipeline = _build_authenticity_pipeline()
    result   = await pipeline.run(
        submission.image,
        context={"input_type": "document"},
    )

    logger.info(
        "stage.authenticity verdict=%s risk=%.3f escalation=%s checks=%d early_exit=%s",
        result.verdict.value, result.risk_score,
        result.escalation.value, len(result.checks), result.early_exit,
    )
    return AuthenticitySignal(result=result)


# Stage 3: consistency — Gemini certificate extraction + narrative scoring

async def _stage_consistency(submission: Submission) -> ConsistencySignal:
    """Extract certificate facts with Gemini and score them against the narrative.

    Uses analyze_death_certificate_consistency() which makes a single Gemini
    call to both extract visible certificate fields and compare them against
    the claimant's chat_history (submission.narrative).

    Requires GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT. If neither is set the
    stage returns a zero-confidence signal with a note — the pipeline still
    completes and the score reflects the missing data.
    """
    has_credentials = bool(
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )

    if not has_credentials:
        logger.warning("stage.consistency skipped — no Gemini credentials")
        return ConsistencySignal(
            consistency_score=0.0,
            consistency_label="unknown",
            confidence=0.0,
            summary=(
                "Consistency check skipped — "
                "set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT to enable."
            ),
        )

    try:
        result = analyze_death_certificate_consistency(
            chat_history=submission.narrative,
            image_bytes=submission.image,
        )
    except Exception as exc:
        logger.exception("stage.consistency failed")
        return ConsistencySignal(
            consistency_score=0.0,
            consistency_label="unknown",
            confidence=0.0,
            summary=f"Consistency check failed: {exc}",
        )

    logger.info(
        "stage.consistency score=%.3f label=%s confidence=%.3f fields=%d",
        result["consistency_score"], result["consistency_label"],
        result["confidence"], len(result.get("certificate") or {}),
    )

    return ConsistencySignal(
        consistency_score=result["consistency_score"],
        consistency_label=result["consistency_label"],
        confidence=result["confidence"],
        extracted_fields=result.get("certificate") or {},
        matches=result.get("matches", []),
        contradictions=result.get("mismatches", []),   # tool uses "mismatches"
        uncertain_points=result.get("uncertain_points", []),
        summary=result.get("summary", ""),
    )


# Stage 4: score — combine signals into a ReliabilityResult

async def _stage_score(
    doc:  DocumentSignal,
    auth: AuthenticitySignal,
    con:  ConsistencySignal,
) -> ReliabilityResult:
    """Combine the three stage signals into a weighted reliability score.

    Hard escalation from authenticity overrides the numeric band — any
    HUMAN_REVIEW or AUTO_REJECT from fake_image_detector forces ESCALATE
    regardless of how strong the document and consistency signals are.
    """
    doc_score  = 1.0 if doc.legible else 0.3
    auth_score = round(1.0 - auth.result.risk_score, 3)
    con_score  = con.consistency_score

    sub_scores: dict[str, float] = {
        "document":     round(doc_score, 3),
        "authenticity": auth_score,
        "consistency":  round(con_score, 3),
    }

    weighted  = sum(sub_scores[k] * _STAGE_WEIGHTS[k] for k in _STAGE_WEIGHTS)
    raw_score = max(1, min(100, round(weighted * 100)))
    flags:    list[str] = []

    if auth.result.escalation in (Escalation.HUMAN_REVIEW, Escalation.AUTO_REJECT):
        band = Band.ESCALATE
        flags.append("HARD_ESCALATION")
    elif raw_score >= 75:
        band = Band.HIGH
    elif raw_score >= 50:
        band = Band.MEDIUM
    elif raw_score >= 25:
        band = Band.LOW
    else:
        band = Band.ESCALATE

    justification = (
        f"document={'legible' if doc.legible else 'illegible'} "
        f"(type={doc.document_type or 'unknown'}), "
        f"authenticity={auth.result.verdict.value} "
        f"(risk={auth.result.risk_score:.2f}), "
        f"consistency={con.consistency_label} "
        f"(score={con.consistency_score:.2f})"
    )

    logger.info("stage.score raw=%d band=%s flags=%s", raw_score, band.value, flags)

    return ReliabilityResult(
        score=raw_score,
        band=band,
        sub_scores=sub_scores,
        weights=dict(_STAGE_WEIGHTS),
        flags=flags,
        justification=justification,
        extracted_fields=con.extracted_fields,
        # Required by tests/unit/test_whatsapp_dc_scenarios.py and the live
        # Gemini webhook eval so consistency evidence survives final scoring.
        matches=con.matches,
        mismatches=con.contradictions,
        uncertain_points=con.uncertain_points,
    )


# Entry point

async def run_pipeline_with_diagnostics(submission: Submission) -> PipelineExecution:
    """Run the pipeline and retain its authenticity result for E2E diagnostics."""
    doc_signal = await _stage_document(submission)
    auth_signal = await _stage_authenticity(submission)
    con_signal = await _stage_consistency(submission)
    result = await _stage_score(doc_signal, auth_signal, con_signal)
    return PipelineExecution(result=result, authenticity=auth_signal.result)


async def run_pipeline(submission: Submission) -> ReliabilityResult:
    """Run the full death certificate reliability pipeline.

    Input:  Submission(image: bytes, narrative: str, case_fields: dict)
    Output: ReliabilityResult(score, band, sub_scores, flags, extracted_fields)

    Called by poc/api.py (FastAPI POST /score) and poc/cli.py (CLI).
    """
    execution = await run_pipeline_with_diagnostics(submission)
    return execution.result
