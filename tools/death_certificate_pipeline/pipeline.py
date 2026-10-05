"""Death certificate reliability pipeline.

Workflow: Submission → document → authenticity → consistency → ReliabilityResult

Tools used:
  tools.fake_image_detector.pipeline   — Stage 2: image authenticity (EXIF, checksum,
                                         Gemini extract, reverse image search, SynthID)
  tools.death_certificate_pipeline     — Stage 3: Gemini extracts certificate facts and
  .death_certificate_consistency         scores them against the claimant narrative

Stage weights, band cutoffs, the acceptance rule and the consistency minimum
live in config/scoring.yaml (see config_loader.py). Hard escalation from
authenticity, an unavailable consistency stage, or a consistency score below
the minimum each override the numeric band.

Environment:
  GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT  — required for authenticity (Gemini checks)
                                            and consistency stage
  Without credentials, Gemini checks are skipped inside fake_image_detector and
  the consistency stage is marked unavailable, which escalates to human review.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import os

from tools.fake_image_detector.file_formats import sniff
from tools.fake_image_detector.google_clients import attempts_budget_seconds
from tools.fake_image_detector.models import Escalation, ToolResult, Verdict
from tools.fake_image_detector.pipeline import build_pipeline as _build_authenticity_pipeline
from tools.death_certificate_pipeline.death_certificate_consistency import (
    analyze_death_certificate_consistency,
)
from tools.death_certificate_pipeline.config_loader import ScoringConfig, default_scoring_config
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




# Stage 1: document — validate image format, determine legibility

async def _stage_document(submission: Submission) -> DocumentSignal:
    """Validate the image and detect its format from its leading bytes.

    Runs without any external dependencies (PIL, Tesseract, Gemini). Formats
    are defined once in tools.fake_image_detector.file_formats.
    """
    image = submission.image

    if not image:
        return DocumentSignal(legible=False, notes=["image is empty"])

    detected = sniff(image)
    fmt = detected.name if detected else "unknown"

    legible = detected is not None
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

    context={"input_type": "document", "doc_type": "death_certificate"}
    bypasses Tesseract-based auto-classification, so the document check suite
    always runs and the document type is never guessed from keywords.
    """
    pipeline = _build_authenticity_pipeline()
    result   = await pipeline.run(
        submission.image,
        # This pipeline only handles death certificates, so the type is stated
        # rather than guessed from English/German keywords, which miss
        # Indonesian, French and Arabic certificates.
        context={"input_type": "document", "doc_type": "death_certificate"},
    )

    logger.info(
        "stage.authenticity verdict=%s risk=%.3f escalation=%s checks=%d early_exit=%s",
        result.verdict.value, result.risk_score,
        result.escalation.value, len(result.checks), result.early_exit,
    )
    return AuthenticitySignal(result=result, config_version=pipeline.config.version)


# Stage 3: consistency — Gemini certificate extraction + narrative scoring

async def _stage_consistency(
    submission: Submission, config: ScoringConfig | None = None
) -> ConsistencySignal:
    """Extract certificate facts with Gemini and score them against the narrative.

    Uses analyze_death_certificate_consistency() which makes a single Gemini
    call to both extract visible certificate fields and compare them against
    the claimant's chat_history (submission.narrative).

    Requires GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT. If neither is set the
    stage returns a zero-confidence signal with a note — the pipeline still
    completes and the score reflects the missing data.
    """
    cfg = config or default_scoring_config()
    timeout = cfg.consistency_timeout_seconds  # per attempt
    budget = attempts_budget_seconds(cfg.consistency_attempts, timeout)

    has_credentials = bool(
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    )

    if not has_credentials:
        logger.warning("stage.consistency skipped — no Gemini credentials")
        return ConsistencySignal(
            consistency_score=0.0,
            available=False,
            consistency_label="unknown",
            confidence=0.0,
            summary=(
                "Consistency check skipped — "
                "set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT to enable."
            ),
        )

    try:
        # analyze_death_certificate_consistency is synchronous and makes a network
        # call; run it off the event loop so one slow request cannot stall every
        # other in-flight case on this instance.
        result = await asyncio.wait_for(
            asyncio.to_thread(
                analyze_death_certificate_consistency,
                chat_history=submission.narrative,
                image_bytes=submission.image,
                timeout_seconds=timeout,
                attempts=cfg.consistency_attempts,
                model=cfg.consistency_model,
                location=cfg.consistency_location,
                thinking_level=cfg.consistency_thinking_level,
            ),
            # Backstop only: each attempt's HTTP timeout is what ends it.
            timeout=budget,
        )
    except TimeoutError:
        logger.warning("stage.consistency timed out after %.0fs", budget)
        return ConsistencySignal(
            consistency_score=0.0,
            available=False,
            consistency_label="unknown",
            confidence=0.0,
            summary=(
                f"Consistency check timed out after {budget:.0f}s."
            ),
        )
    except Exception as exc:
        logger.exception("stage.consistency failed")
        return ConsistencySignal(
            consistency_score=0.0,
            available=False,
            consistency_label="unknown",
            confidence=0.0,
            summary=f"Consistency check failed: {exc}",
        )

    logger.info(
        "stage.consistency score=%.3f label=%s confidence=%.3f fields=%d",
        result["consistency_score"], result["consistency_label"],
        result["confidence"], len(result.get("certificate") or {}),
    )

    if not result.get("claimant_account_present", True):
        logger.info("stage.consistency no claimant account — comparison skipped")
        return ConsistencySignal(
            consistency_score=0.0,
            available=False,
            claimant_account_present=False,
            consistency_label="no_account",
            confidence=result["confidence"],
            extracted_fields=result.get("certificate") or {},
            uncertain_points=result.get("uncertain_points", []),
            summary="The claimant has not described the death; nothing to compare.",
            claimant=result.get("claimant") or {},
            case_note=result.get("case_note", ""),
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
        claimant=result.get("claimant") or {},
        case_note=result.get("case_note", ""),
    )


# Stage 4: score — combine signals into a ReliabilityResult

async def _stage_score(
    doc:  DocumentSignal,
    auth: AuthenticitySignal,
    con:  ConsistencySignal,
    config: ScoringConfig | None = None,
) -> ReliabilityResult:
    """Combine the three stage signals into a weighted reliability score.

    Hard escalation from authenticity overrides the numeric band — any
    HUMAN_REVIEW or AUTO_REJECT from fake_image_detector forces ESCALATE
    regardless of how strong the consistency signal is.
    """
    cfg = config or default_scoring_config()

    auth_score = round(1.0 - auth.result.risk_score, 3)
    con_score  = con.consistency_score

    sub_scores: dict[str, float] = {
        "authenticity": auth_score,
        "consistency":  round(con_score, 3),
    }

    weighted  = sum(sub_scores[k] * w for k, w in cfg.stage_weights.items())
    raw_score = max(1, min(100, round(weighted * 100)))
    flags:    list[str] = []

    if not doc.legible:
        # Recognising the file format is a gate, not evidence of authenticity:
        # it earns no points, but a file we cannot read at all needs a person.
        # The other stages are not run on it (run_pipeline_with_diagnostics),
        # so this is the only reason given and there is nothing to score.
        flags.append("UNREADABLE_DOCUMENT")
        sub_scores = {}
        raw_score = 1
    else:
        # Every reason to escalate is recorded, not just the first, so a
        # reviewer sees the whole picture.
        if auth.result.escalation in (Escalation.HUMAN_REVIEW, Escalation.AUTO_REJECT):
            flags.append("HARD_ESCALATION")
        if not con.claimant_account_present:
            flags.append("NO_CLAIMANT_ACCOUNT")
        elif not con.available:
            # The consistency stage never ran. Its 0.0 is absence of evidence,
            # not a contradiction — scoring it as one still cleared MEDIUM and
            # auto-forwarded the case.
            flags.append("CONSISTENCY_UNAVAILABLE")
        elif con.consistency_score < cfg.consistency_min_score:
            # The story and the certificate disagree too much to auto-accept,
            # however strong the other stages are.
            flags.append("CONSISTENCY_BELOW_MINIMUM")

    if flags:
        band = Band.ESCALATE
    elif raw_score >= cfg.band_high:
        band = Band.HIGH
    elif raw_score >= cfg.band_medium:
        band = Band.MEDIUM
    elif raw_score >= cfg.band_low:
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
        weights=dict(cfg.stage_weights),
        config_versions={"scoring": cfg.version, "detector": auth.config_version},
        flags=flags,
        justification=justification,
        extracted_fields=con.extracted_fields,
        # Required by tests/unit/test_whatsapp_dc_scenarios.py and the live
        # Gemini webhook eval so consistency evidence survives final scoring.
        matches=con.matches,
        mismatches=con.contradictions,
        uncertain_points=con.uncertain_points,
        story_summary=con.summary,
        claimant=con.claimant,
        case_note=con.case_note,
    )


def _not_run_on_unreadable_file() -> tuple[AuthenticitySignal, ConsistencySignal]:
    reason = "not run: the file could not be read as an image or PDF"
    authenticity = ToolResult(
        verdict=Verdict.FLAG,
        risk_score=0.0,
        escalation=Escalation.HUMAN_REVIEW,
        checks=[],
        early_exit=True,
        early_exit_reason=reason,
    )
    consistency = ConsistencySignal(
        consistency_score=0.0,
        available=False,
        consistency_label="unknown",
        confidence=0.0,
        summary=f"Consistency check {reason}.",
    )
    return AuthenticitySignal(result=authenticity), consistency


# Entry point

async def run_pipeline_with_diagnostics(
    submission: Submission, config: ScoringConfig | None = None
) -> PipelineExecution:
    """Run the pipeline and retain its authenticity result for E2E diagnostics.

    config defaults to config/scoring.yaml; pass one to evaluate alternatives.
    """
    cfg = config or default_scoring_config()
    doc_signal = await _stage_document(submission)
    if not doc_signal.legible:
        # Gemini rejects a file we cannot recognise, so both checks would fail
        # and the reviewer would see "could not be completed" reasons on top of
        # the one real problem. Neither is run.
        auth_signal, con_signal = _not_run_on_unreadable_file()
    else:
        # The fraud check and the story check are independent and both wait on
        # Gemini; run together, the claimant waits for the slower one, not both.
        auth_signal, con_signal = await asyncio.gather(
            _stage_authenticity(submission),
            _stage_consistency(submission, cfg),
        )
    result = await _stage_score(doc_signal, auth_signal, con_signal, cfg)
    return PipelineExecution(result=result, authenticity=auth_signal.result)


async def run_pipeline(
    submission: Submission, config: ScoringConfig | None = None
) -> ReliabilityResult:
    """Run the full death certificate reliability pipeline.

    Input:  Submission(image: bytes, narrative: str, case_fields: dict)
    Output: ReliabilityResult(score, band, sub_scores, flags, extracted_fields)

    Called by poc/api.py (FastAPI POST /score) and poc/cli.py (CLI).
    """
    execution = await run_pipeline_with_diagnostics(submission, config)
    return execution.result


_WARM_UP_TIMEOUT_SECONDS = 10.0


def warm_up_google_clients() -> None:
    """Create the shared Google clients and their credentials before the first case.

    Otherwise the first verification after a start does it inside worker threads,
    where on a gcloud-signed-in machine credential discovery starts a subprocess
    while other threads may be mid-call to Google, which can freeze the process.
    The Gemini SDK only loads credentials on a client's first request, so each
    model gets one token count (free, and served by the same endpoint as real
    calls). Each step is independent; failures are logged and left for the
    first real call to report.
    """
    import importlib

    from tools.fake_image_detector.config_loader import load_pipeline_config
    from tools.fake_image_detector.google_clients import gemini_client, request_timeout, vision_client

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        return
    fallback = os.environ.get("VERTEX_LOCATION", "us-central1")
    scoring = default_scoring_config()
    detector = load_pipeline_config()

    models = {(scoring.consistency_location or fallback, scoring.consistency_model)}
    if detector.gemini.enabled:
        models.add((detector.gemini.location or fallback, detector.gemini.model))
    enabled = {check.id: check for check in detector.checks if check.enabled}
    if "gemini_extract" in enabled:
        params = enabled["gemini_extract"].params
        models.add((params.get("location") or fallback, params["model"]))

    def warm_gemini(location: str, model: str) -> None:
        genai = importlib.import_module("google.genai")
        types = importlib.import_module("google.genai.types")
        gemini_client(genai, project=project, location=location).models.count_tokens(
            model=model,
            contents="warm-up",
            # Startup waits for this, so it must not hang: a stuck call would
            # keep the instance from ever serving and Cloud Run restarting it.
            config=types.CountTokensConfig(
                http_options=request_timeout(types, _WARM_UP_TIMEOUT_SECONDS)
            ),
        )

    steps = [
        (f"gemini {model} @ {location}", lambda location=location, model=model: warm_gemini(location, model))
        for location, model in sorted(models)
    ]
    if "reverse_image" in enabled:
        steps.append(("vision", vision_client))

    for name, step in steps:
        try:
            step()
            logger.info("warm_up.ok %s", name)
        except Exception as exc:
            logger.warning("warm_up.failed %s error=%.200s", name, exc)
