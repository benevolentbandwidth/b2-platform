"""Death certificate pipeline data models.

Submission  — input contract (image + narrative + case_fields)
Band        — customer-facing reliability band (HIGH/MEDIUM/LOW/ESCALATE)
Signal types — DocumentSignal, AuthenticitySignal, ConsistencySignal
ReliabilityResult — final output contract
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from tools.fake_image_detector.models import ToolResult


class Band(str, Enum):
    """Customer-facing reliability band.

    HIGH     — score 75–100: strong evidence, proceed with confidence.
    MEDIUM   — score 50–74: reasonable evidence, minor gaps present.
    LOW      — score 25–49: significant gaps, manual follow-up advised.
    ESCALATE — score 0–24 OR hard flag: requires immediate human review.
    """

    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    ESCALATE = "escalate"


class Submission(BaseModel):
    """Incoming case submission.

    narrative is a first-class field — it is the primary claim being
    cross-referenced against the certificate image.
    image is always bytes — conversion from Path or upload happens at the
    API/CLI boundary before Submission is constructed.
    """

    image: bytes
    narrative: str
    case_fields: dict[str, Any] = Field(default_factory=dict)

    @field_validator("narrative")
    @classmethod
    def narrative_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("narrative must not be blank")
        return v

    model_config = {"arbitrary_types_allowed": True}


class DocumentSignal(BaseModel):
    """Output of the document analysis stage.

    Answers: can we read this document and what type is it?
    """

    legible: bool = False
    document_type: str | None = None
    page_count: int = 1
    language_hint: str | None = None
    notes: list[str] = Field(default_factory=list)
    stage: str = "document_analysis"


class AuthenticitySignal(BaseModel):
    """Output of the authenticity stage.

    Wraps ToolResult from the fake_image_detector pipeline directly.
    """

    result: ToolResult
    # pipeline.yaml version the detector ran with.
    config_version: int | None = None
    stage: str = "authenticity"

    model_config = {"arbitrary_types_allowed": True}


class ConsistencySignal(BaseModel):
    """Output of the narrative consistency stage.

    Maps from analyze_death_certificate_consistency() output.
    extracted_fields carries certificate data through to ReliabilityResult.

    Note: the tool returns 'mismatches' — map to 'contradictions' on construction.
    """

    consistency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # False when the stage could not run at all (no credentials, or the call
    # raised). Distinguishes "no evidence" from a genuine score of 0.0, which
    # otherwise both read as a perfect contradiction worth 0 points.
    available: bool = True
    # False when the claimant never described the death, so there was nothing
    # to compare the certificate with. Extracted fields are still kept.
    claimant_account_present: bool = True
    # From the claimant's own messages: relationship, dependants, other details.
    claimant: dict[str, Any] = Field(default_factory=dict)
    # Short plain-language note for a GiveLight caseworker.
    case_note: str = ""
    consistency_label: str = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    matches: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    uncertain_points: list[str] = Field(default_factory=list)
    summary: str = ""
    stage: str = "consistency"


class ReliabilityResult(BaseModel):
    """Final customer-facing reliability verdict.

    score        — 1–100 integer computed from weighted sub_scores.
    band         — customer-facing quality band (see Band enum).
    sub_scores   — per-stage scores (0.0–1.0).
    weights      — per-stage weights summing to 1.0.
    flags        — named concerns, e.g. ["HARD_ESCALATION"].
    extracted_fields — certificate data from ConsistencySignal.
    """

    score: int = Field(ge=1, le=100)
    band: Band
    sub_scores: dict[str, float] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    # Settings versions that produced this verdict:
    # {"scoring": scoring.yaml version, "detector": pipeline.yaml version}.
    config_versions: dict[str, int | None] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    justification: str
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    # Required by WhatsApp full-pipeline tests to report why the claimant
    # narrative matched, conflicted with, or could not be compared to the image.
    matches: list[str] = Field(default_factory=list)
    mismatches: list[str] = Field(default_factory=list)
    uncertain_points: list[str] = Field(default_factory=list)
    # Story-check output that used to be dropped before the handoff.
    story_summary: str = ""
    claimant: dict[str, Any] = Field(default_factory=dict)
    case_note: str = ""
