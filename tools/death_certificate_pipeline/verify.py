"""Context-aware death-certificate verification tool.

Registered as the ``death_certificate_verification`` agent tool. The model calls
it with no arguments once a user has sent a document; the image itself is never
seen by the model. This handler pulls the user's most recent image from the
transient session store (via the run's SessionContext), runs the full
reliability pipeline, and hands every case to GiveLight: marked `accepted` when
it clears the acceptance policy, `needs_review` (with reasons) when it does not.
GiveLight's reviewers are the human review; nothing is held back here.

Flow: SessionContext → latest image → run_pipeline → (policy) → handoff.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from tools.death_certificate_pipeline.config_loader import ScoringConfig, default_scoring_config
from tools.death_certificate_pipeline.debug import build_verification_debug_event
from tools.death_certificate_pipeline.handoff import (
    DECISION_ACCEPTED,
    DECISION_NEEDS_REVIEW,
    build_handoff_payload,
    deliver_to_gl,
    new_case_reference,
)
from tools.death_certificate_pipeline.models import ReliabilityResult, Submission
from tools.death_certificate_pipeline.pipeline import run_pipeline_with_diagnostics
from tools.fake_image_detector.models import ToolResult

logger = logging.getLogger(__name__)

# Sent when there is no chat at all. Stated plainly so Gemini reports that
# there is no account rather than scoring against an invented one.
_NARRATIVE_FALLBACK = "(No messages from the claimant. They have not described the death.)"


def _contact_identifier(session_id: str | None) -> str | None:
    """Stable join key for GiveLight (placeholder until GL-20).

    Pseudonymous, not anonymous: this is an unsalted SHA-256 of the WhatsApp
    ID, which is a phone number and can be recovered by brute force. Do not
    treat anything carrying it as de-identified. The keyed equivalent lives in
    tools/privacy/pii_scrubber.py (_make_contact_identifier).
    """
    if not session_id:
        return None
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


def _accepted(result: ReliabilityResult, config: ScoringConfig) -> bool:
    """Whether the case is forwarded automatically (accept_bands in scoring.yaml)."""
    return result.band in config.accept_bands and "HARD_ESCALATION" not in result.flags


def _append_debug_event(
    deps: Any,
    payload: dict[str, Any],
    accepted: bool | None,
    authenticity: ToolResult | None = None,
) -> None:
    debug_events = getattr(deps, "debug_events", None)
    if debug_events is None:
        return
    debug_events.append(build_verification_debug_event(payload, accepted, authenticity))


# Plain words for flags a reviewer may see, so the headline says what was
# found rather than naming an internal flag.
_FLAG_WORDS = {
    "INTERNAL_INCONSISTENCY": "details on the certificate contradict each other",
    "FOUND_ONLINE": "the document was found published online",
    "POSSIBLE_STOCK": "the image matches a stock-photo site",
    "SYNTHID_WATERMARK_DETECTED": "the image carries Google's AI-generation watermark",
    "CHECKSUM_FAIL": "a check digit on the document does not validate",
    "FORGED_DOCUMENT": "the document looks forged",
    "DIGITALLY_MANIPULATED": "the document looks digitally altered",
    "EDITING_ARTIFACTS": "there are signs of editing",
    "TEMPLATE_DETECTED": "the document looks made from a template",
    "INCONSISTENT_SECURITY_FEATURES": "expected official markings are missing or inconsistent",
    "LANGUAGE_INCONSISTENCY": "the language does not match the issuing country",
    "PHOTO_OF_PHOTO": "it may be a photo of another photo or a screen",
}
_NOTES_PER_CHECK = 3


def _authenticity_reasons(authenticity: ToolResult) -> list[str]:
    """What the fraud checks found, in terms a reviewer can confirm or dismiss.

    A headline in plain words, then each flagging check's own observations
    quoted (Gemini names the exact details it objects to), so a false flag can
    be dismissed in seconds. A check that could not run is reported as such, so
    an outage is never mistaken for suspicion.
    """
    headline_words: list[str] = []
    details: list[str] = []
    flagged = errored = False
    for check in authenticity.checks:
        if check.skipped or (check.passed and not check.human_escalate and not check.error):
            continue
        if check.error:
            errored = True
            details.append(
                f"the {check.check} check could not run ({check.error[:120]}), "
                "so the document was not fully checked"
            )
            continue
        flagged = True
        for flag in check.flags:
            words = _FLAG_WORDS.get(flag)
            if words and words not in headline_words:
                headline_words.append(words)
        notes = (check.signals or {}).get("signals") or []
        details += [f"{check.check} noted: {str(note)[:200]}" for note in notes[:_NOTES_PER_CHECK]]
        domains = (check.signals or {}).get("domains") or []
        if domains:
            details.append(f"{check.check} found it on: {', '.join(map(str, domains[:5]))}")
        details += [str(reason)[:200] for reason in check.escalation_reasons]

    if errored and not flagged:
        # Nothing was found against the document; a check simply could not run.
        headline = "fraud check could not be completed, so the document was not fully checked"
    else:
        headline = f"fraud check flagged it (risk {authenticity.risk_score:.2f})"
        if headline_words:
            headline += ": " + "; ".join(headline_words)
    return [headline, *details]


def _review_reasons(
    result: ReliabilityResult, authenticity: ToolResult | None, config: ScoringConfig
) -> list[str]:
    """Plain reasons a reviewer can act on, for a case that was not auto-accepted."""
    reasons: list[str] = []
    if "UNREADABLE_DOCUMENT" in result.flags:
        reasons.append("the file could not be read as an image or PDF")
    if "HARD_ESCALATION" in result.flags and authenticity is not None:
        reasons.extend(_authenticity_reasons(authenticity))
    if "NO_CLAIMANT_ACCOUNT" in result.flags:
        reasons.append(
            "the claimant has not described what happened, so there was nothing "
            "to compare the certificate with"
        )
    if "CONSISTENCY_UNAVAILABLE" in result.flags:
        reasons.append("the claimant's account could not be compared with the certificate")
    if "CONSISTENCY_BELOW_MINIMUM" in result.flags:
        reasons.append(
            "the claimant's account and the certificate disagree "
            f"(consistency {result.sub_scores.get('consistency', 0.0):.2f} "
            f"< minimum {config.consistency_min_score:.2f})"
        )
    if not reasons:
        reasons.append(
            f"score {result.score} ({result.band.value}) is below automatic acceptance"
        )
    return reasons


def _summary(accepted: bool, handed_off: bool, case_reference: str | None) -> str:
    """Outcome text for the agent to relay to the claimant.

    A failed upload means GiveLight does not have the case and nothing else
    will pick it up, so the claimant is asked to resend rather than promised a
    handover that will not happen. No reference is given then: it would not
    point at anything.
    """
    if not handed_off:
        return (
            "The certificate could not be passed to GiveLight because of a technical "
            "problem on our side. Ask the user to send it again a little later."
        )
    if accepted:
        return (
            "Verification passed and the case was forwarded to GiveLight. "
            f"Case reference: {case_reference}."
        )
    return (
        "The case was passed to GiveLight for a person to review. "
        f"Case reference: {case_reference}. If the photo was unclear or cropped, "
        "a clearer photo of the full certificate may help."
    )


async def verify_death_certificate(ctx: Any) -> dict[str, Any]:
    """Verify the user's most recent document and hand off to GiveLight if it passes."""
    deps = getattr(ctx, "deps", None)
    session_id = getattr(deps, "session_id", None)
    store = getattr(deps, "store", None)

    media = store.load_latest_media(session_id) if (store is not None and session_id) else None
    if media is None:
        logger.info("verify.no_media session=%.8s", _contact_identifier(session_id) or "")
        payload = {
            "status": "no_document",
            "handed_off": False,
            "summary": "No document has been received yet — ask the user to send a photo or PDF of the death certificate.",
        }
        _append_debug_event(deps, payload, accepted=None)
        return payload

    image_bytes, mime_type = media
    narrative = (getattr(deps, "history_text", "") or "").strip() or _NARRATIVE_FALLBACK

    submission = Submission(
        image=image_bytes,
        narrative=narrative,
        case_fields={"channel": "whatsapp", "contact_identifier": _contact_identifier(session_id)},
    )

    execution = await run_pipeline_with_diagnostics(submission)
    result = execution.result

    # Same cached config the pipeline scored with.
    config = default_scoring_config()
    accepted = _accepted(result, config)
    decision = DECISION_ACCEPTED if accepted else DECISION_NEEDS_REVIEW
    case_reference = new_case_reference()

    handoff_payload = build_handoff_payload(
        result,
        contact_identifier=_contact_identifier(session_id),
        contact_phone=session_id,
        case_fields=submission.case_fields,
        case_reference=case_reference,
        decision=decision,
        review_reasons=[] if accepted else _review_reasons(result, execution.authenticity, config),
        authenticity=execution.authenticity,
        claimant_messages=list(getattr(deps, "claimant_messages", None) or []),
    )
    handed_off = await deliver_to_gl(handoff_payload, image_bytes, mime_type)
    # Only a delivered case has a reference that points at anything.
    reference = case_reference if handed_off else None

    logger.info(
        "verify.done session=%.8s score=%d band=%s decision=%s handed_off=%s case=%s",
        _contact_identifier(session_id) or "",
        result.score,
        result.band.value,
        decision,
        handed_off,
        reference or "-",
    )

    payload = {
        "status": "verified",
        "score": result.score,
        "band": result.band.value,
        "config_versions": result.config_versions,
        # Required by tests/unit/test_whatsapp_dc_scenarios.py so webhook/tool
        # responses expose the score breakdown and consistency evidence.
        "sub_scores": result.sub_scores,
        "accepted": accepted,
        "decision": decision,
        "handed_off": handed_off,
        "case_reference": reference,
        "flags": result.flags,
        "extracted_fields": result.extracted_fields,
        "justification": result.justification,
        "matches": getattr(result, "matches", []),
        "mismatches": getattr(result, "mismatches", []),
        "uncertain_points": getattr(result, "uncertain_points", []),
        "summary": _summary(accepted, handed_off, reference),
    }
    _append_debug_event(deps, payload, accepted=accepted, authenticity=execution.authenticity)
    return payload
