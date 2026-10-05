"""Bounded E2E diagnostics for death-certificate verification."""

from __future__ import annotations

from typing import Any

from tools.fake_image_detector.models import ToolResult


def check_status(check: Any) -> str:
    """Plain status for one fraud check.

    Several report skipped=True for different reasons: some ran and only gather
    information, some never apply to a death certificate, some are waiting on
    access or setup. Without this they all read identically as "skipped".
    """
    signals = check.signals or {}
    if not check.skipped:
        if check.error:
            return f"error: {check.error[:120]}"
        return "passed" if check.passed else "flagged"
    if "reason" in signals:
        return f"unavailable: {signals['reason']}"
    if any(flag.endswith("_UNAVAILABLE") for flag in check.flags):
        # The check's own error says why (timeout, outage, API switched off,
        # missing library); a fixed guess sent people looking in the wrong place.
        errors = signals.get("errors") or []
        detail = signals.get("error") or (errors[-1] if errors else None) or check.error
        return f"unavailable: {str(detail)[:120]}" if detail else "unavailable"
    if signals:
        return "ran (information only, not scored)"
    return "not applicable to this document"


def authenticity_summary(authenticity: ToolResult) -> dict[str, Any]:
    """Fraud-check outcome without raw document-derived signals.

    Shared by the debug event and the GiveLight handoff, where it tells a
    reviewer which checks ran, which flagged and which errored.
    """
    return {
        "verdict": authenticity.verdict.value,
        "risk_score": authenticity.risk_score,
        "escalation": authenticity.escalation.value,
        "early_exit": authenticity.early_exit,
        "early_exit_reason": authenticity.early_exit_reason,
        "checks": [
            {
                "check": check.check,
                "status": check_status(check),
                "passed": check.passed,
                "skipped": check.skipped,
                "fake_score": check.fake_score,
                "confidence": check.confidence,
                "flags": list(check.flags),
                "human_escalate": check.human_escalate,
                "escalation_reasons": list(check.escalation_reasons),
                "error": check.error,
            }
            for check in authenticity.checks
        ],
    }


def build_verification_debug_event(
    payload: dict[str, Any],
    accepted: bool | None,
    authenticity: ToolResult | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe debug event without raw document-derived signals."""
    event: dict[str, Any] = {
        "tool": "death_certificate_verification",
        "status": payload.get("status"),
        "score": payload.get("score"),
        "band": payload.get("band"),
        "config_versions": dict(payload.get("config_versions") or {}),
        "decision": payload.get("decision"),
        "case_reference": payload.get("case_reference"),
        "accepted": accepted,
        "handed_off": bool(payload.get("handed_off", False)),
        "flags": list(payload.get("flags", [])),
        "extracted_fields": dict(payload.get("extracted_fields", {})),
        "summary": str(payload.get("summary", "")),
    }
    if authenticity is not None:
        event["authenticity"] = authenticity_summary(authenticity)
    return event
