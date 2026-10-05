"""Run one stage of the death-certificate pipeline on one file and print what came out.

Stages, smallest first:

  document      Can the file be read, what format is it, what Gemini would be sent.
                Runs offline.
  consistency   Does Gemini find the claimant's story, and does the certificate match it?
  authenticity  Fraud checks: which ran, which flagged, which errored.
  verify        The whole thing, exactly as the WhatsApp tool runs it. Nothing is
                uploaded: the GiveLight payload is saved to --out instead.

Examples:

  python scripts/try_stage.py document     cert.jpeg
  python scripts/try_stage.py consistency  cert.jpeg "My father Ahmad died in Takengon in June 2022."
  python scripts/try_stage.py consistency  cert.jpeg                     # no story at all
  python scripts/try_stage.py authenticity cert.jpeg
  python scripts/try_stage.py verify       cert.jpeg --story-file narrative.txt --out payload.json
  python scripts/try_stage.py verify       cert.jpeg "..." --settings my_scoring.yaml

Stories are sent the way the live chat sends them ("user: ..."). A story file
that is already a transcript (lines starting "user:" / "assistant:") is used as is.

Gemini credentials come from the environment or .env: GOOGLE_CLOUD_PROJECT for
Vertex (needed by the agent and both Gemini fraud checks), or GEMINI_API_KEY
(enough for the consistency stage only).

Output can include names and other details read off the certificate. Keep any
--out file outside git (e2e_cases/ is ignored).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import mimetypes
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

STAGES = ("document", "consistency", "authenticity", "verify")
# Stands in for the claimant's WhatsApp number in the verify stage.
_PLACEHOLDER_PHONE = "000000000000"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one pipeline stage on one certificate.",
        epilog="See the module docstring for examples.",
    )
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("file", type=Path, help="certificate image or PDF")
    parser.add_argument("story", nargs="?", help="the claimant's account, in quotes")
    parser.add_argument("--story-file", type=Path, help="read the account from a file instead")
    parser.add_argument("--settings", type=Path, help="alternative scoring.yaml to use")
    parser.add_argument("--out", type=Path, help="verify: save the GiveLight payload here")
    parser.add_argument(
        "--simulate-upload-failure",
        action="store_true",
        help="verify: behave as if the Drive upload failed",
    )
    parser.add_argument("--json", action="store_true", help="print the raw result as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="show pipeline logs")
    return parser.parse_args(argv)


def _transcript(args: argparse.Namespace) -> str:
    """The claimant's account in the form the live chat produces, or "" for none."""
    text = args.story_file.read_text(encoding="utf-8") if args.story_file else (args.story or "")
    text = text.strip()
    if not text:
        return ""
    if any(line.startswith(("user:", "assistant:")) for line in text.splitlines()):
        return text
    return f"user: {text}"


def _banner() -> None:
    from tools.death_certificate_pipeline.config_loader import default_scoring_config
    from tools.fake_image_detector.config_loader import load_pipeline_config

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project:
        gemini = f"Vertex AI, project {project}"
    elif os.environ.get("GEMINI_API_KEY"):
        gemini = (
            "API key only — enough for consistency; the Gemini fraud checks need "
            "Vertex (set GOOGLE_CLOUD_PROJECT)"
        )
    else:
        gemini = "not configured — Gemini stages will report themselves unavailable"
    scoring = default_scoring_config()
    detector = load_pipeline_config()
    fallback = os.environ.get("VERTEX_LOCATION", "us-central1")

    def model(name: str, location: str | None, level: str | None) -> str:
        return f"{name} @ {location or fallback}, thinking {level or 'default'}"

    print(f"Gemini    {gemini}")
    print(f"Settings  scoring v{scoring.version}, detector v{detector.version}")
    print(f"Story     {model(scoring.consistency_model, scoring.consistency_location, scoring.consistency_thinking_level)}")
    print(f"Fraud     {model(detector.gemini.model, detector.gemini.location, detector.gemini.thinking_level)}")
    print()


def _row(label: str, value: Any) -> None:
    print(f"{label:<16}{value}")


def _bullets(label: str, items: list[str]) -> None:
    if items:
        print(f"{label}:")
        for item in items:
            print(f"  - {item}")


async def _document(data: bytes, path: Path, as_json: bool) -> None:
    from tools.death_certificate_pipeline.models import Submission
    from tools.death_certificate_pipeline.pipeline import _stage_document
    from tools.fake_image_detector.file_formats import gemini_payload, sniff

    signal = await _stage_document(Submission(image=data, narrative="n/a"))
    fmt = sniff(data)
    sent_bytes, sent_mime = gemini_payload(data)
    if as_json:
        print(json.dumps({
            **signal.model_dump(),
            "format": fmt.name if fmt else None,
            "gemini_mime_type": sent_mime,
            "converted_for_gemini": sent_bytes is not data,
        }, indent=2))
        return
    _row("File", f"{path.name} ({len(data):,} bytes)")
    _row("Format", f"{fmt.name if fmt else 'unrecognised'}")
    _row("Readable", "yes" if signal.legible else "NO — would go to review")
    if sent_bytes is not data:
        _row("Sent to Gemini", f"{sent_mime} (converted from {fmt.name if fmt else '?'})")
    else:
        _row("Sent to Gemini", f"{sent_mime}, unchanged")
    _bullets("Notes", signal.notes)


async def _consistency(data: bytes, transcript: str, as_json: bool) -> None:
    from tools.death_certificate_pipeline.models import Submission
    from tools.death_certificate_pipeline.pipeline import _stage_consistency
    from tools.death_certificate_pipeline.verify import _NARRATIVE_FALLBACK

    signal = await _stage_consistency(
        Submission(image=data, narrative=transcript or _NARRATIVE_FALLBACK)
    )
    if as_json:
        print(json.dumps(signal.model_dump(), indent=2, ensure_ascii=False))
        return
    _row("Story sent", "yes" if transcript else "none")
    _row("Account found", "yes" if signal.claimant_account_present else "NO")
    _row("Comparison", "ran" if signal.available else "did not run")
    if signal.available:
        _row(
            "Consistency",
            f"{signal.consistency_score:.2f} ({signal.consistency_label}), "
            f"confidence {signal.confidence:.2f}",
        )
    _bullets("Matches", signal.matches)
    _bullets("Mismatches", signal.contradictions)
    _bullets("Uncertain", signal.uncertain_points)
    if signal.summary:
        _row("Summary", signal.summary)
    if signal.extracted_fields:
        print("Read off the certificate:")
        for key, value in signal.extracted_fields.items():
            print(f"  {key}: {value}")


def _check_line(check: Any) -> str:
    # Same wording as the GiveLight payload, so the two never disagree.
    from tools.death_certificate_pipeline.debug import check_status

    status = check_status(check)
    if not check.skipped:
        detail = f"fake {check.fake_score:.2f} conf {check.confidence:.2f}"
    elif status.startswith("ran"):
        signals = check.signals or {}
        detail = ", ".join(f"{k}={v}" for k, v in signals.items() if k != "extracted")
    else:
        detail = ""
    flags = f"flags: {', '.join(check.flags)}" if check.flags else ""
    # check_status already carries the error for a check that ran.
    error = f"— {check.error[:100]}" if check.error and check.skipped else ""
    return "  ".join(part for part in (f"{check.check:<15}", status, detail, flags, error) if part)


async def _authenticity(data: bytes, as_json: bool) -> None:
    from tools.death_certificate_pipeline.debug import authenticity_summary
    from tools.death_certificate_pipeline.models import Submission
    from tools.death_certificate_pipeline.pipeline import _stage_authenticity

    signal = await _stage_authenticity(Submission(image=data, narrative="n/a"))
    result = signal.result
    if as_json:
        print(json.dumps(authenticity_summary(result), indent=2, ensure_ascii=False))
        return
    _row(
        "Verdict",
        f"{result.verdict.value}   risk {result.risk_score:.2f}   {result.escalation.value}",
    )
    if result.early_exit_reason:
        _row("Stopped early", result.early_exit_reason)
    print("Checks:")
    for check in result.checks:
        print(f"  {_check_line(check)}")


async def _verify(data: bytes, path: Path, transcript: str, args: argparse.Namespace) -> None:
    from tools.death_certificate_pipeline import verify as verify_module

    captured: dict[str, Any] = {}

    async def deliver_locally(payload: dict[str, Any], image_bytes: bytes, mime_type: str) -> bool:
        captured["payload"] = payload
        return not args.simulate_upload_failure

    verify_module.deliver_to_gl = deliver_locally

    declared_mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    store = SimpleNamespace(load_latest_media=lambda _session_id: (data, declared_mime))
    events: list[dict[str, Any]] = []
    ctx = SimpleNamespace(deps=SimpleNamespace(
        session_id=_PLACEHOLDER_PHONE, store=store, history_text=transcript, debug_events=events,
    ))

    result = await verify_module.verify_death_certificate(ctx)
    payload = captured.get("payload")

    if args.out and payload is not None:
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.json:
        print(json.dumps({"tool_result": result, "givelight_payload": payload}, indent=2, ensure_ascii=False))
        return

    _row("Decision", f"{result.get('decision')}   (score {result.get('score')}, band {result.get('band')})")
    _row("Flags", ", ".join(result.get("flags") or []) or "none")
    if payload:
        _bullets("Why it needs review", payload.get("review_reasons") or [])
    _row("Sub-scores", result.get("sub_scores"))
    _row("Case reference", result.get("case_reference") or "none (not delivered)")
    _row("Agent is told", result.get("summary"))
    if args.out and payload is not None:
        _row("Payload", f"saved to {args.out}")
    elif payload is not None:
        _row("Payload", "not saved; pass --out FILE to keep the GiveLight JSON")


async def _run(args: argparse.Namespace) -> int:
    if not args.file.is_file():
        print(f"No such file: {args.file}", file=sys.stderr)
        return 2
    data = args.file.read_bytes()
    transcript = _transcript(args)

    if not args.json:
        _banner()

    if args.stage == "document":
        await _document(data, args.file, args.json)
    elif args.stage == "consistency":
        await _consistency(data, transcript, args.json)
    elif args.stage == "authenticity":
        await _authenticity(data, args.json)
    else:
        await _verify(data, args.file, transcript, args)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Must be set before the scoring config is first loaded (it is cached).
    if args.settings:
        os.environ["B2_SCORING_CONFIG"] = str(args.settings.resolve())

    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:
        pass

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        # passporteye / scikit-image deprecation noise on import
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
