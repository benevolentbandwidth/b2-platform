from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from e2e.stats import calculate_stats


def write_reports(out_dir: Path, results: list[dict[str, Any]], base_url: str) -> tuple[Path, Path, dict[str, Any]]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = out_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    passed = sum(1 for result in results if result["passed"])
    report = {
        "timestamp": timestamp,
        "base_url": base_url,
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "stats": calculate_stats(results),
        "results": results,
    }
    json_path = run_dir / "results.json"
    md_path = run_dir / "summary.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path, report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# E2E Run Summary",
        "",
        f"- Base URL: `{report['base_url']}`",
        f"- Total: {report['total']}",
        f"- Passed: {report['passed']}",
        f"- Failed: {report['failed']}",
        "",
        "## Stats",
        "",
        "| Scope | Total | Classified | Accuracy | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    overall = report["stats"]["overall"]
    lines.append(_stats_row("overall", overall))
    for country, stats in report["stats"]["by_country"].items():
        lines.append(_stats_row(country, stats))

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Result | Expected | Actual | Class | Authenticity Verdict | Risk Score | Escalation | Early Exit | Early Exit Reason | Final Response Chars | Summary Tool Response | Errors |",
            "| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for result in report["results"]:
        status = "PASS" if result["passed"] else "FAIL"
        errors = _markdown_cell("<br>".join(result["errors"]) if result["errors"] else "")
        summary_tool_response = _markdown_cell(result.get("summary_tool_response"))
        authenticity = _authenticity(result)
        verdict = _diagnostic_value(authenticity, "verdict")
        risk_score = _diagnostic_value(authenticity, "risk_score")
        escalation = _diagnostic_value(authenticity, "escalation")
        early_exit = _early_exit_value(authenticity)
        early_exit_reason = _early_exit_reason(authenticity)
        lines.append(
            f"| `{result['case']}` | {status} | {result['expected_outcome']} | "
            f"{result['actual_outcome']} | {result['classification']} | "
            f"{_markdown_cell(verdict)} | {_markdown_cell(risk_score)} | "
            f"{_markdown_cell(escalation)} | {_markdown_cell(early_exit)} | "
            f"{_markdown_cell(early_exit_reason)} | {len(result['final_response'])} | "
            f"{summary_tool_response} | {errors} |"
        )
    lines.append("")
    return "\n".join(lines)


def _stats_row(label: str, stats: dict[str, Any]) -> str:
    return (
        f"| `{label}` | {stats['total']} | {stats['classified']} | "
        f"{_percent(stats['accuracy'])} | {stats['false_positives']} | "
        f"{stats['false_negatives']} |"
    )


def _percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def _authenticity(result: dict[str, Any]) -> dict[str, Any] | None:
    tool_result = result.get("tool_result")
    if not isinstance(tool_result, dict):
        return None
    authenticity = tool_result.get("authenticity")
    return authenticity if isinstance(authenticity, dict) else None


def _diagnostic_value(authenticity: dict[str, Any] | None, key: str) -> Any:
    if authenticity is None:
        return "n/a"
    value = authenticity.get(key)
    return "n/a" if value is None else value


def _early_exit_value(authenticity: dict[str, Any] | None) -> str:
    if authenticity is None or "early_exit" not in authenticity:
        return "n/a"
    return "Yes" if authenticity["early_exit"] else "No"


def _early_exit_reason(authenticity: dict[str, Any] | None) -> str:
    if authenticity is None:
        return "n/a"
    if not authenticity.get("early_exit"):
        return ""
    return str(authenticity.get("early_exit_reason") or "n/a")
