"""CLI entrypoint for the filesystem-driven /message e2e harness."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/e2e_cases.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Do not read local .env files when src.chat imports python-dotenv.
os.environ.setdefault("PYTHON_DOTENV_DISABLED", "1")

from e2e.cases import discover_cases
from e2e.local_app import start_local_server, wait_for_health
from e2e.reports import write_reports

DEFAULT_CASES_ROOT = "e2e_cases"
DEFAULT_OUT_DIR = "e2e_runs"
DEFAULT_PORT = 8765


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run filesystem-driven /message e2e cases.")
    parser.add_argument("--base-url", help="Existing service URL. If omitted, starts local uvicorn.")
    parser.add_argument("--cases-root", default=DEFAULT_CASES_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--secret", default=os.getenv("WEBHOOK_SECRET", ""))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--country")
    parser.add_argument("--kind", choices=("real", "fake"))
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Local uvicorn port when --base-url is omitted.")
    parser.add_argument(
        "--debug-tools",
        action="store_true",
        help="Request structured tool debug output for verdict-based metrics.",
    )
    return parser.parse_args()


def run_all(
    cases,
    *,
    base_url: str,
    secret: str,
    timeout: float,
    local_mode: bool,
    debug_tools: bool,
):
    import httpx

    from e2e.runner import run_case

    results = []
    total = len(cases)
    with httpx.Client(timeout=timeout) as client:
        for index, case in enumerate(cases, start=1):
            _print_progress(index, total, case.label)
            result = run_case(
                client,
                case,
                base_url=base_url,
                secret=secret,
                local_mode=local_mode,
                debug_tools=debug_tools,
            )
            results.append(result)
            _clear_progress_line()
            print(_format_realtime_result(result))
            for error in result["errors"]:
                print(f"  - {error}")
    return results


def _format_realtime_result(result: dict) -> str:
    status = _green("PASS") if result["passed"] else _red("FAIL")
    return (
        f"{status} {result['case']} "
        f"expected={result['expected_outcome']} actual={result['actual_outcome']} "
        f"class={result['classification']} ({len(result['final_response'])} response chars)"
    )


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[----------------------------] 0/0 0%"
    filled = round(width * done / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = round(100 * done / total)
    return f"[{bar}] {done}/{total} {percent}%"


def _print_progress(index: int, total: int, label: str) -> None:
    text = f"{_progress_bar(index - 1, total)} RUNNING {index}/{total} {label}"
    print(f"\r{text}", end="", flush=True)


def _clear_progress_line() -> None:
    print("\r\033[2K", end="", flush=True)


def _green(value: str) -> str:
    return _color(value, "32")


def _red(value: str) -> str:
    return _color(value, "31")


def _color(value: str, code: str) -> str:
    if os.getenv("NO_COLOR"):
        return value
    return f"\033[{code}m{value}\033[0m"


def main() -> int:
    args = parse_args()
    cases_root = (REPO_ROOT / args.cases_root).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()
    cases = discover_cases(cases_root, country=args.country, kind=args.kind)
    if not cases:
        print(f"No e2e cases found under {cases_root}")
        return 1

    local_mode = args.base_url is None
    if local_mode:
        base_url, _thread = start_local_server(args.port, args.secret, debug_tools=args.debug_tools)
        wait_for_health(base_url, args.timeout)
    else:
        base_url = args.base_url.rstrip("/")

    results = run_all(
        cases,
        base_url=base_url,
        secret=args.secret,
        timeout=args.timeout,
        local_mode=local_mode,
        debug_tools=args.debug_tools,
    )
    print()
    print("=" * 80)
    print()
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"{status} {result['case']} "
            f"expected={result['expected_outcome']} actual={result['actual_outcome']} "
            f"class={result['classification']} ({len(result['final_response'])} response chars)"
        )
        for error in result["errors"]:
            print(f"  - {error}")

    json_path, md_path, report = write_reports(out_dir, results, base_url)
    failed = report["failed"]
    stats = report["stats"]["overall"]
    accuracy = stats["accuracy"]
    accuracy_text = "n/a" if accuracy is None else f"{accuracy:.1%}"
    
    print()
    print("=" * 80)
    print(f"\nSummary: {report['passed']}/{report['total']} passed, {failed} failed")
    print(
        "Stats: "
        f"accuracy={accuracy_text}, FP={stats['false_positives']}, "
        f"FN={stats['false_negatives']}, unknown={stats['unknown']}"
    )
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
