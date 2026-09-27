#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-testplan"
PYTHON_BIN="${VENV_DIR}/bin/python"
OUT_DIR="${E2E_OUT_DIR:-e2e_runs/local}"
if [[ "$OUT_DIR" = /* ]]; then
  OUT_DIR_PATH="$OUT_DIR"
else
  OUT_DIR_PATH="${ROOT_DIR}/${OUT_DIR}"
fi

cd "$ROOT_DIR"

if [[ "${E2E_LOAD_ENV:-1}" != "0" && -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

export ENABLE_E2E_DEBUG=true
export E2E_DISABLE_WHATSAPP_OUTBOUND=true
export ALLOW_UNAUTHENTICATED_WEBHOOK=true

if ! command -v uv >/dev/null 2>&1; then
  echo "FAIL: uv is not installed or not on PATH" >&2
  exit 1
fi

uv venv --allow-existing "$VENV_DIR"
uv pip install --python "$PYTHON_BIN" -r requirements-api.txt pytest pytest-asyncio

echo
echo "================================================================================"
echo "Running local e2e harness with structured tool debug verdicts"
echo "ENABLE_E2E_DEBUG is enabled for this local harness process only."
echo "================================================================================"
echo

latest_results_json() {
  if [[ ! -d "$OUT_DIR_PATH" ]]; then
    return 0
  fi
  find "$OUT_DIR_PATH" -mindepth 2 -maxdepth 2 -type f -name results.json -print \
    | sort \
    | tail -n 1
}

PREVIOUS_RESULTS_JSON="$(latest_results_json)"

if "$PYTHON_BIN" scripts/e2e_cases.py --debug-tools --out-dir "$OUT_DIR" "$@"; then
  E2E_STATUS=0
else
  E2E_STATUS=$?
fi

RESULTS_JSON="$(latest_results_json)"
MARKDOWN_STATUS=0
if [[ -n "$RESULTS_JSON" && "$RESULTS_JSON" != "$PREVIOUS_RESULTS_JSON" ]]; then
  echo
  echo "Generating clean Markdown report from ${RESULTS_JSON}"
  if "$PYTHON_BIN" tests/scripts/clean_results_to_markdown.py "$RESULTS_JSON"; then
    MARKDOWN_STATUS=0
  else
    MARKDOWN_STATUS=$?
  fi
else
  echo "FAIL: the E2E run did not produce a new results.json under ${OUT_DIR_PATH}" >&2
  MARKDOWN_STATUS=1
fi

if [[ "$E2E_STATUS" -ne 0 ]]; then
  exit "$E2E_STATUS"
fi
exit "$MARKDOWN_STATUS"
