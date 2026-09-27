#!/usr/bin/env python3
"""Gate a Cloud Run deploy on the service's live env vars and secret bindings.

Runs before the image is built, against the service's *current* revision --
env vars and secrets are never touched by the deploy step (it only swaps the
image), so whatever is missing now will still be missing after deploy.

Reads deploy/cloud_run_config.json for the required names per service.
Exits 1 (fails the build) if a required env var or secret binding is absent,
or if the service has no entry in the manifest at all. warn_env entries are
reported but never fail the build.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "deploy" / "cloud_run_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a Cloud Run service's env vars and secrets before deploy.")
    parser.add_argument("--service", required=True, help="Cloud Run service name, e.g. $_SERVICE_NAME.")
    parser.add_argument("--region", required=True, help="Cloud Run region, e.g. $_DEPLOY_REGION.")
    parser.add_argument("--project", required=True, help="GCP project id, e.g. $PROJECT_ID.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def access_token() -> str:
    result = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAIL: could not obtain an access token:\n{result.stderr}")
        sys.exit(1)
    return result.stdout.strip()


def describe_service(service: str, region: str, project: str) -> dict[str, Any]:
    """Fetch the service via the Cloud Run Admin API v2 REST endpoint.

    Not gcloud's own `describe` output: that command's JSON shape has changed
    across gcloud versions, while this REST schema (template.containers[].env)
    is the one every other check in this project already relies on.
    """
    url = f"https://run.googleapis.com/v2/projects/{project}/locations/{region}/services/{service}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {access_token()}"})
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        print(f"FAIL: could not describe service '{service}' in {project}/{region}: HTTP {exc.code} {exc.reason}")
        sys.exit(1)


def container_env(service_json: dict[str, Any]) -> list[dict[str, Any]]:
    containers = service_json.get("template", {}).get("containers", [])
    return containers[0].get("env", []) if containers else []


def is_secret_bound(entry: dict[str, Any]) -> bool:
    return "secretKeyRef" in entry.get("valueSource", {})


def main() -> int:
    args = parse_args()

    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = manifest.get(args.service)
    if config is None:
        print(f"FAIL: '{args.service}' has no entry in {args.manifest}. Add one before deploying.")
        return 1

    service_json = describe_service(args.service, args.region, args.project)
    env = container_env(service_json)
    names = {e["name"] for e in env}
    secret_names = {e["name"] for e in env if is_secret_bound(e)}

    failures: list[str] = []
    warnings: list[str] = []

    for name in config.get("required_env", []):
        if not any(e.get("name") == name and e.get("value") for e in env):

    for name in config.get("required_secrets", []):
        if name not in names:
            failures.append(f"missing required secret binding: {name}")
        elif name not in secret_names:
            failures.append(f"{name} is set as a plain value, not a Secret Manager reference")

    for name in config.get("warn_env", []):
        if name not in names:
            warnings.append(f"missing optional env var: {name}")

    print(f"Checked '{args.service}' against {args.manifest.name}:")
    for w in warnings:
        print(f"  WARN: {w}")
    for f in failures:
        print(f"  FAIL: {f}")
    if not failures and not warnings:
        print("  all required config present")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
