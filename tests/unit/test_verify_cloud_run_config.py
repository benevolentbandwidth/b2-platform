"""Unit tests for scripts/verify_cloud_run_config.py."""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import verify_cloud_run_config as vcrc  # noqa: E402


def _service_json(env: list[dict]) -> dict:
    return {"template": {"containers": [{"env": env}]}}


def _fake_run(returncode: int, stdout: str = "token-123", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def _fake_urlopen(payload: dict):
    return lambda request: _FakeResponse(payload)


def _write_manifest(tmp_path: Path, config: dict) -> Path:
    manifest = tmp_path / "cloud_run_config.json"
    manifest.write_text(json.dumps(config), encoding="utf-8")
    return manifest


def _args(service: str, manifest: Path) -> list[str]:
    return ["prog", "--service", service, "--region=us-east1", "--project=b2-platform", f"--manifest={manifest}"]


def test_passes_when_all_required_config_present(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path, {
        "svc": {"required_env": ["GOOGLE_CLOUD_PROJECT"], "required_secrets": ["WEBHOOK_SECRET"], "warn_env": []},
    })
    env = [
        {"name": "GOOGLE_CLOUD_PROJECT", "value": "benevolent-bandwidth"},
        {"name": "WEBHOOK_SECRET", "valueSource": {"secretKeyRef": {"secret": "webhook-secret", "version": "latest"}}},
    ]
    monkeypatch.setattr(vcrc.subprocess, "run", lambda *a, **k: _fake_run(0))
    monkeypatch.setattr(vcrc.urllib.request, "urlopen", _fake_urlopen(_service_json(env)))
    monkeypatch.setattr(sys, "argv", _args("svc", manifest))

    assert vcrc.main() == 0


def test_fails_when_required_env_missing(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path, {
        "svc": {"required_env": ["GOOGLE_CLOUD_PROJECT"], "required_secrets": [], "warn_env": []},
    })
    monkeypatch.setattr(vcrc.subprocess, "run", lambda *a, **k: _fake_run(0))
    monkeypatch.setattr(vcrc.urllib.request, "urlopen", _fake_urlopen(_service_json([])))
    monkeypatch.setattr(sys, "argv", _args("svc", manifest))

    assert vcrc.main() == 1


def test_fails_when_secret_is_a_plain_value_not_a_binding(tmp_path, monkeypatch):
    """A hand-edited literal value where a secret reference should be is exactly the drift this guards against."""
    manifest = _write_manifest(tmp_path, {
        "svc": {"required_env": [], "required_secrets": ["WEBHOOK_SECRET"], "warn_env": []},
    })
    env = [{"name": "WEBHOOK_SECRET", "value": "hardcoded-oops"}]
    monkeypatch.setattr(vcrc.subprocess, "run", lambda *a, **k: _fake_run(0))
    monkeypatch.setattr(vcrc.urllib.request, "urlopen", _fake_urlopen(_service_json(env)))
    monkeypatch.setattr(sys, "argv", _args("svc", manifest))

    assert vcrc.main() == 1


def test_warn_env_missing_does_not_fail_the_build(tmp_path, monkeypatch, capsys):
    manifest = _write_manifest(tmp_path, {
        "svc": {"required_env": [], "required_secrets": [], "warn_env": ["GOOGLE_DRIVE_FOLDER_ID"]},
    })
    monkeypatch.setattr(vcrc.subprocess, "run", lambda *a, **k: _fake_run(0))
    monkeypatch.setattr(vcrc.urllib.request, "urlopen", _fake_urlopen(_service_json([])))
    monkeypatch.setattr(sys, "argv", _args("svc", manifest))

    assert vcrc.main() == 0
    assert "WARN" in capsys.readouterr().out


def test_fails_when_service_missing_from_manifest(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path, {"other-service": {"required_env": [], "required_secrets": []}})
    monkeypatch.setattr(sys, "argv", _args("svc", manifest))

    assert vcrc.main() == 1


def test_fails_when_access_token_call_errors(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path, {"svc": {"required_env": [], "required_secrets": []}})
    monkeypatch.setattr(vcrc.subprocess, "run", lambda *a, **k: _fake_run(1, stderr="not logged in"))
    monkeypatch.setattr(sys, "argv", _args("svc", manifest))

    try:
        vcrc.main()
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert exc.code == 1


def test_fails_when_describe_call_errors(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path, {"svc": {"required_env": [], "required_secrets": []}})

    def raise_http_error(request):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", hdrs=None, fp=io.BytesIO(b""))

    monkeypatch.setattr(vcrc.subprocess, "run", lambda *a, **k: _fake_run(0))
    monkeypatch.setattr(vcrc.urllib.request, "urlopen", raise_http_error)
    monkeypatch.setattr(sys, "argv", _args("svc", manifest))

    try:
        vcrc.main()
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert exc.code == 1
