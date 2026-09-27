"""Read-through cache for secrets held in another GCP project's Secret Manager.

Components:
- get_secret(secret_name, project) -- fetch and cache one secret version
- resolve(env_var, secret_name, project) -- env var override, else Secret Manager

Invariant: this project (b2-platform) is not granted IAM access to browse or
list anything in the secret's home project. It can only call
accessSecretVersion on the exact secret names the project owner granted, via
this service's own runtime service account (ADC) -- no credentials in code.

Runtime requirement: WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID must
exist as Secret Manager secrets in the project passed as `project`
(default benevolent-bandwidth), readable by this service's service account.
"""

from __future__ import annotations

import base64
import logging
import os

import httpx

logger = logging.getLogger(__name__)

_SECRET_MANAGER_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_TIMEOUT = 10.0
DEFAULT_SECRET_PROJECT = "benevolent-bandwidth"

# Cached for the life of the process -- these values do not change within a
# single Cloud Run instance's lifetime, and re-fetching per request would add
# a network round trip to every inbound message.
_cache: dict[str, str] = {}


def _access_token() -> str:
    import google.auth
    from google.auth.exceptions import GoogleAuthError
    from google.auth.transport.requests import Request

    try:
        credentials, _ = google.auth.default(scopes=[_SECRET_MANAGER_SCOPE])
        credentials.refresh(Request())
    except GoogleAuthError as exc:
        raise ValueError("Secret Manager authentication failed") from exc
    if not credentials.token:
        raise ValueError("Secret Manager authentication returned no access token")
    return credentials.token


def get_secret(secret_name: str, *, project: str = DEFAULT_SECRET_PROJECT) -> str | None:
    """Return the latest version of secret_name from project, or None on failure."""
    cache_key = f"{project}/{secret_name}"
    if cache_key in _cache:
        return _cache[cache_key]

    url = (
        f"https://secretmanager.googleapis.com/v1/projects/{project}"
        f"/secrets/{secret_name}/versions/latest:access"
    )
    try:
        token = _access_token()
        response = httpx.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=_TIMEOUT)
        response.raise_for_status()
        encoded = response.json()["payload"]["data"]
        value = base64.b64decode(encoded).decode("utf-8")
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        logger.error("gcp_secrets.get_secret failed name=%s project=%s error=%s", secret_name, project, exc)
        return None

    _cache[cache_key] = value
    return value


def resolve(env_var: str, secret_name: str, *, project: str = DEFAULT_SECRET_PROJECT) -> str:
    """Return os.environ[env_var] if set, else the Secret Manager value, else "".

    The env var always wins so local development and tests can override
    without needing Secret Manager access or network calls.
    """
    value = os.getenv(env_var, "")
    if value:
        return value
    return get_secret(secret_name, project=project) or ""
