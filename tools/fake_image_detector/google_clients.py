"""Google clients and credentials, created once per process and shared, and the
timeout and retry rules for every call made through them (Gemini and Vision).

Creating them per call ran Google's credential discovery every time. On a
machine signed in with the gcloud CLI that discovery starts a `gcloud`
subprocess (about 1 s per Gemini call), and starting a subprocess while other
threads were mid-call to the Vision API over gRPC froze the whole process for
about a minute: every in-flight Gemini call stalled with it and timed out
together. Clients are thread-safe, so one per configuration is shared; per-call
timeouts are set on each request instead of on the client.
"""

from __future__ import annotations

import threading
from typing import Any

# Waits between attempts. Immediate retries under a rate limit only hit it again.
RETRY_BACKOFF_SECONDS: tuple[float, ...] = (1.0, 2.0)
# Extra wait beyond every attempt's own timeout before a caller gives up.
TIMEOUT_GRACE_SECONDS = 5.0

# One lock per client or credential, so building or refreshing one never makes
# callers of another wait (a slow Drive token refresh used to hold up every
# Gemini call). _registry only guards the maps of locks and objects.
_registry = threading.Lock()
_locks: dict[tuple, threading.Lock] = {}
_objects: dict[tuple, Any] = {}


def _lock_for(key: tuple) -> threading.Lock:
    with _registry:
        return _locks.setdefault(key, threading.Lock())


def _shared(key: tuple, build: Any) -> Any:
    """The object cached under `key`, built once by `build()` if missing."""
    obj = _objects.get(key)
    if obj is not None:
        return obj
    with _lock_for(key):
        obj = _objects.get(key)
        if obj is None:
            obj = build()
            _objects[key] = obj
        return obj


def gemini_client(
    genai: Any,
    *,
    project: str | None = None,
    location: str | None = None,
    api_key: str | None = None,
) -> Any:
    """The shared google.genai client for Vertex (project, location) or an API key.

    `genai` is the module the caller imported, part of the key so a test that
    swaps the module gets its own client.
    """

    def build() -> Any:
        if project:
            return genai.Client(vertexai=True, project=project, location=location)
        return genai.Client(api_key=api_key)

    return _shared(("gemini", id(genai), project, location, api_key), build)


def request_timeout(gentypes: Any, timeout_seconds: float | None) -> Any:
    """Per-request HttpOptions ending one attempt after `timeout_seconds`, or None."""
    return gentypes.HttpOptions(timeout=int(timeout_seconds * 1000)) if timeout_seconds else None


def vision_client() -> Any:
    """The shared Vision ImageAnnotatorClient."""

    def build() -> Any:
        from google.cloud import vision

        return vision.ImageAnnotatorClient()

    return _shared(("vision",), build)


def access_token(scope: str) -> str:
    """A valid OAuth access token for `scope`, refreshed only when it expires.

    Raises google.auth's own errors on failure; callers decide what that means.
    """
    import google.auth
    from google.auth.transport.requests import Request

    credentials = _shared(
        ("credentials", scope), lambda: google.auth.default(scopes=[scope])[0]
    )
    if not credentials.valid:
        with _lock_for(("refresh", scope)):
            if not credentials.valid:
                credentials.refresh(Request())
    return credentials.token


def reset() -> None:
    """Forget every shared client and credential (for tests)."""
    with _registry:
        _objects.clear()
        _locks.clear()


def retry_wait(attempt: int) -> float:
    """Seconds to wait after failed attempt number `attempt` (0-based)."""
    return RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]


def retryable(exc: BaseException) -> bool:
    """Whether a failed Google call is worth another attempt.

    Timeouts, rate limits (429) and server errors (5xx) can clear on their own.
    Anything else, such as 400 for a setting the model rejects, 403 for an API
    that is switched off or 404 for a wrong model name, fails the same way
    again, so retrying only doubles the cost and the wait. google.genai and
    google.api_core errors both carry the HTTP status as an int `code`.
    """
    import httpx

    code = getattr(exc, "code", None)
    if isinstance(code, int) and not isinstance(code, bool):
        return code in (408, 429) or code >= 500
    return isinstance(exc, (TimeoutError, ConnectionError, httpx.TransportError))


def attempts_budget_seconds(attempts: int, timeout_seconds: float) -> float:
    """Longest a call with retries can take: every attempt at its full timeout,
    the waits between them, and a grace.

    A timeout applies to each attempt (the HTTP timeout ends it), so a stalled
    call is retried. An outer limit of a single timeout fired first and threw
    every retry away, while the thread went on making them.
    """
    waits = sum(retry_wait(i) for i in range(attempts - 1))
    return attempts * timeout_seconds + waits + TIMEOUT_GRACE_SECONDS
