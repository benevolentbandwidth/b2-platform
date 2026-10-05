from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from e2e.payloads import mime_for_path
from src.session_store import MAX_MEDIA_BYTES


class InMemoryStore:
    """Stand-in for FirestoreSessionStore, shared across all local test turns."""

    def __init__(self) -> None:
        self._history: dict[str, list[Any]] = {}
        self._media: dict[str, tuple[bytes, str]] = {}

    def load_history(self, session_id: str) -> list[Any]:
        return list(self._history.get(session_id, []))

    def save_history(
        self,
        session_id: str,
        history: list[Any],
        *,
        agent_name: str | None = None,
        channel: str | None = None,
    ) -> None:
        self._history[session_id] = list(history)

    def save_media(self, session_id: str, image_bytes: bytes, *, mime_type: str) -> bool:
        # Same size limit as FirestoreSessionStore, so local runs refuse exactly
        # what production refuses.
        if len(image_bytes) > MAX_MEDIA_BYTES:
            return False
        self._media[session_id] = (image_bytes, mime_type)
        return True

    def load_latest_media(self, session_id: str) -> tuple[bytes, str] | None:
        return self._media.get(session_id)


def start_local_server(port: int, secret: str, *, debug_tools: bool = False) -> tuple[str, threading.Thread]:
    from src import api as api_module
    from src import chat as chat_module

    shared_store = InMemoryStore()
    media_by_id: dict[str, tuple[bytes, str]] = {}

    async def fake_download(media_id: str) -> tuple[bytes, str] | None:
        return media_by_id.get(media_id)

    chat_module.FirestoreSessionStore = lambda *a, **k: shared_store  # type: ignore[assignment]
    api_module.download_media = fake_download  # type: ignore[assignment]
    api_module._WEBHOOK_SECRET = secret
    if debug_tools:
        api_module._ENABLE_E2E_DEBUG = "true"
    api_module.app.state.e2e_media_by_id = media_by_id

    def serve() -> None:
        import uvicorn

        config = uvicorn.Config(
            api_module.app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            log_config=None,
        )
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None
        asyncio.run(server.serve())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{port}", thread


def wait_for_health(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + min(timeout, 30.0)
    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                response = client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
    raise RuntimeError(f"server did not become healthy at {base_url}")


def register_local_media(media_id: str, certificate_path: Path) -> None:
    from src import api as api_module

    media_by_id = getattr(api_module.app.state, "e2e_media_by_id")
    media_by_id[media_id] = (certificate_path.read_bytes(), mime_for_path(certificate_path))
