"""PII scrubbing layer — GL-1.

Single-file privacy toolkit. Import via the package:
    from tools.privacy import PiiScrubber, SessionRegistry, graceful_shutdown

What this module does:
  1. Scrubs PII from CanonicalMessage (Presidio-backed, config-driven).
  2. Derives a stable anonymous contact ID from the raw channel user ID (BLOCK-1).
  3. Writes PII audit entries — hashes in plaintext, originals encrypted for GL (BLOCK-2).
  4. Tracks session state and buffers PII in memory per conversation (BLOCK-3).
  5. Flushes buffers to encrypted disk on shutdown; re-delivers on startup (BLOCK-4).

Runtime env vars:
  B2_PII_HMAC_SECRET   — HMAC secret for contact_identifier (BLOCK-1)
  gl_public_key_path   — set in pii_config.yaml; path to GL's RSA public key (BLOCK-2)
"""

from __future__ import annotations

import asyncio
import base64
import hmac as _hmac
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

if TYPE_CHECKING:
    from src.envelope import CanonicalMessage

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "pii_config.yaml"
_SCHEMA_VERSION = 1


# BLOCK-2: Crypto helpers (AES-256-GCM + RSA-OAEP)
# B2 encrypts with GL's public key — only GL can decrypt with their private key.

def load_public_key(pem_path: str | Path) -> Any:
    return serialization.load_pem_public_key(Path(pem_path).read_bytes())


def encrypt_audit_entry(entry: dict[str, Any], public_key: Any) -> str:
    """Encrypt one audit entry → JSON line. Fresh AES key per call."""
    plaintext     = json.dumps(entry, ensure_ascii=False).encode()
    aes_key, nonce = os.urandom(32), os.urandom(12)
    ciphertext    = AESGCM(aes_key).encrypt(nonce, plaintext, None)
    wrapped_key   = public_key.encrypt(
        aes_key,
        padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )
    return json.dumps({
        "v": _SCHEMA_VERSION,
        "k": base64.b64encode(wrapped_key).decode(),
        "n": base64.b64encode(nonce).decode(),
        "c": base64.b64encode(ciphertext).decode(),
    })


def decrypt_audit_entry(line: str, private_key: Any) -> dict[str, Any]:
    """GL-side only. Decrypts one encrypted audit line."""
    p = json.loads(line)
    if p.get("v") != _SCHEMA_VERSION:
        raise ValueError(f"unsupported audit schema version: {p.get('v')}")
    aes_key = private_key.decrypt(
        base64.b64decode(p["k"]),
        padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
    )
    return json.loads(AESGCM(aes_key).decrypt(base64.b64decode(p["n"]), base64.b64decode(p["c"]), None))


# Config

@dataclass(frozen=True)
class EntityConfig:
    name: str
    enabled: bool = True
    recognizer: str = "regex"
    allow_list: tuple[str, ...] = ()
    deny_list: tuple[str, ...] = ()


@dataclass(frozen=True)
class NlpEngineConfig:
    provider: str = "spacy"
    model_name: str = "en_core_web_sm"


@dataclass(frozen=True)
class AuditStoreConfig:
    memory_ttl_hours: int = 24
    file_path: str = "data/pii_audit.jsonl"
    store_originals: bool = False           # True only during POC/QA
    gl_public_key_path: str | None = None   # path to GL's RSA PEM (gitignored)
    encrypted_audit_path: str = "data/pii_audit_encrypted.jsonl"


@dataclass(frozen=True)
class ContactIdentifierConfig:
    hmac_secret_env_var: str = "B2_PII_HMAC_SECRET"


@dataclass
class PiiConfig:
    score_threshold: float = 0.5
    max_retries: int = 3
    retry_timeout_ms: int = 200
    language: str = "en"
    entities: list[EntityConfig] = field(default_factory=list)
    nlp_engine: NlpEngineConfig = field(default_factory=NlpEngineConfig)
    audit_store: AuditStoreConfig = field(default_factory=AuditStoreConfig)
    contact_id: ContactIdentifierConfig = field(default_factory=ContactIdentifierConfig)

    @property
    def enabled_entity_names(self) -> list[str]:
        return [e.name for e in self.entities if e.enabled]

    @classmethod
    def from_yaml(cls, path: Path = _DEFAULT_CONFIG_PATH) -> PiiConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        s, a, n, c = (raw.get(k, {}) for k in ("scrubber", "audit_store", "nlp_engine", "contact_identifier"))
        return cls(
            score_threshold=s.get("score_threshold", 0.5),
            max_retries=s.get("max_retries", 3),
            retry_timeout_ms=s.get("retry_timeout_ms", 200),
            language=s.get("language", "en"),
            entities=[
                EntityConfig(
                    name=e["name"], enabled=e.get("enabled", True),
                    recognizer=e.get("recognizer", "regex"),
                    allow_list=tuple(e.get("allow_list") or []),
                    deny_list=tuple(e.get("deny_list") or []),
                )
                for e in raw.get("entities", [])
            ],
            nlp_engine=NlpEngineConfig(
                provider=n.get("provider", "spacy"),
                model_name=n.get("model_name", "en_core_web_sm"),
            ),
            audit_store=AuditStoreConfig(
                memory_ttl_hours=a.get("memory_ttl_hours", 24),
                file_path=a.get("file_path", "data/pii_audit.jsonl"),
                store_originals=a.get("store_originals", False),
                gl_public_key_path=a.get("gl_public_key_path"),
                encrypted_audit_path=a.get("encrypted_audit_path", "data/pii_audit_encrypted.jsonl"),
            ),
            contact_id=ContactIdentifierConfig(
                hmac_secret_env_var=c.get("hmac_secret_env_var", "B2_PII_HMAC_SECRET"),
            ),
        )


# Data types

@dataclass(frozen=True)
class PiiEntity:
    """One detected PII item. Goes to audit store only — never the Agentic OS."""
    field_path: str
    entity_type: str
    original_value: str
    sha256_hash: str
    placeholder: str
    score: float
    start: int
    end: int


@dataclass
class ScrubResult:
    """Output of PiiScrubber.scrub().

    clean_message      → PII-free, safe for the Agentic OS.
    entities           → audit store only.
    contact_identifier → HMAC-SHA256 of channel_user_id; stable across sessions (BLOCK-1).
    """
    clean_message: CanonicalMessage
    entities: list[PiiEntity]
    scrub_failed: bool = False
    failure_reason: str | None = None
    scrubbed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    contact_identifier: str | None = None

    @property
    def found_pii(self) -> bool:
        return bool(self.entities)


# BLOCK-1: Contact identifier

def _make_contact_identifier(channel_user_id: str, secret: str) -> str:
    """HMAC-SHA256(channel_user_id, secret) → stable 64-char hex. Not reversible."""
    return _hmac.new(secret.encode(), channel_user_id.encode(), "sha256").hexdigest()


# Audit store

@dataclass
class _AuditEntry:
    session_id_hash: str
    scrubbed_at: str
    scrub_failed: bool
    failure_reason: str | None
    entities: list[dict[str, Any]]

    def to_json_line(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class PiiAuditStore:
    """Append-only PII audit store. Never read by the Agentic OS.

    store_originals=True  → plaintext (POC/QA only).
    store_originals=False + GL key → hashes in plaintext, originals encrypted (production).
    store_originals=False + no key → hashes only (no originals stored).
    """

    def __init__(self, config: AuditStoreConfig) -> None:
        self._config = config
        self._memory: list[_AuditEntry] = []
        self._day = datetime.now(timezone.utc).date()
        self._path = Path(config.file_path)
        self._enc  = Path(config.encrypted_audit_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._enc.parent.mkdir(parents=True, exist_ok=True)
        self._gl_key = self._load_gl_key()
        self._migrate_plaintext_audit()

    def record(self, session_id_hash: str, result: ScrubResult) -> None:
        self._rotate_if_new_day()
        plain_entities, enc_entities = [], []
        for e in result.entities:
            base = {
                "field_path": e.field_path, "entity_type": e.entity_type,
                "sha256_hash": e.sha256_hash, "placeholder": e.placeholder, "score": e.score,
            }
            if self._config.store_originals:
                base["original_value"] = e.original_value
            plain_entities.append(base)
            if not self._config.store_originals:
                enc_entities.append({**base, "original_value": e.original_value})

        entry = _AuditEntry(session_id_hash, result.scrubbed_at.isoformat(),
                            result.scrub_failed, result.failure_reason, plain_entities)
        self._memory.append(entry)
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(entry.to_json_line() + "\n")
        except OSError:
            logger.exception("pii_audit.write_failed")

        if enc_entities and self._gl_key:
            self._write_encrypted(session_id_hash, result, enc_entities)
        elif enc_entities:
            logger.warning("pii_audit.encrypted_skipped — GL key absent session=%.8s", session_id_hash)

    def today_count(self) -> int:
        self._rotate_if_new_day()
        return len(self._memory)

    def _load_gl_key(self) -> Any | None:
        path = self._config.gl_public_key_path
        if not path:
            logger.warning("pii_audit.gl_key_not_configured — encrypted audit disabled")
            return None
        try:
            return load_public_key(path)
        except Exception:
            logger.exception("pii_audit.gl_key_load_failed path=%s", path)
            return None

    def _write_encrypted(self, session_id_hash: str, result: ScrubResult, enc_entities: list) -> None:
        try:
            payload = {
                "session_id_hash": session_id_hash, "scrubbed_at": result.scrubbed_at.isoformat(),
                "scrub_failed": result.scrub_failed, "failure_reason": result.failure_reason,
                "entities": enc_entities,
            }
            with self._enc.open("a", encoding="utf-8") as fh:
                fh.write(encrypt_audit_entry(payload, self._gl_key) + "\n")
        except Exception:
            logger.exception("pii_audit.encrypted_write_failed session=%.8s", session_id_hash)

    def _migrate_plaintext_audit(self) -> None:
        """On startup: encrypt any plaintext original_value entries left from POC/QA mode."""
        if not self._path.exists():
            return
        try:
            lines = self._path.read_text(encoding="utf-8").strip().splitlines()
        except OSError:
            return
        stale = [l for l in lines if '"original_value"' in l]
        if not stale:
            return
        if not self._gl_key:
            logger.critical("pii_audit.migration_deferred %d entries — GL key absent", len(stale))
            return
        clean, migrated = [], 0
        for raw in lines:
            if '"original_value"' not in raw:
                clean.append(raw)
                continue
            try:
                rec = json.loads(raw)
                with self._enc.open("a", encoding="utf-8") as fh:
                    fh.write(encrypt_audit_entry(rec, self._gl_key) + "\n")
                for e in rec.get("entities", []):
                    e.pop("original_value", None)
                clean.append(json.dumps(rec, ensure_ascii=False))
                migrated += 1
            except Exception:
                logger.exception("pii_audit.migration_failed — entry kept")
                clean.append(raw)
        self._path.write_text("\n".join(clean) + ("\n" if clean else ""), encoding="utf-8")
        logger.info("pii_audit.migration_complete migrated=%d", migrated)

    def _rotate_if_new_day(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._day:
            self._memory.clear()
            self._day = today


# Scrubber

class _FailureKind:
    ENCODING         = "encoding"
    MODEL_NOT_LOADED = "model_not_loaded"
    MEMORY           = "memory"
    TIMEOUT          = "timeout"
    UNKNOWN          = "unknown"


def _classify_failure(exc: Exception) -> str:
    if isinstance(exc, (UnicodeDecodeError, UnicodeEncodeError)):
        return _FailureKind.ENCODING
    if isinstance(exc, MemoryError):
        return _FailureKind.MEMORY
    if isinstance(exc, TimeoutError):
        return _FailureKind.TIMEOUT
    if any(kw in str(exc).lower() for kw in ("model", "spacy", "not found", "pipeline")):
        return _FailureKind.MODEL_NOT_LOADED
    return _FailureKind.UNKNOWN


class PiiScrubber:
    """Presidio-backed scrubber for CanonicalMessage.

    Stateless — one instance per process. Scrubs text_content, session_context,
    prior_context. Fails open (nulls text_content) after exhausting retries.
    """

    def __init__(self, config: PiiConfig | None = None) -> None:
        self._config = config or PiiConfig.from_yaml()
        self._analyzer = self._build_analyzer()
        self._hmac_secret: str | None = os.environ.get(self._config.contact_id.hmac_secret_env_var)
        if not self._hmac_secret:
            logger.warning("pii.hmac_secret not set — contact_identifier will be None")

    def scrub(self, message: CanonicalMessage, channel_user_id: str | None = None) -> ScrubResult:
        """Scrub PII from message, attach contact_identifier if channel_user_id provided."""
        contact_id = (
            _make_contact_identifier(channel_user_id, self._hmac_secret)
            if channel_user_id and self._hmac_secret else None
        )
        deadline  = time.monotonic() + self._config.retry_timeout_ms / 1000.0
        last_exc: Exception | None = None
        last_kind = _FailureKind.UNKNOWN

        for attempt in range(1, self._config.max_retries + 1):
            if time.monotonic() >= deadline:
                break
            try:
                result = self._do_scrub(message)
                return replace(result, contact_identifier=contact_id)
            except Exception as exc:
                last_exc, last_kind = exc, _classify_failure(exc)
                logger.warning("pii.scrub attempt=%d/%d kind=%s", attempt, self._config.max_retries, last_kind)
                if last_kind == _FailureKind.MODEL_NOT_LOADED:
                    try:
                        self._analyzer = self._build_analyzer()
                    except Exception:
                        pass

        logger.error("pii.scrub fail_open kind=%s", last_kind)
        return ScrubResult(
            clean_message=replace(message, text_content=None),
            entities=[], scrub_failed=True,
            failure_reason=f"{last_kind}: {last_exc}",
            contact_identifier=contact_id,
        )

    def _do_scrub(self, message: CanonicalMessage) -> ScrubResult:
        entities: list[PiiEntity] = []
        enabled  = self._config.enabled_entity_names

        new_text = message.text_content
        if new_text is not None:
            new_text, found = self._scrub_text(self._safe(new_text), "text_content", enabled)
            entities.extend(found)

        ctx = deepcopy(message.session_context)
        self._scrub_dict(ctx, "session_context", enabled, entities)

        prior = deepcopy(message.prior_context)
        for i, item in enumerate(prior):
            if isinstance(item, dict):
                self._scrub_dict(item, f"prior_context[{i}]", enabled, entities)

        return ScrubResult(
            clean_message=replace(message, text_content=new_text, session_context=ctx, prior_context=prior),
            entities=entities,
        )

    def _scrub_text(self, text: str, path: str, enabled: list[str]) -> tuple[str, list[PiiEntity]]:
        results = self._analyzer.analyze(
            text=text, language=self._config.language,
            entities=enabled, score_threshold=self._config.score_threshold,
        )
        allow_map = {e.name: e.allow_list for e in self._config.entities if e.enabled}

        # Filter allow-list hits, deduplicate overlapping spans (keep highest score)
        filtered = [
            (r, text[r.start:r.end]) for r in results
            if not any(a.lower() == text[r.start:r.end].lower() for a in allow_map.get(r.entity_type, ()))
        ]
        filtered.sort(key=lambda x: x[0].score, reverse=True)
        accepted, occupied = [], []
        for r, orig in filtered:
            if not any(r.start < e and r.end > s for s, e in occupied):
                accepted.append((r, orig))
                occupied.append((r.start, r.end))

        entities = []
        for r, orig in sorted(accepted, key=lambda x: x[0].start, reverse=True):
            h = sha256(orig.encode()).hexdigest()
            ph = f"[{r.entity_type}_{h[:8]}]"
            entities.append(PiiEntity(path, r.entity_type, orig, h, ph, r.score, r.start, r.end))
            text = text[:r.start] + ph + text[r.end:]
        return text, entities

    def _scrub_dict(self, d: dict, path: str, enabled: list[str], out: list[PiiEntity]) -> None:
        for k, v in list(d.items()):
            cp = f"{path}.{k}"
            if isinstance(v, str):
                scrubbed, found = self._scrub_text(self._safe(v), cp, enabled)
                if found:
                    d[k] = scrubbed
                out.extend(found)
            elif isinstance(v, dict):
                self._scrub_dict(v, cp, enabled, out)
            elif isinstance(v, list):
                d[k] = self._scrub_list(v, cp, enabled, out)

    def _scrub_list(self, lst: list, path: str, enabled: list[str], out: list[PiiEntity]) -> list:
        result = []
        for i, item in enumerate(lst):
            ip = f"{path}[{i}]"
            if isinstance(item, str):
                scrubbed, found = self._scrub_text(self._safe(item), ip, enabled)
                out.extend(found)
                result.append(scrubbed)
            elif isinstance(item, dict):
                item = deepcopy(item)
                self._scrub_dict(item, ip, enabled, out)
                result.append(item)
            else:
                result.append(item)
        return result

    def _build_analyzer(self) -> Any:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        nlp_cfg  = self._config.nlp_engine
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": nlp_cfg.provider,
            "models": [{"lang_code": self._config.language, "model_name": nlp_cfg.model_name}],
        })
        nlp_engine = provider.create_engine()
        registry   = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)
        for ec in self._config.entities:
            if ec.enabled and ec.deny_list:
                from presidio_analyzer import PatternRecognizer
                registry.add_recognizer(PatternRecognizer(
                    supported_entity=ec.name, deny_list=list(ec.deny_list),
                    name=f"{ec.name}_deny_list",
                ))
        return AnalyzerEngine(registry=registry, nlp_engine=nlp_engine)

    @staticmethod
    def _safe(v: str) -> str:
        return v.encode("utf-8", errors="replace").decode("utf-8")


# BLOCK-3: In-memory PII buffer per conversation

class EncryptedPiiBuffer:
    """Thread-safe in-memory PII buffer. Flushed to encrypted disk on session close."""

    def __init__(self) -> None:
        self._buffers: dict[str, list[PiiEntity]] = {}
        self._lock = asyncio.Lock()

    async def append(self, conv_id: str, entities: list[PiiEntity]) -> None:
        if not entities:
            return
        async with self._lock:
            self._buffers.setdefault(conv_id, []).extend(entities)

    async def flush(self, conv_id: str) -> list[PiiEntity]:
        async with self._lock:
            return self._buffers.pop(conv_id, [])

    async def discard(self, conv_id: str) -> None:
        async with self._lock:
            self._buffers.pop(conv_id, None)

    def entity_count(self, conv_id: str) -> int:
        return len(self._buffers.get(conv_id, []))

    def active_conversation_ids(self) -> list[str]:
        return list(self._buffers.keys())


# BLOCK-3: Session state machine

class SessionState(StrEnum):
    ACTIVE   = "active"
    EXPIRING = "expiring"   # TTL hit — new messages create a fresh session
    FLUSHING = "flushing"   # delivery in progress
    FAILED   = "failed"     # delivery failed — bundle on disk, sweeper retries
    WIPED    = "wiped"      # terminal — entry removed


@dataclass
class SessionEntry:
    conversation_id: str
    session_id: str
    state: SessionState
    last_activity: datetime
    session_start: datetime
    contact_identifier: str | None = None


class SessionRegistry:
    """Async session state machine. Lock held only for in-memory ops, never across I/O."""

    def __init__(self) -> None:
        self._sessions:     dict[str, SessionEntry] = {}  # conv_id → entry
        self._active_index: dict[str, str]          = {}  # session_id → conv_id
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: str, contact_identifier: str | None = None) -> SessionEntry:
        """Return ACTIVE session or create a fresh one (even if old session is mid-flush)."""
        async with self._lock:
            conv_id = self._active_index.get(session_id)
            if conv_id:
                self._sessions[conv_id].last_activity = datetime.now(timezone.utc)
                return self._sessions[conv_id]
            now     = datetime.now(timezone.utc)
            new_id  = str(uuid.uuid4())
            entry   = SessionEntry(new_id, session_id, SessionState.ACTIVE, now, now, contact_identifier)
            self._sessions[new_id] = entry
            self._active_index[session_id] = new_id
            return entry

    async def sweep_expired(self, ttl_minutes: int) -> list[SessionEntry]:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=ttl_minutes)
        async with self._lock:
            expired = [
                e for e in self._sessions.values()
                if e.state == SessionState.ACTIVE and e.last_activity <= cutoff
            ]
            for e in expired:
                e.state = SessionState.EXPIRING
                self._active_index.pop(e.session_id, None)
            return expired

    async def mark_flushing(self, conv_id: str) -> bool:
        async with self._lock:
            e = self._sessions.get(conv_id)
            if not e or e.state not in (SessionState.EXPIRING, SessionState.FAILED):
                return False
            e.state = SessionState.FLUSHING
            return True

    async def mark_delivered(self, conv_id: str) -> None:
        async with self._lock:
            e = self._sessions.pop(conv_id, None)
            if e:
                self._active_index.pop(e.session_id, None)

    async def mark_failed(self, conv_id: str) -> None:
        async with self._lock:
            e = self._sessions.get(conv_id)
            if e:
                e.state = SessionState.FAILED

    def get_entry(self, conv_id: str) -> SessionEntry | None:
        return self._sessions.get(conv_id)

    def active_count(self) -> int:
        return len(self._active_index)

    def all_entries(self) -> list[SessionEntry]:
        return list(self._sessions.values())

    def failed_entries(self) -> list[SessionEntry]:
        return [e for e in self._sessions.values() if e.state == SessionState.FAILED]


# BLOCK-4: Lifecycle — disk persistence across restarts

class PiiDeliveryChannel(ABC):
    @abstractmethod
    async def deliver(self, conv_id: str, encrypted_bundle: bytes) -> bool: ...


class StubPiiDeliveryChannel(PiiDeliveryChannel):
    async def deliver(self, conv_id: str, encrypted_bundle: bytes) -> bool:
        logger.info("pii.delivery_stub conv_id=%.8s bytes=%d — GL-20 not wired", conv_id, len(encrypted_bundle))
        return True


class PiiBundleStore:
    """Append-only JSONL store for encrypted bundles awaiting delivery."""

    def __init__(self, path: str = "data/pii_pending_delivery.jsonl") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, conv_id: str, bundle: bytes) -> None:
        line = json.dumps({"conversation_id": conv_id, "bundle": base64.b64encode(bundle).decode()})
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            logger.exception("pii_bundle.save_failed conv_id=%.8s", conv_id)

    def load_pending(self) -> list[tuple[str, bytes]]:
        if not self._path.exists():
            return []
        out = []
        for raw in self._path.read_text(encoding="utf-8").strip().splitlines():
            try:
                r = json.loads(raw)
                out.append((r["conversation_id"], base64.b64decode(r["bundle"])))
            except Exception:
                logger.warning("pii_bundle.corrupt_line skipped")
        return out

    def remove(self, conv_id: str) -> None:
        if not self._path.exists():
            return
        lines = [l for l in self._path.read_text(encoding="utf-8").strip().splitlines()
                 if f'"{conv_id}"' not in l]
        self._path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def pending_count(self) -> int:
        return len(self.load_pending())


def _make_bundle(entry: SessionEntry, entities: list[PiiEntity], gl_key: Any) -> bytes:
    bundle = {
        "conversation_id": entry.conversation_id, "session_id_hash": entry.session_id,
        "contact_identifier": entry.contact_identifier,
        "session_start": entry.session_start.isoformat(),
        "session_end": datetime.now(timezone.utc).isoformat(),
        "entities": [{"entity_type": e.entity_type, "sha256_hash": e.sha256_hash,
                      "original_value": e.original_value} for e in entities],
    }
    return encrypt_audit_entry(bundle, gl_key).encode()


async def graceful_shutdown(
    registry: SessionRegistry, buffer: EncryptedPiiBuffer,
    bundle_store: PiiBundleStore, gl_public_key: Any,
) -> int:
    """Flush ACTIVE session buffers to encrypted disk before process exit."""
    if not gl_public_key:
        logger.critical("pii.shutdown GL key absent — buffers NOT persisted")
        return 0
    flushed = 0
    for entry in [e for e in registry.all_entries() if e.state == SessionState.ACTIVE]:
        entities = await buffer.flush(entry.conversation_id)
        if not entities:
            await registry.mark_delivered(entry.conversation_id)
            continue
        try:
            bundle_store.save(entry.conversation_id, _make_bundle(entry, entities, gl_public_key))
            await registry.mark_delivered(entry.conversation_id)
            flushed += 1
        except Exception:
            logger.exception("pii.shutdown encrypt_failed conv_id=%.8s", entry.conversation_id)
            await buffer.append(entry.conversation_id, entities)
    logger.info("pii.shutdown complete flushed=%d", flushed)
    return flushed


async def startup_recovery(bundle_store: PiiBundleStore, channel: PiiDeliveryChannel) -> tuple[int, int]:
    """Re-deliver bundles saved by a prior shutdown."""
    pending = bundle_store.load_pending()
    if not pending:
        return 0, 0
    delivered = failed = 0
    for conv_id, enc in pending:
        try:
            if await channel.deliver(conv_id, enc):
                bundle_store.remove(conv_id)
                delivered += 1
            else:
                failed += 1
        except Exception:
            logger.exception("pii.recovery error conv_id=%.8s", conv_id)
            failed += 1
    logger.info("pii.recovery delivered=%d failed=%d", delivered, failed)
    return delivered, failed


async def _flush_and_deliver(
    entry: SessionEntry, buffer: EncryptedPiiBuffer, channel: PiiDeliveryChannel,
    bundle_store: PiiBundleStore, registry: SessionRegistry, gl_public_key: Any,
) -> None:
    if not await registry.mark_flushing(entry.conversation_id):
        return
    entities = await buffer.flush(entry.conversation_id)
    if not entities:
        await registry.mark_delivered(entry.conversation_id)
        return
    if not gl_public_key:
        await registry.mark_failed(entry.conversation_id)
        return
    try:
        enc = _make_bundle(entry, entities, gl_public_key)
        if await channel.deliver(entry.conversation_id, enc):
            await registry.mark_delivered(entry.conversation_id)
        else:
            bundle_store.save(entry.conversation_id, enc)
            await registry.mark_failed(entry.conversation_id)
    except Exception:
        logger.exception("pii.sweeper flush_error conv_id=%.8s", entry.conversation_id)
        await buffer.append(entry.conversation_id, entities)
        await registry.mark_failed(entry.conversation_id)


async def run_sweeper(
    registry: SessionRegistry, buffer: EncryptedPiiBuffer, channel: PiiDeliveryChannel,
    bundle_store: PiiBundleStore, gl_public_key: Any,
    ttl_minutes: int = 30, interval_minutes: int = 15,
) -> None:
    """Background sweeper task — flush expired sessions, deliver to GL."""
    while True:
        await asyncio.sleep(interval_minutes * 60)
        try:
            expired = await registry.sweep_expired(ttl_minutes)
            for e in expired:
                await _flush_and_deliver(e, buffer, channel, bundle_store, registry, gl_public_key)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("pii.sweeper iteration_error")
