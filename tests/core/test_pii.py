"""PII scrubbing layer — consolidated deployment test suite.

Covers all critical paths for production deployment:
  - Config loading and production defaults
  - Core scrubbing (mocked Presidio — fast)
  - Fail-open safety guarantees
  - Privacy invariants
  - BLOCK-1: contact_identifier (channel_user_id → HMAC-SHA256)
  - BLOCK-2: encrypted audit log + migration guard + deployment script
  - BLOCK-3: session state machine + PII buffer (race condition fix)
  - WhatsApp real-Presidio scenarios (slow — uses actual spaCy model)

Run fast (mocked only):
    pytest tests/core/test_pii.py -v -k "not Scenario"

Run full (includes real Presidio):
    pytest tests/core/test_pii.py -v
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.envelope import (
    CanonicalMessage,
    InputType,
    LocationContext,
    LocationSource,
    SubmissionType,
    make_session_id,
)
from tools.privacy.pii_scrubber import (
    AuditStoreConfig,
    EntityConfig,
    NlpEngineConfig,
    PiiAuditStore,
    PiiConfig,
    PiiEntity,
    PiiScrubber,
    ScrubResult,
    _make_contact_identifier,
)
from tools.privacy import decrypt_audit_entry, encrypt_audit_entry


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_HMAC_SECRET = "test-secret-not-for-production"
_PHONE       = "+254712345678"


@pytest.fixture()
def minimal_config() -> PiiConfig:
    return PiiConfig(
        score_threshold=0.5,
        max_retries=3,
        retry_timeout_ms=200,
        language="en",
        entities=[
            EntityConfig(name="PHONE_NUMBER", enabled=True, recognizer="regex"),
            EntityConfig(name="EMAIL_ADDRESS", enabled=True, recognizer="regex"),
        ],
        nlp_engine=NlpEngineConfig(provider="spacy", model_name="en_core_web_sm"),
        audit_store=AuditStoreConfig(
            file_path="data/test_pii_audit.jsonl",
            store_originals=True,
        ),
    )


@pytest.fixture()
def mock_analyzer() -> MagicMock:
    a = MagicMock()
    a.analyze.return_value = []
    return a


@pytest.fixture()
def scrubber(minimal_config: PiiConfig, mock_analyzer: MagicMock) -> PiiScrubber:
    s = PiiScrubber.__new__(PiiScrubber)
    s._config   = minimal_config
    s._analyzer = mock_analyzer
    s._hmac_secret = _HMAC_SECRET
    return s


@pytest.fixture()
def whatsapp_msg() -> CanonicalMessage:
    return CanonicalMessage(
        session_id=make_session_id(_PHONE, "whatsapp"),
        channel="whatsapp",
        input_type=InputType.TEXT,
        text_content=f"My number is {_PHONE} and email is me@example.com.",
    )


@pytest.fixture(scope="module")
def rsa_keypair():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return priv, priv.public_key()


@pytest.fixture(scope="module")
def gl_public_key_pem_file(tmp_path_factory, rsa_keypair):
    _, pub = rsa_keypair
    pem  = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    path = tmp_path_factory.mktemp("keys") / "gl_public_key.pem"
    path.write_bytes(pem)
    return str(path)


def _pii_entity(original: str = _PHONE) -> PiiEntity:
    return PiiEntity(
        field_path="text_content", entity_type="PHONE_NUMBER",
        original_value=original, sha256_hash="a" * 64,
        placeholder="[PHONE_NUMBER_aaaaaaaa]", score=0.85, start=0, end=len(original),
    )


def _scrub_result(msg: CanonicalMessage, has_entity: bool = True) -> ScrubResult:
    return ScrubResult(
        clean_message=msg,
        entities=[_pii_entity()] if has_entity else [],
    )


# ---------------------------------------------------------------------------
# 1. Config — production defaults
# ---------------------------------------------------------------------------

class TestConfig:
    def test_phase1_entities_enabled(self) -> None:
        cfg = PiiConfig.from_yaml()
        enabled = cfg.enabled_entity_names
        assert all(e in enabled for e in
                   ["PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS"])

    def test_phase2_entities_disabled(self) -> None:
        cfg = PiiConfig.from_yaml()
        disabled = {e.name for e in cfg.entities if not e.enabled}
        assert {"PERSON", "LOCATION"} <= disabled

    def test_store_originals_defaults_false(self) -> None:
        # BLOCK-2: plaintext originals OFF in production.
        assert PiiConfig.from_yaml().audit_store.store_originals is False

    def test_hmac_secret_env_var_configured(self) -> None:
        # BLOCK-1: env var name present in config.
        assert PiiConfig.from_yaml().contact_id.hmac_secret_env_var == "B2_PII_HMAC_SECRET"


# ---------------------------------------------------------------------------
# 2. Core scrubbing (mocked Presidio)
# ---------------------------------------------------------------------------

class TestScrubbing:
    def test_phone_replaced_in_text(
        self, scrubber: PiiScrubber, whatsapp_msg: CanonicalMessage
    ) -> None:
        from presidio_analyzer import RecognizerResult
        scrubber._analyzer.analyze.return_value = [
            RecognizerResult("PHONE_NUMBER", 13, 26, 0.85)
        ]
        result = scrubber.scrub(whatsapp_msg)
        assert _PHONE not in (result.clean_message.text_content or "")
        assert result.found_pii

    def test_pii_in_session_context_stripped(
        self, scrubber: PiiScrubber
    ) -> None:
        from presidio_analyzer import RecognizerResult
        msg = CanonicalMessage(
            session_id=make_session_id(_PHONE, "whatsapp"),
            channel="whatsapp", input_type=InputType.TEXT,
            text_content="safe",
            session_context={"contact": _PHONE, "region": "Nairobi"},
        )
        scrubber._analyzer.analyze.side_effect = [
            [],
            [RecognizerResult("PHONE_NUMBER", 0, len(_PHONE), 0.9)],
            [],
        ]
        result = scrubber.scrub(msg)
        assert _PHONE not in result.clean_message.session_context["contact"]
        assert result.clean_message.session_context["region"] == "Nairobi"

    def test_pii_in_prior_context_stripped(
        self, scrubber: PiiScrubber
    ) -> None:
        from presidio_analyzer import RecognizerResult
        content = f"call {_PHONE}"
        msg = CanonicalMessage(
            session_id=make_session_id(_PHONE, "whatsapp"),
            channel="whatsapp", input_type=InputType.TEXT,
            text_content="",
            prior_context=[{"role": "user", "content": content}],
        )
        # analyze called 3x: text_content, prior_context[0].role, prior_context[0].content
        scrubber._analyzer.analyze.side_effect = [
            [],   # text_content ""
            [],   # "user" (role value)
            [RecognizerResult("PHONE_NUMBER", 5, 5 + len(_PHONE), 0.9)],
        ]
        result = scrubber.scrub(msg)
        assert _PHONE not in result.clean_message.prior_context[0]["content"]

    def test_original_message_not_mutated(
        self, scrubber: PiiScrubber, whatsapp_msg: CanonicalMessage
    ) -> None:
        from presidio_analyzer import RecognizerResult
        original_text = whatsapp_msg.text_content
        scrubber._analyzer.analyze.return_value = [
            RecognizerResult("PHONE_NUMBER", 13, 26, 0.85)
        ]
        scrubber.scrub(whatsapp_msg)
        assert whatsapp_msg.text_content == original_text

    def test_image_message_no_text_scrub(
        self, scrubber: PiiScrubber
    ) -> None:
        msg = CanonicalMessage(
            session_id=make_session_id(_PHONE, "whatsapp"),
            channel="whatsapp", input_type=InputType.IMAGE, text_content=None,
        )
        result = scrubber.scrub(msg)
        assert result.clean_message.text_content is None
        scrubber._analyzer.analyze.assert_not_called()

    def test_allow_list_prevents_flagging(
        self, scrubber: PiiScrubber
    ) -> None:
        from presidio_analyzer import RecognizerResult
        from tools.privacy import EntityConfig, PiiConfig
        cfg = PiiConfig(
            entities=[EntityConfig("LOCATION", True, "ner", allow_list=("Kenya",))],
            nlp_engine=scrubber._config.nlp_engine,
            audit_store=scrubber._config.audit_store,
        )
        s = PiiScrubber.__new__(PiiScrubber)
        s._config = cfg
        s._analyzer = scrubber._analyzer
        s._hmac_secret = _HMAC_SECRET
        msg = CanonicalMessage(
            session_id=make_session_id("u", "whatsapp"),
            channel="whatsapp", input_type=InputType.TEXT, text_content="I am in Kenya.",
        )
        scrubber._analyzer.analyze.return_value = [
            RecognizerResult("LOCATION", 8, 13, 0.9)
        ]
        result = s.scrub(msg)
        assert not result.found_pii  # Kenya in allow_list → not flagged


# ---------------------------------------------------------------------------
# 3. Fail-open safety
# ---------------------------------------------------------------------------

class TestFailOpen:
    def test_fail_open_nulls_text_content(
        self, minimal_config: PiiConfig, whatsapp_msg: CanonicalMessage
    ) -> None:
        s = PiiScrubber.__new__(PiiScrubber)
        s._config      = replace(minimal_config, max_retries=1)
        s._do_scrub    = MagicMock(side_effect=RuntimeError("Presidio down"))
        s._recover     = MagicMock()
        s._hmac_secret = _HMAC_SECRET

        result = s.scrub(whatsapp_msg)
        assert result.scrub_failed is True
        assert result.clean_message.text_content is None
        assert not result.found_pii

    def test_fail_open_preserves_non_text_fields(
        self, minimal_config: PiiConfig
    ) -> None:
        msg = CanonicalMessage(
            session_id=make_session_id(_PHONE, "whatsapp"),
            channel="whatsapp",
            input_type=InputType.IMAGE,
            language_hint="sw",
            location_context=LocationContext(
                country_code="KE", source=LocationSource.DEVICE, confidence=0.9
            ),
        )
        s = PiiScrubber.__new__(PiiScrubber)
        s._config      = replace(minimal_config, max_retries=1)
        s._do_scrub    = MagicMock(side_effect=MemoryError("OOM"))
        s._recover     = MagicMock()
        s._hmac_secret = _HMAC_SECRET

        result = s.scrub(msg)
        assert result.scrub_failed
        assert result.clean_message.language_hint == "sw"
        assert result.clean_message.location_context.country_code == "KE"


# ---------------------------------------------------------------------------
# 4. Privacy invariants
# ---------------------------------------------------------------------------

class TestPrivacyInvariants:
    def test_entities_not_on_clean_message(
        self, scrubber: PiiScrubber, whatsapp_msg: CanonicalMessage
    ) -> None:
        from presidio_analyzer import RecognizerResult
        scrubber._analyzer.analyze.return_value = [
            RecognizerResult("PHONE_NUMBER", 13, 26, 0.85)
        ]
        result = scrubber.scrub(whatsapp_msg)
        assert not hasattr(result.clean_message, "entities")
        assert not hasattr(result.clean_message, "channel_user_id")

    def test_agent_prompt_excludes_pii_keys(
        self, scrubber: PiiScrubber
    ) -> None:
        msg = CanonicalMessage(
            session_id=make_session_id(_PHONE, "whatsapp"),
            channel="whatsapp", input_type=InputType.TEXT,
            text_content="safe text",
        )
        scrubber._analyzer.analyze.return_value = []
        result = scrubber.scrub(msg)
        prompt = result.clean_message.to_agent_prompt()
        assert "session_id" not in prompt
        assert "channel" not in prompt
        assert "media_url" not in prompt


# ---------------------------------------------------------------------------
# 5. BLOCK-1 — contact identifier
# ---------------------------------------------------------------------------

class TestBlock1ContactIdentifier:
    def test_stable_same_inputs_same_output(self) -> None:
        assert (_make_contact_identifier(_PHONE, _HMAC_SECRET) ==
                _make_contact_identifier(_PHONE, _HMAC_SECRET))

    def test_different_users_different_ids(self) -> None:
        a = _make_contact_identifier("+254712000001", _HMAC_SECRET)
        b = _make_contact_identifier("+254712000002", _HMAC_SECRET)
        assert a != b

    def test_not_plain_sha256_of_phone(self) -> None:
        import hashlib
        plain = hashlib.sha256(_PHONE.encode()).hexdigest()
        assert _make_contact_identifier(_PHONE, _HMAC_SECRET) != plain

    def test_scrub_attaches_contact_id(
        self, scrubber: PiiScrubber, whatsapp_msg: CanonicalMessage
    ) -> None:
        result = scrubber.scrub(whatsapp_msg, channel_user_id=_PHONE)
        assert result.contact_identifier is not None
        assert len(result.contact_identifier) == 64

    def test_no_channel_user_id_gives_none(
        self, scrubber: PiiScrubber, whatsapp_msg: CanonicalMessage
    ) -> None:
        assert scrubber.scrub(whatsapp_msg).contact_identifier is None

    def test_no_secret_gives_none(
        self, scrubber: PiiScrubber, whatsapp_msg: CanonicalMessage
    ) -> None:
        scrubber._hmac_secret = None
        result = scrubber.scrub(whatsapp_msg, channel_user_id=_PHONE)
        scrubber._hmac_secret = _HMAC_SECRET  # restore
        assert result.contact_identifier is None

    def test_contact_id_present_on_fail_open(
        self, minimal_config: PiiConfig, whatsapp_msg: CanonicalMessage
    ) -> None:
        s = PiiScrubber.__new__(PiiScrubber)
        s._config      = replace(minimal_config, max_retries=1)
        s._do_scrub    = MagicMock(side_effect=RuntimeError("boom"))
        s._recover     = MagicMock()
        s._hmac_secret = _HMAC_SECRET
        result = s.scrub(whatsapp_msg, channel_user_id=_PHONE)
        assert result.scrub_failed and result.contact_identifier is not None

    def test_contact_id_not_on_clean_message(
        self, scrubber: PiiScrubber, whatsapp_msg: CanonicalMessage
    ) -> None:
        result = scrubber.scrub(whatsapp_msg, channel_user_id=_PHONE)
        assert not hasattr(result.clean_message, "contact_identifier")
        assert not hasattr(result.clean_message, "channel_user_id")


# ---------------------------------------------------------------------------
# 6. BLOCK-2 — encrypted audit write
# ---------------------------------------------------------------------------

class TestBlock2EncryptedAudit:
    def test_encrypt_decrypt_roundtrip(self, rsa_keypair) -> None:
        priv, pub = rsa_keypair
        entry = {"session_id_hash": "abc", "entities": [{"original_value": _PHONE}]}
        assert decrypt_audit_entry(encrypt_audit_entry(entry, pub), priv) == entry

    def test_different_ciphertexts_per_call(self, rsa_keypair) -> None:
        _, pub = rsa_keypair
        entry = {"x": "y"}
        assert encrypt_audit_entry(entry, pub) != encrypt_audit_entry(entry, pub)

    def test_hash_only_when_no_gl_key(
        self, tmp_path: Path, whatsapp_msg: CanonicalMessage
    ) -> None:
        cfg = AuditStoreConfig(
            file_path=str(tmp_path / "a.jsonl"),
            store_originals=False, gl_public_key_path=None,
            encrypted_audit_path=str(tmp_path / "enc.jsonl"),
        )
        store = PiiAuditStore(cfg)
        store.record(whatsapp_msg.session_id, _scrub_result(whatsapp_msg))
        data = json.loads((tmp_path / "a.jsonl").read_text())
        assert "original_value" not in data["entities"][0]
        assert not (tmp_path / "enc.jsonl").exists()

    def test_encrypted_when_gl_key_present(
        self, tmp_path: Path, whatsapp_msg: CanonicalMessage,
        gl_public_key_pem_file: str, rsa_keypair,
    ) -> None:
        priv, _ = rsa_keypair
        cfg = AuditStoreConfig(
            file_path=str(tmp_path / "a.jsonl"),
            store_originals=False, gl_public_key_path=gl_public_key_pem_file,
            encrypted_audit_path=str(tmp_path / "enc.jsonl"),
        )
        store = PiiAuditStore(cfg)
        store.record(whatsapp_msg.session_id, _scrub_result(whatsapp_msg))

        plain = json.loads((tmp_path / "a.jsonl").read_text())
        assert "original_value" not in plain["entities"][0]

        recovered = decrypt_audit_entry(
            (tmp_path / "enc.jsonl").read_text().strip(), priv
        )
        assert recovered["entities"][0]["original_value"] == _PHONE

    def test_store_originals_true_writes_plaintext(
        self, tmp_path: Path, whatsapp_msg: CanonicalMessage
    ) -> None:
        cfg = AuditStoreConfig(
            file_path=str(tmp_path / "a.jsonl"),
            store_originals=True,
            encrypted_audit_path=str(tmp_path / "enc.jsonl"),
        )
        store = PiiAuditStore(cfg)
        store.record(whatsapp_msg.session_id, _scrub_result(whatsapp_msg))
        data = json.loads((tmp_path / "a.jsonl").read_text())
        assert data["entities"][0]["original_value"] == _PHONE


# ---------------------------------------------------------------------------
# Shared helper — stale plaintext audit file
# ---------------------------------------------------------------------------

def _write_stale_audit(path: Path, session_id_hash: str = "test") -> None:
    entry = {
        "session_id_hash": session_id_hash,
        "scrubbed_at": "2026-06-07T00:00:00+00:00",
        "scrub_failed": False,
        "failure_reason": None,
        "entities": [{"original_value": _PHONE, "sha256_hash": "a" * 64,
                       "entity_type": "PHONE_NUMBER"}],
    }
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 7. BLOCK-2 — migration guard
# ---------------------------------------------------------------------------

class TestBlock2MigrationGuard:

    def test_migration_encrypts_and_strips_plaintext(
        self, tmp_path: Path, gl_public_key_pem_file: str, rsa_keypair
    ) -> None:
        priv, _ = rsa_keypair
        audit = tmp_path / "audit.jsonl"
        enc   = tmp_path / "enc.jsonl"
        _write_stale_audit(audit)

        PiiAuditStore(AuditStoreConfig(
            file_path=str(audit), store_originals=False,
            gl_public_key_path=gl_public_key_pem_file,
            encrypted_audit_path=str(enc),
        ))

        assert "original_value" not in json.loads(audit.read_text())["entities"][0]
        recovered = decrypt_audit_entry(enc.read_text().strip(), priv)
        assert recovered["entities"][0]["original_value"] == _PHONE

    def test_migration_deferred_without_gl_key(self, tmp_path: Path) -> None:
        audit = tmp_path / "audit.jsonl"
        enc   = tmp_path / "enc.jsonl"
        _write_stale_audit(audit)
        original = audit.read_text()

        PiiAuditStore(AuditStoreConfig(
            file_path=str(audit), store_originals=False,
            gl_public_key_path=None, encrypted_audit_path=str(enc),
        ))

        assert audit.read_text() == original   # NOT wiped
        assert not enc.exists()                # NOT encrypted (key absent)


# ---------------------------------------------------------------------------
# 8. BLOCK-2 — deployment script
# ---------------------------------------------------------------------------

class TestBlock2DeployScript:
    def test_script_migrates_and_returns_zero(
        self, tmp_path: Path, gl_public_key_pem_file: str, rsa_keypair
    ) -> None:
        from scripts.pre_deploy_pii_audit import migrate
        priv, _ = rsa_keypair
        audit = tmp_path / "audit.jsonl"
        enc   = tmp_path / "enc.jsonl"
        _write_stale_audit(audit, session_id_hash="deploy_test")

        assert migrate(gl_public_key_pem_file, str(audit), str(enc)) == 0
        assert "original_value" not in json.loads(audit.read_text())["entities"][0]
        recovered = decrypt_audit_entry(enc.read_text().strip(), priv)
        assert recovered["entities"][0]["original_value"] == _PHONE

    def test_script_returns_one_on_bad_key(self, tmp_path: Path) -> None:
        from scripts.pre_deploy_pii_audit import migrate
        audit = tmp_path / "audit.jsonl"
        _write_stale_audit(audit)
        assert migrate("/nonexistent/key.pem", str(audit), str(tmp_path / "enc.jsonl")) == 1


# ---------------------------------------------------------------------------
# 9. BLOCK-3 — session state machine + PII buffer (race condition fix)
# ---------------------------------------------------------------------------

class TestBlock3SessionRegistry:
    """Session state machine — prevents sweeper ↔ new-message race."""

    def test_get_or_create_returns_same_entry_for_active_session(self) -> None:
        from tools.privacy import SessionRegistry, SessionState
        registry = SessionRegistry()
        sid = "session-abc"
        e1 = asyncio.run(registry.get_or_create(sid))
        e2 = asyncio.run(registry.get_or_create(sid))
        assert e1.conversation_id == e2.conversation_id
        assert e1.state == SessionState.ACTIVE

    def test_get_or_create_new_session_on_unknown_session_id(self) -> None:
        from tools.privacy import SessionRegistry
        registry = SessionRegistry()
        e1 = asyncio.run(registry.get_or_create("sid-1"))
        e2 = asyncio.run(registry.get_or_create("sid-2"))
        assert e1.conversation_id != e2.conversation_id
        assert registry.active_count() == 2

    def test_get_or_create_creates_fresh_session_during_flushing(self) -> None:
        # BLOCK-3 core: new message during FLUSHING → new session immediately.
        from tools.privacy import SessionRegistry, SessionState
        registry = SessionRegistry()
        sid = "session-flush"

        old = asyncio.run(registry.get_or_create(sid))
        old_conv_id = old.conversation_id

        # Simulate session expiring + flushing
        asyncio.run(registry.sweep_expired(ttl_minutes=0))   # expires immediately
        asyncio.run(registry.mark_flushing(old_conv_id))

        # New message arrives while old session is FLUSHING
        new_entry = asyncio.run(registry.get_or_create(sid))

        assert new_entry.conversation_id != old_conv_id      # fresh session
        assert new_entry.state == SessionState.ACTIVE
        assert registry.get_entry(old_conv_id).state == SessionState.FLUSHING  # old still tracked

    def test_sweep_expired_transitions_stale_active_sessions(self) -> None:
        from tools.privacy import SessionRegistry, SessionState
        registry = SessionRegistry()
        e = asyncio.run(registry.get_or_create("stale-sid"))
        # Back-date last_activity to simulate inactivity
        e.last_activity = datetime.now(timezone.utc) - timedelta(minutes=40)

        expired = asyncio.run(registry.sweep_expired(ttl_minutes=30))

        assert len(expired) == 1
        assert expired[0].conversation_id == e.conversation_id
        assert expired[0].state == SessionState.EXPIRING
        assert registry.active_count() == 0  # removed from active index

    def test_sweep_expired_ignores_recent_sessions(self) -> None:
        from tools.privacy import SessionRegistry
        registry = SessionRegistry()
        asyncio.run(registry.get_or_create("recent-sid"))  # last_activity = now
        expired = asyncio.run(registry.sweep_expired(ttl_minutes=30))
        assert expired == []
        assert registry.active_count() == 1

    def test_mark_delivered_wipes_entry(self) -> None:
        from tools.privacy import SessionRegistry
        registry = SessionRegistry()
        e = asyncio.run(registry.get_or_create("sid-del"))
        conv_id = e.conversation_id
        e.last_activity = datetime.now(timezone.utc) - timedelta(minutes=40)
        asyncio.run(registry.sweep_expired(ttl_minutes=30))
        asyncio.run(registry.mark_flushing(conv_id))
        asyncio.run(registry.mark_delivered(conv_id))

        assert registry.get_entry(conv_id) is None
        assert registry.active_count() == 0

    def test_mark_failed_keeps_entry_as_failed(self) -> None:
        from tools.privacy import SessionRegistry, SessionState
        registry = SessionRegistry()
        e = asyncio.run(registry.get_or_create("sid-fail"))
        conv_id = e.conversation_id
        e.last_activity = datetime.now(timezone.utc) - timedelta(minutes=40)
        asyncio.run(registry.sweep_expired(ttl_minutes=30))
        asyncio.run(registry.mark_flushing(conv_id))
        asyncio.run(registry.mark_failed(conv_id))

        assert registry.get_entry(conv_id).state == SessionState.FAILED
        assert len(registry.failed_entries()) == 1

    def test_failed_session_can_retry_via_mark_flushing(self) -> None:
        from tools.privacy import SessionRegistry, SessionState
        registry = SessionRegistry()
        e = asyncio.run(registry.get_or_create("sid-retry"))
        conv_id = e.conversation_id
        e.last_activity = datetime.now(timezone.utc) - timedelta(minutes=40)
        asyncio.run(registry.sweep_expired(ttl_minutes=30))
        asyncio.run(registry.mark_flushing(conv_id))
        asyncio.run(registry.mark_failed(conv_id))

        # Retry: FAILED → FLUSHING again
        ok = asyncio.run(registry.mark_flushing(conv_id))
        assert ok is True
        assert registry.get_entry(conv_id).state == SessionState.FLUSHING

    def test_contact_identifier_preserved_through_transitions(self) -> None:
        from tools.privacy import SessionRegistry
        registry = SessionRegistry()
        e = asyncio.run(registry.get_or_create("sid-ci", contact_identifier="hmac-xyz"))
        assert e.contact_identifier == "hmac-xyz"
        conv_id = e.conversation_id
        e.last_activity = datetime.now(timezone.utc) - timedelta(minutes=40)
        asyncio.run(registry.sweep_expired(ttl_minutes=30))
        assert registry.get_entry(conv_id).contact_identifier == "hmac-xyz"


class TestBlock3PiiBuffer:
    """EncryptedPiiBuffer — in-memory PII accumulator."""

    def test_append_and_flush_roundtrip(self) -> None:
        from tools.privacy import EncryptedPiiBuffer
        buf = EncryptedPiiBuffer()
        entity = _pii_entity()
        asyncio.run(buf.append("conv-1", [entity]))
        entities = asyncio.run(buf.flush("conv-1"))
        assert len(entities) == 1
        assert entities[0].original_value == _PHONE

    def test_flush_empty_for_unknown_conv_id(self) -> None:
        from tools.privacy import EncryptedPiiBuffer
        buf = EncryptedPiiBuffer()
        assert asyncio.run(buf.flush("nonexistent")) == []

    def test_flush_clears_buffer(self) -> None:
        from tools.privacy import EncryptedPiiBuffer
        buf = EncryptedPiiBuffer()
        asyncio.run(buf.append("conv-2", [_pii_entity()]))
        asyncio.run(buf.flush("conv-2"))
        assert buf.entity_count("conv-2") == 0

    def test_discard_removes_without_returning(self) -> None:
        from tools.privacy import EncryptedPiiBuffer
        buf = EncryptedPiiBuffer()
        asyncio.run(buf.append("conv-3", [_pii_entity()]))
        asyncio.run(buf.discard("conv-3"))
        assert buf.entity_count("conv-3") == 0

    def test_multiple_appends_accumulate(self) -> None:
        from tools.privacy import EncryptedPiiBuffer
        buf = EncryptedPiiBuffer()
        asyncio.run(buf.append("conv-4", [_pii_entity("+254712000001")]))
        asyncio.run(buf.append("conv-4", [_pii_entity("+254712000002")]))
        entities = asyncio.run(buf.flush("conv-4"))
        assert len(entities) == 2

    def test_empty_entities_list_no_op(self) -> None:
        from tools.privacy import EncryptedPiiBuffer
        buf = EncryptedPiiBuffer()
        asyncio.run(buf.append("conv-5", []))
        assert "conv-5" not in buf.active_conversation_ids()

    def test_separate_conversations_isolated(self) -> None:
        from tools.privacy import EncryptedPiiBuffer
        buf = EncryptedPiiBuffer()
        asyncio.run(buf.append("conv-a", [_pii_entity("+111")]))
        asyncio.run(buf.append("conv-b", [_pii_entity("+222")]))
        a = asyncio.run(buf.flush("conv-a"))
        b = asyncio.run(buf.flush("conv-b"))
        assert a[0].original_value == "+111"
        assert b[0].original_value == "+222"


# ---------------------------------------------------------------------------
# 10. BLOCK-4 — graceful shutdown, startup recovery, bundle store, sweeper
# ---------------------------------------------------------------------------

class TestBlock4BundleStore:
    """PiiBundleStore — encrypted bundle persistence across restarts."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        from tools.privacy import PiiBundleStore
        store = PiiBundleStore(str(tmp_path / "bundles.jsonl"))
        store.save("conv-abc", b"encrypted-payload")
        pending = store.load_pending()
        assert len(pending) == 1
        assert pending[0] == ("conv-abc", b"encrypted-payload")

    def test_remove_deletes_specific_entry(self, tmp_path: Path) -> None:
        from tools.privacy import PiiBundleStore
        store = PiiBundleStore(str(tmp_path / "bundles.jsonl"))
        store.save("conv-1", b"payload-1")
        store.save("conv-2", b"payload-2")
        store.remove("conv-1")
        pending = store.load_pending()
        assert len(pending) == 1
        assert pending[0][0] == "conv-2"

    def test_load_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        from tools.privacy import PiiBundleStore
        store = PiiBundleStore(str(tmp_path / "nonexistent.jsonl"))
        assert store.load_pending() == []

    def test_pending_count(self, tmp_path: Path) -> None:
        from tools.privacy import PiiBundleStore
        store = PiiBundleStore(str(tmp_path / "bundles.jsonl"))
        assert store.pending_count() == 0
        store.save("c1", b"x")
        store.save("c2", b"y")
        assert store.pending_count() == 2


class TestBlock4GracefulShutdown:
    """graceful_shutdown — flushes ACTIVE sessions to encrypted disk."""

    def test_shutdown_encrypts_active_sessions_to_disk(
        self, tmp_path: Path, rsa_keypair
    ) -> None:
        from tools.privacy import PiiBundleStore, graceful_shutdown
        from tools.privacy import SessionRegistry
        from tools.privacy import EncryptedPiiBuffer
        _, pub = rsa_keypair

        registry = SessionRegistry()
        buffer   = EncryptedPiiBuffer()
        store    = PiiBundleStore(str(tmp_path / "bundles.jsonl"))

        entry = asyncio.run(registry.get_or_create("sid-shutdown", "hmac-xyz"))
        asyncio.run(buffer.append(entry.conversation_id, [_pii_entity()]))

        flushed = asyncio.run(graceful_shutdown(registry, buffer, store, pub))

        assert flushed == 1
        assert store.pending_count() == 1
        assert registry.active_count() == 0  # session wiped from registry
        assert buffer.entity_count(entry.conversation_id) == 0  # buffer cleared

    def test_shutdown_skips_sessions_with_no_pii(
        self, tmp_path: Path, rsa_keypair
    ) -> None:
        from tools.privacy import PiiBundleStore, graceful_shutdown
        from tools.privacy import SessionRegistry
        from tools.privacy import EncryptedPiiBuffer
        _, pub = rsa_keypair

        registry = SessionRegistry()
        buffer   = EncryptedPiiBuffer()
        store    = PiiBundleStore(str(tmp_path / "bundles.jsonl"))

        asyncio.run(registry.get_or_create("sid-empty"))  # no PII added

        flushed = asyncio.run(graceful_shutdown(registry, buffer, store, pub))

        assert flushed == 0
        assert store.pending_count() == 0  # nothing to save

    def test_shutdown_no_gl_key_returns_zero(
        self, tmp_path: Path
    ) -> None:
        from tools.privacy import PiiBundleStore, graceful_shutdown
        from tools.privacy import SessionRegistry
        from tools.privacy import EncryptedPiiBuffer

        registry = SessionRegistry()
        buffer   = EncryptedPiiBuffer()
        store    = PiiBundleStore(str(tmp_path / "bundles.jsonl"))

        entry = asyncio.run(registry.get_or_create("sid-nokey"))
        asyncio.run(buffer.append(entry.conversation_id, [_pii_entity()]))

        flushed = asyncio.run(graceful_shutdown(registry, buffer, store, gl_public_key=None))

        assert flushed == 0
        assert store.pending_count() == 0  # nothing saved without key


class TestBlock4StartupRecovery:
    """startup_recovery — re-attempts delivery of bundles from prior shutdown."""

    def test_recovery_delivers_pending_and_clears_store(self, tmp_path: Path) -> None:
        from tools.privacy import (
            PiiBundleStore, StubPiiDeliveryChannel, startup_recovery
        )
        store = PiiBundleStore(str(tmp_path / "bundles.jsonl"))
        store.save("conv-recover", b"encrypted-bundle")

        delivered, failed = asyncio.run(
            startup_recovery(store, StubPiiDeliveryChannel())
        )

        assert delivered == 1 and failed == 0
        assert store.pending_count() == 0

    def test_recovery_keeps_bundle_on_delivery_failure(self, tmp_path: Path) -> None:
        from tools.privacy import PiiBundleStore, PiiDeliveryChannel, startup_recovery

        class FailingChannel(PiiDeliveryChannel):
            async def deliver(self, conv_id, enc) -> bool:
                return False

        store = PiiBundleStore(str(tmp_path / "bundles.jsonl"))
        store.save("conv-fail", b"encrypted-bundle")

        delivered, failed = asyncio.run(startup_recovery(store, FailingChannel()))

        assert delivered == 0 and failed == 1
        assert store.pending_count() == 1  # still on disk

    def test_recovery_returns_zero_when_nothing_pending(self, tmp_path: Path) -> None:
        from tools.privacy import (
            PiiBundleStore, StubPiiDeliveryChannel, startup_recovery
        )
        store = PiiBundleStore(str(tmp_path / "empty.jsonl"))
        delivered, failed = asyncio.run(startup_recovery(store, StubPiiDeliveryChannel()))
        assert delivered == 0 and failed == 0

    def test_shutdown_then_recovery_full_roundtrip(
        self, tmp_path: Path, rsa_keypair
    ) -> None:
        from tools.privacy import (
            PiiBundleStore, StubPiiDeliveryChannel,
            graceful_shutdown, startup_recovery,
            SessionRegistry, EncryptedPiiBuffer,
        )
        _, pub = rsa_keypair

        # Simulate process A: session active, then shutdown
        registry = SessionRegistry()
        buffer   = EncryptedPiiBuffer()
        store    = PiiBundleStore(str(tmp_path / "bundles.jsonl"))

        entry = asyncio.run(registry.get_or_create("sid-roundtrip", "hmac-abc"))
        asyncio.run(buffer.append(entry.conversation_id, [_pii_entity()]))
        asyncio.run(graceful_shutdown(registry, buffer, store, pub))

        assert store.pending_count() == 1

        # Simulate process B: startup, recover
        delivered, failed = asyncio.run(
            startup_recovery(store, StubPiiDeliveryChannel())
        )
        assert delivered == 1 and failed == 0
        assert store.pending_count() == 0  # clean after recovery


class TestBlock4Sweeper:
    """run_sweeper — periodic lifecycle task."""

    def test_sweeper_flushes_expired_session(
        self, tmp_path: Path, rsa_keypair
    ) -> None:
        from tools.privacy import (
            PiiBundleStore, StubPiiDeliveryChannel, SessionRegistry, EncryptedPiiBuffer,
        )
        from datetime import timedelta
        _, pub = rsa_keypair

        registry = SessionRegistry()
        buffer   = EncryptedPiiBuffer()
        store    = PiiBundleStore(str(tmp_path / "bundles.jsonl"))
        channel  = StubPiiDeliveryChannel()

        entry = asyncio.run(registry.get_or_create("sid-sweep"))
        asyncio.run(buffer.append(entry.conversation_id, [_pii_entity()]))
        # Back-date so session appears expired
        entry.last_activity = datetime.now(timezone.utc) - timedelta(minutes=40)

        async def run_once():
            expired = await registry.sweep_expired(ttl_minutes=30)
            from tools.privacy.pii_scrubber import _flush_and_deliver
            for e in expired:
                await _flush_and_deliver(e, buffer, channel, store, registry, pub)

        asyncio.run(run_once())

        assert registry.get_entry(entry.conversation_id) is None  # delivered → wiped
        assert buffer.entity_count(entry.conversation_id) == 0


# ---------------------------------------------------------------------------
# 11. WhatsApp real-Presidio scenarios  (slow — uses actual spaCy model)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wa_scrubber() -> PiiScrubber:
    cfg = PiiConfig(
        score_threshold=0.4, max_retries=3, retry_timeout_ms=1000, language="en",
        entities=[
            EntityConfig(name="PHONE_NUMBER", enabled=True, recognizer="regex"),
            EntityConfig(name="EMAIL_ADDRESS", enabled=True, recognizer="regex"),
            EntityConfig(name="CREDIT_CARD",   enabled=True, recognizer="regex"),
            EntityConfig(name="IBAN_CODE",     enabled=True, recognizer="regex"),
            EntityConfig(name="IP_ADDRESS",    enabled=True, recognizer="regex"),
        ],
        nlp_engine=NlpEngineConfig(provider="spacy", model_name="en_core_web_sm"),
        audit_store=AuditStoreConfig(
            file_path="data/scenario_pii_audit.jsonl", store_originals=True,
            encrypted_audit_path="data/scenario_pii_audit_encrypted.jsonl",
        ),
    )
    s = PiiScrubber(config=cfg)
    s._hmac_secret = _HMAC_SECRET
    return s


def _wa_msg(phone: str, text: str, **kwargs) -> CanonicalMessage:
    return CanonicalMessage(
        session_id=make_session_id(phone, "whatsapp"),
        channel="whatsapp", input_type=InputType.TEXT,
        text_content=text, **kwargs,
    )


class TestWhatsAppScenarios:
    def test_intake_phone_and_email_stripped(self, wa_scrubber: PiiScrubber) -> None:
        msg = _wa_msg(
            _PHONE,
            f"My phone is {_PHONE}, email joseph.kamau@gmail.com. Plants turning yellow.",
            language_hint="en",
            submission_type=SubmissionType.INTAKE,
            location_context=LocationContext(
                country_code="KE", region="Nyanza", city="Kisumu",
                source=LocationSource.DEVICE, confidence=0.95,
            ),
        )
        result = wa_scrubber.scrub(msg)
        assert result.found_pii and not result.scrub_failed
        full = (result.clean_message.text_content or "")
        assert _PHONE not in full
        assert "joseph.kamau@gmail.com" not in full
        assert result.clean_message.location_context.city == "Kisumu"

    def test_multi_turn_prior_context_stripped(self, wa_scrubber: PiiScrubber) -> None:
        msg = _wa_msg(
            _PHONE,
            f"Call me at {_PHONE} for fertiliser delivery.",
            prior_context=[
                {"role": "user", "content": f"My number is {_PHONE} if needed."},
                {"role": "assistant", "content": "What crop issue are you facing?"},
                {"role": "user", "content": "Email joseph.kamau@gmail.com for reports."},
            ],
        )
        result = wa_scrubber.scrub(msg)
        assert result.found_pii
        full = (result.clean_message.text_content or "") + json.dumps(
            result.clean_message.prior_context
        )
        assert _PHONE not in full and "joseph.kamau@gmail.com" not in full
        assert "What crop issue are you facing?" in json.dumps(
            result.clean_message.prior_context
        )

    def test_farming_numbers_no_false_positives(self, wa_scrubber: PiiScrubber) -> None:
        msg = _wa_msg(
            "+254700000010",
            "Yield: 3.5 tons/hectare. Row spacing: 45cm. DAP: 50kg/acre.",
        )
        result = wa_scrubber.scrub(msg)
        for e in result.entities:
            assert e.original_value not in {"3.5", "45", "50"}

    def test_audio_session_context_email_stripped(self, wa_scrubber: PiiScrubber) -> None:
        msg = CanonicalMessage(
            session_id=make_session_id("+254700000020", "whatsapp"),
            channel="whatsapp", input_type=InputType.AUDIO,
            text_content=None,
            session_context={"transcription_by": "agent@b2.ai", "locale": "sw-KE"},
        )
        result = wa_scrubber.scrub(msg)
        assert "agent@b2.ai" not in result.clean_message.session_context.get(
            "transcription_by", ""
        )
        assert result.clean_message.session_context["locale"] == "sw-KE"

    def test_fail_open_real_presidio(self, wa_scrubber: PiiScrubber) -> None:
        msg = _wa_msg("+254791123456", f"My number is {_PHONE}")
        original = wa_scrubber._do_scrub
        try:
            wa_scrubber._do_scrub = MagicMock(side_effect=RuntimeError("Presidio down"))
            result = wa_scrubber.scrub(msg)
        finally:
            wa_scrubber._do_scrub = original
        assert result.scrub_failed and result.clean_message.text_content is None

    def test_prompt_format_excludes_pii_keys(self, wa_scrubber: PiiScrubber) -> None:
        msg = _wa_msg(
            "+254712000001", f"Call me at +254 712 000 001.",
            language_hint="en",
            submission_type=SubmissionType.INTAKE,
            location_context=LocationContext(
                country_code="KE", city="Kisumu",
                source=LocationSource.DEVICE, confidence=0.95,
            ),
            session_context={"program": "farming-kenya"},
        )
        result = wa_scrubber.scrub(msg)
        prompt = result.clean_message.to_agent_prompt()
        assert "session_id" not in prompt and "channel" not in prompt
        assert "+254 712 000 001" not in prompt.get("text_content", "")
        assert prompt["location_context"]["city"] == "Kisumu"

    def test_contact_id_present_and_phone_not_leaked(
        self, wa_scrubber: PiiScrubber
    ) -> None:
        msg = _wa_msg(_PHONE, f"My number is {_PHONE}.")
        result = wa_scrubber.scrub(msg, channel_user_id=_PHONE)
        assert result.contact_identifier is not None and len(result.contact_identifier) == 64
        assert _PHONE not in result.contact_identifier
        assert _PHONE not in (result.clean_message.text_content or "")

    def test_contact_id_stable_same_user(self, wa_scrubber: PiiScrubber) -> None:
        r1 = wa_scrubber.scrub(_wa_msg(_PHONE, "First message."), channel_user_id=_PHONE)
        r2 = wa_scrubber.scrub(_wa_msg(_PHONE, "Second message."), channel_user_id=_PHONE)
        assert r1.contact_identifier == r2.contact_identifier

    def test_contact_id_not_in_agent_prompt(self, wa_scrubber: PiiScrubber) -> None:
        msg = _wa_msg("+254712999888", "Farming advice needed.")
        result = wa_scrubber.scrub(msg, channel_user_id="+254712999888")
        assert result.contact_identifier is not None
        prompt_str = json.dumps(result.clean_message.to_agent_prompt())
        assert result.contact_identifier not in prompt_str
