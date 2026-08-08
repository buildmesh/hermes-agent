from __future__ import annotations

import hashlib
import asyncio
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.interaction_session import (
    InteractionSessionError,
    canonical_session_state,
    load_interaction_session_declaration,
    require_codex_additional_context_support,
    validate_turn_control,
)
from hermes_cli.persistent_specialist_worker import PersistentSpecialistWorker, _bridge_prompt, _fingerprint, _validate_v3_request


@pytest.fixture(autouse=True)
def _restore_interaction_session_env():
    original = os.environ.get("HERMES_INTERACTION_SESSION_ENABLED")
    yield
    if original is None:
        os.environ.pop("HERMES_INTERACTION_SESSION_ENABLED", None)
    else:
        os.environ["HERMES_INTERACTION_SESSION_ENABLED"] = original


def _write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(raw)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def install_declaration(root: Path, *, bound: bool = True) -> tuple[object, dict]:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["period"],
        "properties": {"period": {"type": "string"}},
    }
    schema_digest = _write_json(root / "contract/session.schema.json", schema)
    declaration = {
        "schema_version": "telegram.bridge.specialist_interaction_sessions.v1",
        "contexts": [{
            "context_type": "report", "version": "1.0.0", "description": "Report parameters",
            "schema": "contract/session.schema.json", "schema_sha256": schema_digest,
            "ttl_seconds": 1800, "consumers": [],
        }],
    }
    declaration_digest = _write_json(root / "contract/interaction-sessions.json", declaration)
    manifest = {
        "schema_version": "telegram.bridge.specialist_distribution.v1",
        "profile": {
            "name": "reports", "version": "1.0.0",
            "required_files": ["AGENTS.md"],
            "memory_templates": ["memories/MEMORY.md"],
            "runtime_directories": ["state"],
        },
        "dependencies": [],
    }
    if bound:
        manifest["profile"]["interaction_session_contract"] = {
            "path": "contract/interaction-sessions.json", "sha256": declaration_digest,
        }
    _write_json(root / "state/installed-distribution.json", {
        "schema_version": "telegram.bridge.specialist_install_receipt.v1",
        "profile": "reports", "profile_version": "1.0.0", "mode": "production-copy",
        "installed_at": 1, "manifest": manifest, "dependencies": [],
    })
    _write_json(root / "distribution.json", manifest)
    loaded = load_interaction_session_declaration(root, supported_consumers=set())
    return loaded, manifest


def test_unbound_declaration_is_ignored(tmp_path: Path) -> None:
    declaration, _ = install_declaration(tmp_path, bound=False)
    assert declaration is None


def test_unbound_profile_does_not_load_runtime_contract_schemas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_declaration(tmp_path, bound=False)
    monkeypatch.setattr(
        "hermes_cli.interaction_session._validate_packaged_schema",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("schema I/O")),
    )
    assert load_interaction_session_declaration(tmp_path, supported_consumers=set()) is None


def test_bound_declaration_validates_snapshot_and_identity(tmp_path: Path) -> None:
    declaration, _ = install_declaration(tmp_path)
    assert declaration is not None
    state = {"period": "2026-Q3"}
    raw = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    control = {
        "profile": "reports", "profile_version": "1.0.0",
        "installed_profile_sha256": declaration.installed_profile_sha256,
        "conversation_generation": 3, "active": True,
        "snapshot": {
            "session_id": "sctx_one", "context_type": "report", "context_version": "1.0.0",
            "schema_sha256": declaration.contexts["report"].schema_sha256, "revision": 2,
            "state": state, "state_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
        },
    }
    assert validate_turn_control(control, declaration, now=datetime.now(timezone.utc)) is control
    changed = {**control, "conversation_generation": 4}
    request = {"protocol_version": "telegram.bridge.persistent_service.v3", "operation": "turn", "conversation_id": "conv_" + "a" * 64, "envelope": {}, "interaction_session": control}
    assert _fingerprint(request) != _fingerprint({**request, "interaction_session": changed})


def test_bound_declaration_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    install_declaration(tmp_path)
    (tmp_path / "contract/interaction-sessions.json").write_text("{}", encoding="utf-8")
    with pytest.raises(InteractionSessionError, match="digest mismatch"):
        load_interaction_session_declaration(tmp_path, supported_consumers=set())


def test_opt_in_receipt_requires_distribution_parity(tmp_path: Path) -> None:
    install_declaration(tmp_path)
    distribution = json.loads((tmp_path / "distribution.json").read_text())
    distribution["profile"]["version"] = "2.0.0"
    _write_json(tmp_path / "distribution.json", distribution)
    with pytest.raises(InteractionSessionError, match="does not match receipt"):
        load_interaction_session_declaration(tmp_path, supported_consumers=set())


def test_packaged_task220_schemas_reject_invalid_receipt_dependency(tmp_path: Path) -> None:
    install_declaration(tmp_path)
    receipt_path = tmp_path / "state/installed-distribution.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["dependencies"] = [{"package": "bad"}]
    _write_json(receipt_path, receipt)
    with pytest.raises(InteractionSessionError, match="installed profile receipt is invalid"):
        load_interaction_session_declaration(tmp_path, supported_consumers=set())


@pytest.mark.parametrize("mutation", ["empty_required_files", "invalid_fast_path"])
def test_packaged_task220_schema_rejects_invalid_distribution_inventory(tmp_path: Path, mutation: str) -> None:
    install_declaration(tmp_path)
    receipt_path = tmp_path / "state/installed-distribution.json"
    receipt = json.loads(receipt_path.read_text())
    if mutation == "empty_required_files":
        receipt["manifest"]["profile"]["required_files"] = []
    else:
        receipt["manifest"]["fast_path"] = {}
    _write_json(receipt_path, receipt)
    _write_json(tmp_path / "distribution.json", receipt["manifest"])
    with pytest.raises(InteractionSessionError, match="installed distribution manifest is invalid"):
        load_interaction_session_declaration(tmp_path, supported_consumers=set())


def test_installed_codex_experimental_schema_supports_additional_context() -> None:
    if shutil.which("codex") is None:
        pytest.skip("installed codex binary is unavailable")
    require_codex_additional_context_support()


def test_opted_app_server_fails_startup_when_schema_preflight_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_declaration(tmp_path)
    (tmp_path / "config.yaml").write_text("model:\n  provider: openai-codex\n  openai_runtime: codex_app_server\n", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.persistent_specialist_worker.require_codex_additional_context_support",
        lambda: (_ for _ in ()).throw(InteractionSessionError("SESSION_CAPABILITY_UNAVAILABLE", "unsupported schema")),
    )
    with pytest.raises(InteractionSessionError, match="unsupported schema"):
        PersistentSpecialistWorker(profile_root=tmp_path, socket_path=tmp_path / "worker.sock", agent_id="reports")


def test_model_tool_projection_is_service_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    from model_tools import _clear_tool_defs_cache, get_tool_definitions
    from tools.registry import invalidate_check_fn_cache

    def names() -> set[str]:
        return {item["function"]["name"] for item in get_tool_definitions(enabled_toolsets=["terminal"], quiet_mode=True)}

    monkeypatch.delenv("HERMES_INTERACTION_SESSION_ENABLED", raising=False)
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    assert "update_interaction_session" not in names()
    monkeypatch.setenv("HERMES_INTERACTION_SESSION_ENABLED", "1")
    invalidate_check_fn_cache()
    _clear_tool_defs_cache()
    assert "update_interaction_session" in names()


@pytest.mark.parametrize(
    ("protocol_version", "expected_payload_field"),
    [
        ("telegram.bridge.persistent_service.v1", "render_payloads"),
        ("telegram.bridge.persistent_service.v2", "render_candidate"),
        ("telegram.bridge.persistent_service.v3", "render_candidate"),
    ],
)
@pytest.mark.asyncio
async def test_unchanged_profile_stays_on_pre_session_wire_and_filesystem_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol_version: str,
    expected_payload_field: str,
) -> None:
    """A profile with no declaration must not observe session machinery."""
    monkeypatch.setenv("HERMES_INTERACTION_SESSION_ENABLED", "stale")
    prompts: list[str] = []

    class Agent:
        api_mode = "codex_responses"

        def run_conversation(self, prompt, conversation_history=None):
            prompts.append(prompt)
            candidate = json.dumps([{
                "schema_version": "telegram.bridge.render_payload.v1",
                "message_id": "msg_compat",
                "correlation_id": "compat-turn",
                "action": "send",
                "target": {"chat_id": 123},
                "render": {"text": "unchanged"},
            }])
            return {
                "completed": True,
                "partial": False,
                "final_response": candidate,
                "messages": [
                    *(conversation_history or []),
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": candidate},
                ],
            }

        def close(self):
            return None

    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="unchanged-profile",
        agent_factory=Agent,
    )
    assert worker._interaction_session_declaration is None
    assert "HERMES_INTERACTION_SESSION_ENABLED" not in os.environ

    envelope = {
        "schema_version": "telegram.bridge.user_input.v1",
        "event_id": "compat-turn",
        "chat_id": 123,
        "routing": {"resolved_agent": "unchanged-profile"},
        "message": {"text": "hello"},
    }
    turn = {
        "protocol_version": protocol_version,
        "request_id": "req_compat_turn",
        "operation": "turn",
        "event_id": "compat-turn",
        "conversation_id": "conv_" + "7" * 64,
        "deadline": "2099-01-01T00:00:00+00:00",
        "envelope": envelope,
    }
    reset = {
        **turn,
        "request_id": "req_compat_reset",
        "event_id": "compat-reset",
        "operation": "reset",
        "envelope": {
            **envelope,
            "event_id": "compat-reset",
            "message": {"command": {"mode": "reset"}},
        },
    }

    await worker.start()
    try:
        first = await worker.handle_request(turn)
        replay = await worker.handle_request(turn)
        reset_response = await worker.handle_request(reset)
    finally:
        await worker.stop()

    assert first["status"] == reset_response["status"] == "completed"
    assert expected_payload_field in first
    assert replay == first
    assert prompts == [_bridge_prompt(envelope)]
    for response in (first, reset_response):
        assert "interaction_session_transition" not in response
    assert worker._active_interaction_session_turn is None
    assert not worker._interaction_session_endpoint_path.exists()
    assert not list(tmp_path.rglob("*interaction-session*"))


@pytest.mark.parametrize("state", [
    {"value": 1.5}, {"value": 9007199254740992}, {"value": "\ud800"},
])
def test_canonical_state_rejects_nonportable_values(state: dict) -> None:
    with pytest.raises(InteractionSessionError):
        canonical_session_state(state)


def test_prompt_projects_only_active_untrusted_semantic_state() -> None:
    inactive = {"profile": "secret-profile", "profile_version": "1.0.0", "installed_profile_sha256": "sha256:" + "a" * 64, "conversation_generation": 9, "active": False}
    inactive_prompt = _bridge_prompt({"event_id": "one"}, interaction_session=inactive)
    assert "interaction-session" not in inactive_prompt
    active = {**inactive, "active": True, "snapshot": {
        "session_id": "sctx_secret", "context_type": "report", "context_version": "1.0.0",
        "schema_sha256": "sha256:" + "b" * 64, "revision": 2,
        "state": {"period": "2026-Q3"}, "state_sha256": "sha256:" + "c" * 64,
        "expires_at": "2026-08-05T12:30:00+00:00",
    }}
    active_prompt = _bridge_prompt({"event_id": "one"}, interaction_session=active)
    assert "structured prior user data" in active_prompt
    assert '"period":"2026-Q3"' in active_prompt
    for private in ("secret-profile", "sctx_secret", "schema_sha256", "state_sha256", "conversation_generation"):
        assert private not in active_prompt


def test_native_history_persists_sanitized_base_prompt(tmp_path: Path) -> None:
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path, socket_path=tmp_path / "worker.sock", agent_id="reports",
        agent_factory=lambda: None,
    )
    worker._agent = SimpleNamespace(api_mode="codex_responses")
    prior = {"role": "assistant", "content": "prior"}
    worker._native_conversation_history = [prior]
    result = {"messages": [prior, {"role": "user", "content": "base\nPRIVATE"}, {"role": "assistant", "content": "done"}]}
    worker._accept_native_conversation_history(result, enriched_prompt="base\nPRIVATE", base_prompt="base", prior_history_length=1)
    assert worker._native_conversation_history[0] == prior
    assert worker._native_conversation_history[1] == {"role": "user", "content": "base"}
    assert result["messages"][1]["content"] == "base\nPRIVATE"


@pytest.mark.asyncio
async def test_completed_replay_precedes_current_expiry_validation(tmp_path: Path) -> None:
    declaration, _ = install_declaration(tmp_path)
    (tmp_path / "config.yaml").write_text("model:\n  provider: openai-codex\n  api_mode: codex_responses\n", encoding="utf-8")
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path, socket_path=tmp_path / "worker.sock", agent_id="reports",
        agent_factory=lambda: None,
    )
    state = {"period": "2026-Q3"}
    raw = canonical_session_state(state)
    control = {
        "profile": "reports", "profile_version": "1.0.0",
        "installed_profile_sha256": declaration.installed_profile_sha256,
        "conversation_generation": 0, "active": True,
        "snapshot": {
            "session_id": "sctx_old", "context_type": "report", "context_version": "1.0.0",
            "schema_sha256": declaration.contexts["report"].schema_sha256, "revision": 1,
            "state": state, "state_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "expires_at": "2026-08-05T00:00:00+00:00",
        },
    }
    request = {
        "protocol_version": "telegram.bridge.persistent_service.v3", "request_id": "req_replay",
        "operation": "turn", "event_id": "expired-replay", "conversation_id": "conv_" + "a" * 64,
        "deadline": "2099-01-01T00:00:00+00:00", "envelope": {},
        "interaction_session": control, "accepted_capabilities": ["specialist_interaction_session.v1"],
        "_interaction_session_negotiated": True,
    }
    path = worker._ledger_path(request["conversation_id"], request["event_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = {"protocol_version": request["protocol_version"], "request_id": request["request_id"], "operation": "turn", "status": "completed"}
    path.write_text(json.dumps({"fingerprint": _fingerprint(request), "response": stored}), encoding="utf-8")
    assert await worker.handle_request(request) == stored
    conflict = await worker.handle_request({**request, "interaction_session": {**control, "conversation_generation": 1}})
    assert conflict["error"]["code"] == "IDEMPOTENCY_CONFLICT"


def test_v3_wire_validator_enforces_closed_inactive_control() -> None:
    request = {
        "protocol_version": "telegram.bridge.persistent_service.v3", "request_id": "req_wire",
        "operation": "turn", "event_id": "wire", "conversation_id": "conv_" + "a" * 64,
        "deadline": "2099-01-01T00:00:00+00:00", "envelope": {},
        "accepted_capabilities": ["specialist_interaction_session.v1"],
        "interaction_session": {
            "profile": "reports", "profile_version": "1.0.0",
            "installed_profile_sha256": "sha256:" + "a" * 64,
            "conversation_generation": 0, "active": False,
        },
    }
    _validate_v3_request(request)
    with pytest.raises(ValueError):
        _validate_v3_request({**request, "interaction_session": {**request["interaction_session"], "snapshot": {}}})
    with pytest.raises(ValueError):
        _validate_v3_request({**request, "interaction_session": {**request["interaction_session"], "conversation_generation": True}})


@pytest.mark.asyncio
async def test_real_socket_v3_negotiates_and_scopes_endpoint(tmp_path: Path) -> None:
    declaration, _ = install_declaration(tmp_path)
    (tmp_path / "config.yaml").write_text("model:\n  provider: openai-codex\n  api_mode: codex_responses\n", encoding="utf-8")
    observed = {}

    class Agent:
        api_mode = "codex_responses"
        model = "test"
        provider = "openai-codex"
        base_url = ""
        reasoning_effort = "medium"

        def run_conversation(self, prompt, conversation_history=None):
            observed["endpoint_during_turn"] = (tmp_path / "state/persistent-runtime/interaction-session-endpoint.json").is_file()
            return {"completed": True, "final_response": "[]", "messages": [*(conversation_history or []), {"role": "user", "content": prompt}, {"role": "assistant", "content": "[]"}]}

        def close(self):
            return None

    worker = PersistentSpecialistWorker(profile_root=tmp_path, socket_path=tmp_path / "worker.sock", agent_id="reports", agent_factory=Agent)
    await worker.start()
    serve = asyncio.create_task(worker.serve_forever())
    try:
        reader, writer = await asyncio.open_unix_connection(str(worker.socket_path))
        hello = {"protocol_version": "telegram.bridge.persistent_service.v3", "request_id": "req_hello", "operation": "hello", "client": {"name": "telegram-bridge", "revision": "test"}, "accepted_capabilities": ["specialist_interaction_session.v1"]}
        writer.write(json.dumps(hello).encode() + b"\n")
        await writer.drain()
        negotiated = json.loads(await reader.readline())
        assert "specialist_interaction_session.v1" in negotiated["capabilities"]
        request = {
            "protocol_version": "telegram.bridge.persistent_service.v3", "request_id": "req_turn",
            "operation": "turn", "event_id": "socket-turn", "conversation_id": "conv_" + "a" * 64,
            "deadline": "2099-01-01T00:00:00+00:00", "envelope": {"event_id": "socket-turn"},
            "accepted_capabilities": ["specialist_interaction_session.v1"],
            "interaction_session": {"profile": "reports", "profile_version": "1.0.0", "installed_profile_sha256": declaration.installed_profile_sha256, "conversation_generation": 0, "active": False},
        }
        writer.write(json.dumps(request).encode() + b"\n")
        await writer.drain()
        response = json.loads(await reader.readline())
        writer.close()
        await writer.wait_closed()
        assert response["status"] == "completed"
        assert observed["endpoint_during_turn"] is True
        assert not (tmp_path / "state/persistent-runtime/interaction-session-endpoint.json").exists()
    finally:
        serve.cancel()
        await asyncio.gather(serve, return_exceptions=True)
        await worker.stop()


@pytest.mark.asyncio
async def test_transition_replace_replays_and_changed_second_call_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HERMES_INTERACTION_SESSION_ENABLED", raising=False)
    declaration, _ = install_declaration(tmp_path)
    (tmp_path / "config.yaml").write_text("model:\n  provider: openai\n  api_mode: codex_responses\n", encoding="utf-8")
    worker = PersistentSpecialistWorker(profile_root=tmp_path, socket_path=tmp_path / "worker.sock", agent_id="reports")
    worker._interaction_session_declaration = declaration
    control = {
        "profile": "reports", "profile_version": "1.0.0",
        "installed_profile_sha256": declaration.installed_profile_sha256,
        "conversation_generation": 0, "active": False,
    }
    worker._active_interaction_session_turn = {"turn_token": "token", "event_id": "event-1", "control": control, "transition": None}
    request = {"protocol_version": "hermes.interaction_session_endpoint.v1", "operation": "replace", "turn_token": "token", "context_type": "report", "state": {"period": "2026-Q3"}}
    first = await worker._handle_interaction_session_request(request)
    assert first["status"] == "completed"
    assert first["transition"]["revision"] == 1
    assert await worker._handle_interaction_session_request(request) == first
    conflict = await worker._handle_interaction_session_request({**request, "state": {"period": "2026-Q4"}})
    assert conflict["error"]["code"] == "SESSION_TRANSITION_ALREADY_PREPARED"


@pytest.mark.asyncio
async def test_transition_inherits_exact_control_generation_and_revision(tmp_path: Path) -> None:
    declaration, _ = install_declaration(tmp_path)
    (tmp_path / "config.yaml").write_text("model:\n  provider: openai-codex\n  api_mode: codex_responses\n", encoding="utf-8")
    worker = PersistentSpecialistWorker(profile_root=tmp_path, socket_path=tmp_path / "worker.sock", agent_id="reports")
    state = {"period": "2026-Q3"}
    raw = canonical_session_state(state)
    control = {
        "profile": "reports", "profile_version": "1.0.0",
        "installed_profile_sha256": declaration.installed_profile_sha256,
        "conversation_generation": 47, "active": True,
        "snapshot": {
            "session_id": "sctx_current", "context_type": "report", "context_version": "1.0.0",
            "schema_sha256": declaration.contexts["report"].schema_sha256, "revision": 12,
            "state": state, "state_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "expires_at": "2099-01-01T00:00:00+00:00",
        },
    }
    worker._active_interaction_session_turn = {"turn_token": "token", "event_id": "event", "control": control, "transition": None}
    result = await worker._handle_interaction_session_request({
        "protocol_version": "hermes.interaction_session_endpoint.v1", "operation": "replace",
        "turn_token": "token", "context_type": "report", "state": {"period": "2026-Q4"},
    })
    transition = result["transition"]
    assert transition["conversation_generation"] == 47
    assert transition["prior_revision"] == 12
    assert transition["revision"] == 13
