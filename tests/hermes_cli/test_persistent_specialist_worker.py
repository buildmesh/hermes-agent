import asyncio
import hashlib
import json
import os
import signal
import shutil
import stat
import sys
import threading
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jsonschema
import pytest

import hermes_cli.persistent_specialist_worker as persistent_worker
from agent.transports.codex_app_server import CodexAppServerClient
from hermes_cli.codex_runtime_plugin_migration import (
    hermes_tools_mcp_app_server_args,
)
from hermes_cli.persistent_specialist_worker import (
    PROTOCOL,
    PROTOCOL_V2,
    PROTOCOL_V3,
    PersistentSpecialistWorker,
    _bridge_prompt,
    _initialize_worker_process_context,
    _extract_render_payloads,
    _terminal_presenter_is_final_and_unbatched,
)


def test_extract_render_payloads_repairs_one_missing_container_close() -> None:
    malformed = (
        '[{"schema_version":"telegram.bridge.render_payload.v1",'
        '"render":{"text":"Corner contains 23 items.","blocks":['
        '{"type":"table","columns":["Item","Quantity"],'
        '"rows":[["Citric Acid, qt","2 jars"],["Paper towels","8 rolls"]}'
        ']}}]'
    )

    payloads = _extract_render_payloads(malformed)

    assert payloads[0]["render"]["blocks"][0]["rows"] == [
        ["Citric Acid, qt", "2 jars"],
        ["Paper towels", "8 rolls"],
    ]


def test_extract_render_payloads_rejects_other_malformed_json() -> None:
    with pytest.raises(ValueError):
        _extract_render_payloads('[{"render":{"text":"broken" "blocks":[]}}]')


def test_terminal_presenter_finality_rejects_batched_tool_call() -> None:
    result = {"messages": [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_domain", "function": {"name": "domain_tool"}},
            {"id": "call_present", "function": {"name": "finalize_telegram_presentation"}},
        ]},
        {"role": "tool", "tool_call_id": "call_present", "content": "artifact"},
    ]}

    assert not _terminal_presenter_is_final_and_unbatched(result, "prompt", "artifact")


def test_terminal_presenter_finality_allows_failed_retry_but_rejects_calls_after_success() -> None:
    result = {"messages": [
        {"role": "user", "content": "prompt"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_failed", "function": {"name": "finalize_telegram_presentation"}},
        ]},
        {"role": "tool", "tool_call_id": "call_failed", "content": "{\"error\":true}"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_success", "function": {"name": "finalize_telegram_presentation"}},
        ]},
        {"role": "tool", "tool_call_id": "call_success", "content": "artifact"},
    ]}

    assert _terminal_presenter_is_final_and_unbatched(result, "prompt", "artifact")
    result["messages"].extend([
        {"role": "assistant", "tool_calls": [
            {"id": "call_late", "function": {"name": "finalize_telegram_presentation"}},
        ]},
        {"role": "tool", "tool_call_id": "call_late", "content": "{\"error\":true}"},
    ])
    assert not _terminal_presenter_is_final_and_unbatched(result, "prompt", "artifact")


def test_bridge_prompt_without_terminal_presenter_requires_render_json() -> None:
    prompt = _bridge_prompt({"event_id": "evt_1", "chat_id": 123})

    assert "Use this render envelope contract for every response" in prompt
    assert "finalize_telegram_presentation" not in prompt


def test_bridge_prompt_with_terminal_presenter_has_non_conflicting_completion_paths() -> None:
    prompt = _bridge_prompt(
        {"event_id": "evt_1", "chat_id": 123},
        terminal_presenter=True,
    )

    assert "Choose exactly one completion path" in prompt
    assert "you must invoke finalize_telegram_presentation" in prompt
    assert "as the final, unbatched tool call" in prompt
    assert "only when the profile does not declare a supported" in prompt
    assert "For the model-render completion path, return only" in prompt
    assert "for every response" not in prompt
    assert "Return only a JSON array" not in prompt


class FakeAgent:
    def __init__(self) -> None:
        self.turns = 0
        self.closed = False
        self._codex_session = None
        self.session_cwd = None
        self.prompts: list[str] = []
        self.effective_cwds: list[tuple[str, str, str]] = []
        self.loaded_profile_context: list[tuple[str, ...]] = []

    def run_conversation(self, prompt: str) -> dict:
        self.turns += 1
        self.prompts.append(prompt)
        from agent.runtime_cwd import resolve_agent_cwd, resolve_context_cwd

        self.effective_cwds.append((
            str(resolve_agent_cwd()),
            str(resolve_context_cwd()),
            os.getcwd(),
        ))
        context_root = resolve_context_cwd() or Path.cwd()
        self.loaded_profile_context.append(tuple(
            str(path.relative_to(context_root))
            for path in (
                context_root / "SOUL.md",
                context_root / "memories/USER.md",
                context_root / "memories/MEMORY.md",
                context_root / "AGENTS.md",
                context_root / "skills/index.md",
            )
            if path.exists()
        ))
        envelope = json.loads(prompt.split("Envelope JSON:\n", 1)[1])
        payload = [{
            "schema_version": "telegram.bridge.render_payload.v1",
            "message_id": f"msg_{envelope['event_id']}",
            "correlation_id": envelope["event_id"],
            "action": "send",
            "target": {"chat_id": envelope["chat_id"]},
            "render": {"text": f"turn {self.turns}"},
        }]
        return {"completed": True, "partial": False, "final_response": json.dumps(payload)}

    def close(self) -> None:
        self.closed = True


class FakeCodexSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FailingAgent(FakeAgent):
    def run_conversation(self, prompt: str) -> dict:
        raise RuntimeError("model failure")


class BlockingAgent(FakeAgent):
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self.entered = entered
        self.release = release

    def run_conversation(self, prompt: str) -> dict:
        self.entered.set()
        self.release.wait(timeout=2)
        return super().run_conversation(prompt)


class ClosingBlockingAgent(BlockingAgent):
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        super().__init__(entered, release)
        self.close_before_release = False
        self.close_called = threading.Event()

    def close(self) -> None:
        self.close_before_release = not self.release.is_set()
        self.close_called.set()
        super().close()


def request(event_id: str, operation: str = "turn") -> dict:
    return {
        "protocol_version": PROTOCOL,
        "request_id": f"req_{event_id}",
        "operation": operation,
        "event_id": event_id,
        "conversation_id": "conv_" + "1" * 64,
        "deadline": "2099-01-01T00:00:00+00:00",
        "envelope": {
            "schema_version": "telegram.bridge.user_input.v1",
            "event_id": event_id,
            "chat_id": 123,
            "routing": {"resolved_agent": "hello_world"},
            "message": {"command": {"mode": "reset"}} if operation == "reset" else {"text": "hello"},
        },
    }


def v2_request(event_id: str, operation: str = "turn") -> dict:
    value = request(event_id, operation)
    value["protocol_version"] = PROTOCOL_V2
    return value


def v3_request(event_id: str, operation: str = "turn") -> dict:
    value = request(event_id, operation)
    value["protocol_version"] = PROTOCOL_V3
    value["_terminal_presenter_negotiated"] = True
    return value


def install_test_presenter(root: Path) -> None:
    handler = root / "skills/test/present.py"
    schema = root / "contract/test.input.schema.json"
    handler.parent.mkdir(parents=True)
    schema.parent.mkdir(parents=True)
    handler.write_text(
        "import json,sys\n"
        "value=json.load(sys.stdin)\n"
        "sys.stdout.write(json.dumps(value['payloads'], separators=(',', ':')))\n",
        encoding="utf-8",
    )
    schema.write_text(json.dumps({
        "type": "object",
        "additionalProperties": False,
        "properties": {"payloads": {"type": "array"}},
    }), encoding="utf-8")
    manifest = {
        "schema_version": "telegram.bridge.terminal_presenters.v1",
        "presenters": [{
            "presenter_id": "test-presenter",
            "version": "1.0.0",
            "description": "Render a test payload.",
            "handler": "skills/test/present.py",
            "sha256": "sha256:" + hashlib.sha256(handler.read_bytes()).hexdigest(),
            "input_schema": "contract/test.input.schema.json",
            "input_schema_sha256": "sha256:" + hashlib.sha256(schema.read_bytes()).hexdigest(),
            "command": ["python3", "{handler}"],
            "timeout_seconds": 2,
            "max_output_bytes": 4096,
        }],
    }
    (root / "contract/terminal-presenters.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.skipif(
    shutil.which("codex") is None,
    reason="real Codex app-server acceptance test requires the codex binary",
)
async def test_codex_app_server_projects_and_invokes_active_terminal_presenter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_test_presenter(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="projection-test",
        agent_factory=FakeAgent,
    )
    await worker.start()
    serve = asyncio.create_task(worker.serve_forever())
    worker._begin_presenter_turn(v3_request("codex-projection"))
    payloads = [{"schema_version": "telegram.bridge.render_payload.v1"}]

    def exercise_codex_projection(
        *,
        invoke: bool,
    ) -> tuple[dict, dict | None]:
        client = CodexAppServerClient(
            extra_args=hermes_tools_mcp_app_server_args() if invoke else None,
        )
        try:
            client.initialize(
                client_name="hermes-test",
                client_title="Hermes Test",
                client_version="test",
            )
            started = client.request(
                "thread/start",
                {"cwd": str(tmp_path)},
                timeout=15,
            )
            thread_id = started["thread"]["id"]
            inventory = client.request(
                "mcpServerStatus/list",
                {"threadId": thread_id, "detail": "toolsAndAuthOnly"},
                timeout=30,
            )
            result = None
            if invoke:
                result = client.request(
                    "mcpServer/tool/call",
                    {
                        "server": "hermes-tools",
                        "tool": "finalize_telegram_presentation",
                        "arguments": {
                            "presenter_id": "test-presenter",
                            "input": {"payloads": payloads},
                        },
                        "threadId": thread_id,
                    },
                    timeout=30,
                )
            return inventory, result
        finally:
            client.close()

    try:
        inventory, result = await asyncio.to_thread(
            exercise_codex_projection,
            invoke=True,
        )
        worker._end_presenter_turn()
        downgraded_inventory, _ = await asyncio.to_thread(
            exercise_codex_projection,
            invoke=False,
        )
    finally:
        serve.cancel()
        await asyncio.gather(serve, return_exceptions=True)
        await worker.stop()

    hermes_server = next(
        item for item in inventory["data"] if item["name"] == "hermes-tools"
    )
    assert "finalize_telegram_presentation" in hermes_server["tools"]
    assert result is not None
    assert result.get("isError") is not True
    assert any(
        item.get("text") == json.dumps(payloads, separators=(",", ":"))
        for item in result["content"]
        if isinstance(item, dict)
    )
    assert all(
        "finalize_telegram_presentation" not in item.get("tools", {})
        for item in downgraded_inventory["data"]
    )


class TerminalPresenterAgent(FakeAgent):
    def run_conversation(self, prompt: str) -> dict:
        from hermes_cli.terminal_presenter import invoke_presenter_endpoint

        self.turns += 1
        envelope = json.loads(prompt.split("Envelope JSON:\n", 1)[1])
        payload = [{
            "schema_version": "telegram.bridge.render_payload.v1",
            "message_id": f"msg_{envelope['event_id']}",
            "correlation_id": envelope["event_id"],
            "action": "send",
            "target": {"chat_id": envelope["chat_id"]},
            "render": {"text": "exact presenter"},
        }]
        content = invoke_presenter_endpoint(
            "test-presenter",
            {"payloads": payload},
            profile_root=Path(self.session_cwd),
        )
        return {
            "completed": True,
            "partial": False,
            "final_response": "model prose must not win",
            "messages": [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_terminal_presenter",
                        "function": {
                            "name": "finalize_telegram_presentation",
                            "arguments": "{}",
                        }
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_terminal_presenter",
                    "content": content,
                },
                {"role": "assistant", "content": "model prose must not win"},
            ],
        }


def correction_request(event_id: str, attempt_id: str = "corr_attempt_1") -> dict:
    return {
        "protocol_version": PROTOCOL_V2,
        "request_id": f"req_{attempt_id}",
        "operation": "correct_render",
        "event_id": event_id,
        "conversation_id": "conv_" + "1" * 64,
        "deadline": "2099-01-01T00:00:00+00:00",
        "correction_attempt_id": attempt_id,
        "validation_errors": [{
            "code": "RENDER_JSON_INVALID",
            "path": "$",
            "message": "candidate is not valid JSON",
        }],
        "target_constraints": {
            "correlation_id": event_id,
            "chat_id": 123,
            "authorized_telegram_message_ids": [],
            "callback_namespace": "hello",
        },
    }


class CorrectionAgent:
    def __init__(self, *, api_mode: str = "chat_completions") -> None:
        self.api_mode = api_mode
        self.turns = 0
        self.tools = [{"type": "function", "function": {"name": "terminal"}}]
        self.valid_tool_names = {"terminal"}
        self.max_iterations = 8
        self._fallback_chain = [{"provider": "fallback", "model": "fallback"}]
        self._codex_session = None
        self.session_cwd = None
        self.tool_snapshots: list[list[dict]] = []
        self.denied_execution = False
        self.executed_tools = 0
        self.closed = False

    def _execute_tool_calls(self, *_args: object, **_kwargs: object) -> None:
        self.executed_tools += 1

    def run_conversation(self, prompt: str) -> dict:
        self.turns += 1
        self.tool_snapshots.append(list(self.tools))
        if self.turns == 1:
            return {
                "completed": True,
                "partial": False,
                "api_calls": 1,
                "final_response": '[{"schema_version":"telegram.bridge.render_payload.v1"',
            }
        try:
            self._execute_tool_calls(object(), [], "correction")
        except Exception as exc:
            self.denied_execution = "denied" in str(exc).lower()
        payload = [{
            "schema_version": "telegram.bridge.render_payload.v1",
            "message_id": "msg_corrected",
            "correlation_id": "completed-malformed",
            "action": "send",
            "target": {"chat_id": 123},
            "render": {"text": "corrected without rerunning domain work"},
        }]
        return {
            "completed": True,
            "partial": False,
            "api_calls": 1,
            "final_response": json.dumps(payload),
        }

    def close(self) -> None:
        self.closed = True


class SlowCorrectionAgent(CorrectionAgent):
    def __init__(self, entered: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self.entered = entered
        self.release = release

    def run_conversation(self, prompt: str) -> dict:
        if self.turns:
            self.entered.set()
            self.release.wait(timeout=2)
        return super().run_conversation(prompt)


def companion_descriptor(profile_root: Path | None = None) -> dict:
    value = {
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "model": "gpt-test",
    }
    if profile_root is not None:
        value["profile_root"] = str(profile_root.resolve())
    return value


def companion_config(token: str = "test-token") -> dict:
    return {
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "model": "gpt-test",
        "base_url": "https://chatgpt.example/backend-api/codex",
        "api_key": token,
    }


class FakeCorrectionCompanion:
    def __init__(self, *, entered: threading.Event | None = None, release: threading.Event | None = None) -> None:
        self.calls: list[tuple[dict, dict, float]] = []
        self.entered = entered
        self.release = release

    def __call__(self, config: dict, repair: dict, timeout: float) -> str:
        self.calls.append((config, repair, timeout))
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            self.release.wait(timeout=2)
        payload = [{
            "schema_version": "telegram.bridge.render_payload.v1",
            "message_id": "msg_corrected",
            "correlation_id": "completed-malformed",
            "action": "send",
            "target": {"chat_id": 123},
            "render": {"text": "corrected without rerunning domain work"},
        }]
        return json.dumps(payload)


def assert_complete_timing(response: dict) -> None:
    assert response["trace_id"].startswith("trace_")
    assert set(response["timing_ms"]) == {
        "queue",
        "initialization",
        "retirement",
        "model_and_tools",
        "render_preparation",
        "total",
    }
    assert all(duration >= 0 for duration in response["timing_ms"].values())
    assert sum(
        response["timing_ms"][phase]
        for phase in ("queue", "initialization", "retirement", "model_and_tools", "render_preparation")
    ) <= response["timing_ms"]["total"]


def assert_terminal_log(profile_root: Path, response: dict, event_id: str, event: str) -> None:
    events = [json.loads(line) for line in (profile_root / "logs/persistent-specialist.jsonl").read_text().splitlines()]
    terminal = [entry for entry in events if entry.get("event_id") == event_id and entry["event"] == event][-1]
    assert terminal["trace_id"] == response["trace_id"]
    assert terminal["timing_ms"] == response["timing_ms"]


def test_worker_process_initialization_sets_profile_context_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile_root = tmp_path / "profile"
    profile_root.mkdir()
    bridge_cwd = tmp_path / "bridge"
    bridge_cwd.mkdir()
    monkeypatch.chdir(bridge_cwd)
    monkeypatch.setenv("HERMES_HOME", str(bridge_cwd))
    monkeypatch.setenv("TERMINAL_CWD", str(bridge_cwd))
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_ACCEPT_HOOKS", raising=False)

    _initialize_worker_process_context(profile_root)

    assert (os.getcwd(), os.environ["HERMES_HOME"], os.environ["TERMINAL_CWD"]) == (
        str(profile_root), str(profile_root), str(profile_root)
    )
    assert os.environ["HERMES_YOLO_MODE"] == "1"
    assert os.environ["HERMES_ACCEPT_HOOKS"] == "1"


def test_default_agent_factory_does_not_mutate_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sentinel = object()
    oneshot = types.ModuleType("hermes_cli.oneshot")
    oneshot.create_noninteractive_agent = lambda: sentinel
    monkeypatch.setitem(sys.modules, "hermes_cli.oneshot", oneshot)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_ACCEPT_HOOKS", raising=False)
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "w.sock",
        agent_id="hello_world",
    )

    assert worker._default_agent_factory() is sentinel
    assert "HERMES_YOLO_MODE" not in os.environ
    assert "HERMES_ACCEPT_HOOKS" not in os.environ


def test_app_server_companion_uses_specialist_profile_config_and_auth_not_global(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import render_correction_companion as companion
    from hermes_constants import get_hermes_home

    global_home = tmp_path / "global"
    specialist_home = tmp_path / "profiles/specialist"
    for home, model, token in (
        (global_home, "gpt-global", "global-token"),
        (specialist_home, "gpt-specialist", "specialist-startup-token"),
    ):
        home.mkdir(parents=True)
        (home / "test-config.json").write_text(json.dumps({"model": {"default": model}}))
        (home / "test-auth.json").write_text(json.dumps({"api_key": token}))
    original_global_auth = (global_home / "test-auth.json").read_text()
    monkeypatch.setenv("HERMES_HOME", str(global_home))

    monkeypatch.setattr(
        companion,
        "load_config",
        lambda: json.loads((get_hermes_home() / "test-config.json").read_text()),
    )

    def scoped_runtime(**_kwargs: object) -> dict[str, str]:
        auth = json.loads((get_hermes_home() / "test-auth.json").read_text())
        return {
            "provider": "openai-codex",
            "api_mode": "codex_app_server",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": auth["api_key"],
        }

    monkeypatch.setattr(companion, "resolve_runtime_provider", scoped_runtime)
    health_configs: list[dict] = []
    monkeypatch.setattr(companion, "_direct_responses_client", lambda config: (
        health_configs.append(dict(config))
        or types.SimpleNamespace(close=lambda: None)
    ))

    descriptor = companion.resolve_companion_descriptor(specialist_home)
    (specialist_home / "test-auth.json").write_text(
        json.dumps({"api_key": "specialist-rotated-token"})
    )
    correction_config = companion.resolve_companion_credentials(descriptor)

    assert descriptor == {
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "model": "gpt-specialist",
        "profile_root": str(specialist_home.resolve()),
    }
    assert correction_config == {
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "model": "gpt-specialist",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_key": "specialist-rotated-token",
    }
    assert health_configs[0]["api_key"] == "specialist-startup-token"
    assert "specialist-startup-token" not in json.dumps(descriptor)
    assert (global_home / "test-auth.json").read_text() == original_global_auth


def test_companion_descriptor_rejects_relative_unexpected_or_unbound_profile(
    tmp_path: Path,
) -> None:
    from agent import render_correction_companion as companion

    expected = tmp_path / "expected"
    unexpected = tmp_path / "unexpected"
    expected.mkdir()
    unexpected.mkdir()

    with pytest.raises(companion.CorrectionCompanionUnavailable):
        companion.validate_companion_descriptor(
            {**companion_descriptor(), "profile_root": "relative/profile"},
            expected,
        )
    with pytest.raises(companion.CorrectionCompanionUnavailable):
        companion.validate_companion_descriptor(
            companion_descriptor(unexpected),
            expected,
        )
    with pytest.raises(companion.CorrectionCompanionUnavailable):
        companion.validate_companion_descriptor(
            {**companion_descriptor(expected), "provider": "anthropic"},
            expected,
        )


def test_companion_model_call_has_no_tools_or_parent_context(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent import render_correction_companion as companion

    calls: list[dict] = []

    class Completions:
        def create(self, **kwargs: object) -> object:
            calls.append(dict(kwargs))
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='[{"ok":true}]'))]
            )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=Completions()),
        close=lambda: None,
    )
    monkeypatch.setattr(companion, "_direct_responses_client", lambda _config: client)
    candidate = json.dumps([
        {
            "schema_version": "telegram.bridge.render_payload.v1",
            "message_id": "msg_evt_1",
            "correlation_id": "evt_1",
            "action": "send",
            "target": {"chat_id": 123456789},
            "render": {
                "text": "User-facing response.",
                "blocks": [
                    {
                        "type": "table",
                        "columns": ["Item", "Qty"],
                        "rows": [["Citric Acid, qt", "2 jars"], ["Paper towels", "8 rolls"]],
                    }
                ],
            },
        }
    ], ensure_ascii=False, separators=(",", ":"))
    repair = {
        "candidate": candidate,
        "validation_errors": [{
            "code": "RENDER_TARGET_MISMATCH",
            "path": "$[0].target.chat_id",
            "message": "candidate target does not match the authoritative chat",
        }],
        "target_constraints": {
            "correlation_id": "evt_1",
            "chat_id": 123,
            "authorized_telegram_message_ids": [],
            "callback_namespace": "hello",
        },
    }

    result = companion.run_companion_model_call(companion_config(), repair, timeout=3)

    assert result == '[{"ok":true}]'
    assert len(calls) == 1
    assert calls[0]["model"] == "gpt-test"
    assert calls[0]["tools"] == []
    assert calls[0]["tool_choice"] == "none"
    assert len(calls[0]["messages"]) == 2
    repair_payload = json.loads(calls[0]["messages"][1]["content"])
    assert repair_payload == repair
    assert json.loads(repair_payload["candidate"])[0]["render"]["blocks"][0]["type"] == "table"
    assert "Envelope JSON" not in json.dumps(calls[0])
    correction_instructions = calls[0]["messages"][0]["content"]
    lowered_instructions = correction_instructions.lower()
    for required_fragment in (
        '"message_id":"msg_<event_id>_result"',
        '"action":"send"',
        '"target":{"chat_id":123456789}',
        '"render":{"text":"User-facing response."}',
        "Never put chat_id or text at the payload top level",
        "Never wrap the object in payload",
    ):
        assert required_fragment in correction_instructions
    assert "Do not repeat, continue, verify, or infer any domain work." in correction_instructions
    assert "preserve the candidate's intended supported presentation" in lowered_instructions
    assert "preserve any valid structured blocks" in lowered_instructions
    assert "tables, lists, or buttons" in lowered_instructions
    assert "plain text" in lowered_instructions


def test_credential_bootstrap_child_is_profile_scoped_and_separate_from_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import render_correction_companion as companion

    profile_root = tmp_path.resolve()
    descriptor = companion_descriptor(profile_root)
    observed: dict[str, object] = {}

    class Process:
        returncode = 0

        def communicate(self, payload: str, timeout: float | None = None):
            observed["payload"] = json.loads(payload)
            observed["timeout"] = timeout
            return json.dumps({"status": "completed", "config": companion_config("profile-token")}), None

        def wait(self) -> int:
            observed["waited"] = True
            return 0

    def popen(*_args: object, **kwargs: object) -> Process:
        observed["kwargs"] = kwargs
        return Process()

    monkeypatch.setenv("HERMES_HOME", "/global/home")
    monkeypatch.setattr(companion.subprocess, "Popen", popen)

    config = companion.resolve_companion_credentials_subprocess(descriptor, 1.5)

    assert config["api_key"] == "profile-token"
    assert observed["payload"] == {"descriptor": descriptor}
    assert observed["kwargs"]["env"]["HERMES_HOME"] == str(profile_root)
    assert observed["kwargs"]["start_new_session"] is True
    assert observed["timeout"] <= 1.5
    assert observed["waited"] is True


def test_credential_bootstrap_timeout_kills_and_reaps_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import render_correction_companion as companion

    observed: dict[str, object] = {}

    class Process:
        pid = 4241
        returncode = None

        def communicate(self, _payload: str | None = None, timeout: float | None = None):
            observed.setdefault("communicate", []).append(timeout)
            if timeout is not None:
                raise companion.subprocess.TimeoutExpired("credentials", timeout)
            self.returncode = -9
            return "", None

        def kill(self) -> None:
            observed["process_kill"] = True

        def wait(self) -> int:
            observed["waited"] = True
            return int(self.returncode or 0)

    monkeypatch.setattr(companion.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(companion.os, "killpg", lambda pid, sig: observed.update(killpg=(pid, sig)))

    with pytest.raises(companion.CorrectionCompanionTimeout, match="credential bootstrap"):
        companion.resolve_companion_credentials_subprocess(
            companion_descriptor(tmp_path.resolve()),
            0.01,
        )

    assert observed["killpg"] == (4241, signal.SIGKILL)
    assert observed["communicate"] == [pytest.approx(0.01, abs=0.01), None]
    assert observed["waited"] is True


def test_companion_does_not_spawn_subprocess_without_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import render_correction_companion as companion

    spawned: list[bool] = []
    monkeypatch.setattr(companion.subprocess, "Popen", lambda *_args, **_kwargs: spawned.append(True))

    with pytest.raises(companion.CorrectionCompanionTimeout):
        companion.run_companion_subprocess(
            companion_config(),
            {"candidate": "bad", "validation_errors": [], "target_constraints": {}},
            0,
        )

    assert spawned == []


def test_companion_timeout_kills_and_reaps_sanitized_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import render_correction_companion as companion

    observed: dict[str, object] = {}

    class Process:
        pid = 4242
        returncode = None

        def communicate(self, payload: str | None = None, timeout: float | None = None):
            observed.setdefault("communicate", []).append((payload, timeout))
            if len(observed["communicate"]) == 1:
                raise companion.subprocess.TimeoutExpired("companion", timeout)
            self.returncode = -9
            return "", None

        def kill(self) -> None:
            observed["process_kill"] = True

        def wait(self) -> int:
            observed["waited"] = True
            return int(self.returncode or 0)

    def popen(*args: object, **kwargs: object) -> Process:
        observed["popen_args"] = args
        observed["popen_kwargs"] = kwargs
        return Process()

    monkeypatch.setenv("HERMES_HOME", "/private/profile")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-inherited")
    monkeypatch.setattr(companion.subprocess, "Popen", popen)
    monkeypatch.setattr(companion.os, "killpg", lambda pid, sig: observed.update(killpg=(pid, sig)))

    with pytest.raises(companion.CorrectionCompanionTimeout):
        companion.run_companion_subprocess(
            companion_config(),
            {"candidate": "bad", "validation_errors": [], "target_constraints": {}},
            0.1,
        )

    kwargs = observed["popen_kwargs"]
    assert kwargs["start_new_session"] is True
    assert kwargs["close_fds"] is True
    assert Path(kwargs["cwd"]).name.startswith("hermes-render-correction-")
    assert "HERMES_HOME" not in kwargs["env"]
    assert "OPENAI_API_KEY" not in kwargs["env"]
    model_payload = json.loads(observed["communicate"][0][0])
    assert model_payload["config"] == companion_config()
    assert "descriptor" not in model_payload
    assert "profile_root" not in json.dumps(model_payload)
    assert observed["killpg"] == (4242, signal.SIGKILL)
    assert observed["waited"] is True


def test_companion_timeout_falls_back_to_child_kill_when_process_group_kill_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import render_correction_companion as companion

    killed = threading.Event()
    waited = threading.Event()

    class Process:
        pid = 4243
        returncode = None

        def communicate(self, _payload: str | None = None, timeout: float | None = None):
            if not killed.is_set():
                raise companion.subprocess.TimeoutExpired("companion", timeout)
            self.returncode = -9
            return "", None

        def kill(self) -> None:
            killed.set()

        def wait(self) -> int:
            waited.set()
            return int(self.returncode or 0)

    monkeypatch.setattr(companion.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(companion.os, "killpg", lambda _pid, _sig: (_ for _ in ()).throw(PermissionError()))

    with pytest.raises(companion.CorrectionCompanionTimeout):
        companion.run_companion_subprocess(
            companion_config(),
            {"candidate": "bad", "validation_errors": [], "target_constraints": {}},
            0.1,
        )

    assert killed.is_set() and waited.is_set()


def test_companion_timeout_suppresses_kill_races_and_always_reaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent import render_correction_companion as companion

    drained = threading.Event()
    waited = threading.Event()
    direct_kill_attempted = threading.Event()

    class Process:
        pid = 4244
        returncode = None

        def communicate(self, _payload: str | None = None, timeout: float | None = None):
            if timeout is not None:
                raise companion.subprocess.TimeoutExpired("companion", timeout)
            drained.set()
            self.returncode = -9
            return "", None

        def kill(self) -> None:
            direct_kill_attempted.set()
            raise ProcessLookupError()

        def wait(self) -> int:
            waited.set()
            return int(self.returncode or 0)

    monkeypatch.setattr(companion.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(companion.os, "killpg", lambda _pid, _sig: (_ for _ in ()).throw(PermissionError()))

    with pytest.raises(companion.CorrectionCompanionTimeout, match="hard timeout"):
        companion.run_companion_subprocess(
            companion_config(),
            {"candidate": "bad", "validation_errors": [], "target_constraints": {}},
            0.1,
        )

    assert direct_kill_attempted.is_set()
    assert drained.is_set()
    assert waited.is_set()


@pytest.mark.asyncio
async def test_worker_reuses_agent_deduplicates_and_resets(tmp_path: Path) -> None:
    agents: list[FakeAgent] = []

    def factory() -> FakeAgent:
        agent = FakeAgent()
        agents.append(agent)
        return agent

    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        turn_timeout=2,
        reset_timeout=3,
        agent_factory=factory,
    )
    await worker.start()
    try:
        first = await worker.handle_request(request("event1"))
        second = await worker.handle_request(request("event2"))
        duplicate = await worker.handle_request(request("event1"))
        assert first["render_payloads"][0]["render"]["text"] == "turn 1"
        assert second["render_payloads"][0]["render"]["text"] == "turn 2"
        assert duplicate == first
        assert len(agents) == 1
        assert len(agents[0].prompts) == 2
        assert agents[0].prompts[0].count('"event_id": "event1"') == 1
        assert agents[0].prompts[1].count('"event_id": "event2"') == 1
        for prompt in agents[0].prompts:
            for required_fragment in (
                '"message_id":"msg_<event_id>_result"',
                '"action":"send"',
                '"target":{"chat_id":123456789}',
                '"render":{"text":"User-facing response."}',
                "Never put chat_id or text at the payload top level",
                "Never wrap the object in payload",
            ):
                assert required_fragment in prompt
            lowered_prompt = prompt.lower()
            assert "text-only, non-exclusive example" in lowered_prompt
            assert "tables and lists" in lowered_prompt
            assert "render.blocks" in prompt
            assert "buttons use render.buttons" in lowered_prompt
            assert "profile and task instructions" in lowered_prompt
            assert "exactly this object shape" not in lowered_prompt
        assert "turn_completed" in (tmp_path / "logs/persistent-specialist.jsonl").read_text()

        old_thread = second["conversation_instance_id"]
        codex_session = FakeCodexSession()
        agents[0]._codex_session = codex_session
        reset = await worker.handle_request(request("reset1", "reset"))
        assert reset["status"] == "completed"
        assert_complete_timing(reset)
        assert reset["timing_ms"]["retirement"] >= 0
        assert reset["timing_ms"]["model_and_tools"] == 0
        events = [json.loads(line) for line in (tmp_path / "logs/persistent-specialist.jsonl").read_text().splitlines()]
        reset_event = [event for event in events if event.get("event_id") == "reset1"][-1]
        assert reset_event["trace_id"] == reset["trace_id"]
        assert reset_event["timing_ms"] == reset["timing_ms"]
        assert reset["conversation_instance_id"] != old_thread
        assert agents[0].closed is True
        assert codex_session.closed is True

        third = await worker.handle_request(request("event3"))
        assert third["render_payloads"][0]["render"]["text"] == "turn 1"
        assert len(agents) == 2
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_reset_projects_new_agent_prompt_into_new_codex_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reset retires the old agent so its successor prompt reaches a new thread.

    Fresh-agent prompt assembly is covered by the conversation-loop tests. This
    test covers the worker boundary from reset retirement through projection.
    """
    from agent.codex_runtime import run_codex_app_server_turn

    profile_prompt = {"value": "profile prompt v1"}
    agents = []
    sessions = []

    class CapturingSession:
        def __init__(self, **kwargs) -> None:
            self.developer_instructions = kwargs.get("developer_instructions")
            self.closed = False
            sessions.append(self)

        def run_turn(self, *, user_input: str) -> types.SimpleNamespace:
            envelope = json.loads(user_input.split("Envelope JSON:\n", 1)[1])
            payload = [{
                "schema_version": "telegram.bridge.render_payload.v1",
                "message_id": f"msg_{envelope['event_id']}",
                "correlation_id": envelope["event_id"],
                "action": "send",
                "target": {"chat_id": envelope["chat_id"]},
                "render": {"text": "ok"},
            }]
            return types.SimpleNamespace(
                interrupted=False,
                error=None,
                thread_id=f"thread-{len(sessions)}",
                turn_id="turn-1",
                projected_messages=[],
                tool_iterations=0,
                final_text=json.dumps(payload),
                should_retire=False,
            )

        def close(self) -> None:
            self.closed = True

    class RuntimeAgent(FakeAgent):
        def __init__(self) -> None:
            super().__init__()
            self._cached_system_prompt = profile_prompt["value"]
            self.tool_progress_callback = None
            self._session_db = None
            self._session_db_created = False
            self.session_id = None
            self.session_api_calls = 0
            self._iters_since_skill = 0
            self._skill_nudge_interval = 0
            self.valid_tool_names = set()

        def run_conversation(self, prompt: str) -> dict:
            return run_codex_app_server_turn(
                self,
                user_message=prompt,
                original_user_message=prompt,
                messages=[{"role": "user", "content": prompt}],
                effective_task_id="persistent-specialist",
            )

        def _sync_external_memory_for_turn(self, **kwargs) -> None:
            return None

    def factory() -> RuntimeAgent:
        agent = RuntimeAgent()
        agents.append(agent)
        return agent

    monkeypatch.setattr(
        "agent.transports.codex_app_server_session.CodexAppServerSession",
        CapturingSession,
    )
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        turn_timeout=2,
        reset_timeout=3,
        agent_factory=factory,
    )
    await worker.start()
    try:
        first = await worker.handle_request(request("profile-v1"))
        assert first["status"] == "completed"
        assert sessions[0].developer_instructions == "profile prompt v1"

        profile_prompt["value"] = "profile prompt v2"
        reset = await worker.handle_request(request("reset-profile", "reset"))
        assert reset["status"] == "completed"
        assert sessions[0].closed is True

        second = await worker.handle_request(request("profile-v2"))
        assert second["status"] == "completed"
        assert len(agents) == 2
        assert len(sessions) == 2
        assert sessions[1].developer_instructions == "profile prompt v2"
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_v2_completed_turn_persists_raw_malformed_render_candidate(tmp_path: Path) -> None:
    agent = CorrectionAgent()
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=lambda: agent,
    )
    await worker.start()
    try:
        response = await worker.handle_request(v2_request("completed-malformed"))
    finally:
        await worker.stop()

    candidate = '[{"schema_version":"telegram.bridge.render_payload.v1"'
    assert response["protocol_version"] == PROTOCOL_V2
    assert response["status"] == "completed", response
    assert response["execution_state"] == "completed"
    assert response["presentation_state"] == "candidate_unvalidated"
    assert response["render_candidate"]["content"] == candidate
    assert response["render_candidate"]["sha256"] == persistent_worker._candidate_hash(candidate)
    assert "render_payloads" not in response
    ledger_path = worker._ledger_path("conv_" + "1" * 64, "completed-malformed")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["execution_state"] == "completed"
    assert ledger["presentation_state"] == "candidate_unvalidated"
    assert ledger["render_candidate"] == candidate
    assert ledger["render_candidate_sha256"] == response["render_candidate"]["sha256"]


@pytest.mark.asyncio
async def test_v3_terminal_presenter_promotes_exact_output_and_replays(
    tmp_path: Path,
) -> None:
    install_test_presenter(tmp_path)
    agents: list[TerminalPresenterAgent] = []

    def factory() -> TerminalPresenterAgent:
        agent = TerminalPresenterAgent()
        agents.append(agent)
        return agent

    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=factory,
    )
    await worker.start()
    request_value = v3_request("terminal-presenter")
    try:
        response = await worker.handle_request(request_value)
        replay = await worker.handle_request(request_value)
    finally:
        await worker.stop()

    expected = json.dumps([{
        "schema_version": "telegram.bridge.render_payload.v1",
        "message_id": "msg_terminal-presenter",
        "correlation_id": "terminal-presenter",
        "action": "send",
        "target": {"chat_id": 123},
        "render": {"text": "exact presenter"},
    }], sort_keys=True, separators=(",", ":"))
    assert response["status"] == "completed", response
    assert response["render_candidate"]["content"] == expected
    assert response["candidate_source"] == {
        "kind": "terminal_presenter",
        "presenter_id": "test-presenter",
        "presenter_version": "1.0.0",
        "presenter_sha256": hashlib.sha256(
            (tmp_path / "skills/test/present.py").read_bytes()
        ).hexdigest(),
        "invocation_id": response["candidate_source"]["invocation_id"],
    }
    assert response["candidate_source"]["invocation_id"].startswith("present_")
    assert replay == response
    assert len(agents) == 1 and agents[0].turns == 1


class ModelReportingAgent(FakeAgent):
    """Mirrors an AIAgent on the codex app-server runtime: configured model
    attributes (the claim) plus the transport's runtime_model observation in
    the run_conversation result. The runtime model deliberately differs so
    tests prove the observation is forwarded verbatim, not restated from
    configuration."""

    model = "gpt-5.3-codex-spark"
    provider = "openai-codex"
    api_mode = "codex_app_server"

    def run_conversation(self, prompt: str) -> dict:
        result = super().run_conversation(prompt)
        result["runtime_model"] = "gpt-5.6-sol"
        result["runtime_model_provider"] = "openai"
        return result


class RequestedOnlyAgent(FakeAgent):
    """Configured attributes but no runtime observation (e.g. a runtime that
    predates thread/start model reporting)."""

    model = "gpt-5.3-codex-spark"
    provider = "openai-codex"
    api_mode = "codex_app_server"


def _v2_response_schema() -> dict:
    return json.loads(
        (Path(__file__).parents[1] / "fixtures/telegram.bridge.persistent_service_response.v2.schema.json")
        .read_text(encoding="utf-8")
    )


@pytest.mark.asyncio
async def test_v2_completed_turn_emits_model_resolution_metadata(tmp_path: Path) -> None:
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=ModelReportingAgent,
    )
    await worker.start()
    try:
        response = await worker.handle_request(v2_request("model-metadata-1"))
    finally:
        await worker.stop()

    assert response["status"] == "completed"
    assert response["requested_model"] == "gpt-5.3-codex-spark"
    assert response["requested_provider"] == "openai-codex"
    assert response["requested_api_mode"] == "codex_app_server"
    # The observation is forwarded verbatim — here it disagrees with the
    # requested model, which is exactly the substitution signal the bridge
    # verifies. Provider/api_mode stay in Hermes naming for comparability.
    assert response["runtime_model"] == "gpt-5.6-sol"
    assert response["runtime_provider"] == "openai-codex"
    assert response["runtime_api_mode"] == "codex_app_server"
    jsonschema.Draft202012Validator(_v2_response_schema()).validate(response)
    # The durable ledger record replays the same metadata.
    ledger_path = worker._ledger_path("conv_" + "1" * 64, "model-metadata-1")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["response"]["runtime_model"] == "gpt-5.6-sol"


@pytest.mark.asyncio
async def test_v2_turn_without_runtime_observation_emits_requested_only(tmp_path: Path) -> None:
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=RequestedOnlyAgent,
    )
    await worker.start()
    try:
        response = await worker.handle_request(v2_request("model-metadata-2"))
    finally:
        await worker.stop()

    assert response["status"] == "completed"
    assert response["requested_model"] == "gpt-5.3-codex-spark"
    for absent in ("runtime_model", "runtime_provider", "runtime_api_mode"):
        assert absent not in response
    jsonschema.Draft202012Validator(_v2_response_schema()).validate(response)


@pytest.mark.asyncio
async def test_v2_turn_without_any_model_attributes_emits_no_metadata(tmp_path: Path) -> None:
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=FakeAgent,
    )
    await worker.start()
    try:
        response = await worker.handle_request(v2_request("model-metadata-3"))
    finally:
        await worker.stop()

    assert response["status"] == "completed"
    for absent in (
        "requested_model", "requested_provider", "requested_api_mode",
        "runtime_model", "runtime_provider", "runtime_api_mode",
    ):
        assert absent not in response
    jsonschema.Draft202012Validator(_v2_response_schema()).validate(response)


@pytest.mark.asyncio
async def test_v2_duplicate_in_flight_turn_never_reaches_the_model(tmp_path: Path) -> None:
    """The runtime ledger, not the agent, owns mutation idempotency: a
    duplicate of an event whose turn is still running must be rejected
    without a second model run (task 150 evidence)."""
    entered = threading.Event()
    release = threading.Event()
    agents: list[BlockingAgent] = []

    def factory() -> BlockingAgent:
        agent = BlockingAgent(entered, release)
        agents.append(agent)
        return agent

    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        turn_timeout=5,
        reset_timeout=6,
        agent_factory=factory,
    )
    await worker.start()
    try:
        owner = asyncio.create_task(worker.handle_request(v2_request("dup-in-flight")))
        await asyncio.to_thread(entered.wait, 2)
        duplicate = await worker.handle_request(v2_request("dup-in-flight"))
        release.set()
        owner_response = await owner
    finally:
        release.set()
        await worker.stop()

    assert owner_response["status"] == "completed"
    assert duplicate["status"] == "failed"
    assert duplicate["error"]["code"] == "DUPLICATE_IN_FLIGHT"
    assert duplicate["execution_state"] == "in_flight"
    assert len(agents) == 1 and agents[0].turns == 1


@pytest.mark.asyncio
async def test_v2_event_id_reuse_with_different_content_is_a_conflict(tmp_path: Path) -> None:
    agents: list[FakeAgent] = []

    def factory() -> FakeAgent:
        agent = FakeAgent()
        agents.append(agent)
        return agent

    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=factory,
    )
    await worker.start()
    try:
        first = await worker.handle_request(v2_request("reused-event"))
        tampered = v2_request("reused-event")
        tampered["envelope"]["message"] = {"text": "use 99 instead"}
        conflict = await worker.handle_request(tampered)
    finally:
        await worker.stop()

    assert first["status"] == "completed"
    assert conflict["status"] == "failed"
    assert conflict["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    assert agents and agents[0].turns == 1


@pytest.mark.asyncio
async def test_ledger_retention_keeps_replay_until_horizon_then_prunes(tmp_path: Path) -> None:
    """Terminal ledger records survive worker restarts (so redelivery keeps
    replaying, never re-running) until the 30-day retention horizon, far
    beyond any Telegram redelivery window; only then are they compacted."""
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=FakeAgent,
    )
    await worker.start()
    try:
        first = await worker.handle_request(v2_request("retained-event"))
    finally:
        await worker.stop()
    assert first["status"] == "completed"
    ledger_path = worker._ledger_path("conv_" + "1" * 64, "retained-event")

    agents: list[FakeAgent] = []

    def factory() -> FakeAgent:
        agent = FakeAgent()
        agents.append(agent)
        return agent

    restarted = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=factory,
    )
    await restarted.start()
    try:
        replay = await restarted.handle_request(v2_request("retained-event"))
    finally:
        await restarted.stop()
    assert replay == first
    assert not agents  # replay came from the ledger; no agent was built
    assert ledger_path.exists()

    record = json.loads(ledger_path.read_text(encoding="utf-8"))
    record["accepted_at"] = "2026-06-01T00:00:00+00:00"
    ledger_path.write_text(json.dumps(record), encoding="utf-8")
    expired = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=FakeAgent,
    )
    await expired.start()
    try:
        assert not ledger_path.exists()
    finally:
        await expired.stop()


@pytest.mark.asyncio
async def test_v2_failed_turn_emits_no_model_metadata(tmp_path: Path) -> None:
    class FailingModelAgent(FailingAgent):
        model = "gpt-5.3-codex-spark"
        provider = "openai-codex"
        api_mode = "codex_app_server"

    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=FailingModelAgent,
    )
    await worker.start()
    try:
        response = await worker.handle_request(v2_request("model-metadata-4"))
    finally:
        await worker.stop()

    # Only completed turns are verifiable; failure responses must not carry
    # model claims (the v2 schema permits them, the producer refuses).
    assert response["status"] == "failed"
    for absent in (
        "requested_model", "requested_provider", "requested_api_mode",
        "runtime_model", "runtime_provider", "runtime_api_mode",
    ):
        assert absent not in response


@pytest.mark.asyncio
async def test_v2_reset_uses_render_candidate_contract(tmp_path: Path) -> None:
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=FakeAgent,
    )
    await worker.start()
    try:
        response = await worker.handle_request(v2_request("reset-v2", "reset"))
    finally:
        await worker.stop()

    assert response["protocol_version"] == PROTOCOL_V2
    assert response["execution_state"] == "completed"
    assert response["presentation_state"] == "candidate_unvalidated"
    payloads = json.loads(response["render_candidate"]["content"])
    assert payloads[0]["correlation_id"] == "reset-v2"
    assert "render_payloads" not in response


@pytest.mark.asyncio
async def test_v2_app_server_correction_uses_isolated_companion_once_and_is_durable(tmp_path: Path) -> None:
    agent = CorrectionAgent()
    companion = FakeCorrectionCompanion()
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=lambda: agent,
        correction_companion_resolver=companion_descriptor,
        correction_companion_credentials_resolver=lambda _descriptor, _timeout: companion_config("rotated-token"),
        correction_companion_runner=companion,
    )
    await worker.start()
    try:
        original = await worker.handle_request(v2_request("completed-malformed"))
        correction = await worker.handle_request(correction_request("completed-malformed"))
        correction_replay = await worker.handle_request(correction_request("completed-malformed"))
        original_replay = await worker.handle_request(v2_request("completed-malformed"))
        second_attempt = await worker.handle_request(
            correction_request("completed-malformed", "corr_attempt_2")
        )
    finally:
        await worker.stop()

    assert original["execution_state"] == "completed"
    assert correction["status"] == "completed"
    assert correction["execution_state"] == "completed"
    assert correction["presentation_state"] == "correction_candidate_unvalidated"
    corrected = json.loads(correction["render_candidate"]["content"])
    assert corrected[0]["render"]["text"] == "corrected without rerunning domain work"
    assert correction_replay == correction
    assert original_replay == original
    assert second_attempt["status"] == "failed"
    assert second_attempt["execution_state"] == "completed"
    assert second_attempt["presentation_state"] == "correction_candidate_unvalidated"
    assert second_attempt["error"]["code"] == "CORRECTION_ALREADY_ATTEMPTED"
    assert agent.turns == 1
    assert agent.tool_snapshots == [agent.tool_snapshots[0]]
    assert agent.tool_snapshots[0]
    assert agent.executed_tools == 0
    assert len(companion.calls) == 1
    config, repair, _timeout = companion.calls[0]
    assert config == companion_config("rotated-token")
    assert set(repair) == {"candidate", "validation_errors", "target_constraints"}
    assert repair["candidate"] == original["render_candidate"]["content"]
    assert "Envelope JSON" not in json.dumps(repair)
    assert "rotated-token" not in json.dumps(worker.__dict__, default=str)

    ledger = json.loads(
        worker._ledger_path("conv_" + "1" * 64, "completed-malformed").read_text(encoding="utf-8")
    )
    assert ledger["correction_attempt_id"] == "corr_attempt_1"
    assert ledger["correction_attempt_count"] == 1
    assert ledger["corrected_candidate"] == correction["render_candidate"]["content"]
    assert ledger["correction_response"] == correction


@pytest.mark.asyncio
async def test_v2_correction_capability_is_not_advertised_when_companion_is_unavailable(tmp_path: Path) -> None:
    agent = CorrectionAgent(api_mode="codex_app_server")
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=lambda: agent,
        correction_companion_resolver=lambda _profile_root: None,
    )
    await worker.start()
    serve = asyncio.create_task(worker.serve_forever())
    try:
        await worker.handle_request(v2_request("completed-malformed"))
        response = await worker.handle_request(correction_request("completed-malformed"))
        replay = await worker.handle_request(correction_request("completed-malformed"))
        reader, writer = await asyncio.open_unix_connection(str(worker.socket_path))
        writer.write(json.dumps({
            "protocol_version": PROTOCOL_V2,
            "request_id": "req_hello_unavailable",
            "operation": "hello",
            "client": {"name": "telegram-bridge", "revision": "test"},
        }).encode() + b"\n")
        await writer.drain()
        hello = json.loads(await reader.readline())
        writer.close()
        await writer.wait_closed()
    finally:
        serve.cancel()
        await asyncio.gather(serve, return_exceptions=True)
        await worker.stop()

    assert response == replay
    assert response["status"] == "failed"
    assert response["execution_state"] == "completed"
    assert response["presentation_state"] == "correction_failed"
    assert response["error"]["code"] == "TOOL_FREE_CORRECTION_UNAVAILABLE"
    assert "tool_free_render_correction" not in hello["capabilities"]
    assert agent.turns == 1


@pytest.mark.asyncio
async def test_v2_concurrent_correction_replay_waits_for_one_terminal_result(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    companion = FakeCorrectionCompanion(entered=entered, release=release)
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=CorrectionAgent,
        correction_companion_resolver=companion_descriptor,
        correction_companion_credentials_resolver=lambda _descriptor, _timeout: companion_config(),
        correction_companion_runner=companion,
    )
    await worker.start()
    try:
        await worker.handle_request(v2_request("completed-malformed"))
        first = asyncio.create_task(worker.handle_request(correction_request("completed-malformed")))
        assert await asyncio.to_thread(entered.wait, 1)
        replay = asyncio.create_task(worker.handle_request(correction_request("completed-malformed")))
        await asyncio.sleep(0.05)
        assert replay.done() is False
        release.set()
        first_response, replay_response = await asyncio.gather(first, replay)
    finally:
        release.set()
        await worker.stop()

    assert first_response == replay_response
    assert first_response["status"] == "completed"
    assert len(companion.calls) == 1


@pytest.mark.asyncio
async def test_v2_shorter_deadline_duplicate_does_not_terminalize_owner_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(persistent_worker, "CORRECTION_CREDENTIAL_PREFLIGHT_SECONDS", 0.001)
    entered = threading.Event()
    release = threading.Event()
    companion = FakeCorrectionCompanion(entered=entered, release=release)
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        turn_timeout=0.5,
        agent_factory=CorrectionAgent,
        correction_companion_resolver=companion_descriptor,
        correction_companion_credentials_resolver=lambda _descriptor, _timeout: companion_config(),
        correction_companion_runner=companion,
    )
    await worker.start()
    try:
        await worker.handle_request(v2_request("completed-malformed"))
        owner_request = correction_request("completed-malformed")
        owner_request["deadline"] = (
            datetime.now(timezone.utc) + timedelta(seconds=0.3)
        ).isoformat()
        owner = asyncio.create_task(worker.handle_request(owner_request))
        assert await asyncio.to_thread(entered.wait, 1)

        duplicate_request = dict(owner_request)
        duplicate_request["request_id"] = "req_correction_short_duplicate"
        duplicate_request["deadline"] = (
            datetime.now(timezone.utc) + timedelta(seconds=0.03)
        ).isoformat()
        duplicate_response = await asyncio.wait_for(
            worker.handle_request(duplicate_request),
            timeout=0.15,
        )
        in_flight_record = json.loads(
            worker._ledger_path(owner_request["conversation_id"], owner_request["event_id"]).read_text()
        )

        release.set()
        owner_response = await asyncio.wait_for(owner, timeout=0.2)
        replay_response = await worker.handle_request(duplicate_request)
        terminal_record = json.loads(
            worker._ledger_path(owner_request["conversation_id"], owner_request["event_id"]).read_text()
        )
    finally:
        release.set()
        await worker.stop()

    assert duplicate_response["status"] == "failed"
    assert duplicate_response["error"]["code"] == "CORRECTION_DEADLINE_EXPIRED"
    assert in_flight_record["presentation_state"] == "correction_in_flight"
    assert "correction_response" not in in_flight_record
    assert in_flight_record["correction_deadline"] == owner_request["deadline"]
    assert owner_response["status"] == "completed"
    assert replay_response == owner_response
    assert terminal_record["correction_response"] == owner_response
    assert len(companion.calls) == 1


@pytest.mark.asyncio
async def test_v2_nonterminating_credential_refresh_completes_owner_and_duplicate_at_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(persistent_worker, "CORRECTION_CREDENTIAL_PREFLIGHT_SECONDS", 0.001)
    refresh_entered = threading.Event()
    release_refresh = threading.Event()

    def credentials(_descriptor: dict, _timeout: float) -> dict:
        refresh_entered.set()
        release_refresh.wait()
        return companion_config()

    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        turn_timeout=0.05,
        agent_factory=CorrectionAgent,
        correction_companion_resolver=companion_descriptor,
        correction_companion_credentials_resolver=credentials,
        correction_companion_runner=FakeCorrectionCompanion(),
    )
    await worker.start()
    try:
        await worker.handle_request(v2_request("completed-malformed"))
        request = correction_request("completed-malformed")
        first = asyncio.create_task(worker.handle_request(request))
        assert await asyncio.to_thread(refresh_entered.wait, 1)
        duplicate = asyncio.create_task(worker.handle_request(dict(request)))
        duplicate_response = await asyncio.wait_for(duplicate, timeout=0.2)
        in_flight_record = json.loads(
            worker._ledger_path(request["conversation_id"], request["event_id"]).read_text()
        )
        release_refresh.set()
        first_response = await asyncio.wait_for(first, timeout=0.2)
        replay_response = await worker.handle_request(dict(request))
        final_record = json.loads(
            worker._ledger_path(request["conversation_id"], request["event_id"]).read_text()
        )
    finally:
        release_refresh.set()
        await worker.stop()

    assert first_response["status"] == "failed"
    assert first_response["error"]["code"] == "CORRECTION_DEADLINE_EXPIRED"
    assert duplicate_response["status"] == "failed"
    assert duplicate_response["error"]["code"] == "CORRECTION_DEADLINE_EXPIRED"
    assert in_flight_record["presentation_state"] == "correction_in_flight"
    assert "correction_response" not in in_flight_record
    assert replay_response == first_response
    assert final_record["correction_response"] == first_response


@pytest.mark.asyncio
async def test_v2_correction_does_not_spawn_companion_after_lock_wait_exhausts_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(persistent_worker, "CORRECTION_CREDENTIAL_PREFLIGHT_SECONDS", 0.001)
    companion = FakeCorrectionCompanion()
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=CorrectionAgent,
        correction_companion_resolver=companion_descriptor,
        correction_companion_credentials_resolver=lambda _descriptor, _timeout: companion_config(),
        correction_companion_runner=companion,
    )
    await worker.start()
    try:
        await worker.handle_request(v2_request("completed-malformed"))
        await worker._conversation_lock.acquire()
        correction = correction_request("completed-malformed")
        correction["deadline"] = (
            datetime.now(timezone.utc) + timedelta(seconds=0.03)
        ).isoformat()
        pending = asyncio.create_task(worker.handle_request(correction))
        await asyncio.sleep(0.05)
        worker._conversation_lock.release()
        response = await pending
    finally:
        if worker._conversation_lock.locked():
            worker._conversation_lock.release()
        await worker.stop()

    assert response["status"] == "failed"
    assert response["error"]["code"] == "CORRECTION_DEADLINE_EXPIRED"
    assert companion.calls == []


@pytest.mark.asyncio
async def test_v2_refresh_quiesces_and_persists_after_deadline_without_blocking_domain_or_spawning_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(persistent_worker, "CORRECTION_CREDENTIAL_PREFLIGHT_SECONDS", 0.001)
    refresh_entered = threading.Event()
    persisted = threading.Event()
    token_path = tmp_path / "state/rotated-token.txt"
    credential_calls: list[dict] = []
    model = FakeCorrectionCompanion()
    agent = CorrectionAgent()

    def credentials(descriptor: dict, _timeout: float = 1) -> dict:
        credential_calls.append(dict(descriptor))
        if not token_path.exists():
            refresh_entered.set()
            time.sleep(0.08)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = token_path.with_suffix(".tmp")
            temporary.write_text("rotated-token", encoding="utf-8")
            os.replace(temporary, token_path)
            persisted.set()
        return companion_config(token_path.read_text(encoding="utf-8"))

    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        turn_timeout=0.05,
        agent_factory=lambda: agent,
        correction_companion_resolver=companion_descriptor,
        correction_companion_credentials_resolver=credentials,
        correction_companion_runner=model,
    )
    await worker.start()
    try:
        await worker.handle_request(v2_request("completed-malformed"))
        correction_task = asyncio.create_task(
            worker.handle_request(correction_request("completed-malformed"))
        )
        assert await asyncio.to_thread(refresh_entered.wait, 1)
        domain_response = await asyncio.wait_for(
            worker.handle_request(v2_request("domain-during-refresh")),
            timeout=0.04,
        )
        correction_response = await correction_task
        subsequent = credentials(worker._correction_companion_descriptor)
    finally:
        await worker.stop()

    assert domain_response["status"] == "completed"
    assert correction_response["status"] == "failed"
    assert correction_response["error"]["code"] == "CORRECTION_DEADLINE_EXPIRED"
    assert persisted.is_set()
    assert token_path.read_text(encoding="utf-8") == "rotated-token"
    assert subsequent["api_key"] == "rotated-token"
    assert model.calls == []
    assert len(credential_calls) == 2
    assert agent.turns == 2


@pytest.mark.asyncio
async def test_v2_in_flight_correction_recovers_failed_without_model_replay(tmp_path: Path) -> None:
    original_agent = CorrectionAgent()
    first = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=lambda: original_agent,
    )
    await first.start()
    try:
        await first.handle_request(v2_request("completed-malformed"))
    finally:
        await first.stop()

    ledger_path = first._ledger_path("conv_" + "1" * 64, "completed-malformed")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    pending = correction_request("completed-malformed")
    ledger.update(
        correction_attempt_id="corr_attempt_1",
        correction_attempt_count=1,
        correction_fingerprint=persistent_worker._correction_fingerprint(pending),
        presentation_state="correction_in_flight",
    )
    ledger.pop("correction_response", None)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    replacement_agent = CorrectionAgent()
    recovered = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=lambda: replacement_agent,
    )
    await recovered.start()
    try:
        response = await recovered.handle_request(pending)
    finally:
        await recovered.stop()

    assert response["status"] == "failed"
    assert response["execution_state"] == "completed"
    assert response["presentation_state"] == "correction_failed"
    assert response["error"]["code"] == "WORKER_RESTARTED_DURING_CORRECTION"
    assert replacement_agent.turns == 0


@pytest.mark.asyncio
async def test_v2_correction_timeout_quiesces_companion_before_rejecting_prequeued_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(persistent_worker, "CORRECTION_CREDENTIAL_PREFLIGHT_SECONDS", 0.001)
    entered = threading.Event()
    terminated = threading.Event()
    joined = threading.Event()
    agent = CorrectionAgent()

    model_spawned = threading.Event()

    def timed_out_companion(_descriptor: dict, _repair: dict, _timeout: float) -> str:
        entered.set()
        time.sleep(0.05)
        terminated.set()
        joined.set()
        raise persistent_worker.CorrectionCompanionTimeout("credential refresh timed out")

    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        turn_timeout=0.01,
        agent_factory=lambda: agent,
        correction_companion_resolver=companion_descriptor,
        correction_companion_credentials_resolver=lambda _descriptor, _timeout: companion_config(),
        correction_companion_runner=timed_out_companion,
    )
    await worker.start()
    try:
        await worker.handle_request(v2_request("completed-malformed"))
        correction = asyncio.create_task(worker.handle_request(correction_request("completed-malformed")))
        assert await asyncio.to_thread(entered.wait, 1)
        queued = asyncio.create_task(worker.handle_request(v2_request("queued-before-timeout")))
        response, later = await asyncio.gather(correction, queued)
    finally:
        await worker.stop()

    assert response["status"] == "failed"
    assert response["execution_state"] == "completed"
    assert response["presentation_state"] == "correction_failed"
    assert response["error"]["code"] == "CORRECTION_TIMEOUT"
    assert later["error"]["code"] == "CONVERSATION_UNHEALTHY"
    assert later["execution_state"] == "not_started"
    assert terminated.is_set() and joined.is_set()
    assert model_spawned.is_set() is False
    assert agent.turns == 1


@pytest.mark.asyncio
async def test_worker_callbacks_do_not_mutate_process_context_and_set_agent_session_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_root = tmp_path / "profiles" / "hello-world"
    profile_root.mkdir(parents=True)
    bridge_cwd = tmp_path / "profiles" / "telegram-bridge"
    bridge_cwd.mkdir(parents=True)
    monkeypatch.chdir(bridge_cwd)
    monkeypatch.setenv("HERMES_HOME", str(bridge_cwd))
    monkeypatch.setenv("TERMINAL_CWD", str(bridge_cwd))
    agents: list[FakeAgent] = []
    constructed_in: list[tuple[str, str, str]] = []

    def factory() -> FakeAgent:
        constructed_in.append((os.getcwd(), os.environ["HERMES_HOME"], os.environ["TERMINAL_CWD"]))
        agent = FakeAgent()
        agents.append(agent)
        return agent

    worker = PersistentSpecialistWorker(
        profile_root=profile_root,
        socket_path=profile_root / "w.sock",
        agent_id="hello_world",
        agent_factory=factory,
    )
    await worker.start()
    try:
        response = await worker.handle_request(request("profile-context"))
    finally:
        await worker.stop()

    assert constructed_in == [(str(bridge_cwd), str(bridge_cwd), str(bridge_cwd))]
    assert (os.getcwd(), os.environ["HERMES_HOME"], os.environ["TERMINAL_CWD"]) == (
        str(bridge_cwd), str(bridge_cwd), str(bridge_cwd)
    )
    assert agents[0].session_cwd == str(profile_root)
    assert agents[0].effective_cwds == [(str(bridge_cwd), str(bridge_cwd), str(bridge_cwd))]
    assert_complete_timing(response)


@pytest.mark.asyncio
async def test_worker_callback_threads_do_not_cross_contaminate_process_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge_cwd = tmp_path / "bridge"
    bridge_cwd.mkdir()
    monkeypatch.chdir(bridge_cwd)
    monkeypatch.setenv("HERMES_HOME", str(bridge_cwd))
    monkeypatch.setenv("TERMINAL_CWD", str(bridge_cwd))
    barrier = threading.Barrier(2)
    observed: list[tuple[str, str, str]] = []

    def factory() -> FakeAgent:
        barrier.wait(timeout=1)
        observed.append((os.getcwd(), os.environ["HERMES_HOME"], os.environ["TERMINAL_CWD"]))
        return FakeAgent()

    workers = [
        PersistentSpecialistWorker(
            profile_root=tmp_path / name,
            socket_path=tmp_path / name / "w.sock",
            agent_id=name,
            agent_factory=factory,
        )
        for name in ("first", "second")
    ]
    for worker in workers:
        worker.profile_root.mkdir()
        await worker.start()
    try:
        await asyncio.gather(*(worker.handle_request(request(f"race-{worker.agent_id}")) for worker in workers))
    finally:
        await asyncio.gather(*(worker.stop() for worker in workers))

    assert observed == [(str(bridge_cwd), str(bridge_cwd), str(bridge_cwd))] * 2
    assert (os.getcwd(), os.environ["HERMES_HOME"], os.environ["TERMINAL_CWD"]) == (
        str(bridge_cwd), str(bridge_cwd), str(bridge_cwd)
    )


@pytest.mark.asyncio
async def test_worker_queued_turn_reports_actual_queue_wait(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "w.sock",
        agent_id="hello_world",
        agent_factory=lambda: BlockingAgent(entered, release),
    )
    await worker.start()
    try:
        active = asyncio.create_task(worker.handle_request(request("active-queue")))
        assert await asyncio.to_thread(entered.wait, 1)
        queued = asyncio.create_task(worker.handle_request(request("queued-wait")))
        await asyncio.sleep(0.05)
        release.set()
        assert (await active)["status"] == "completed"
        queued_response = await queued
    finally:
        release.set()
        await worker.stop()

    assert_complete_timing(queued_response)
    assert queued_response["timing_ms"]["queue"] >= 40


def test_profile_context_loaders_read_specialist_soul_and_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent.prompt_builder import build_context_files_prompt, load_soul_md

    profile_root = tmp_path / "profile"
    profile_root.mkdir()
    (profile_root / "SOUL.md").write_text("Specialist identity", encoding="utf-8")
    (profile_root / "AGENTS.md").write_text("Specialist operating instructions", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(profile_root))
    monkeypatch.setenv("TERMINAL_CWD", str(profile_root))

    assert load_soul_md() == "Specialist identity"
    assert "Specialist operating instructions" in build_context_files_prompt(
        cwd=profile_root,
        skip_soul=True,
    )


@pytest.mark.asyncio
async def test_worker_failure_responses_and_logs_include_complete_timing(tmp_path: Path) -> None:
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "w.sock",
        agent_id="hello_world",
        agent_factory=FailingAgent,
    )
    await worker.start()
    try:
        response = await worker.handle_request(request("failure"))
    finally:
        await worker.stop()

    assert response["error"]["code"] == "TURN_FAILED"
    assert_complete_timing(response)
    events = [json.loads(line) for line in (tmp_path / "logs/persistent-specialist.jsonl").read_text().splitlines()]
    failure = next(event for event in events if event["event"] == "turn_failed")
    assert failure["trace_id"] == response["trace_id"]
    assert failure["timing_ms"] == response["timing_ms"]


@pytest.mark.asyncio
async def test_worker_timeout_response_and_log_include_complete_timing(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "w.sock",
        agent_id="hello_world",
        turn_timeout=0.01,
        agent_factory=lambda: BlockingAgent(entered, release),
    )
    await worker.start()
    try:
        response = await worker.handle_request(request("timeout"))
    finally:
        release.set()
        await worker.stop()

    assert response["error"]["code"] == "TURN_TIMEOUT"
    assert_complete_timing(response)
    events = [json.loads(line) for line in (tmp_path / "logs/persistent-specialist.jsonl").read_text().splitlines()]
    failure = next(event for event in events if event["event"] == "turn_failed")
    assert failure["trace_id"] == response["trace_id"]
    assert failure["timing_ms"] == response["timing_ms"]


@pytest.mark.asyncio
async def test_worker_shutdown_rejects_subsequent_turns(tmp_path: Path) -> None:
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "w.sock",
        agent_id="hello_world",
        agent_factory=FakeAgent,
    )
    await worker.start()
    try:
        wait_for_owner = asyncio.create_task(worker.wait_for_shutdown())
        shutdown = await worker.handle_request({
            "protocol_version": PROTOCOL,
            "request_id": "req_shutdown",
            "operation": "shutdown",
        })
        assert worker.shutdown_requested is False
        worker.acknowledge_shutdown()
        await asyncio.wait_for(wait_for_owner, timeout=1)
        rejected = await worker.handle_request(request("after-shutdown"))
    finally:
        await worker.stop()

    assert shutdown["status"] == "stopping"
    assert_complete_timing(shutdown)
    assert worker.shutdown_requested is True
    assert rejected["error"]["code"] == "WORKER_STOPPING"
    assert_complete_timing(rejected)


@pytest.mark.asyncio
async def test_worker_socket_negotiation_permissions_and_restart_recovery(tmp_path: Path) -> None:
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=FakeAgent,
    )
    await worker.start()
    serve = asyncio.create_task(worker.serve_forever())
    try:
        assert stat.S_IMODE(worker.socket_path.stat().st_mode) == 0o600
        reader, writer = await asyncio.open_unix_connection(str(worker.socket_path))
        hello = {
            "protocol_version": PROTOCOL,
            "request_id": "req_hello",
            "operation": "hello",
            "client": {"name": "telegram-bridge", "revision": "test"},
        }
        writer.write(json.dumps(hello).encode() + b"\n")
        await writer.drain()
        negotiated = json.loads(await reader.readline())
        assert negotiated["status"] == "ready"
        writer.write(json.dumps(request("socket-event")).encode() + b"\n")
        await writer.drain()
        response = json.loads(await reader.readline())
        assert response["status"] == "completed"
        writer.close()
        await writer.wait_closed()
    finally:
        serve.cancel()
        await asyncio.gather(serve, return_exceptions=True)
        await worker.stop()

    ledger = next((tmp_path / "state/persistent-runtime/ledger").glob("*.json"))
    record = json.loads(ledger.read_text())
    record["state"] = "in_flight"
    record.pop("response", None)
    ledger.write_text(json.dumps(record))
    accepted_request = request("accepted-recovery")
    accepted_record = {
        **record,
        "request_id": accepted_request["request_id"],
        "event_id": accepted_request["event_id"],
        "fingerprint": persistent_worker._fingerprint(accepted_request),
        "operation": "turn",
        "state": "accepted",
    }
    accepted_path = tmp_path / "state/persistent-runtime/ledger" / (
        f"{persistent_worker._ledger_key(accepted_request['conversation_id'], accepted_request['event_id'])}.json"
    )
    accepted_path.write_text(json.dumps(accepted_record))
    recovered = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=FakeAgent,
    )
    await recovered.start()
    try:
        replay = await recovered.handle_request(request("socket-event"))
        accepted_replay = await recovered.handle_request(accepted_request)
        assert replay["execution_state"] == "outcome_unknown"
        assert replay["error"]["code"] == "WORKER_RESTARTED_IN_FLIGHT"
        assert_complete_timing(replay)
        assert_terminal_log(tmp_path, replay, "socket-event", "turn_recovered")
        assert_terminal_log(tmp_path, replay, "socket-event", "turn_replayed")
        assert accepted_replay["execution_state"] == "not_started"
        assert accepted_replay["error"]["code"] == "WORKER_RESTARTED_BEFORE_START"
        assert_complete_timing(accepted_replay)
        assert_terminal_log(tmp_path, accepted_replay, "accepted-recovery", "turn_recovered")
        assert_terminal_log(tmp_path, accepted_replay, "accepted-recovery", "turn_replayed")
    finally:
        await recovered.stop()


@pytest.mark.asyncio
async def test_v3_declined_presenter_capability_is_not_enabled(tmp_path: Path) -> None:
    install_test_presenter(tmp_path)
    agents: list[FakeAgent] = []

    def factory() -> FakeAgent:
        agent = FakeAgent()
        agents.append(agent)
        return agent

    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=factory,
    )
    await worker.start()
    serve = asyncio.create_task(worker.serve_forever())
    try:
        reader, writer = await asyncio.open_unix_connection(str(worker.socket_path))
        writer.write(json.dumps({
            "protocol_version": PROTOCOL_V3,
            "request_id": "req_hello_v3_declined",
            "operation": "hello",
            "client": {"name": "telegram-bridge", "revision": "test"},
            "accepted_capabilities": [],
        }).encode() + b"\n")
        await writer.drain()
        negotiated = json.loads(await reader.readline())
        assert "terminal_presenter_finalization.v1" not in negotiated["capabilities"]
        turn = v3_request("v3-declined")
        turn.pop("_terminal_presenter_negotiated")
        writer.write(json.dumps(turn).encode() + b"\n")
        await writer.drain()
        response = json.loads(await reader.readline())
        writer.close()
        await writer.wait_closed()
    finally:
        serve.cancel()
        await asyncio.gather(serve, return_exceptions=True)
        await worker.stop()

    assert response["status"] == "completed"
    assert response["candidate_source"] == {"kind": "model_final"}
    assert "finalize_telegram_presentation" not in agents[0].prompts[0]
    endpoint = json.loads(
        (tmp_path / "state/persistent-runtime/presenter-endpoint.json").read_text()
    ) if (tmp_path / "state/persistent-runtime/presenter-endpoint.json").exists() else {}
    assert "turn_token" not in endpoint


@pytest.mark.asyncio
async def test_socket_invalid_protocol_response_and_log_have_terminal_contract(tmp_path: Path) -> None:
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "w.sock",
        agent_id="hello_world",
        agent_factory=FakeAgent,
    )
    await worker.start()
    serve = asyncio.create_task(worker.serve_forever())
    try:
        reader, writer = await asyncio.open_unix_connection(str(worker.socket_path))
        writer.write(json.dumps({
            "protocol_version": "wrong",
            "request_id": "req_bad_protocol",
            "operation": "hello",
        }).encode() + b"\n")
        await writer.drain()
        response = json.loads(await reader.readline())
        writer.close()
        await writer.wait_closed()
    finally:
        serve.cancel()
        await asyncio.gather(serve, return_exceptions=True)
        await worker.stop()

    assert response["error"]["code"] == "PROTOCOL_ERROR"
    assert_complete_timing(response)
    assert_terminal_log(tmp_path, response, "", "protocol_failed")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "expected_request_id", "expected_operation"),
    [
        (lambda value: value.update({"unexpected": True}), "req_strict_v2", "health"),
        (lambda value: value.update({"conversation_id": "not-a-conversation-id"}), "req_strict_v2", "health"),
        (lambda value: value.update({"deadline": "not-a-date"}), "req_strict_v2", "health"),
        (lambda value: value.update({"deadline": "2099-01-01 00:00:00+00:00"}), "req_strict_v2", "health"),
        (lambda value: value.update({"deadline": "2099-01-01T00:00:00"}), "req_strict_v2", "health"),
        (lambda value: value.update({"request_id": "malformed"}), "req_error", "health"),
        (lambda value: value.update({"operation": "unsupported"}), "req_strict_v2", "health"),
        (
            lambda value: value.update({
                "operation": "correct_render",
                "correction_attempt_id": "malformed",
                "validation_errors": [],
                "target_constraints": {},
            }) or value.pop("envelope"),
            "req_strict_v2",
            "health",
        ),
    ],
)
async def test_socket_rejects_closed_or_malformed_v2_requests_with_stable_error(
    tmp_path: Path,
    mutate,
    expected_request_id: str,
    expected_operation: str,
) -> None:
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "w.sock",
        agent_id="hello_world",
        agent_factory=FakeAgent,
        correction_companion_resolver=lambda _profile_root: None,
    )
    await worker.start()
    serve = asyncio.create_task(worker.serve_forever())
    try:
        reader, writer = await asyncio.open_unix_connection(str(worker.socket_path))
        hello = {
            "protocol_version": PROTOCOL_V2,
            "request_id": "req_hello_strict",
            "operation": "hello",
            "client": {"name": "telegram-bridge", "revision": "test"},
        }
        writer.write(json.dumps(hello).encode() + b"\n")
        await writer.drain()
        assert json.loads(await reader.readline())["status"] == "ready"
        invalid = v2_request("strict_v2")
        mutate(invalid)
        writer.write(json.dumps(invalid).encode() + b"\n")
        await writer.drain()
        response = json.loads(await reader.readline())
        writer.close()
        await writer.wait_closed()
    finally:
        serve.cancel()
        await asyncio.gather(serve, return_exceptions=True)
        await worker.stop()

    assert response["protocol_version"] == PROTOCOL_V2
    assert response["request_id"] == expected_request_id
    assert response["operation"] == expected_operation
    assert response["error"] == {
        "code": "PROTOCOL_ERROR",
        "message": "invalid protocol request",
        "retryable": False,
    }
    response_schema = json.loads(
        (Path(__file__).parents[1] / "fixtures/telegram.bridge.persistent_service_response.v2.schema.json")
        .read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(response_schema).validate(response)
    assert worker._agent is None


@pytest.mark.asyncio
async def test_main_shutdown_waits_for_active_socket_turn_before_retiring_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    agents: list[ClosingBlockingAgent] = []
    workers: list[PersistentSpecialistWorker] = []
    profile_root = tmp_path / "profile"
    profile_root.mkdir()
    socket_path = profile_root / "w.sock"

    class TestWorker(PersistentSpecialistWorker):
        def __init__(self, **kwargs) -> None:
            super().__init__(
                **kwargs,
                agent_factory=lambda: agents.append(ClosingBlockingAgent(entered, release)) or agents[-1],
            )
            workers.append(self)

    monkeypatch.setattr(persistent_worker, "PersistentSpecialistWorker", TestWorker)
    main_task = asyncio.create_task(persistent_worker._main_async(types.SimpleNamespace(
        profile_root=str(profile_root),
        socket="w.sock",
        agent_id="hello_world",
        turn_timeout=2,
        reset_timeout=2,
        queue_limit=1,
        idle_conversation_seconds=3600,
    )))
    while not socket_path.exists():
        await asyncio.sleep(0.01)
    active_reader, active_writer = await asyncio.open_unix_connection(str(socket_path))
    active_writer.write(json.dumps({
        "protocol_version": PROTOCOL,
        "request_id": "req_hello_active",
        "operation": "hello",
        "client": {"name": "test", "revision": "test"},
    }).encode() + b"\n")
    await active_writer.drain()
    await active_reader.readline()
    active_writer.write(json.dumps(request("active-shutdown")).encode() + b"\n")
    await active_writer.drain()
    assert await asyncio.to_thread(entered.wait, 1)

    shutdown_reader, shutdown_writer = await asyncio.open_unix_connection(str(socket_path))
    shutdown_writer.write(json.dumps({
        "protocol_version": PROTOCOL,
        "request_id": "req_hello_shutdown",
        "operation": "hello",
        "client": {"name": "test", "revision": "test"},
    }).encode() + b"\n")
    await shutdown_writer.drain()
    await shutdown_reader.readline()
    shutdown_writer.write(json.dumps({
        "protocol_version": PROTOCOL,
        "request_id": "req_shutdown",
        "operation": "shutdown",
    }).encode() + b"\n")
    await shutdown_writer.drain()
    try:
        shutdown = json.loads(await shutdown_reader.readline())
        assert shutdown["status"] == "stopping"
        assert workers[0]._stopping is True
        rejected = await workers[0].handle_request(request("after-shutdown-socket"))
        assert rejected["error"]["code"] == "WORKER_STOPPING"
        await asyncio.sleep(0.05)
        assert agents[0].close_called.is_set() is False

        release.set()
        assert (json.loads(await active_reader.readline()))["status"] == "completed"
        await asyncio.wait_for(main_task, timeout=2)
        assert agents[0].close_before_release is False
        assert agents[0].close_called.is_set() is True
    finally:
        release.set()
        if not main_task.done():
            await asyncio.wait_for(main_task, timeout=2)
        active_writer.close()
        shutdown_writer.close()
        await active_writer.wait_closed()
        await shutdown_writer.wait_closed()


@pytest.mark.asyncio
async def test_main_shutdown_waits_for_timed_out_model_thread_before_retiring_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    agents: list[ClosingBlockingAgent] = []
    workers: list[PersistentSpecialistWorker] = []
    profile_root = tmp_path / "profile"
    profile_root.mkdir()
    socket_path = profile_root / "w.sock"

    class TestWorker(PersistentSpecialistWorker):
        def __init__(self, **kwargs) -> None:
            super().__init__(
                **kwargs,
                agent_factory=lambda: agents.append(ClosingBlockingAgent(entered, release)) or agents[-1],
            )
            workers.append(self)

    monkeypatch.setattr(persistent_worker, "PersistentSpecialistWorker", TestWorker)
    main_task = asyncio.create_task(persistent_worker._main_async(types.SimpleNamespace(
        profile_root=str(profile_root),
        socket="w.sock",
        agent_id="hello_world",
        turn_timeout=0.01,
        reset_timeout=2,
        queue_limit=1,
        idle_conversation_seconds=3600,
    )))
    while not socket_path.exists():
        await asyncio.sleep(0.01)
    active_reader, active_writer = await asyncio.open_unix_connection(str(socket_path))
    active_writer.write(json.dumps({
        "protocol_version": PROTOCOL,
        "request_id": "req_hello_timeout",
        "operation": "hello",
        "client": {"name": "test", "revision": "test"},
    }).encode() + b"\n")
    await active_writer.drain()
    await active_reader.readline()
    active_writer.write(json.dumps(request("timeout-shutdown")).encode() + b"\n")
    await active_writer.drain()
    assert await asyncio.to_thread(entered.wait, 1)
    timed_out = json.loads(await active_reader.readline())
    assert timed_out["error"]["code"] == "TURN_TIMEOUT"

    shutdown_reader, shutdown_writer = await asyncio.open_unix_connection(str(socket_path))
    shutdown_writer.write(json.dumps({
        "protocol_version": PROTOCOL,
        "request_id": "req_hello_shutdown",
        "operation": "hello",
        "client": {"name": "test", "revision": "test"},
    }).encode() + b"\n")
    await shutdown_writer.drain()
    await shutdown_reader.readline()
    shutdown_writer.write(json.dumps({
        "protocol_version": PROTOCOL,
        "request_id": "req_shutdown",
        "operation": "shutdown",
    }).encode() + b"\n")
    await shutdown_writer.drain()
    try:
        assert json.loads(await shutdown_reader.readline())["status"] == "stopping"
        rejected = await workers[0].handle_request(request("after-timeout-shutdown"))
        assert rejected["error"]["code"] == "WORKER_STOPPING"
        await asyncio.sleep(0.05)
        assert agents[0].close_called.is_set() is False

        release.set()
        await asyncio.wait_for(main_task, timeout=2)
        assert agents[0].close_before_release is False
        assert agents[0].close_called.is_set() is True
    finally:
        release.set()
        if not main_task.done():
            await asyncio.wait_for(main_task, timeout=2)
        active_writer.close()
        shutdown_writer.close()
        await active_writer.wait_closed()
        await shutdown_writer.wait_closed()


@pytest.mark.asyncio
async def test_stop_quiescence_timeout_keeps_profile_locked_without_closing_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    agents: list[ClosingBlockingAgent] = []
    monkeypatch.setattr(persistent_worker, "MODEL_QUIESCE_TIMEOUT_SECONDS", 0.01)
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "w.sock",
        agent_id="hello_world",
        turn_timeout=0.01,
        agent_factory=lambda: agents.append(ClosingBlockingAgent(entered, release)) or agents[-1],
    )
    await worker.start()
    try:
        timed_out = await worker.handle_request(request("bounded-stop"))
        assert timed_out["error"]["code"] == "TURN_TIMEOUT"
        await worker.stop()
        assert agents[0].close_called.is_set() is False
        assert worker.health()["quiesce_failed"] is True
        assert worker._instance_lock is not None

        release.set()
        await worker._await_active_model_task()
        await worker.stop()
        assert agents[0].close_called.is_set() is True
        assert worker._instance_lock is None
    finally:
        release.set()
        if worker._instance_lock is not None:
            await worker.stop()


@pytest.mark.asyncio
async def test_reset_waits_for_active_turn_and_blocks_new_admission(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingAgent(FakeAgent):
        def run_conversation(self, prompt: str) -> dict:
            entered.set()
            release.wait(timeout=2)
            return super().run_conversation(prompt)

    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        turn_timeout=3,
        reset_timeout=4,
        agent_factory=BlockingAgent,
    )
    await worker.start()
    try:
        active = asyncio.create_task(worker.handle_request(request("active")))
        assert await asyncio.to_thread(entered.wait, 1)
        queued = asyncio.create_task(worker.handle_request(request("queued")))
        await asyncio.sleep(0.05)
        assert worker.health()["queued_turns"] == 1
        reset_task = asyncio.create_task(worker.handle_request(request("reset-active", "reset")))
        await asyncio.sleep(0.05)
        rejected = await worker.handle_request(request("too-soon"))
        assert rejected["error"]["code"] == "RESET_IN_PROGRESS"
        assert reset_task.done() is False
        release.set()
        assert (await active)["status"] == "completed"
        canceled = await queued
        assert canceled["error"]["code"] == "RESET"
        assert canceled["execution_state"] == "not_started"
        assert (await reset_task)["status"] == "completed"
    finally:
        release.set()
        await worker.stop()
