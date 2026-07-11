import asyncio
import json
import os
import stat
import threading
from pathlib import Path

import pytest

from hermes_cli.persistent_specialist_worker import (
    PROTOCOL,
    PersistentSpecialistWorker,
    _extract_render_payloads,
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
        context_root = resolve_context_cwd()
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
        assert "turn_completed" in (tmp_path / "logs/persistent-specialist.jsonl").read_text()

        old_thread = second["conversation_instance_id"]
        codex_session = FakeCodexSession()
        agents[0]._codex_session = codex_session
        reset = await worker.handle_request(request("reset1", "reset"))
        assert reset["status"] == "completed"
        assert reset["conversation_instance_id"] != old_thread
        assert agents[0].closed is True
        assert codex_session.closed is True

        third = await worker.handle_request(request("event3"))
        assert third["render_payloads"][0]["render"]["text"] == "turn 1"
        assert len(agents) == 2
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_worker_constructs_and_runs_agent_in_profile_context_with_phase_timing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_root = tmp_path / "profiles" / "hello-world"
    profile_root.mkdir(parents=True)
    for relative_path in ("SOUL.md", "memories/USER.md", "memories/MEMORY.md", "AGENTS.md", "skills/index.md"):
        path = profile_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative_path, encoding="utf-8")
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

    assert constructed_in == [(str(profile_root), str(profile_root), str(profile_root))]
    assert (os.getcwd(), os.environ["HERMES_HOME"], os.environ["TERMINAL_CWD"]) == (
        str(bridge_cwd), str(bridge_cwd), str(bridge_cwd)
    )
    assert agents[0].session_cwd == str(profile_root)
    assert agents[0].effective_cwds == [(str(profile_root), str(profile_root), str(profile_root))]
    assert agents[0].loaded_profile_context == [
        ("SOUL.md", "memories/USER.md", "memories/MEMORY.md", "AGENTS.md", "skills/index.md")
    ]
    assert response["trace_id"].startswith("trace_")
    assert set(response["timing_ms"]) == {
        "queue",
        "initialization",
        "model_and_tools",
        "render_preparation",
        "total",
    }
    assert all(duration >= 0 for duration in response["timing_ms"].values())
    assert sum(
        response["timing_ms"][phase]
        for phase in ("queue", "initialization", "model_and_tools", "render_preparation")
    ) <= response["timing_ms"]["total"]


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
    recovered = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=FakeAgent,
    )
    await recovered.start()
    try:
        replay = await recovered.handle_request(request("socket-event"))
        assert replay["execution_state"] == "outcome_unknown"
        assert replay["error"]["code"] == "DUPLICATE_IN_FLIGHT"
    finally:
        await recovered.stop()


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
