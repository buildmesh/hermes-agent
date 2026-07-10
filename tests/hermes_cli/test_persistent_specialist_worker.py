import asyncio
import json
import stat
import threading
from pathlib import Path

import pytest

from hermes_cli.persistent_specialist_worker import PROTOCOL, PersistentSpecialistWorker


class FakeAgent:
    def __init__(self) -> None:
        self.turns = 0
        self.closed = False
        self._codex_session = None

    def run_conversation(self, prompt: str) -> dict:
        self.turns += 1
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
        reset_task = asyncio.create_task(worker.handle_request(request("reset-active", "reset")))
        await asyncio.sleep(0.05)
        rejected = await worker.handle_request(request("too-soon"))
        assert rejected["error"]["code"] == "QUEUE_FULL"
        assert reset_task.done() is False
        release.set()
        assert (await active)["status"] == "completed"
        assert (await reset_task)["status"] == "completed"
    finally:
        release.set()
        await worker.stop()
