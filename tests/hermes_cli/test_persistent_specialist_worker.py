import asyncio
import json
import os
import stat
import sys
import threading
import types
from pathlib import Path

import pytest

import hermes_cli.persistent_specialist_worker as persistent_worker
from hermes_cli.persistent_specialist_worker import (
    PROTOCOL,
    PROTOCOL_V2,
    PersistentSpecialistWorker,
    _initialize_worker_process_context,
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
    assert response["status"] == "completed"
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
async def test_v2_correction_is_same_conversation_tool_free_once_and_durable(tmp_path: Path) -> None:
    agent = CorrectionAgent()
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=lambda: agent,
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
    assert agent.turns == 2
    assert agent.tool_snapshots[0]
    assert agent.tool_snapshots[1] == []
    assert agent.denied_execution is True
    assert agent.executed_tools == 0

    ledger = json.loads(
        worker._ledger_path("conv_" + "1" * 64, "completed-malformed").read_text(encoding="utf-8")
    )
    assert ledger["correction_attempt_id"] == "corr_attempt_1"
    assert ledger["correction_attempt_count"] == 1
    assert ledger["corrected_candidate"] == correction["render_candidate"]["content"]
    assert ledger["correction_response"] == correction


@pytest.mark.asyncio
async def test_v2_correction_rejects_runtime_without_tool_free_enforcement(tmp_path: Path) -> None:
    agent = CorrectionAgent(api_mode="codex_app_server")
    worker = PersistentSpecialistWorker(
        profile_root=tmp_path,
        socket_path=tmp_path / "state/runtime/worker.sock",
        agent_id="hello_world",
        agent_factory=lambda: agent,
    )
    await worker.start()
    try:
        await worker.handle_request(v2_request("completed-malformed"))
        response = await worker.handle_request(correction_request("completed-malformed"))
        replay = await worker.handle_request(correction_request("completed-malformed"))
    finally:
        await worker.stop()

    assert response == replay
    assert response["status"] == "failed"
    assert response["execution_state"] == "completed"
    assert response["presentation_state"] == "correction_failed"
    assert response["error"]["code"] == "TOOL_FREE_CORRECTION_UNAVAILABLE"
    assert agent.turns == 1


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
