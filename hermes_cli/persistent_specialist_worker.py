"""Profile-scoped persistent runtime for Telegram Bridge specialists."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import re
import signal
import stat
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable


PROTOCOL = "telegram.bridge.persistent_service.v1"
MAX_LINE_BYTES = 1024 * 1024
LEDGER_RETENTION = timedelta(days=30)
TERMINAL_STATES = {"completed", "failed", "outcome_unknown"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _fingerprint(request: dict[str, Any]) -> str:
    canonical = {
        "protocol_version": request["protocol_version"],
        "operation": request["operation"],
        "conversation_id": request["conversation_id"],
        "envelope": request["envelope"],
    }
    raw = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ledger_key(conversation_id: str, event_id: str) -> str:
    return hashlib.sha256(f"{conversation_id}\0{event_id}".encode("utf-8")).hexdigest()


def _extract_render_payloads(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    if not stripped.startswith("["):
        start, end = stripped.find("["), stripped.rfind("]")
        if start < 0 or end <= start:
            raise ValueError("specialist response did not contain a JSON array")
        stripped = stripped[start : end + 1]
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        repaired = _repair_missing_container_close(stripped, exc.pos)
        if repaired is None:
            raise
        value = json.loads(repaired)
    if not isinstance(value, list) or not value or not all(isinstance(item, dict) for item in value):
        raise ValueError("specialist response must be a non-empty JSON object array")
    return value


def _repair_missing_container_close(text: str, error_pos: int) -> str | None:
    """Repair one provably missing array/object close at the decoder boundary."""
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"[": "]", "{": "}"}
    for char in text[:error_pos]:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(char)
        elif char in "]}":
            if not stack or pairs[stack[-1]] != char:
                return None
            stack.pop()

    if in_string or not stack or error_pos >= len(text):
        return None
    expected = pairs[stack[-1]]
    encountered = text[error_pos]
    if encountered not in "]}" or encountered == expected:
        return None
    return text[:error_pos] + expected + text[error_pos:]


def _bridge_prompt(envelope: dict[str, Any]) -> str:
    return (
        "Handle the following Telegram Bridge specialist envelope using this profile's skills and "
        "instructions. Return only a JSON array of telegram.bridge.render_payload.v1 objects. "
        "Do not call Telegram directly. Preserve event_id as correlation_id and use the envelope "
        "chat_id as the send target.\n\nEnvelope JSON:\n"
        + json.dumps(envelope, ensure_ascii=False, sort_keys=True)
    )


class PersistentSpecialistWorker:
    def __init__(
        self,
        *,
        profile_root: Path,
        socket_path: Path,
        agent_id: str,
        turn_timeout: float = 90,
        reset_timeout: float = 105,
        queue_limit: int = 4,
        idle_conversation_seconds: float = 21600,
        agent_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.profile_root = profile_root.resolve()
        self.socket_path = socket_path.resolve()
        self.agent_id = agent_id
        self.turn_timeout = float(turn_timeout)
        self.reset_timeout = float(reset_timeout)
        self.queue_limit = max(1, int(queue_limit))
        self.idle_conversation_seconds = max(60.0, float(idle_conversation_seconds))
        self.agent_factory = agent_factory or self._default_agent_factory
        self.runtime_instance_id = f"runtime_{uuid.uuid4().hex}"
        self.conversation_instance_id = f"thread_{uuid.uuid4().hex}"
        self.started_at = _utcnow()
        self.last_success_at: str | None = None
        self.last_error_code: str | None = None
        self._agent: Any | None = None
        self._conversation_id: str | None = None
        self._conversation_lock = asyncio.Lock()
        self._resetting = False
        self._queued_turns = 0
        self._reset_generation = 0
        self._last_conversation_activity = time.monotonic()
        self._server: asyncio.AbstractServer | None = None
        self._instance_lock: Any | None = None
        self._stopping = False
        self._unhealthy = False
        self._ledger_dir = self.profile_root / "state/persistent-runtime/ledger"
        self._log_path = self.profile_root / "logs/persistent-specialist.jsonl"

    def _log_event(self, event: str, **fields: Any) -> None:
        value = {
            "timestamp": _iso(_utcnow()),
            "event": event,
            "agent_id": self.agent_id,
            "runtime_instance_id": self.runtime_instance_id,
            **{key: item for key, item in fields.items() if item is not None},
        }
        self._log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self._log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    def _default_agent_factory(self) -> Any:
        os.environ.setdefault("HERMES_YOLO_MODE", "1")
        os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")
        from hermes_cli.oneshot import create_noninteractive_agent

        return create_noninteractive_agent()

    def _run_in_profile_context(self, callback: Callable[..., Any], *args: Any) -> Any:
        """Run agent work with an isolated specialist runtime context."""
        previous_cwd = Path.cwd()
        previous_env = {key: os.environ.get(key) for key in ("HERMES_HOME", "TERMINAL_CWD")}
        try:
            os.chdir(self.profile_root)
            os.environ["HERMES_HOME"] = str(self.profile_root)
            os.environ["TERMINAL_CWD"] = str(self.profile_root)
            from agent.runtime_cwd import clear_session_cwd

            clear_session_cwd()
            return callback(*args)
        finally:
            os.chdir(previous_cwd)
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _construct_agent(self) -> Any:
        agent = self._run_in_profile_context(self.agent_factory)
        agent.session_cwd = str(self.profile_root)
        return agent

    def _ledger_path(self, conversation_id: str, event_id: str) -> Path:
        return self._ledger_dir / f"{_ledger_key(conversation_id, event_id)}.json"

    def _recover_and_compact_ledger(self) -> None:
        now = _utcnow()
        self._ledger_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        for path in self._ledger_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                accepted = _parse_time(record["accepted_at"])
            except Exception:
                continue
            if record.get("state") == "in_flight":
                record["state"] = "outcome_unknown"
                record["updated_at"] = _iso(now)
                record["error"] = {
                    "code": "WORKER_RESTARTED_IN_FLIGHT",
                    "message": "The prior worker stopped while this turn was running.",
                    "retryable": False,
                }
                _atomic_json(path, record)
            elif record.get("state") == "accepted":
                record["state"] = "failed"
                record["updated_at"] = _iso(now)
                record["response"] = {
                    "protocol_version": PROTOCOL,
                    "request_id": record["request_id"],
                    "operation": record["operation"],
                    "status": "failed",
                    "execution_state": "not_started",
                    "event_id": record["event_id"],
                    "runtime_instance_id": self.runtime_instance_id,
                    "error": {
                        "code": "WORKER_RESTARTED_BEFORE_START",
                        "message": "The worker restarted before this queued turn began.",
                        "retryable": True,
                    },
                }
                _atomic_json(path, record)
            if now >= accepted + LEDGER_RETENTION and record.get("state") in TERMINAL_STATES:
                path.unlink(missing_ok=True)

    async def start(self) -> None:
        try:
            self.socket_path.relative_to(self.profile_root)
        except ValueError as exc:
            raise ValueError("socket path must resolve inside the specialist profile") from exc
        self.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.socket_path.parent, 0o700)
        lock_path = self.profile_root / "state/persistent-runtime/worker.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        stream = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise RuntimeError("another persistent specialist worker owns this profile") from exc
        self._instance_lock = stream
        self._recover_and_compact_ledger()
        if self.socket_path.exists():
            if stat.S_ISSOCK(self.socket_path.lstat().st_mode):
                self.socket_path.unlink()
            else:
                raise RuntimeError("refusing to replace a non-socket runtime path")
        self._server = await asyncio.start_unix_server(self._handle_connection, path=str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        self._log_event("worker_ready", socket=str(self.socket_path))

    async def stop(self) -> None:
        self._stopping = True
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self.socket_path.unlink(missing_ok=True)
        await self._retire_agent()
        if self._instance_lock is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._instance_lock.fileno(), fcntl.LOCK_UN)
            self._instance_lock.close()
            self._instance_lock = None
        self._log_event("worker_stopped")

    async def serve_forever(self) -> None:
        if self._server is None:
            raise RuntimeError("worker is not started")
        async with self._server:
            await self._server.serve_forever()

    async def _retire_agent(self) -> None:
        agent, self._agent = self._agent, None
        if agent is not None:
            session = getattr(agent, "_codex_session", None)
            if session is not None:
                await asyncio.to_thread(session.close)
                agent._codex_session = None
            await asyncio.to_thread(agent.close)

    def health(self) -> dict[str, Any]:
        return {
            "ready": not self._stopping and not self._unhealthy,
            "unhealthy": self._unhealthy,
            "uptime_seconds": max(0, int((_utcnow() - self.started_at).total_seconds())),
            "resident_conversations": 1 if self._conversation_id else 0,
            "active_turns": 1 if self._conversation_lock.locked() else 0,
            "queued_turns": self._queued_turns,
            "last_success_at": self.last_success_at,
            "last_error_code": self.last_error_code,
        }

    async def _read(self, reader: asyncio.StreamReader) -> dict[str, Any]:
        raw = await reader.readline()
        if not raw or len(raw) > MAX_LINE_BYTES:
            raise ValueError("invalid or oversized protocol line")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("protocol message must be an object")
        return value

    async def _write(self, writer: asyncio.StreamWriter, value: dict[str, Any]) -> None:
        writer.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
        await writer.drain()

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            hello = await self._read(reader)
            if hello.get("protocol_version") != PROTOCOL or hello.get("operation") != "hello":
                raise ValueError("hello negotiation required")
            await self._write(writer, {
                "protocol_version": PROTOCOL,
                "request_id": hello["request_id"],
                "operation": "hello",
                "status": "ready" if not self._unhealthy else "failed",
                "runtime_instance_id": self.runtime_instance_id,
                "framework": "hermes",
                "profile": self.profile_root.name,
                "supported_protocol_versions": [PROTOCOL],
                "capabilities": ["turn", "reset", "health", "shutdown", "durable_event_ledger"],
            })
            request = await self._read(reader)
            response = await self.handle_request(request)
            await self._write(writer, response)
        except (asyncio.IncompleteReadError, BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            with contextlib.suppress(Exception):
                await self._write(writer, {
                    "protocol_version": PROTOCOL,
                    "request_id": "req_error",
                    "operation": "health",
                    "status": "failed",
                    "execution_state": "not_started",
                    "error": {"code": "PROTOCOL_ERROR", "message": str(exc)[:500], "retryable": False},
                })
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    def _base_response(self, request: dict[str, Any], **extra: Any) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL,
            "request_id": request["request_id"],
            "operation": request["operation"],
            "request_id": request["request_id"],
            "runtime_instance_id": self.runtime_instance_id,
            "trace_id": request.get("_trace_id"),
            **extra,
        }

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request = {**request, "_trace_id": f"trace_{uuid.uuid4().hex}"}
        if request.get("protocol_version") != PROTOCOL:
            raise ValueError("unsupported protocol_version")
        operation = request.get("operation")
        if operation == "health":
            return self._base_response(request, status="ready", health=self.health())
        if operation == "shutdown":
            self._stopping = True
            return self._base_response(request, status="stopping")
        if operation not in {"turn", "reset"}:
            raise ValueError("unsupported operation")
        return await self._handle_turn_or_reset(request)

    async def _handle_turn_or_reset(self, request: dict[str, Any]) -> dict[str, Any]:
        conversation_id = request["conversation_id"]
        event_id = request["event_id"]
        fingerprint = _fingerprint(request)
        path = self._ledger_path(conversation_id, event_id)
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["fingerprint"] != fingerprint:
                return self._failure(request, "IDEMPOTENCY_CONFLICT", "event_id was reused with different content", False, "not_started")
            if record.get("response"):
                return record["response"]
            state = "outcome_unknown" if record.get("state") == "outcome_unknown" else "in_flight"
            return self._failure(request, "DUPLICATE_IN_FLIGHT", "the event has already been accepted", False, state)
        if self._unhealthy:
            return self._failure(request, "CONVERSATION_UNHEALTHY", "the specialist conversation requires recovery", False, "not_started")
        if (
            self._conversation_id not in {None, conversation_id}
            and not self._conversation_lock.locked()
            and (time.monotonic() - self._last_conversation_activity) >= self.idle_conversation_seconds
        ):
            await self._retire_agent()
            self._conversation_id = None
            self.conversation_instance_id = f"thread_{uuid.uuid4().hex}"
            self._log_event("conversation_evicted", reason="idle")
        if self._conversation_id not in {None, conversation_id}:
            return self._failure(request, "CONVERSATION_LIMIT", "this v1 worker already owns another conversation", True, "not_started")
        queued = False
        generation = self._reset_generation
        if request["operation"] == "turn":
            if self._resetting:
                return self._failure(request, "RESET_IN_PROGRESS", "the specialist conversation is resetting", True, "not_started")
            if self._conversation_lock.locked():
                if self._queued_turns >= self.queue_limit:
                    return self._failure(request, "QUEUE_FULL", "the specialist conversation queue is full", True, "not_started")
                self._queued_turns += 1
                queued = True
        else:
            self._resetting = True
            self._reset_generation += 1

        now = _utcnow()
        record = {
            "schema_version": "telegram.bridge.persistent_event_record.v1",
            "conversation_id": conversation_id,
            "event_id": event_id,
            "fingerprint": fingerprint,
            "operation": request["operation"],
            "state": "accepted" if queued or request["operation"] == "reset" else "in_flight",
            "accepted_at": _iso(now),
            "updated_at": _iso(now),
        }
        _atomic_json(path, record)
        started = time.monotonic()
        wait_timeout = self.reset_timeout if request["operation"] == "reset" else max(
            0.1, (_parse_time(request["deadline"]) - _utcnow()).total_seconds()
        )
        try:
            await asyncio.wait_for(self._conversation_lock.acquire(), timeout=wait_timeout)
        except asyncio.TimeoutError:
            if queued:
                self._queued_turns -= 1
            code = "RESET_UNCONFIRMED" if request["operation"] == "reset" else "DEADLINE_EXPIRED"
            if request["operation"] == "reset":
                self._unhealthy = True
                self._resetting = False
            self.last_error_code = code
            response = self._failure(request, code, "the specialist request did not start before its deadline", False, "not_started")
            record["state"] = "failed"
            record["response"] = response
            record["updated_at"] = _iso(_utcnow())
            _atomic_json(path, record)
            self._log_event("turn_failed", event_id=event_id, code=code, execution_state="not_started")
            return response
        try:
            queue_ms = int((time.monotonic() - started) * 1000)
            if queued:
                self._queued_turns -= 1
                if generation != self._reset_generation:
                    response = self._failure(request, "RESET", "the queued turn was canceled by reset", False, "not_started")
                    record["state"] = "failed"
                    record["response"] = response
                    record["updated_at"] = _iso(_utcnow())
                    _atomic_json(path, record)
                    return response
            self._conversation_id = conversation_id
            record["state"] = "in_flight"
            record["updated_at"] = _iso(_utcnow())
            _atomic_json(path, record)
            self._log_event(
                "turn_started",
                event_id=event_id,
                operation=request["operation"],
                trace_id=request["_trace_id"],
            )
            try:
                if request["operation"] == "reset":
                    remaining = max(0.1, self.reset_timeout - (time.monotonic() - started))
                    await asyncio.wait_for(self._retire_agent(), timeout=remaining)
                    self.conversation_instance_id = f"thread_{uuid.uuid4().hex}"
                    response = self._reset_response(request, started)
                else:
                    initialization_started = time.monotonic()
                    if self._agent is None:
                        self._agent = await asyncio.to_thread(self._construct_agent)
                    initialization_ms = int((time.monotonic() - initialization_started) * 1000)
                    prompt = _bridge_prompt(request["envelope"])
                    model_started = time.monotonic()
                    result = await asyncio.wait_for(
                        asyncio.to_thread(self._run_in_profile_context, self._agent.run_conversation, prompt),
                        timeout=self.turn_timeout,
                    )
                    model_and_tools_ms = int((time.monotonic() - model_started) * 1000)
                    if not result.get("completed", True) or result.get("partial"):
                        raise RuntimeError(result.get("error") or "model turn did not complete")
                    render_started = time.monotonic()
                    payloads = _extract_render_payloads(result.get("final_response") or "")
                    render_preparation_ms = int((time.monotonic() - render_started) * 1000)
                    total_ms = int((time.monotonic() - started) * 1000)
                    response = self._base_response(
                        request,
                        status="completed",
                        execution_state="completed",
                        event_id=event_id,
                        conversation_instance_id=self.conversation_instance_id,
                        render_payloads=payloads,
                        timing_ms={
                            "queue": queue_ms,
                            "initialization": initialization_ms,
                            "model_and_tools": model_and_tools_ms,
                            "render_preparation": render_preparation_ms,
                            "total": total_ms,
                        },
                    )
                self.last_success_at = _iso(_utcnow())
                self._last_conversation_activity = time.monotonic()
                record["state"] = "completed"
                record["response"] = response
                self._log_event(
                    "turn_completed",
                    event_id=event_id,
                    operation=request["operation"],
                    trace_id=request["_trace_id"],
                    timing_ms=response.get("timing_ms"),
                )
            except asyncio.TimeoutError:
                self._unhealthy = True
                self.last_error_code = "TURN_TIMEOUT" if request["operation"] == "turn" else "RESET_UNCONFIRMED"
                self._interrupt_agent()
                response = self._failure(request, self.last_error_code, "the specialist did not stop before its deadline", False, "outcome_unknown")
                record["state"] = "outcome_unknown"
                record["response"] = response
                self._log_event("turn_failed", event_id=event_id, code=self.last_error_code, execution_state="outcome_unknown")
            except Exception as exc:
                self.last_error_code = "TURN_FAILED"
                response = self._failure(request, "TURN_FAILED", str(exc)[:500], False, "outcome_unknown")
                record["state"] = "outcome_unknown"
                record["response"] = response
                self._log_event("turn_failed", event_id=event_id, code="TURN_FAILED", execution_state="outcome_unknown")
            record["updated_at"] = _iso(_utcnow())
            _atomic_json(path, record)
            return response
        finally:
            self._conversation_lock.release()
            if request["operation"] == "reset":
                self._resetting = False

    def _interrupt_agent(self) -> None:
        session = getattr(self._agent, "_codex_session", None)
        if session is not None:
            with contextlib.suppress(Exception):
                session.request_interrupt()

    def _failure(self, request: dict[str, Any], code: str, message: str, retryable: bool, state: str) -> dict[str, Any]:
        return self._base_response(
            request,
            status="failed",
            execution_state=state,
            event_id=request.get("event_id"),
            error={"code": code, "message": message[:500], "retryable": retryable},
        )

    def _reset_response(self, request: dict[str, Any], started: float) -> dict[str, Any]:
        envelope = request["envelope"]
        payload = {
            "schema_version": "telegram.bridge.render_payload.v1",
            "message_id": f"msg_{request['event_id']}_reset",
            "correlation_id": request["event_id"],
            "action": "send",
            "target": {"chat_id": int(envelope["chat_id"])},
            "render": {"text": f"{self.agent_id} conversation context reset."},
        }
        elapsed = int((time.monotonic() - started) * 1000)
        return self._base_response(
            request,
            status="completed",
            execution_state="completed",
            event_id=request["event_id"],
            conversation_instance_id=self.conversation_instance_id,
            render_payloads=[payload],
            timing_ms={"queue": 0, "model_and_tools": 0, "render_preparation": elapsed, "total": elapsed},
        )


async def _main_async(args: argparse.Namespace) -> int:
    profile_root = Path(args.profile_root).resolve()
    os.chdir(profile_root)
    os.environ["HERMES_HOME"] = str(profile_root)
    os.environ["TERMINAL_CWD"] = str(profile_root)
    socket_path = (profile_root / args.socket).resolve()
    worker = PersistentSpecialistWorker(
        profile_root=profile_root,
        socket_path=socket_path,
        agent_id=args.agent_id,
        turn_timeout=args.turn_timeout,
        reset_timeout=args.reset_timeout,
        queue_limit=args.queue_limit,
        idle_conversation_seconds=args.idle_conversation_seconds,
    )
    await worker.start()
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signum, stop.set)
    serve = asyncio.create_task(worker.serve_forever())
    wait = asyncio.create_task(stop.wait())
    done, _ = await asyncio.wait({serve, wait}, return_when=asyncio.FIRST_COMPLETED)
    if serve in done and not serve.cancelled():
        exception = serve.exception()
        if exception and not isinstance(exception, asyncio.CancelledError):
            raise exception
    serve.cancel()
    wait.cancel()
    await asyncio.gather(serve, wait, return_exceptions=True)
    await worker.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--turn-timeout", type=float, default=90)
    parser.add_argument("--reset-timeout", type=float, default=105)
    parser.add_argument("--queue-limit", type=int, default=4)
    parser.add_argument("--idle-conversation-seconds", type=float, default=21600)
    args = parser.parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
