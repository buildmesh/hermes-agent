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

from agent.render_correction_companion import (
    CorrectionCompanionTimeout,
    CorrectionCompanionUnavailable,
    resolve_companion_config,
    run_companion_subprocess,
)


PROTOCOL = "telegram.bridge.persistent_service.v1"
PROTOCOL_V2 = "telegram.bridge.persistent_service.v2"
MAX_LINE_BYTES = 1024 * 1024
MAX_CANDIDATE_BYTES = 512 * 1024
LEDGER_RETENTION = timedelta(days=30)
TERMINAL_STATES = {"completed", "failed", "outcome_unknown"}
TIMING_PHASES = ("queue", "initialization", "retirement", "model_and_tools", "render_preparation")
MODEL_QUIESCE_TIMEOUT_SECONDS = 5.0
_UNQUIESCED_WORKERS: set[Any] = set()


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


def _candidate_hash(candidate: str) -> str:
    return hashlib.sha256(candidate.encode("utf-8")).hexdigest()


def _bounded_candidate(candidate: str) -> dict[str, Any]:
    raw = candidate.encode("utf-8")
    bounded = raw[:MAX_CANDIDATE_BYTES]
    while bounded:
        try:
            content = bounded.decode("utf-8")
            break
        except UnicodeDecodeError:
            bounded = bounded[:-1]
    else:
        content = ""
    return {
        "content": content,
        "sha256": _candidate_hash(candidate),
        "truncated": len(raw) > len(bounded),
    }


def _correction_fingerprint(request: dict[str, Any]) -> str:
    canonical = {
        key: request[key]
        for key in (
            "protocol_version",
            "operation",
            "event_id",
            "conversation_id",
            "correction_attempt_id",
            "validation_errors",
            "target_constraints",
        )
    }
    raw = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_v2_request(request: dict[str, Any]) -> None:
    """Validate the closed v2 wire shape without importing TBA schemas."""
    if request.get("protocol_version") != PROTOCOL_V2:
        raise ValueError("invalid protocol version")
    operation = request.get("operation")
    shapes = {
        "hello": {"protocol_version", "request_id", "operation", "client"},
        "turn": {"protocol_version", "request_id", "operation", "event_id", "conversation_id", "deadline", "envelope"},
        "reset": {"protocol_version", "request_id", "operation", "event_id", "conversation_id", "deadline", "envelope"},
        "correct_render": {
            "protocol_version", "request_id", "operation", "event_id", "conversation_id", "deadline",
            "correction_attempt_id", "validation_errors", "target_constraints",
        },
        "health": {"protocol_version", "request_id", "operation"},
        "shutdown": {"protocol_version", "request_id", "operation"},
    }
    expected = shapes.get(operation)
    if expected is None or set(request) != expected:
        raise ValueError("invalid closed request shape")
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not re.fullmatch(r"req_[A-Za-z0-9._:-]{1,200}", request_id):
        raise ValueError("invalid request id")
    if operation == "hello":
        client = request.get("client")
        if not isinstance(client, dict) or set(client) != {"name", "revision"}:
            raise ValueError("invalid hello client")
        revision = client.get("revision")
        if client.get("name") != "telegram-bridge" or not isinstance(revision, str) or not 1 <= len(revision) <= 80:
            raise ValueError("invalid hello client")
        return
    if operation in {"health", "shutdown"}:
        return
    event_id = request.get("event_id")
    conversation_id = request.get("conversation_id")
    deadline = request.get("deadline")
    if not isinstance(event_id, str) or not 1 <= len(event_id) <= 240:
        raise ValueError("invalid event id")
    if not isinstance(conversation_id, str) or not re.fullmatch(r"conv_[0-9a-f]{64}", conversation_id):
        raise ValueError("invalid conversation id")
    if not isinstance(deadline, str):
        raise ValueError("invalid deadline")
    try:
        parsed_deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid deadline") from exc
    if parsed_deadline.tzinfo is None:
        raise ValueError("invalid deadline")
    if operation in {"turn", "reset"}:
        if not isinstance(request.get("envelope"), dict):
            raise ValueError("invalid envelope")
        return
    attempt_id = request.get("correction_attempt_id")
    if not isinstance(attempt_id, str) or not re.fullmatch(r"corr_[A-Za-z0-9._:-]{1,200}", attempt_id):
        raise ValueError("invalid correction attempt id")
    errors = request.get("validation_errors")
    if not isinstance(errors, list) or not 1 <= len(errors) <= 16:
        raise ValueError("invalid validation errors")
    for error in errors:
        if not isinstance(error, dict) or set(error) != {"code", "path", "message"}:
            raise ValueError("invalid validation error")
        code, path, message = error.get("code"), error.get("path"), error.get("message")
        if not isinstance(code, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", code):
            raise ValueError("invalid validation error code")
        if not isinstance(path, str) or not 1 <= len(path) <= 240:
            raise ValueError("invalid validation error path")
        if not isinstance(message, str) or not 1 <= len(message) <= 240:
            raise ValueError("invalid validation error message")
    constraints = request.get("target_constraints")
    required = {"correlation_id", "chat_id", "authorized_telegram_message_ids", "callback_namespace"}
    allowed = required | {"callback_query_id"}
    if not isinstance(constraints, dict) or not required.issubset(constraints) or not set(constraints).issubset(allowed):
        raise ValueError("invalid target constraints")
    if not isinstance(constraints.get("correlation_id"), str) or not 1 <= len(constraints["correlation_id"]) <= 240:
        raise ValueError("invalid correlation constraint")
    if type(constraints.get("chat_id")) is not int:
        raise ValueError("invalid chat constraint")
    message_ids = constraints.get("authorized_telegram_message_ids")
    if (
        not isinstance(message_ids, list) or len(message_ids) > 64
        or any(type(value) is not int for value in message_ids) or len(message_ids) != len(set(message_ids))
    ):
        raise ValueError("invalid authorized message constraints")
    namespace = constraints.get("callback_namespace")
    if not isinstance(namespace, str) or not 1 <= len(namespace) <= 64:
        raise ValueError("invalid callback namespace")
    callback_id = constraints.get("callback_query_id")
    if callback_id is not None and (not isinstance(callback_id, str) or not 1 <= len(callback_id) <= 240):
        raise ValueError("invalid callback query constraint")


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


def _timing_ms(started: float, **phases: int) -> dict[str, int]:
    timing = {phase: max(0, int(phases.get(phase, 0))) for phase in TIMING_PHASES}
    timing["total"] = max(0, int((time.monotonic() - started) * 1000))
    return timing


def _initialize_worker_process_context(profile_root: Path) -> None:
    """Bind this dedicated worker process to its installed specialist profile."""
    os.chdir(profile_root)
    os.environ["HERMES_HOME"] = str(profile_root)
    os.environ["TERMINAL_CWD"] = str(profile_root)
    os.environ.setdefault("HERMES_YOLO_MODE", "1")
    os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")
    from agent.runtime_cwd import clear_session_cwd

    clear_session_cwd()


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
        correction_companion_resolver: Callable[[], dict[str, str] | None] | None = None,
        correction_companion_runner: Callable[[dict[str, str], dict[str, Any], float], str] | None = None,
    ) -> None:
        self.profile_root = profile_root.resolve()
        self.socket_path = socket_path.resolve()
        self.agent_id = agent_id
        self.turn_timeout = float(turn_timeout)
        self.reset_timeout = float(reset_timeout)
        self.queue_limit = max(1, int(queue_limit))
        self.idle_conversation_seconds = max(60.0, float(idle_conversation_seconds))
        self.agent_factory = agent_factory or self._default_agent_factory
        resolver = correction_companion_resolver or resolve_companion_config
        try:
            self._correction_companion_config = resolver()
        except Exception:
            self._correction_companion_config = None
        self._correction_companion_runner = correction_companion_runner or run_companion_subprocess
        self.runtime_instance_id = f"runtime_{uuid.uuid4().hex}"
        self.conversation_instance_id = f"thread_{uuid.uuid4().hex}"
        self.started_at = _utcnow()
        self.last_success_at: str | None = None
        self.last_error_code: str | None = None
        self._agent: Any | None = None
        self._active_model_task: asyncio.Task[Any] | None = None
        self._conversation_id: str | None = None
        self._conversation_lock = asyncio.Lock()
        self._correction_state_lock = asyncio.Lock()
        self._correction_events: dict[Path, asyncio.Event] = {}
        self._resetting = False
        self._queued_turns = 0
        self._reset_generation = 0
        self._last_conversation_activity = time.monotonic()
        self._server: asyncio.AbstractServer | None = None
        self._instance_lock: Any | None = None
        self._stopping = False
        self._quiesce_failed = False
        self._shutdown_event = asyncio.Event()
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
        from hermes_cli.oneshot import create_noninteractive_agent

        return create_noninteractive_agent()

    def _construct_agent(self) -> Any:
        agent = self.agent_factory()
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
            recovered = False
            if (
                record.get("presentation_state") == "correction_in_flight"
                and record.get("correction_attempt_id")
                and not record.get("correction_response")
            ):
                correction_response = {
                    "protocol_version": PROTOCOL_V2,
                    "request_id": record.get("correction_request_id")
                    or f"req_recovered_correction_{record['event_id']}",
                    "operation": "correct_render",
                    "status": "failed",
                    "execution_state": "completed",
                    "presentation_state": "correction_failed",
                    "event_id": record["event_id"],
                    "correction_attempt_id": record["correction_attempt_id"],
                    "runtime_instance_id": self.runtime_instance_id,
                    "trace_id": f"trace_{uuid.uuid4().hex}",
                    "error": {
                        "code": "WORKER_RESTARTED_DURING_CORRECTION",
                        "message": "the one render correction attempt was interrupted by worker restart",
                        "retryable": False,
                    },
                    "timing_ms": _timing_ms(time.monotonic()),
                }
                record["presentation_state"] = "correction_failed"
                record["correction_response"] = correction_response
                record["updated_at"] = _iso(now)
                _atomic_json(path, record)
            if record.get("state") == "in_flight":
                recovered = True
                record["state"] = "outcome_unknown"
                record["updated_at"] = _iso(now)
                record["response"] = self._recovered_failure_response(
                    record,
                    "WORKER_RESTARTED_IN_FLIGHT",
                    "The prior worker stopped while this turn was running.",
                    False,
                    "outcome_unknown",
                )
                _atomic_json(path, record)
            elif record.get("state") == "accepted":
                recovered = True
                record["state"] = "failed"
                record["updated_at"] = _iso(now)
                record["response"] = self._recovered_failure_response(
                    record,
                    "WORKER_RESTARTED_BEFORE_START",
                    "The worker restarted before this queued turn began.",
                    True,
                    "not_started",
                )
                _atomic_json(path, record)
            response = record.get("response")
            if recovered and response is not None:
                self._log_event(
                    "turn_recovered",
                    event_id=record.get("event_id", ""),
                    operation=record.get("operation", "turn"),
                    trace_id=response.get("trace_id"),
                    execution_state=response.get("execution_state"),
                    code=(response.get("error") or {}).get("code"),
                    timing_ms=response.get("timing_ms"),
                )
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
        self._shutdown_event.set()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        await self._conversation_lock.acquire()
        quiesced = False
        try:
            self.socket_path.unlink(missing_ok=True)
            quiesced = await self._retire_agent(quiesce_timeout=MODEL_QUIESCE_TIMEOUT_SECONDS)
        finally:
            self._conversation_lock.release()
        if not quiesced:
            self._unhealthy = True
            self._quiesce_failed = True
            _UNQUIESCED_WORKERS.add(self)
            self.last_error_code = "MODEL_QUIESCE_TIMEOUT"
            self._log_event("worker_quiesce_timeout", timeout_seconds=MODEL_QUIESCE_TIMEOUT_SECONDS)
            return
        _UNQUIESCED_WORKERS.discard(self)
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

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_event.is_set()

    def acknowledge_shutdown(self) -> None:
        """Notify the main owner after the shutdown response is written."""
        self._shutdown_event.set()

    async def wait_for_shutdown(self) -> None:
        await self._shutdown_event.wait()

    async def _await_active_model_task(self, *, timeout: float | None = None) -> bool:
        task = self._active_model_task
        if task is None:
            return True
        try:
            if timeout is None:
                await asyncio.shield(task)
            else:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        except Exception:
            # A completed model task may fail after its protocol response timed out.
            pass
        finally:
            if task.done() and self._active_model_task is task:
                self._active_model_task = None
        return task.done()

    async def _retire_agent(self, *, quiesce_timeout: float | None = None) -> bool:
        if not await self._await_active_model_task(timeout=quiesce_timeout):
            return False
        agent, self._agent = self._agent, None
        if agent is not None:
            session = getattr(agent, "_codex_session", None)
            if session is not None:
                await asyncio.to_thread(session.close)
                agent._codex_session = None
            await asyncio.to_thread(agent.close)
        return True

    def health(self) -> dict[str, Any]:
        return {
            "ready": not self._stopping and not self._unhealthy,
            "unhealthy": self._unhealthy,
            "quiesce_failed": self._quiesce_failed,
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
        hello: dict[str, Any] | None = None
        request: dict[str, Any] | None = None
        try:
            hello = await self._read(reader)
            if hello.get("protocol_version") not in {PROTOCOL, PROTOCOL_V2} or hello.get("operation") != "hello":
                raise ValueError("hello negotiation required")
            if hello["protocol_version"] == PROTOCOL_V2:
                _validate_v2_request(hello)
            capabilities = [
                "turn", "reset", "health", "shutdown", "durable_event_ledger", "render_candidate",
            ]
            if self._correction_companion_config is not None:
                capabilities.append("tool_free_render_correction")
            await self._write(writer, {
                "protocol_version": hello["protocol_version"],
                "request_id": hello["request_id"],
                "operation": "hello",
                "status": "ready" if not self._unhealthy else "failed",
                "runtime_instance_id": self.runtime_instance_id,
                "framework": "hermes",
                "profile": self.profile_root.name,
                "supported_protocol_versions": [PROTOCOL, PROTOCOL_V2],
                "capabilities": capabilities,
            })
            request = await self._read(reader)
            if request.get("protocol_version") == PROTOCOL_V2:
                _validate_v2_request(request)
            response = await self.handle_request(request)
            await self._write(writer, response)
            if request.get("operation") == "shutdown":
                self.acknowledge_shutdown()
        except (asyncio.IncompleteReadError, BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            with contextlib.suppress(Exception):
                await self._write(
                    writer,
                    self._protocol_failure_response(
                        request_id=str((request or hello or {}).get("request_id") or "req_error"),
                        operation=str((request or hello or {}).get("operation") or "health"),
                        protocol_version=str((request or hello or {}).get("protocol_version") or PROTOCOL),
                    ),
                )
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    def _base_response(self, request: dict[str, Any], **extra: Any) -> dict[str, Any]:
        return {
            "protocol_version": request.get("protocol_version", PROTOCOL),
            "request_id": request["request_id"],
            "operation": request["operation"],
            "request_id": request["request_id"],
            "runtime_instance_id": self.runtime_instance_id,
            "trace_id": request.get("_trace_id"),
            **extra,
        }

    def _recovered_failure_response(
        self,
        record: dict[str, Any],
        code: str,
        message: str,
        retryable: bool,
        state: str,
    ) -> dict[str, Any]:
        started = time.monotonic()
        return {
            "protocol_version": PROTOCOL,
            "request_id": record.get("request_id") or f"req_recovered_{record['event_id']}",
            "operation": record["operation"],
            "status": "failed",
            "execution_state": state,
            "event_id": record["event_id"],
            "runtime_instance_id": self.runtime_instance_id,
            "trace_id": f"trace_{uuid.uuid4().hex}",
            "error": {"code": code, "message": message, "retryable": retryable},
            "timing_ms": _timing_ms(started),
        }

    def _protocol_failure_response(
        self,
        *,
        request_id: str,
        operation: str,
        protocol_version: str = PROTOCOL,
    ) -> dict[str, Any]:
        started = time.monotonic()
        if not re.fullmatch(r"req_[A-Za-z0-9._:-]{1,200}", request_id):
            request_id = "req_error"
        if operation not in {"hello", "turn", "reset", "correct_render", "health", "shutdown"}:
            operation = "health"
        response = {
            "protocol_version": protocol_version if protocol_version in {PROTOCOL, PROTOCOL_V2} else PROTOCOL,
            "request_id": request_id,
            "operation": operation,
            "status": "failed",
            "execution_state": "not_started",
            "trace_id": f"trace_{uuid.uuid4().hex}",
            "error": {
                "code": "PROTOCOL_ERROR",
                "message": "invalid protocol request",
                "retryable": False,
            },
            "timing_ms": _timing_ms(started),
        }
        self._log_event(
            "protocol_failed",
            event_id="",
            operation=operation,
            trace_id=response["trace_id"],
            execution_state=response["execution_state"],
            code=response["error"]["code"],
            timing_ms=response["timing_ms"],
        )
        return response

    def _log_terminal_response(self, event: str, request: dict[str, Any], response: dict[str, Any]) -> None:
        self._log_event(
            event,
            event_id=request.get("event_id"),
            operation=request["operation"],
            trace_id=response.get("trace_id"),
            execution_state=response.get("execution_state"),
            code=(response.get("error") or {}).get("code"),
            timing_ms=response.get("timing_ms"),
        )

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request = {
            **request,
            "_trace_id": f"trace_{uuid.uuid4().hex}",
            "_started_at": time.monotonic(),
        }
        if request.get("protocol_version") not in {PROTOCOL, PROTOCOL_V2}:
            return self._protocol_failure_response(
                request_id=str(request.get("request_id") or "req_error"),
                operation=str(request.get("operation") or "health"),
            )
        operation = request.get("operation")
        if operation == "health":
            return self._base_response(
                request,
                status="ready" if not self._stopping else "stopping",
                health=self.health(),
                timing_ms=_timing_ms(request["_started_at"]),
            )
        if operation == "shutdown":
            self._stopping = True
            self._reset_generation += 1
            response = self._base_response(
                request,
                status="stopping",
                execution_state="not_started",
                timing_ms=_timing_ms(request["_started_at"]),
            )
            self._log_terminal_response("worker_shutdown_requested", request, response)
            return response
        if operation == "correct_render":
            if request["protocol_version"] != PROTOCOL_V2:
                raise ValueError("correct_render requires protocol v2")
            return await self._handle_correct_render(request)
        if operation not in {"turn", "reset"}:
            raise ValueError("unsupported operation")
        if self._stopping:
            response = self._failure(
                request,
                "WORKER_STOPPING",
                "the specialist worker is stopping",
                False,
                "not_started",
            )
            self._log_terminal_response("turn_failed", request, response)
            return response
        return await self._handle_turn_or_reset(request)

    async def _handle_turn_or_reset(self, request: dict[str, Any]) -> dict[str, Any]:
        started = request["_started_at"]
        conversation_id = request["conversation_id"]
        event_id = request["event_id"]
        fingerprint = _fingerprint(request)
        path = self._ledger_path(conversation_id, event_id)
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["fingerprint"] != fingerprint:
                response = self._failure(request, "IDEMPOTENCY_CONFLICT", "event_id was reused with different content", False, "not_started")
                self._log_terminal_response("turn_failed", request, response)
                return response
            if record.get("response"):
                response = record["response"]
                self._log_terminal_response("turn_replayed", request, response)
                return response
            state = "outcome_unknown" if record.get("state") == "outcome_unknown" else "in_flight"
            response = self._failure(request, "DUPLICATE_IN_FLIGHT", "the event has already been accepted", False, state)
            self._log_terminal_response("turn_failed", request, response)
            return response
        if self._unhealthy:
            response = self._failure(request, "CONVERSATION_UNHEALTHY", "the specialist conversation requires recovery", False, "not_started")
            self._log_terminal_response("turn_failed", request, response)
            return response
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
            response = self._failure(request, "CONVERSATION_LIMIT", "this v1 worker already owns another conversation", True, "not_started")
            self._log_terminal_response("turn_failed", request, response)
            return response
        queued = False
        generation = self._reset_generation
        if request["operation"] == "turn":
            if self._resetting:
                response = self._failure(request, "RESET_IN_PROGRESS", "the specialist conversation is resetting", True, "not_started")
                self._log_terminal_response("turn_failed", request, response)
                return response
            if self._conversation_lock.locked():
                if self._queued_turns >= self.queue_limit:
                    response = self._failure(request, "QUEUE_FULL", "the specialist conversation queue is full", True, "not_started")
                    self._log_terminal_response("turn_failed", request, response)
                    return response
                self._queued_turns += 1
                queued = True
        else:
            self._resetting = True
            self._reset_generation += 1

        now = _utcnow()
        record = {
            "schema_version": "telegram.bridge.persistent_event_record.v2",
            "protocol_version": request["protocol_version"],
            "request_id": request["request_id"],
            "conversation_id": conversation_id,
            "event_id": event_id,
            "fingerprint": fingerprint,
            "operation": request["operation"],
            "state": "accepted" if queued or request["operation"] == "reset" else "in_flight",
            "accepted_at": _iso(now),
            "updated_at": _iso(now),
        }
        _atomic_json(path, record)
        queue_started = time.monotonic()
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
            response = self._failure(
                request,
                code,
                "the specialist request did not start before its deadline",
                False,
                "not_started",
                timing_ms=_timing_ms(started, queue=int((time.monotonic() - queue_started) * 1000)),
            )
            record["state"] = "failed"
            record["response"] = response
            record["updated_at"] = _iso(_utcnow())
            _atomic_json(path, record)
            self._log_terminal_response("turn_failed", request, response)
            return response
        try:
            queue_ms = int((time.monotonic() - queue_started) * 1000)
            if queued:
                self._queued_turns -= 1
                if generation != self._reset_generation:
                    response = self._failure(
                        request,
                        "RESET",
                        "the queued turn was canceled by reset",
                        False,
                        "not_started",
                        timing_ms=_timing_ms(started, queue=queue_ms),
                    )
                    record["state"] = "failed"
                    record["response"] = response
                    record["updated_at"] = _iso(_utcnow())
                    _atomic_json(path, record)
                    self._log_terminal_response("turn_failed", request, response)
                    return response
            if self._unhealthy:
                response = self._failure(
                    request,
                    "CONVERSATION_UNHEALTHY",
                    "the specialist conversation requires recovery",
                    False,
                    "not_started",
                    timing_ms=_timing_ms(started, queue=queue_ms),
                )
                record["state"] = "failed"
                record["response"] = response
                record["updated_at"] = _iso(_utcnow())
                _atomic_json(path, record)
                self._log_terminal_response("turn_failed", request, response)
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
                initialization_ms = 0
                retirement_started = time.monotonic()
                model_started = time.monotonic()
                if request["operation"] == "reset":
                    retirement_started = time.monotonic()
                    remaining = max(0.1, self.reset_timeout - (time.monotonic() - started))
                    await asyncio.wait_for(self._retire_agent(), timeout=remaining)
                    retirement_ms = int((time.monotonic() - retirement_started) * 1000)
                    self.conversation_instance_id = f"thread_{uuid.uuid4().hex}"
                    render_started = time.monotonic()
                    response = self._reset_response(
                        request,
                        _timing_ms(
                            started,
                            queue=queue_ms,
                            retirement=retirement_ms,
                            render_preparation=int((time.monotonic() - render_started) * 1000),
                        ),
                    )
                    if request["protocol_version"] == PROTOCOL_V2:
                        reset_candidate = response["render_candidate"]["content"]
                        record.update(
                            execution_state="completed",
                            presentation_state="candidate_unvalidated",
                            render_candidate=reset_candidate,
                            render_candidate_sha256=_candidate_hash(reset_candidate),
                        )
                else:
                    initialization_started = time.monotonic()
                    if self._agent is None:
                        self._agent = await asyncio.to_thread(self._construct_agent)
                        initialization_ms = int((time.monotonic() - initialization_started) * 1000)
                    prompt = _bridge_prompt(request["envelope"])
                    model_started = time.monotonic()
                    model_task: asyncio.Task[Any] | None = None
                    try:
                        model_task = asyncio.create_task(
                            asyncio.to_thread(self._agent.run_conversation, prompt)
                        )
                        self._active_model_task = model_task
                        result = await asyncio.wait_for(
                            asyncio.shield(model_task),
                            timeout=self.turn_timeout,
                        )
                    finally:
                        model_and_tools_ms = int((time.monotonic() - model_started) * 1000)
                        if model_task is not None and model_task.done() and self._active_model_task is model_task:
                            self._active_model_task = None
                    if not result.get("completed", True) or result.get("partial"):
                        raise RuntimeError(result.get("error") or "model turn did not complete")
                    render_started = time.monotonic()
                    candidate = result.get("final_response") or ""
                    payloads = None
                    if request["protocol_version"] == PROTOCOL:
                        payloads = _extract_render_payloads(candidate)
                    render_preparation_ms = int((time.monotonic() - render_started) * 1000)
                    response_fields: dict[str, Any] = {
                        "status": "completed",
                        "execution_state": "completed",
                        "event_id": event_id,
                        "conversation_instance_id": self.conversation_instance_id,
                        "timing_ms": _timing_ms(
                            started,
                            queue=queue_ms,
                            initialization=initialization_ms,
                            model_and_tools=model_and_tools_ms,
                            render_preparation=render_preparation_ms,
                        ),
                    }
                    if request["protocol_version"] == PROTOCOL_V2:
                        response_fields.update(
                            presentation_state="candidate_unvalidated",
                            render_candidate=_bounded_candidate(candidate),
                        )
                        record.update(
                            execution_state="completed",
                            presentation_state="candidate_unvalidated",
                            render_candidate=candidate,
                            render_candidate_sha256=_candidate_hash(candidate),
                        )
                    else:
                        response_fields["render_payloads"] = payloads
                    response = self._base_response(request, **response_fields)
                self.last_success_at = _iso(_utcnow())
                self._last_conversation_activity = time.monotonic()
                record["state"] = "completed"
                record["response"] = response
                self._log_terminal_response("turn_completed", request, response)
            except asyncio.TimeoutError:
                self._unhealthy = True
                self.last_error_code = "TURN_TIMEOUT" if request["operation"] == "turn" else "RESET_UNCONFIRMED"
                self._interrupt_agent()
                response = self._failure(
                    request,
                    self.last_error_code,
                    "the specialist did not stop before its deadline",
                    False,
                    "outcome_unknown",
                    timing_ms=_timing_ms(
                        started,
                        queue=queue_ms,
                        retirement=(int((time.monotonic() - retirement_started) * 1000) if request["operation"] == "reset" else 0),
                        model_and_tools=(int((time.monotonic() - model_started) * 1000) if request["operation"] == "turn" else 0),
                    ),
                )
                record["state"] = "outcome_unknown"
                record["response"] = response
                self._log_terminal_response("turn_failed", request, response)
            except Exception as exc:
                self.last_error_code = "TURN_FAILED"
                response = self._failure(
                    request,
                    "TURN_FAILED",
                    str(exc)[:500],
                    False,
                    "outcome_unknown",
                    timing_ms=_timing_ms(
                        started,
                        queue=queue_ms,
                        initialization=initialization_ms if request["operation"] == "turn" else 0,
                        retirement=(int((time.monotonic() - retirement_started) * 1000) if request["operation"] == "reset" else 0),
                        model_and_tools=(int((time.monotonic() - model_started) * 1000) if request["operation"] == "turn" else 0),
                    ),
                )
                record["state"] = "outcome_unknown"
                record["response"] = response
                self._log_terminal_response("turn_failed", request, response)
            record["updated_at"] = _iso(_utcnow())
            _atomic_json(path, record)
            return response
        finally:
            self._conversation_lock.release()
            if request["operation"] == "reset":
                self._resetting = False

    async def _handle_correct_render(self, request: dict[str, Any]) -> dict[str, Any]:
        conversation_id = request["conversation_id"]
        event_id = request["event_id"]
        path = self._ledger_path(conversation_id, event_id)
        attempt_id = request["correction_attempt_id"]
        fingerprint = _correction_fingerprint(request)
        wait_for_terminal: asyncio.Event | None = None
        async with self._correction_state_lock:
            if not path.exists():
                return self._correction_failure(
                    request, "ORIGINAL_EVENT_NOT_FOUND", "accepted completed event was not found",
                    "correction_failed",
                )
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("execution_state") != "completed" or not isinstance(record.get("render_candidate"), str):
                return self._correction_failure(
                    request, "ORIGINAL_EVENT_NOT_COMPLETED", "render correction requires completed domain execution",
                    "correction_failed",
                )
            existing_attempt = record.get("correction_attempt_id")
            if existing_attempt:
                if existing_attempt != attempt_id or record.get("correction_fingerprint") != fingerprint:
                    return self._correction_failure(
                        request,
                        "CORRECTION_ALREADY_ATTEMPTED",
                        "the completed event already used its one render correction attempt",
                        str(record.get("presentation_state") or "correction_failed"),
                    )
                if isinstance(record.get("correction_response"), dict):
                    return record["correction_response"]
                wait_for_terminal = self._correction_events.get(path)
                if wait_for_terminal is None:
                    return self._correction_failure(
                        request,
                        "CORRECTION_STATE_UNAVAILABLE",
                        "the durable correction attempt has no active owner",
                        "correction_failed",
                    )
            else:
                record.update(
                    correction_attempt_id=attempt_id,
                    correction_attempt_count=1,
                    correction_fingerprint=fingerprint,
                    correction_request_id=request["request_id"],
                    presentation_state="correction_in_flight",
                    updated_at=_iso(_utcnow()),
                )
                _atomic_json(path, record)
                self._correction_events[path] = asyncio.Event()

        if wait_for_terminal is not None:
            await wait_for_terminal.wait()
            async with self._correction_state_lock:
                replay_record = json.loads(path.read_text(encoding="utf-8"))
                response = replay_record.get("correction_response")
                if not isinstance(response, dict):
                    return self._correction_failure(
                        request,
                        "CORRECTION_STATE_UNAVAILABLE",
                        "the durable correction result is unavailable",
                        "correction_failed",
                    )
                return response

        timeout = max(0.1, (_parse_time(request["deadline"]) - _utcnow()).total_seconds())
        started = request["_started_at"]
        try:
            await asyncio.wait_for(self._conversation_lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            response = self._correction_failure(
                request, "CORRECTION_DEADLINE_EXPIRED", "correction did not start before its deadline",
                "correction_failed",
            )
            record["presentation_state"] = "correction_failed"
        else:
            try:
                if self._correction_companion_config is None:
                    raise CorrectionCompanionUnavailable(
                        "isolated render correction companion is not configured"
                    )
                repair = {
                    "candidate": record["render_candidate"],
                    "validation_errors": request["validation_errors"],
                    "target_constraints": request["target_constraints"],
                }
                candidate = await asyncio.to_thread(
                    self._correction_companion_runner,
                    self._correction_companion_config,
                    repair,
                    min(self.turn_timeout, timeout),
                )
                response = self._base_response(
                    request,
                    status="completed",
                    execution_state="completed",
                    presentation_state="correction_candidate_unvalidated",
                    event_id=event_id,
                    correction_attempt_id=attempt_id,
                    conversation_instance_id=self.conversation_instance_id,
                    render_candidate=_bounded_candidate(candidate),
                    timing_ms=_timing_ms(started, model_and_tools=int((time.monotonic() - started) * 1000)),
                )
                record["corrected_candidate"] = candidate
                record["corrected_candidate_sha256"] = _candidate_hash(candidate)
                record["presentation_state"] = "correction_candidate_unvalidated"
            except CorrectionCompanionTimeout:
                self._unhealthy = True
                self.last_error_code = "CORRECTION_TIMEOUT"
                response = self._correction_failure(
                    request,
                    "CORRECTION_TIMEOUT",
                    "the tool-free correction model turn did not stop before its deadline",
                    "correction_failed",
                )
                record["presentation_state"] = "correction_failed"
            except CorrectionCompanionUnavailable as exc:
                response = self._correction_failure(
                    request, "TOOL_FREE_CORRECTION_UNAVAILABLE", str(exc), "correction_failed"
                )
                record["presentation_state"] = "correction_failed"
            except Exception as exc:
                response = self._correction_failure(
                    request, "CORRECTION_FAILED", str(exc), "correction_failed"
                )
                record["presentation_state"] = "correction_failed"
            finally:
                self._conversation_lock.release()

        record["correction_response"] = response
        record["updated_at"] = _iso(_utcnow())
        async with self._correction_state_lock:
            _atomic_json(path, record)
            terminal_event = self._correction_events.pop(path, None)
            if terminal_event is not None:
                terminal_event.set()
        self._log_terminal_response("render_correction_completed", request, response)
        return response

    def _correction_failure(
        self,
        request: dict[str, Any],
        code: str,
        message: str,
        presentation_state: str,
    ) -> dict[str, Any]:
        return self._base_response(
            request,
            status="failed",
            execution_state="completed",
            presentation_state=presentation_state,
            event_id=request.get("event_id"),
            correction_attempt_id=request.get("correction_attempt_id"),
            error={"code": code, "message": message[:500], "retryable": False},
            timing_ms=_timing_ms(request["_started_at"]),
        )

    def _interrupt_agent(self) -> None:
        session = getattr(self._agent, "_codex_session", None)
        if session is not None:
            with contextlib.suppress(Exception):
                session.request_interrupt()

    def _failure(
        self,
        request: dict[str, Any],
        code: str,
        message: str,
        retryable: bool,
        state: str,
        timing_ms: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        return self._base_response(
            request,
            status="failed",
            execution_state=state,
            event_id=request.get("event_id"),
            error={"code": code, "message": message[:500], "retryable": retryable},
            timing_ms=timing_ms or _timing_ms(request["_started_at"]),
        )

    def _reset_response(self, request: dict[str, Any], timing_ms: dict[str, int]) -> dict[str, Any]:
        envelope = request["envelope"]
        payload = {
            "schema_version": "telegram.bridge.render_payload.v1",
            "message_id": f"msg_{request['event_id']}_reset",
            "correlation_id": request["event_id"],
            "action": "send",
            "target": {"chat_id": int(envelope["chat_id"])},
            "render": {"text": f"{self.agent_id} conversation context reset."},
        }
        fields: dict[str, Any] = {
            "status": "completed",
            "execution_state": "completed",
            "event_id": request["event_id"],
            "conversation_instance_id": self.conversation_instance_id,
            "timing_ms": timing_ms,
        }
        if request["protocol_version"] == PROTOCOL_V2:
            candidate = json.dumps([payload], ensure_ascii=False, separators=(",", ":"))
            fields.update(
                presentation_state="candidate_unvalidated",
                render_candidate=_bounded_candidate(candidate),
            )
        else:
            fields["render_payloads"] = [payload]
        return self._base_response(request, **fields)


async def _main_async(args: argparse.Namespace) -> int:
    profile_root = Path(args.profile_root).resolve()
    _initialize_worker_process_context(profile_root)
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
    shutdown = asyncio.create_task(worker.wait_for_shutdown())
    done, _ = await asyncio.wait({serve, wait, shutdown}, return_when=asyncio.FIRST_COMPLETED)
    if serve in done and not serve.cancelled():
        exception = serve.exception()
        if exception and not isinstance(exception, asyncio.CancelledError):
            raise exception
    serve.cancel()
    wait.cancel()
    shutdown.cancel()
    await asyncio.gather(serve, wait, shutdown, return_exceptions=True)
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
