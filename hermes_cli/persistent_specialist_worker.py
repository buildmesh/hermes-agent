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

from agent.telegram_bridge_render_contract import (
    CANONICAL_RENDER_ENVELOPE_INSTRUCTIONS,
    MODEL_RENDER_ENVELOPE_INSTRUCTIONS,
)

from agent.render_correction_companion import (
    CorrectionCompanionTimeout,
    CorrectionCompanionUnavailable,
    resolve_companion_descriptor,
    resolve_companion_credentials_subprocess,
    run_companion_subprocess,
    validate_companion_descriptor,
)
from hermes_cli.terminal_presenter import (
    ENDPOINT_RELATIVE_PATH,
    MAX_PRESENTER_WIRE_BYTES,
    PRESENTER_PROTOCOL,
    TerminalPresenterError,
    execute_terminal_presenter,
    load_terminal_presenters,
)


PROTOCOL = "telegram.bridge.persistent_service.v1"
PROTOCOL_V2 = "telegram.bridge.persistent_service.v2"
PROTOCOL_V3 = "telegram.bridge.persistent_service.v3"
TERMINAL_PRESENTER_CAPABILITY = "terminal_presenter_finalization.v1"
MAX_LINE_BYTES = 1024 * 1024
MAX_CANDIDATE_BYTES = 512 * 1024
LEDGER_RETENTION = timedelta(days=30)
TERMINAL_STATES = {"completed", "failed", "outcome_unknown"}
TIMING_PHASES = ("queue", "initialization", "retirement", "model_and_tools", "render_preparation")
MODEL_QUIESCE_TIMEOUT_SECONDS = 5.0
CORRECTION_CREDENTIAL_PREFLIGHT_SECONDS = 5.0
RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
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
    if not isinstance(deadline, str) or RFC3339_RE.fullmatch(deadline) is None:
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


def _validate_v3_request(request: dict[str, Any]) -> None:
    """Validate v3, which adds explicit hello capability acceptance."""
    if request.get("protocol_version") != PROTOCOL_V3:
        raise ValueError("invalid protocol version")
    if request.get("operation") == "hello":
        if set(request) != {
            "protocol_version", "request_id", "operation", "client", "accepted_capabilities",
        }:
            raise ValueError("invalid closed request shape")
        accepted = request.get("accepted_capabilities")
        if (
            not isinstance(accepted, list)
            or len(accepted) != len(set(accepted))
            or any(not isinstance(item, str) or not 1 <= len(item) <= 128 for item in accepted)
        ):
            raise ValueError("invalid accepted capabilities")
        v2_hello = {key: value for key, value in request.items() if key != "accepted_capabilities"}
        v2_hello["protocol_version"] = PROTOCOL_V2
        _validate_v2_request(v2_hello)
        return
    compatible = dict(request)
    compatible["protocol_version"] = PROTOCOL_V2
    _validate_v2_request(compatible)


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


def _bridge_prompt(envelope: dict[str, Any], *, terminal_presenter: bool = False) -> str:
    if terminal_presenter:
        completion_instruction = (
            "The negotiated finalize_telegram_presentation tool is available for this turn. "
            "Choose exactly one completion path. When the installed profile declares a supported "
            "presentation for the result, you must invoke finalize_telegram_presentation after all "
            "domain work and as the final, unbatched tool call. Do not inspect or invoke the "
            "presenter implementation directly, copy or modify its artifact, or return render JSON "
            "after successful finalization. After the tool result, reply only with a brief "
            "acknowledgement; the runtime uses the captured artifact as the Telegram response. "
            "Use the model-render path below only when the profile does not declare a supported "
            "terminal presentation for the result. Do not call Telegram directly. Preserve "
            "event_id as correlation_id and use the envelope chat_id as the send target. "
            + MODEL_RENDER_ENVELOPE_INSTRUCTIONS
        )
    else:
        completion_instruction = (
            "Do not call Telegram directly. Preserve event_id as correlation_id and use the "
            "envelope chat_id as the send target. "
            + CANONICAL_RENDER_ENVELOPE_INSTRUCTIONS
        )
    return (
        "Handle the following Telegram Bridge specialist envelope using this profile's skills and "
        "instructions. "
        + completion_instruction
        + "\n\nEnvelope JSON:\n"
        + json.dumps(envelope, ensure_ascii=False, sort_keys=True)
    )


def _terminal_presenter_is_final_and_unbatched(
    result: dict[str, Any],
    prompt: str,
    artifact_content: str,
    artifact_sha256: str | None = None,
) -> bool:
    messages = result.get("messages")
    if not isinstance(messages, list):
        return False
    start = next((
        index
        for index in range(len(messages) - 1, -1, -1)
        if isinstance(messages[index], dict)
        and messages[index].get("role") == "user"
        and messages[index].get("content") == prompt
    ), -1)
    if start < 0:
        return False
    calls: list[tuple[str, str, int]] = []
    successful_call_ids: set[str] = set()
    for message in messages[start + 1 :]:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list):
            projected_batch_size = message.get("codex_tool_batch_size")
            batch_size = max(
                len(message["tool_calls"]),
                projected_batch_size
                if type(projected_batch_size) is int and projected_batch_size > 0
                else 1,
            )
            for call in message["tool_calls"]:
                function = call.get("function") if isinstance(call, dict) else None
                call_id = call.get("id") if isinstance(call, dict) else None
                name = function.get("name") if isinstance(function, dict) else None
                if isinstance(call_id, str) and isinstance(name, str):
                    calls.append((call_id, name, batch_size))
        elif (
            message.get("role") == "tool"
            and isinstance(message.get("tool_call_id"), str)
            and (
                message.get("content") == artifact_content
                or (
                    isinstance(artifact_sha256, str)
                    and message.get("terminal_presenter_content_sha256")
                    == artifact_sha256
                )
            )
        ):
            successful_call_ids.add(message["tool_call_id"])
    if not calls or not successful_call_ids:
        return False
    successful = [call for call in calls if call[0] in successful_call_ids]
    return (
        len(successful) == 1
        and successful[0] == calls[-1]
        and successful[0][2] == 1
        and _is_terminal_presenter_tool(successful[0][1])
    )


def _is_terminal_presenter_tool(name: str) -> bool:
    return (
        name == "finalize_telegram_presentation"
        or name.endswith(".finalize_telegram_presentation")
        or name.endswith("__finalize_telegram_presentation")
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
        correction_companion_resolver: Callable[[Path], dict[str, str] | None] | None = None,
        correction_companion_credentials_resolver: Callable[[dict[str, str], float], dict[str, str]] | None = None,
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
        resolver = correction_companion_resolver or resolve_companion_descriptor
        try:
            descriptor = resolver(self.profile_root)
            self._correction_companion_descriptor = (
                validate_companion_descriptor(descriptor, self.profile_root)
                if descriptor is not None else None
            )
        except Exception:
            self._correction_companion_descriptor = None
        self._correction_companion_credentials_resolver = (
            correction_companion_credentials_resolver or resolve_companion_credentials_subprocess
        )
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
        self._presenter_endpoint_path = self.profile_root / ENDPOINT_RELATIVE_PATH
        self._active_presenter_turn: dict[str, Any] | None = None
        try:
            self._terminal_presenters = load_terminal_presenters(self.profile_root)
        except TerminalPresenterError:
            self._terminal_presenters = {}

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

    def _write_presenter_endpoint(self, turn_token: str | None = None) -> None:
        endpoint = {
            "schema_version": "hermes.terminal_presenter_endpoint.v1",
            "socket": str(self.socket_path.relative_to(self.profile_root)),
            "runtime_instance_id": self.runtime_instance_id,
        }
        if turn_token is not None:
            endpoint["turn_token"] = turn_token
        _atomic_json(self._presenter_endpoint_path, endpoint)

    def _begin_presenter_turn(self, request: dict[str, Any]) -> None:
        token = f"present_turn_{uuid.uuid4().hex}"
        self._active_presenter_turn = {
            "turn_token": token,
            "event_id": request["event_id"],
            "conversation_id": request["conversation_id"],
            "artifact": None,
            "violation": None,
        }
        self._write_presenter_endpoint(token)

    def _end_presenter_turn(self) -> None:
        self._active_presenter_turn = None
        if self._server is not None:
            self._write_presenter_endpoint()
        else:
            self._presenter_endpoint_path.unlink(missing_ok=True)

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
                    "protocol_version": (
                        PROTOCOL_V3 if record.get("protocol_version") == PROTOCOL_V3 else PROTOCOL_V2
                    ),
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
        self._write_presenter_endpoint()
        self._log_event("worker_ready", socket=str(self.socket_path))

    async def stop(self) -> None:
        self._stopping = True
        self._shutdown_event.set()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._presenter_endpoint_path.unlink(missing_ok=True)
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

    async def _handle_presenter_request(self, request: dict[str, Any]) -> dict[str, Any]:
        allowed = {"protocol_version", "operation", "turn_token", "presenter_id", "input"}
        active = self._active_presenter_turn
        if set(request) != allowed or request.get("operation") != "present":
            return {
                "protocol_version": PRESENTER_PROTOCOL,
                "status": "failed",
                "error": {"code": "PRESENTER_PROTOCOL_ERROR", "message": "invalid presenter request"},
            }
        if (
            active is None
            or request.get("turn_token") != active.get("turn_token")
            or not self._terminal_presenters
        ):
            return {
                "protocol_version": PRESENTER_PROTOCOL,
                "status": "failed",
                "error": {
                    "code": "PRESENTER_TURN_UNAVAILABLE",
                    "message": "no eligible persistent presenter turn is active",
                },
            }
        if active.get("artifact") is not None:
            active["violation"] = "multiple successful presenter invocations"
            return {
                "protocol_version": PRESENTER_PROTOCOL,
                "status": "failed",
                "error": {
                    "code": "TERMINAL_PRESENTER_NOT_FINAL",
                    "message": "the turn already has a successful terminal presenter",
                },
            }
        presenter_id = request.get("presenter_id")
        presenter_input = request.get("input")
        if not isinstance(presenter_id, str) or not isinstance(presenter_input, dict):
            return {
                "protocol_version": PRESENTER_PROTOCOL,
                "status": "failed",
                "error": {"code": "PRESENTER_INPUT_INVALID", "message": "invalid presenter arguments"},
            }
        try:
            artifact = await asyncio.to_thread(
                execute_terminal_presenter,
                self.profile_root,
                presenter_id,
                presenter_input,
            )
        except TerminalPresenterError as exc:
            return {
                "protocol_version": PRESENTER_PROTOCOL,
                "status": "failed",
                "error": {"code": exc.code, "message": str(exc)[:500]},
            }
        artifact["invocation_id"] = f"present_{uuid.uuid4().hex}"
        completed_response = {
            "protocol_version": PRESENTER_PROTOCOL,
            "status": "completed",
            "content": artifact["content"],
            "sha256": artifact["sha256"],
            "invocation_id": artifact["invocation_id"],
        }
        if len(json.dumps(completed_response, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1 > MAX_PRESENTER_WIRE_BYTES:
            return {
                "protocol_version": PRESENTER_PROTOCOL,
                "status": "failed",
                "error": {
                    "code": "PRESENTER_OUTPUT_TOO_LARGE",
                    "message": "presenter output exceeds the protocol frame limit",
                },
            }
        active["artifact"] = artifact
        return completed_response

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        hello: dict[str, Any] | None = None
        request: dict[str, Any] | None = None
        try:
            hello = await self._read(reader)
            if hello.get("protocol_version") == PRESENTER_PROTOCOL:
                await self._write(writer, await self._handle_presenter_request(hello))
                return
            if hello.get("protocol_version") not in {PROTOCOL, PROTOCOL_V2, PROTOCOL_V3} or hello.get("operation") != "hello":
                raise ValueError("hello negotiation required")
            if hello["protocol_version"] == PROTOCOL_V2:
                _validate_v2_request(hello)
            elif hello["protocol_version"] == PROTOCOL_V3:
                _validate_v3_request(hello)
            capabilities = [
                "turn", "reset", "health", "shutdown", "durable_event_ledger", "render_candidate",
            ]
            if self._correction_companion_descriptor is not None:
                capabilities.append("tool_free_render_correction")
            accepted_raw = hello.get("accepted_capabilities")
            accepted = set(accepted_raw) if isinstance(accepted_raw, list) else None
            if (
                self._terminal_presenters
                and (accepted is None or TERMINAL_PRESENTER_CAPABILITY in accepted)
            ):
                capabilities.append(TERMINAL_PRESENTER_CAPABILITY)
            await self._write(writer, {
                "protocol_version": hello["protocol_version"],
                "request_id": hello["request_id"],
                "operation": "hello",
                "status": "ready" if not self._unhealthy else "failed",
                "runtime_instance_id": self.runtime_instance_id,
                "framework": "hermes",
                "profile": self.profile_root.name,
                "supported_protocol_versions": [PROTOCOL, PROTOCOL_V2, PROTOCOL_V3],
                "capabilities": capabilities,
            })
            request = await self._read(reader)
            if request.get("protocol_version") == PROTOCOL_V2:
                _validate_v2_request(request)
            elif request.get("protocol_version") == PROTOCOL_V3:
                _validate_v3_request(request)
            request["_terminal_presenter_negotiated"] = (
                request.get("protocol_version") == PROTOCOL_V3
                and TERMINAL_PRESENTER_CAPABILITY in capabilities
            )
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
                        operation=(
                            "health"
                            if str((request or hello or {}).get("protocol_version") or "") in {PROTOCOL_V2, PROTOCOL_V3}
                            else str((request or hello or {}).get("operation") or "health")
                        ),
                        protocol_version=str((request or hello or {}).get("protocol_version") or PROTOCOL),
                    ),
                )
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    def _model_metadata(self, result: dict[str, Any]) -> dict[str, str]:
        """Model-resolution metadata for completed turn responses.

        requested_* restate what the profile configuration asked Hermes to
        run — the claim under verification. runtime_model is the model the
        codex app-server reported for the serving thread (thread/start
        response), observed by the transport rather than read from any
        config. runtime_provider/runtime_api_mode identify the transport
        that actually served the turn, emitted in Hermes provider naming so
        the bridge can compare them against profile config; they are only
        emitted alongside a runtime_model observation. All fields are
        optional in the response contract: when the runtime exposes nothing,
        nothing is emitted and the bridge records the turn as unverified.
        """
        fields: dict[str, str] = {}
        agent = self._agent
        for key, value in (
            ("requested_model", getattr(agent, "model", None)),
            ("requested_provider", getattr(agent, "provider", None)),
            ("requested_api_mode", getattr(agent, "api_mode", None)),
        ):
            if isinstance(value, str) and value.strip():
                fields[key] = value.strip()
        runtime_model = result.get("runtime_model")
        if isinstance(runtime_model, str) and runtime_model.strip():
            fields["runtime_model"] = runtime_model.strip()
            provider = getattr(agent, "provider", None)
            if isinstance(provider, str) and provider.strip():
                fields["runtime_provider"] = provider.strip()
            api_mode = getattr(agent, "api_mode", None)
            if isinstance(api_mode, str) and api_mode.strip():
                fields["runtime_api_mode"] = api_mode.strip()
        return fields

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
            "protocol_version": str(record.get("protocol_version") or PROTOCOL),
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
            "protocol_version": (
                protocol_version if protocol_version in {PROTOCOL, PROTOCOL_V2, PROTOCOL_V3} else PROTOCOL
            ),
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
        if request.get("protocol_version") not in {PROTOCOL, PROTOCOL_V2, PROTOCOL_V3}:
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
            if request["protocol_version"] not in {PROTOCOL_V2, PROTOCOL_V3}:
                raise ValueError("correct_render requires protocol v2 or v3")
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
                presenter_turn_active = False
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
                    if request["protocol_version"] in {PROTOCOL_V2, PROTOCOL_V3}:
                        reset_candidate = response["render_candidate"]["content"]
                        record.update(
                            execution_state="completed",
                            presentation_state="candidate_unvalidated",
                            render_candidate=reset_candidate,
                            render_candidate_sha256=_candidate_hash(reset_candidate),
                        )
                        if request["protocol_version"] == PROTOCOL_V3:
                            record["candidate_source"] = response["candidate_source"]
                else:
                    terminal_presenter_enabled = (
                        request["protocol_version"] == PROTOCOL_V3
                        and request.get("_terminal_presenter_negotiated") is True
                        and bool(self._terminal_presenters)
                    )
                    if terminal_presenter_enabled:
                        self._begin_presenter_turn(request)
                        presenter_turn_active = True
                    initialization_started = time.monotonic()
                    if self._agent is None:
                        self._agent = await asyncio.to_thread(self._construct_agent)
                        initialization_ms = int((time.monotonic() - initialization_started) * 1000)
                    prompt = _bridge_prompt(
                        request["envelope"],
                        terminal_presenter=terminal_presenter_enabled,
                    )
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
                    candidate_source: dict[str, Any] = {"kind": "model_final"}
                    active_presenter = self._active_presenter_turn if presenter_turn_active else None
                    artifact = active_presenter.get("artifact") if active_presenter else None
                    if isinstance(artifact, dict):
                        if active_presenter.get("violation"):
                            raise TerminalPresenterError(
                                "TERMINAL_PRESENTER_NOT_FINAL",
                                str(active_presenter["violation"]),
                            )
                        if not _terminal_presenter_is_final_and_unbatched(
                            result,
                            prompt,
                            str(artifact["content"]),
                            str(artifact["sha256"]),
                        ):
                            raise TerminalPresenterError(
                                "TERMINAL_PRESENTER_NOT_FINAL",
                                "terminal presenter was not the final tool invocation",
                            )
                        candidate = str(artifact["content"])
                        candidate_source = {
                            "kind": "terminal_presenter",
                            "presenter_id": artifact["presenter_id"],
                            "presenter_version": artifact["presenter_version"],
                            "presenter_sha256": artifact["presenter_sha256"],
                            "invocation_id": artifact["invocation_id"],
                        }
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
                        **self._model_metadata(result),
                    }
                    if request["protocol_version"] in {PROTOCOL_V2, PROTOCOL_V3}:
                        response_fields.update(
                            presentation_state="candidate_unvalidated",
                            render_candidate=_bounded_candidate(candidate),
                        )
                        if request["protocol_version"] == PROTOCOL_V3:
                            response_fields["candidate_source"] = candidate_source
                        record.update(
                            execution_state="completed",
                            presentation_state="candidate_unvalidated",
                            render_candidate=candidate,
                            render_candidate_sha256=_candidate_hash(candidate),
                        )
                        if request["protocol_version"] == PROTOCOL_V3:
                            record["candidate_source"] = candidate_source
                    else:
                        response_fields["render_payloads"] = payloads
                    response = self._base_response(request, **response_fields)
                self.last_success_at = _iso(_utcnow())
                self._last_conversation_activity = time.monotonic()
                record["state"] = "completed"
                record["response"] = response
                self._log_terminal_response("turn_completed", request, response)
            except TerminalPresenterError as exc:
                self.last_error_code = exc.code
                response = self._failure(
                    request,
                    exc.code,
                    str(exc),
                    False,
                    "completed",
                    timing_ms=_timing_ms(
                        started,
                        queue=queue_ms,
                        initialization=initialization_ms if request["operation"] == "turn" else 0,
                        model_and_tools=(
                            int((time.monotonic() - model_started) * 1000)
                            if request["operation"] == "turn"
                            else 0
                        ),
                    ),
                )
                record["state"] = "failed"
                record["execution_state"] = "completed"
                record["response"] = response
                self._log_terminal_response("turn_failed", request, response)
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
            if self._active_presenter_turn is not None:
                self._end_presenter_turn()
            self._conversation_lock.release()
            if request["operation"] == "reset":
                self._resetting = False

    async def _handle_correct_render(self, request: dict[str, Any]) -> dict[str, Any]:
        caller_deadline = min(
            _parse_time(request["deadline"]),
            _utcnow() + timedelta(seconds=self.turn_timeout),
        )
        correction_deadline = caller_deadline
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
            if (record.get("candidate_source") or {}).get("kind") == "terminal_presenter":
                return self._correction_failure(
                    request,
                    "TERMINAL_PRESENTER_CORRECTION_FORBIDDEN",
                    "terminal presenter candidates cannot receive model correction",
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
                try:
                    correction_deadline = _parse_time(record["correction_deadline"])
                except (KeyError, TypeError, ValueError):
                    return self._correction_failure(
                        request,
                        "CORRECTION_STATE_UNAVAILABLE",
                        "the durable correction attempt has no accepted deadline",
                        "correction_failed",
                    )
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
                    correction_deadline=_iso(correction_deadline),
                    presentation_state="correction_in_flight",
                    updated_at=_iso(_utcnow()),
                )
                _atomic_json(path, record)
                self._correction_events[path] = asyncio.Event()

        if wait_for_terminal is not None:
            remaining = (min(caller_deadline, correction_deadline) - _utcnow()).total_seconds()
            try:
                if remaining <= 0:
                    raise asyncio.TimeoutError
                await asyncio.wait_for(wait_for_terminal.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                async with self._correction_state_lock:
                    replay_record = json.loads(path.read_text(encoding="utf-8"))
                    response = replay_record.get("correction_response")
                    if isinstance(response, dict):
                        return response
                return self._correction_failure(
                    request,
                    "CORRECTION_DEADLINE_EXPIRED",
                    "correction did not complete before its caller deadline",
                    "correction_failed",
                )
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

        started = request["_started_at"]
        conversation_lock_acquired = False
        try:
            if self._correction_companion_descriptor is None:
                raise CorrectionCompanionUnavailable(
                    "isolated render correction companion is not configured"
                )
            repair = {
                "candidate": record["render_candidate"],
                "validation_errors": request["validation_errors"],
                "target_constraints": request["target_constraints"],
            }
            remaining = (correction_deadline - _utcnow()).total_seconds()
            if remaining < CORRECTION_CREDENTIAL_PREFLIGHT_SECONDS:
                raise asyncio.TimeoutError

            # The isolated resolver returns only after any credential rotation is
            # durably persisted, and its hard timeout is the correction deadline.
            try:
                correction_config = await asyncio.to_thread(
                    self._correction_companion_credentials_resolver,
                    self._correction_companion_descriptor,
                    remaining,
                )
            except CorrectionCompanionTimeout as exc:
                raise asyncio.TimeoutError from exc
            remaining = (correction_deadline - _utcnow()).total_seconds()
            if remaining <= 0:
                raise asyncio.TimeoutError

            await asyncio.wait_for(self._conversation_lock.acquire(), timeout=remaining)
            conversation_lock_acquired = True
            remaining = (correction_deadline - _utcnow()).total_seconds()
            if remaining <= 0:
                raise asyncio.TimeoutError
            candidate = await asyncio.to_thread(
                self._correction_companion_runner,
                correction_config,
                repair,
                remaining,
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
            if request["protocol_version"] == PROTOCOL_V3:
                response["candidate_source"] = {"kind": "model_correction"}
            record["corrected_candidate"] = candidate
            record["corrected_candidate_sha256"] = _candidate_hash(candidate)
            record["presentation_state"] = "correction_candidate_unvalidated"
            if request["protocol_version"] == PROTOCOL_V3:
                record["corrected_candidate_source"] = response["candidate_source"]
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
        except asyncio.TimeoutError:
            response = self._correction_failure(
                request,
                "CORRECTION_DEADLINE_EXPIRED",
                "correction did not start before its deadline",
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
            if conversation_lock_acquired:
                self._conversation_lock.release()

        async with self._correction_state_lock:
            persisted_record = json.loads(path.read_text(encoding="utf-8"))
            persisted_response = persisted_record.get("correction_response")
            if isinstance(persisted_response, dict):
                response = persisted_response
                record = persisted_record
            else:
                record["correction_response"] = response
                record["updated_at"] = _iso(_utcnow())
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
        if request["protocol_version"] in {PROTOCOL_V2, PROTOCOL_V3}:
            candidate = json.dumps([payload], ensure_ascii=False, separators=(",", ":"))
            fields.update(
                presentation_state="candidate_unvalidated",
                render_candidate=_bounded_candidate(candidate),
            )
            if request["protocol_version"] == PROTOCOL_V3:
                fields["candidate_source"] = {"kind": "runtime_control"}
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
