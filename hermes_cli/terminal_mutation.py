"""Profile-declared idempotent mutation-to-presenter workflows."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import uuid
import fcntl
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from hermes_cli.terminal_presenter import (
    ENDPOINT_RELATIVE_PATH,
    MAX_PRESENTER_TIMEOUT_SECONDS,
    MAX_PRESENTER_WIRE_BYTES,
    PRESENTER_PROTOCOL,
    TerminalPresenterError,
    _ID_RE,
    _VERSION_RE,
    load_terminal_presenters,
)
from hermes_cli.terminal_workflow import (
    DeclaredProgram,
    TerminalWorkflowError,
    _program,
    _run_program,
)

MUTATION_MANIFEST_RELATIVE_PATH = Path("contract/terminal-mutations.json")
MUTATION_JOURNAL_RELATIVE_PATH = Path(
    "state/persistent-runtime/terminal-mutations"
)
MUTATION_CAPABILITY = "terminal_mutation_chaining.v1"
MUTATION_INVOKE_SCHEMA = "hermes.terminal_mutation.invoke.v1"
MUTATION_JOURNAL_RETENTION = timedelta(days=30)


@dataclass(frozen=True)
class TerminalMutation:
    workflow_id: str
    version: str
    description: str
    handler: DeclaredProgram
    mapper: DeclaredProgram | None
    presenter_id: str
    presenter_version: str
    presenter_sha256: str


class TerminalMutationError(TerminalPresenterError):
    """Typed mutation declaration, execution, or endpoint failure."""

    def __init__(self, code: str, message: str, *, sealed: bool = False) -> None:
        super().__init__(code, message)
        self.sealed = sealed


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
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


def _invocation_key(
    specialist_id: str,
    conversation_id: str,
    event_id: str,
) -> str:
    return hashlib.sha256(
        f"{specialist_id}\0{conversation_id}\0{event_id}".encode()
    ).hexdigest()


def _operation_id(invocation_key: str) -> str:
    return f"tmut_{invocation_key}"


def load_terminal_mutations(
    profile_root: Path,
) -> dict[str, TerminalMutation]:
    root = profile_root.resolve()
    try:
        manifest = json.loads(
            (root / MUTATION_MANIFEST_RELATIVE_PATH).read_text(
                encoding="utf-8"
            )
        )
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalMutationError(
            "MUTATION_CONFIG_INVALID",
            "mutation manifest is unreadable",
        ) from exc
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "workflows"}
        or manifest.get("schema_version")
        != "telegram.bridge.terminal_mutations.v1"
        or not isinstance(manifest.get("workflows"), list)
        or not 1 <= len(manifest["workflows"]) <= 32
    ):
        raise TerminalMutationError(
            "MUTATION_CONFIG_INVALID",
            "mutation manifest shape is invalid",
        )
    presenters = load_terminal_presenters(root)
    mutations: dict[str, TerminalMutation] = {}
    allowed = {
        "workflow_id",
        "version",
        "description",
        "mutation",
        "idempotency",
        "handler",
        "mapper",
        "presenter_id",
    }
    for raw in manifest["workflows"]:
        if not isinstance(raw, dict) or set(raw) != allowed:
            raise TerminalMutationError(
                "MUTATION_CONFIG_INVALID",
                "mutation declaration shape is invalid",
            )
        workflow_id = raw.get("workflow_id")
        version = raw.get("version")
        description = raw.get("description")
        presenter_id = raw.get("presenter_id")
        if (
            not isinstance(workflow_id, str)
            or _ID_RE.fullmatch(workflow_id) is None
        ):
            raise TerminalMutationError(
                "MUTATION_CONFIG_INVALID",
                "workflow_id is invalid",
            )
        if workflow_id in mutations:
            raise TerminalMutationError(
                "MUTATION_CONFIG_INVALID",
                "workflow_id is duplicated",
            )
        if (
            not isinstance(version, str)
            or _VERSION_RE.fullmatch(version) is None
        ):
            raise TerminalMutationError(
                "MUTATION_CONFIG_INVALID",
                "workflow version is invalid",
            )
        if (
            not isinstance(description, str)
            or not 1 <= len(description) <= 500
        ):
            raise TerminalMutationError(
                "MUTATION_CONFIG_INVALID",
                "workflow description is invalid",
            )
        if raw.get("mutation") is not True or raw.get("idempotency") != {
            "mode": "operation_id",
            "guarantee": "transactional",
        }:
            raise TerminalMutationError(
                "MUTATION_CONFIG_INVALID",
                "terminal_mutations.v1 requires transactional operation_id idempotency",
            )
        if not isinstance(presenter_id, str) or presenter_id not in presenters:
            raise TerminalMutationError(
                "MUTATION_CONFIG_INVALID",
                "mutation presenter_id is not declared",
            )
        try:
            handler = _program(root, raw.get("handler"), "mutation handler")
            mapper = (
                None
                if raw.get("mapper") is None
                else _program(root, raw.get("mapper"), "mutation mapper")
            )
        except TerminalWorkflowError as exc:
            raise TerminalMutationError(
                exc.code.replace("WORKFLOW", "MUTATION"),
                str(exc),
            ) from exc
        mutations[workflow_id] = TerminalMutation(
            workflow_id=workflow_id,
            version=version,
            description=description,
            handler=handler,
            mapper=mapper,
            presenter_id=presenter_id,
            presenter_version=presenters[presenter_id].version,
            presenter_sha256=presenters[presenter_id].handler_sha256,
        )
    return mutations


def recover_terminal_mutation_journal(profile_root: Path) -> bool:
    """Recover interrupted private records; false withholds the capability."""
    directory = profile_root.resolve() / MUTATION_JOURNAL_RELATIVE_PATH
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(directory, 0o700)
        for path in directory.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                return False
            changed = False
            if record.get("execution_state") == "executing":
                record["execution_state"] = "outcome_unknown"
                changed = True
            if record.get("presentation_state") == "preparing":
                record["presentation_state"] = "presentation_failed"
                changed = True
            if changed:
                record["updated_at"] = _now()
                _atomic_json(path, record)
            created_at = record.get("created_at")
            if (
                record.get("execution_state")
                in {"committed", "not_applied", "conflict"}
                and isinstance(created_at, str)
                and datetime.now(timezone.utc)
                >= datetime.fromisoformat(
                    created_at.replace("Z", "+00:00")
                )
                + MUTATION_JOURNAL_RETENTION
            ):
                path.unlink()
                path.with_suffix(".lock").unlink(missing_ok=True)
        return os.access(directory, os.W_OK)
    except (OSError, UnicodeDecodeError, ValueError, TypeError):
        return False


def terminal_mutations_available(profile_root: Path | None = None) -> bool:
    root = (profile_root or get_hermes_home()).resolve()
    try:
        endpoint = json.loads(
            (root / ENDPOINT_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        return (
            isinstance(endpoint.get("turn_token"), str)
            and bool(endpoint["turn_token"])
            and endpoint.get("terminal_mutation") is True
            and bool(load_terminal_mutations(root))
        )
    except (
        TerminalPresenterError,
        OSError,
        ValueError,
        TypeError,
        AttributeError,
    ):
        return False


def terminal_mutation_description(
    profile_root: Path | None = None,
) -> str:
    try:
        mutations = load_terminal_mutations(
            (profile_root or get_hermes_home()).resolve()
        )
    except TerminalPresenterError:
        mutations = {}
    entries = "; ".join(
        f"{item.workflow_id}: {item.description} Input schema: "
        f"{json.dumps(item.handler.input_schema, ensure_ascii=False, sort_keys=True)}"
        for item in mutations.values()
    )
    return (
        "Immediately run one installed idempotent mutation workflow. This changes "
        "domain state without a framework confirmation prompt. Call it only after "
        "you determine the user's request is clear and any required confirmation "
        "has already occurred. It must be the final, unbatched tool call. "
        + (f"Available mutations: {entries}" if entries else "")
    )


def execute_terminal_mutation(
    profile_root: Path,
    *,
    specialist_id: str,
    conversation_id: str,
    event_id: str,
    workflow_id: str,
    workflow_input: dict[str, Any],
) -> dict[str, Any]:
    """Serialize the authoritative invocation key across threads/processes."""
    root = profile_root.resolve()
    invocation_key = _invocation_key(
        specialist_id,
        conversation_id,
        event_id,
    )
    directory = root / MUTATION_JOURNAL_RELATIVE_PATH
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = directory / f"{invocation_key}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _execute_terminal_mutation_locked(
            root,
            specialist_id=specialist_id,
            conversation_id=conversation_id,
            event_id=event_id,
            workflow_id=workflow_id,
            workflow_input=workflow_input,
        )


def _execute_terminal_mutation_locked(
    profile_root: Path,
    *,
    specialist_id: str,
    conversation_id: str,
    event_id: str,
    workflow_id: str,
    workflow_input: dict[str, Any],
) -> dict[str, Any]:
    root = profile_root.resolve()
    mutation = load_terminal_mutations(root).get(workflow_id)
    if mutation is None:
        raise TerminalMutationError(
            "MUTATION_NOT_DECLARED",
            "mutation workflow is not declared",
        )
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator(mutation.handler.input_schema).validate(
            workflow_input
        )
    except Exception as exc:
        raise TerminalMutationError(
            "MUTATION_INPUT_INVALID",
            "mutation input failed validation",
        ) from exc
    input_sha256 = hashlib.sha256(_canonical(workflow_input)).hexdigest()
    invocation_key = _invocation_key(
        specialist_id,
        conversation_id,
        event_id,
    )
    operation_id = _operation_id(invocation_key)
    journal_path = (
        root / MUTATION_JOURNAL_RELATIVE_PATH / f"{invocation_key}.json"
    )
    binding = {
        "specialist_id": specialist_id,
        "conversation_id": conversation_id,
        "event_id": event_id,
        "workflow_id": mutation.workflow_id,
        "workflow_version": mutation.version,
        "input_sha256": input_sha256,
        "handler_sha256": mutation.handler.handler_sha256,
        "mapper_sha256": (
            mutation.mapper.handler_sha256 if mutation.mapper else None
        ),
        "presenter_id": mutation.presenter_id,
        "presenter_version": mutation.presenter_version,
        "presenter_sha256": mutation.presenter_sha256,
    }
    if journal_path.exists():
        try:
            record = json.loads(journal_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise TerminalMutationError(
                "MUTATION_JOURNAL_UNAVAILABLE",
                "mutation journal is unreadable",
                sealed=True,
            ) from exc
        if any(record.get(key) != value for key, value in binding.items()):
            raise TerminalMutationError(
                "MUTATION_INVOCATION_CONFLICT",
                "accepted event is already bound to another mutation",
                sealed=True,
            )
        outcome = record.get("execution_state")
        if outcome in {"committed", "not_applied", "conflict"}:
            result = record.get("result")
            if not isinstance(result, dict):
                raise TerminalMutationError(
                    "MUTATION_JOURNAL_UNAVAILABLE",
                    "mutation journal result is invalid",
                    sealed=True,
                )
            return _prepare_mutation_presentation(
                root,
                mutation,
                record,
                journal_path,
                result,
            )
        raise TerminalMutationError(
            "MUTATION_OUTCOME_UNKNOWN",
            "mutation outcome is unknown; inspect authoritative domain state",
            sealed=True,
        )
    record: dict[str, Any] = {
        "schema_version": "hermes.terminal_mutation.journal.v1",
        "operation_id": operation_id,
        **binding,
        "execution_state": "intent_recorded",
        "presentation_state": "not_started",
        "created_at": _now(),
        "updated_at": _now(),
    }
    _atomic_json(journal_path, record)
    record["execution_state"] = "executing"
    record["updated_at"] = _now()
    _atomic_json(journal_path, record)
    invocation = {
        "schema_version": MUTATION_INVOKE_SCHEMA,
        "operation_id": operation_id,
        "workflow_id": mutation.workflow_id,
        "workflow_version": mutation.version,
        "input": workflow_input,
    }
    permissive_input = {
        "type": "object",
        "additionalProperties": True,
    }
    try:
        handler_result = _run_program(
            root,
            replace(mutation.handler, input_schema=permissive_input),
            invocation,
            label="mutation handler",
        )
        outcome = handler_result.get("outcome")
        result = handler_result.get("result")
        if (
            outcome not in {"committed", "not_applied", "conflict"}
            or not isinstance(result, dict)
        ):
            raise TerminalMutationError(
                "MUTATION_OUTPUT_INVALID",
                "mutation handler outcome is invalid",
                sealed=True,
            )
    except Exception as exc:
        record["execution_state"] = "outcome_unknown"
        record["updated_at"] = _now()
        try:
            _atomic_json(journal_path, record)
        except OSError:
            pass
        if isinstance(exc, TerminalMutationError):
            raise
        if isinstance(exc, TerminalWorkflowError):
            code = exc.code.replace("WORKFLOW", "MUTATION")
            message = str(exc)
        else:
            code = "MUTATION_OUTCOME_UNKNOWN"
            message = "mutation outcome is unknown"
        raise TerminalMutationError(code, message, sealed=True) from exc
    record["execution_state"] = outcome
    record["result"] = result
    record["updated_at"] = _now()
    try:
        _atomic_json(journal_path, record)
    except OSError as exc:
        raise TerminalMutationError(
            "MUTATION_OUTCOME_UNKNOWN",
            "mutation result could not be persisted",
            sealed=True,
        ) from exc
    return _prepare_mutation_presentation(
        root,
        mutation,
        record,
        journal_path,
        result,
    )


def _prepare_mutation_presentation(
    root: Path,
    mutation: TerminalMutation,
    record: dict[str, Any],
    journal_path: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    record["presentation_state"] = "preparing"
    record["updated_at"] = _now()
    try:
        _atomic_json(journal_path, record)
    except OSError as exc:
        raise TerminalMutationError(
            "MUTATION_PRESENTATION_FAILED",
            "mutation presentation state could not be persisted",
            sealed=True,
        ) from exc
    try:
        presenter_input = (
            result
            if mutation.mapper is None
            else _run_program(
                root,
                mutation.mapper,
                result,
                label="mutation mapper",
            )
        )
    except Exception as exc:
        record["presentation_state"] = "presentation_failed"
        record["updated_at"] = _now()
        try:
            _atomic_json(journal_path, record)
        except OSError:
            pass
        raise TerminalMutationError(
            "MUTATION_PRESENTATION_FAILED",
            str(exc),
            sealed=True,
        ) from exc
    if not isinstance(presenter_input, dict):
        record["presentation_state"] = "presentation_failed"
        record["updated_at"] = _now()
        try:
            _atomic_json(journal_path, record)
        except OSError:
            pass
        raise TerminalMutationError(
            "MUTATION_PRESENTATION_FAILED",
            "mutation presenter input must be an object",
            sealed=True,
        )
    return {
        "outcome": record["execution_state"],
        "result": result,
        "operation_id": record["operation_id"],
        "workflow_id": mutation.workflow_id,
        "workflow_version": mutation.version,
        "handler_sha256": mutation.handler.handler_sha256,
        "mapper_sha256": (
            mutation.mapper.handler_sha256 if mutation.mapper else None
        ),
        "presenter_id": mutation.presenter_id,
        "presenter_input": presenter_input,
        "journal_path": str(journal_path),
    }


def record_terminal_mutation_presentation(
    profile_root: Path,
    journal_path: str,
    *,
    artifact: dict[str, Any] | None = None,
    error_code: str | None = None,
) -> None:
    root = profile_root.resolve()
    path = Path(journal_path).resolve()
    directory = (root / MUTATION_JOURNAL_RELATIVE_PATH).resolve()
    if path.parent != directory:
        raise TerminalMutationError(
            "MUTATION_JOURNAL_UNAVAILABLE",
            "mutation journal path escapes runtime state",
            sealed=True,
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    if artifact is not None:
        record["presentation_state"] = "presented"
        record["artifact_sha256"] = artifact.get("sha256")
        record["presenter_version"] = artifact.get("presenter_version")
        record["presenter_sha256"] = artifact.get("presenter_sha256")
    else:
        record["presentation_state"] = "presentation_failed"
        record["presentation_error_code"] = error_code
    record["updated_at"] = _now()
    _atomic_json(path, record)


def invoke_terminal_mutation_endpoint(
    workflow_id: str,
    workflow_input: dict[str, Any],
    *,
    profile_root: Path | None = None,
) -> str:
    root = (profile_root or get_hermes_home()).resolve()
    try:
        endpoint = json.loads(
            (root / ENDPOINT_RELATIVE_PATH).read_text(encoding="utf-8")
        )
        socket_path = (root / endpoint["socket"]).resolve()
        turn_token = endpoint["turn_token"]
    except Exception as exc:
        raise TerminalMutationError(
            "MUTATION_TURN_UNAVAILABLE",
            "no eligible persistent mutation turn is active",
        ) from exc
    if root not in socket_path.parents:
        raise TerminalMutationError(
            "MUTATION_ENDPOINT_INVALID",
            "mutation endpoint escapes profile",
        )
    request = {
        "protocol_version": PRESENTER_PROTOCOL,
        "operation": "mutation",
        "turn_token": turn_token,
        "workflow_id": workflow_id,
        "input": workflow_input,
    }
    raw = (
        json.dumps(
            request,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(MAX_PRESENTER_TIMEOUT_SECONDS * 3 + 2)
            client.connect(str(socket_path))
            stream = client.makefile("rwb", buffering=0)
            stream.write(raw)
            response_raw = stream.readline(MAX_PRESENTER_WIRE_BYTES + 1)
        response = json.loads(response_raw)
    except Exception as exc:
        raise TerminalMutationError(
            "MUTATION_ENDPOINT_UNAVAILABLE",
            "mutation endpoint failed",
        ) from exc
    if (
        response.get("status") == "completed"
        and isinstance(response.get("content"), str)
    ):
        return response["content"]
    error = response.get("error") or {}
    raise TerminalMutationError(
        str(error.get("code") or "MUTATION_FAILED"),
        str(error.get("message") or "mutation failed"),
        sealed=response.get("sealed") is True,
    )
