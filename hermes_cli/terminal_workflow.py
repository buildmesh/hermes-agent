"""Profile-declared producer-to-presenter workflows for persistent turns."""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home
from hermes_cli.terminal_presenter import (
    ENDPOINT_RELATIVE_PATH,
    MAX_PRESENTER_INPUT_BYTES,
    MAX_PRESENTER_STDERR_BYTES,
    MAX_PRESENTER_TIMEOUT_SECONDS,
    MAX_PRESENTER_WIRE_BYTES,
    PRESENTER_PROTOCOL,
    TerminalPresenterError,
    _confined_file,
    _contains_remote_ref,
    _declared_sha256,
    _file_sha256,
    _ID_RE,
    _object_schemas_are_closed,
    _VERSION_RE,
    load_terminal_presenters,
)

WORKFLOW_MANIFEST_RELATIVE_PATH = Path("contract/terminal-workflows.json")
MAX_WORKFLOW_OUTPUT_BYTES = 512 * 1024


@dataclass(frozen=True)
class DeclaredProgram:
    command: tuple[str, ...]
    handler_sha256: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    timeout_seconds: int
    max_output_bytes: int


@dataclass(frozen=True)
class TerminalWorkflow:
    workflow_id: str
    version: str
    description: str
    read_only: bool
    producer: DeclaredProgram
    mapper: DeclaredProgram | None
    presenter_id: str


class TerminalWorkflowError(TerminalPresenterError):
    """Typed workflow declaration, execution, or transport failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(code, message)
        self.checkpoint = checkpoint


def _schema(
    root: Path,
    raw_path: Any,
    raw_digest: Any,
    label: str,
) -> tuple[dict[str, Any], str]:
    path = _confined_file(root, raw_path, label)
    digest = _declared_sha256(raw_digest, label)
    if _file_sha256(path) != digest:
        raise TerminalWorkflowError(
            "WORKFLOW_DIGEST_MISMATCH",
            f"{label} digest mismatch",
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        from jsonschema import Draft202012Validator

        Draft202012Validator.check_schema(value)
    except Exception as exc:
        raise TerminalWorkflowError(
            "WORKFLOW_CONFIG_INVALID",
            f"{label} is invalid",
        ) from exc
    if (
        not isinstance(value, dict)
        or _contains_remote_ref(value)
        or not _object_schemas_are_closed(value)
    ):
        raise TerminalWorkflowError(
            "WORKFLOW_CONFIG_INVALID",
            f"{label} must be closed and self-contained",
        )
    return value, digest


def _program(root: Path, raw: Any, label: str) -> DeclaredProgram:
    allowed = {
        "handler",
        "sha256",
        "command",
        "input_schema",
        "input_schema_sha256",
        "output_schema",
        "output_schema_sha256",
        "timeout_seconds",
        "max_output_bytes",
    }
    if not isinstance(raw, dict) or set(raw) != allowed:
        raise TerminalWorkflowError(
            "WORKFLOW_CONFIG_INVALID",
            f"{label} declaration shape is invalid",
        )
    handler = _confined_file(root, raw.get("handler"), f"{label} handler")
    handler_sha256 = _declared_sha256(raw.get("sha256"), f"{label} handler")
    if _file_sha256(handler) != handler_sha256:
        raise TerminalWorkflowError(
            "WORKFLOW_DIGEST_MISMATCH",
            f"{label} handler digest mismatch",
        )
    input_schema, _ = _schema(
        root,
        raw.get("input_schema"),
        raw.get("input_schema_sha256"),
        f"{label} input schema",
    )
    output_schema, _ = _schema(
        root,
        raw.get("output_schema"),
        raw.get("output_schema_sha256"),
        f"{label} output schema",
    )
    command = raw.get("command")
    if (
        not isinstance(command, list)
        or not 1 <= len(command) <= 32
        or any(not isinstance(part, str) or not part or len(part) > 1024 for part in command)
        or command.count("{handler}") != 1
        or any(("{" in part or "}" in part) and part != "{handler}" for part in command)
    ):
        raise TerminalWorkflowError(
            "WORKFLOW_CONFIG_INVALID",
            f"{label} command is invalid",
        )
    timeout = raw.get("timeout_seconds")
    output_limit = raw.get("max_output_bytes")
    if type(timeout) is not int or not 1 <= timeout <= MAX_PRESENTER_TIMEOUT_SECONDS:
        raise TerminalWorkflowError("WORKFLOW_CONFIG_INVALID", f"{label} timeout is invalid")
    if type(output_limit) is not int or not 256 <= output_limit <= MAX_WORKFLOW_OUTPUT_BYTES:
        raise TerminalWorkflowError(
            "WORKFLOW_CONFIG_INVALID",
            f"{label} output limit is invalid",
        )
    return DeclaredProgram(
        command=tuple(str(handler) if part == "{handler}" else part for part in command),
        handler_sha256=handler_sha256,
        input_schema=input_schema,
        output_schema=output_schema,
        timeout_seconds=timeout,
        max_output_bytes=output_limit,
    )


def load_terminal_workflows(profile_root: Path) -> dict[str, TerminalWorkflow]:
    root = profile_root.resolve()
    try:
        manifest = json.loads(
            (root / WORKFLOW_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalWorkflowError(
            "WORKFLOW_CONFIG_INVALID",
            "workflow manifest is unreadable",
        ) from exc
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "workflows"}
        or manifest.get("schema_version") != "telegram.bridge.terminal_workflows.v1"
        or not isinstance(manifest.get("workflows"), list)
        or not 1 <= len(manifest["workflows"]) <= 32
    ):
        raise TerminalWorkflowError(
            "WORKFLOW_CONFIG_INVALID",
            "workflow manifest shape is invalid",
        )
    presenters = load_terminal_presenters(root)
    workflows: dict[str, TerminalWorkflow] = {}
    allowed = {
        "workflow_id",
        "version",
        "description",
        "read_only",
        "producer",
        "mapper",
        "presenter_id",
    }
    for raw in manifest["workflows"]:
        if not isinstance(raw, dict) or set(raw) != allowed:
            raise TerminalWorkflowError(
                "WORKFLOW_CONFIG_INVALID",
                "workflow declaration shape is invalid",
            )
        workflow_id = raw.get("workflow_id")
        version = raw.get("version")
        description = raw.get("description")
        presenter_id = raw.get("presenter_id")
        if not isinstance(workflow_id, str) or _ID_RE.fullmatch(workflow_id) is None:
            raise TerminalWorkflowError("WORKFLOW_CONFIG_INVALID", "workflow_id is invalid")
        if workflow_id in workflows:
            raise TerminalWorkflowError("WORKFLOW_CONFIG_INVALID", "workflow_id is duplicated")
        if not isinstance(version, str) or _VERSION_RE.fullmatch(version) is None:
            raise TerminalWorkflowError("WORKFLOW_CONFIG_INVALID", "workflow version is invalid")
        if not isinstance(description, str) or not 1 <= len(description) <= 500:
            raise TerminalWorkflowError("WORKFLOW_CONFIG_INVALID", "workflow description is invalid")
        if raw.get("read_only") is not True:
            raise TerminalWorkflowError(
                "WORKFLOW_CONFIG_INVALID",
                "terminal_workflows.v1 supports only read_only workflows",
            )
        if not isinstance(presenter_id, str) or presenter_id not in presenters:
            raise TerminalWorkflowError(
                "WORKFLOW_CONFIG_INVALID",
                "workflow presenter_id is not declared",
            )
        workflows[workflow_id] = TerminalWorkflow(
            workflow_id=workflow_id,
            version=version,
            description=description,
            read_only=True,
            producer=_program(root, raw.get("producer"), "producer"),
            mapper=(
                None
                if raw.get("mapper") is None
                else _program(root, raw.get("mapper"), "mapper")
            ),
            presenter_id=presenter_id,
        )
    return workflows


def terminal_workflows_available(profile_root: Path | None = None) -> bool:
    root = (profile_root or get_hermes_home()).resolve()
    try:
        endpoint = json.loads((root / ENDPOINT_RELATIVE_PATH).read_text(encoding="utf-8"))
        return (
            isinstance(endpoint.get("turn_token"), str)
            and bool(endpoint["turn_token"])
            and endpoint.get("terminal_workflow") is True
            and bool(load_terminal_workflows(root))
        )
    except (TerminalPresenterError, OSError, ValueError, TypeError, AttributeError):
        return False


def terminal_workflow_description(profile_root: Path | None = None) -> str:
    try:
        workflows = load_terminal_workflows(
            (profile_root or get_hermes_home()).resolve()
        )
    except TerminalPresenterError:
        workflows = {}
    entries = "; ".join(
        f"{workflow.workflow_id}: {workflow.description} Input schema: "
        f"{json.dumps(workflow.producer.input_schema, ensure_ascii=False, sort_keys=True)}"
        for workflow in workflows.values()
    )
    return (
        "Run one installed read-only terminal workflow. The runtime executes its "
        "digest-bound producer once, validates the result, applies the optional "
        "profile-owned mapper, and invokes the bound terminal presenter without "
        "another model orchestration decision. This must be the final, unbatched "
        "tool call. "
        + (f"Available workflows: {entries}" if entries else "")
    )


def _kill(process: subprocess.Popen[bytes]) -> None:
    if os.name != "nt":
        try:
            os.killpg(process.pid, 9)
            return
        except ProcessLookupError:
            return
    if process.poll() is None:
        process.kill()


def _run_program(
    profile_root: Path,
    program: DeclaredProgram,
    value: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator(program.input_schema).validate(value)
    except Exception as exc:
        raise TerminalWorkflowError(
            "WORKFLOW_INPUT_INVALID",
            f"{label} input failed validation",
        ) from exc
    input_bytes = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(input_bytes) > MAX_PRESENTER_INPUT_BYTES:
        raise TerminalWorkflowError("WORKFLOW_INPUT_TOO_LARGE", f"{label} input exceeds the limit")
    from tools.environments.local import hermes_subprocess_env

    with tempfile.TemporaryFile() as stdin_file:
        stdin_file.write(input_bytes)
        stdin_file.seek(0)
        process = subprocess.Popen(
            list(program.command),
            cwd=str(profile_root),
            env=hermes_subprocess_env(),
            stdin=stdin_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=os.name != "nt",
        )
        assert process.stdout is not None and process.stderr is not None
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        stdout = bytearray()
        stderr = bytearray()
        deadline = time.monotonic() + program.timeout_seconds
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _kill(process)
                    raise TerminalWorkflowError("WORKFLOW_TIMEOUT", f"{label} exceeded its timeout")
                for key, _mask in selector.select(timeout=min(0.1, remaining)):
                    try:
                        chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                    elif key.data == "stdout":
                        stdout.extend(chunk)
                        if len(stdout) > program.max_output_bytes:
                            _kill(process)
                            raise TerminalWorkflowError(
                                "WORKFLOW_OUTPUT_TOO_LARGE",
                                f"{label} output exceeds the declared limit",
                            )
                    elif len(stderr) < MAX_PRESENTER_STDERR_BYTES:
                        stderr.extend(chunk[: MAX_PRESENTER_STDERR_BYTES - len(stderr)])
            return_code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            _kill(process)
            raise TerminalWorkflowError("WORKFLOW_TIMEOUT", f"{label} exceeded its timeout") from exc
        finally:
            selector.close()
            _kill(process)
            process.wait()
    if return_code != 0:
        raise TerminalWorkflowError("WORKFLOW_FAILED", f"{label} exited with status {return_code}")
    try:
        result = json.loads(bytes(stdout).decode("utf-8"))
        from jsonschema import Draft202012Validator

        Draft202012Validator(program.output_schema).validate(result)
    except Exception as exc:
        raise TerminalWorkflowError(
            "WORKFLOW_OUTPUT_INVALID",
            f"{label} output failed validation",
        ) from exc
    if not isinstance(result, dict):
        raise TerminalWorkflowError("WORKFLOW_OUTPUT_INVALID", f"{label} output must be an object")
    return result


def execute_terminal_workflow(
    profile_root: Path,
    workflow_id: str,
    workflow_input: dict[str, Any],
    *,
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workflow = load_terminal_workflows(profile_root).get(workflow_id)
    if workflow is None:
        raise TerminalWorkflowError("WORKFLOW_NOT_DECLARED", "workflow is not declared")
    input_sha256 = hashlib.sha256(
        json.dumps(
            workflow_input,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if checkpoint is None:
        producer_result = _run_program(
            profile_root,
            workflow.producer,
            workflow_input,
            label="producer",
        )
        outcome = producer_result.get("outcome")
        if outcome == "continue":
            return {
                "outcome": "continue",
                "model_result": producer_result.get("model_result"),
                "workflow_id": workflow.workflow_id,
                "workflow_version": workflow.version,
                "producer_sha256": workflow.producer.handler_sha256,
            }
        if outcome != "terminal_ready":
            raise TerminalWorkflowError("WORKFLOW_OUTPUT_INVALID", "producer outcome is invalid")
        domain_result = producer_result.get("result")
        if not isinstance(domain_result, dict):
            raise TerminalWorkflowError("WORKFLOW_OUTPUT_INVALID", "terminal result must be an object")
        checkpoint = {
            "workflow_id": workflow.workflow_id,
            "domain_result": domain_result,
            "input_sha256": input_sha256,
        }
    elif (
        checkpoint.get("workflow_id") != workflow.workflow_id
        or not isinstance(checkpoint.get("domain_result"), dict)
        or checkpoint.get("input_sha256") != input_sha256
    ):
        raise TerminalWorkflowError(
            "WORKFLOW_RETRY_MISMATCH",
            "workflow retry input differs from the completed producer input",
        )
    domain_result = checkpoint["domain_result"]
    try:
        presenter_input = (
            domain_result
            if workflow.mapper is None
            else _run_program(profile_root, workflow.mapper, domain_result, label="mapper")
        )
    except TerminalWorkflowError as exc:
        exc.checkpoint = checkpoint
        raise
    if not isinstance(presenter_input, dict):
        raise TerminalWorkflowError("WORKFLOW_OUTPUT_INVALID", "presenter input must be an object")
    return {
        "outcome": "terminal_ready",
        "presenter_id": workflow.presenter_id,
        "presenter_input": presenter_input,
        "workflow_id": workflow.workflow_id,
        "workflow_version": workflow.version,
        "producer_sha256": workflow.producer.handler_sha256,
        "mapper_sha256": workflow.mapper.handler_sha256 if workflow.mapper else None,
        "checkpoint": checkpoint,
    }


def invoke_terminal_workflow_endpoint(
    workflow_id: str,
    workflow_input: dict[str, Any],
    *,
    profile_root: Path | None = None,
) -> str:
    root = (profile_root or get_hermes_home()).resolve()
    try:
        endpoint = json.loads((root / ENDPOINT_RELATIVE_PATH).read_text(encoding="utf-8"))
        socket_path = (root / endpoint["socket"]).resolve()
        turn_token = endpoint["turn_token"]
    except Exception as exc:
        raise TerminalWorkflowError(
            "WORKFLOW_TURN_UNAVAILABLE",
            "no eligible persistent workflow turn is active",
        ) from exc
    if root not in socket_path.parents:
        raise TerminalWorkflowError("WORKFLOW_ENDPOINT_INVALID", "workflow endpoint escapes profile")
    request = {
        "protocol_version": PRESENTER_PROTOCOL,
        "operation": "workflow",
        "turn_token": turn_token,
        "workflow_id": workflow_id,
        "input": workflow_input,
    }
    raw = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(MAX_PRESENTER_TIMEOUT_SECONDS * 3 + 2)
            client.connect(str(socket_path))
            stream = client.makefile("rwb", buffering=0)
            stream.write(raw)
            response_raw = stream.readline(MAX_PRESENTER_WIRE_BYTES + 1)
        response = json.loads(response_raw)
    except Exception as exc:
        raise TerminalWorkflowError("WORKFLOW_ENDPOINT_UNAVAILABLE", "workflow endpoint failed") from exc
    if response.get("status") == "completed" and isinstance(response.get("content"), str):
        return response["content"]
    if response.get("status") == "continue":
        return json.dumps(
            {
                "outcome": "continue",
                "model_result": response.get("model_result"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    error = response.get("error") or {}
    raise TerminalWorkflowError(
        str(error.get("code") or "WORKFLOW_FAILED"),
        str(error.get("message") or "workflow failed"),
    )
