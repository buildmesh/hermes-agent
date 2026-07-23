"""Declared deterministic presenters for persistent specialist turns."""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


MANIFEST_RELATIVE_PATH = Path("contract/terminal-presenters.json")
ENDPOINT_RELATIVE_PATH = Path("state/persistent-runtime/presenter-endpoint.json")
PRESENTER_PROTOCOL = "hermes.terminal_presenter.v1"
MAX_PRESENTER_OUTPUT_BYTES = 512 * 1024
MAX_PRESENTER_INPUT_BYTES = 512 * 1024
MAX_PRESENTER_STDERR_BYTES = 16 * 1024
MAX_PRESENTER_TIMEOUT_SECONDS = 30
MAX_PRESENTER_WIRE_BYTES = 1024 * 1024
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){2}(?:[-+][0-9A-Za-z.-]+)?$")
_SHA_RE = re.compile(r"^sha256:([0-9a-f]{64})$")


class TerminalPresenterError(RuntimeError):
    """Typed presenter configuration, invocation, or transport failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class TerminalPresenter:
    presenter_id: str
    version: str
    description: str
    handler: Path
    handler_sha256: str
    input_schema: dict[str, Any]
    input_schema_sha256: str
    command: tuple[str, ...]
    timeout_seconds: int
    max_output_bytes: int


def _confined_file(profile_root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise TerminalPresenterError("PRESENTER_CONFIG_INVALID", f"{label} must be a relative path")
    root = profile_root.resolve()
    path = (root / raw).resolve()
    if path == root or root not in path.parents or not path.is_file():
        raise TerminalPresenterError("PRESENTER_CONFIG_INVALID", f"{label} is not a confined file")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_sha256(raw: Any, label: str) -> str:
    if not isinstance(raw, str):
        raise TerminalPresenterError("PRESENTER_CONFIG_INVALID", f"{label} digest is required")
    match = _SHA_RE.fullmatch(raw)
    if match is None:
        raise TerminalPresenterError("PRESENTER_CONFIG_INVALID", f"{label} digest is invalid")
    return match.group(1)


def _contains_remote_ref(value: Any) -> bool:
    if isinstance(value, dict):
        for keyword in ("$ref", "$dynamicRef"):
            ref = value.get(keyword)
            if isinstance(ref, str) and not ref.startswith("#"):
                return True
        return any(_contains_remote_ref(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_remote_ref(item) for item in value)
    return False


def _object_schemas_are_closed(value: Any) -> bool:
    if isinstance(value, dict):
        if (
            value.get("type") == "object" or "properties" in value or "patternProperties" in value
        ) and value.get("additionalProperties") is not False:
            return False
        return all(_object_schemas_are_closed(item) for item in value.values())
    if isinstance(value, list):
        return all(_object_schemas_are_closed(item) for item in value)
    return True


def load_terminal_presenters(profile_root: Path) -> dict[str, TerminalPresenter]:
    """Load and fully validate the active profile's presenter declaration."""
    root = profile_root.resolve()
    manifest_path = root / MANIFEST_RELATIVE_PATH
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalPresenterError("PRESENTER_CONFIG_INVALID", "presenter manifest is unreadable") from exc
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "presenters"}
        or manifest.get("schema_version") != "telegram.bridge.terminal_presenters.v1"
        or not isinstance(manifest.get("presenters"), list)
        or not 1 <= len(manifest["presenters"]) <= 32
    ):
        raise TerminalPresenterError("PRESENTER_CONFIG_INVALID", "presenter manifest shape is invalid")

    presenters: dict[str, TerminalPresenter] = {}
    allowed = {
        "presenter_id", "version", "description", "handler", "sha256",
        "input_schema", "input_schema_sha256", "command", "timeout_seconds",
        "max_output_bytes",
    }
    for raw in manifest["presenters"]:
        if not isinstance(raw, dict) or set(raw) != allowed:
            raise TerminalPresenterError("PRESENTER_CONFIG_INVALID", "presenter declaration shape is invalid")
        presenter_id = raw.get("presenter_id")
        version = raw.get("version")
        description = raw.get("description")
        if not isinstance(presenter_id, str) or _ID_RE.fullmatch(presenter_id) is None:
            raise TerminalPresenterError("PRESENTER_CONFIG_INVALID", "presenter_id is invalid")
        if presenter_id in presenters:
            raise TerminalPresenterError("PRESENTER_CONFIG_INVALID", "presenter_id is duplicated")
        if not isinstance(version, str) or _VERSION_RE.fullmatch(version) is None:
            raise TerminalPresenterError("PRESENTER_CONFIG_INVALID", "presenter version is invalid")
        if not isinstance(description, str) or not 1 <= len(description) <= 500:
            raise TerminalPresenterError("PRESENTER_CONFIG_INVALID", "presenter description is invalid")

        handler = _confined_file(root, raw.get("handler"), "handler")
        handler_sha256 = _declared_sha256(raw.get("sha256"), "handler")
        if _file_sha256(handler) != handler_sha256:
            raise TerminalPresenterError("PRESENTER_DIGEST_MISMATCH", "presenter handler digest mismatch")

        schema_path = _confined_file(root, raw.get("input_schema"), "input_schema")
        input_schema_sha256 = _declared_sha256(raw.get("input_schema_sha256"), "input schema")
        if _file_sha256(schema_path) != input_schema_sha256:
            raise TerminalPresenterError("PRESENTER_DIGEST_MISMATCH", "presenter input schema digest mismatch")
        try:
            input_schema = json.loads(schema_path.read_text(encoding="utf-8"))
            from jsonschema import Draft202012Validator

            Draft202012Validator.check_schema(input_schema)
        except Exception as exc:
            raise TerminalPresenterError("PRESENTER_CONFIG_INVALID", "presenter input schema is invalid") from exc
        if (
            not isinstance(input_schema, dict)
            or _contains_remote_ref(input_schema)
            or not _object_schemas_are_closed(input_schema)
        ):
            raise TerminalPresenterError(
                "PRESENTER_CONFIG_INVALID",
                "presenter input schema must be self-contained",
            )

        command = raw.get("command")
        if (
            not isinstance(command, list)
            or not 1 <= len(command) <= 32
            or any(not isinstance(part, str) or not part or len(part) > 1024 for part in command)
            or command.count("{handler}") != 1
            or any(("{" in part or "}" in part) and part != "{handler}" for part in command)
        ):
            raise TerminalPresenterError("PRESENTER_CONFIG_INVALID", "presenter command is invalid")
        timeout_seconds = raw.get("timeout_seconds")
        max_output_bytes = raw.get("max_output_bytes")
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= MAX_PRESENTER_TIMEOUT_SECONDS:
            raise TerminalPresenterError("PRESENTER_CONFIG_INVALID", "presenter timeout is invalid")
        if type(max_output_bytes) is not int or not 256 <= max_output_bytes <= MAX_PRESENTER_OUTPUT_BYTES:
            raise TerminalPresenterError("PRESENTER_CONFIG_INVALID", "presenter output limit is invalid")

        presenters[presenter_id] = TerminalPresenter(
            presenter_id=presenter_id,
            version=version,
            description=description,
            handler=handler,
            handler_sha256=handler_sha256,
            input_schema=input_schema,
            input_schema_sha256=input_schema_sha256,
            command=tuple(str(handler) if part == "{handler}" else part for part in command),
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
    return presenters


def terminal_presenters_available(profile_root: Path | None = None) -> bool:
    root = (profile_root or get_hermes_home()).resolve()
    try:
        endpoint = json.loads((root / ENDPOINT_RELATIVE_PATH).read_text(encoding="utf-8"))
        return (
            isinstance(endpoint.get("turn_token"), str)
            and bool(endpoint["turn_token"])
            and bool(load_terminal_presenters(root))
        )
    except TerminalPresenterError:
        return False
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def presenter_tool_description(profile_root: Path | None = None) -> str:
    """Build stable agent-facing metadata from the installed declaration."""
    try:
        presenters = load_terminal_presenters((profile_root or get_hermes_home()).resolve())
    except TerminalPresenterError:
        presenters = {}
    entries = "; ".join(
        f"{item.presenter_id}: {item.description} Input schema: "
        f"{json.dumps(item.input_schema, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"
        for item in presenters.values()
    )
    return (
        "Finalize this persistent Telegram specialist turn with one declared deterministic presenter. "
        "Call this only after all domain work and as the final tool invocation. Hermes sends the "
        "presenter's exact output to TBA; do not reproduce it in your final prose."
        + (f" Available presenters: {entries}" if entries else "")
    )


def _kill_presenter_process(process: subprocess.Popen[bytes]) -> None:
    if os.name != "nt":
        try:
            os.killpg(process.pid, 9)
            return
        except ProcessLookupError:
            return
    if process.poll() is None:
        process.kill()


def execute_terminal_presenter(
    profile_root: Path,
    presenter_id: str,
    presenter_input: dict[str, Any],
) -> dict[str, Any]:
    """Validate input and execute one declared presenter with bounded output."""
    presenters = load_terminal_presenters(profile_root)
    presenter = presenters.get(presenter_id)
    if presenter is None:
        raise TerminalPresenterError("PRESENTER_NOT_DECLARED", "presenter is not declared")
    if not isinstance(presenter_input, dict):
        raise TerminalPresenterError("PRESENTER_INPUT_INVALID", "presenter input must be an object")
    try:
        from jsonschema import Draft202012Validator

        Draft202012Validator(presenter.input_schema).validate(presenter_input)
    except Exception as exc:
        raise TerminalPresenterError("PRESENTER_INPUT_INVALID", "presenter input failed validation") from exc
    input_bytes = json.dumps(
        presenter_input,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(input_bytes) > MAX_PRESENTER_INPUT_BYTES:
        raise TerminalPresenterError("PRESENTER_INPUT_TOO_LARGE", "presenter input exceeds the limit")

    from tools.environments.local import hermes_subprocess_env

    with tempfile.TemporaryFile() as stdin_file:
        stdin_file.write(input_bytes)
        stdin_file.seek(0)
        process = subprocess.Popen(
            list(presenter.command),
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
        deadline = time.monotonic() + presenter.timeout_seconds
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _kill_presenter_process(process)
                    raise TerminalPresenterError("PRESENTER_TIMEOUT", "presenter exceeded its timeout")
                events = selector.select(timeout=min(0.1, remaining))
                for key, _mask in events:
                    try:
                        chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "stdout":
                        stdout.extend(chunk)
                        if len(stdout) > presenter.max_output_bytes:
                            _kill_presenter_process(process)
                            raise TerminalPresenterError(
                                "PRESENTER_OUTPUT_TOO_LARGE",
                                "presenter output exceeds the declared limit",
                            )
                    elif len(stderr) < MAX_PRESENTER_STDERR_BYTES:
                        stderr.extend(chunk[: MAX_PRESENTER_STDERR_BYTES - len(stderr)])
            remaining = max(0.01, deadline - time.monotonic())
            return_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            _kill_presenter_process(process)
            raise TerminalPresenterError("PRESENTER_TIMEOUT", "presenter exceeded its timeout") from exc
        finally:
            selector.close()
            _kill_presenter_process(process)
            process.wait()

    if return_code != 0:
        raise TerminalPresenterError("PRESENTER_FAILED", f"presenter exited with status {return_code}")
    if not stdout:
        raise TerminalPresenterError("PRESENTER_OUTPUT_EMPTY", "presenter returned no output")
    try:
        content = bytes(stdout).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TerminalPresenterError("PRESENTER_OUTPUT_INVALID_UTF8", "presenter output is not UTF-8") from exc
    return {
        "content": content,
        "sha256": hashlib.sha256(bytes(stdout)).hexdigest(),
        "presenter_id": presenter.presenter_id,
        "presenter_version": presenter.version,
        "presenter_sha256": presenter.handler_sha256,
    }


def invoke_presenter_endpoint(
    presenter_id: str,
    presenter_input: dict[str, Any],
    *,
    profile_root: Path | None = None,
) -> str:
    """Invoke the active worker's presenter endpoint from a Hermes tool process."""
    root = (profile_root or get_hermes_home()).resolve()
    endpoint_path = root / ENDPOINT_RELATIVE_PATH
    try:
        endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
        relative_socket = endpoint["socket"]
        turn_token = endpoint["turn_token"]
    except Exception as exc:
        raise TerminalPresenterError(
            "PRESENTER_TURN_UNAVAILABLE",
            "no eligible persistent presenter turn is active",
        ) from exc
    socket_path = (root / relative_socket).resolve()
    if root not in socket_path.parents:
        raise TerminalPresenterError("PRESENTER_ENDPOINT_INVALID", "presenter endpoint escapes profile")
    request = {
        "protocol_version": PRESENTER_PROTOCOL,
        "operation": "present",
        "turn_token": turn_token,
        "presenter_id": presenter_id,
        "input": presenter_input,
    }
    raw_request = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(raw_request) > MAX_PRESENTER_INPUT_BYTES + 4096:
        raise TerminalPresenterError("PRESENTER_INPUT_TOO_LARGE", "presenter request exceeds the limit")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(MAX_PRESENTER_TIMEOUT_SECONDS + 2)
            client.connect(str(socket_path))
            stream = client.makefile("rwb", buffering=0)
            stream.write(raw_request)
            raw_response = stream.readline(MAX_PRESENTER_WIRE_BYTES + 1)
    except OSError as exc:
        raise TerminalPresenterError("PRESENTER_ENDPOINT_UNAVAILABLE", "presenter endpoint is unavailable") from exc
    if (
        not raw_response
        or len(raw_response) > MAX_PRESENTER_WIRE_BYTES
        or not raw_response.endswith(b"\n")
    ):
        raise TerminalPresenterError("PRESENTER_ENDPOINT_INVALID", "presenter endpoint returned no complete response")
    try:
        response = json.loads(raw_response)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TerminalPresenterError("PRESENTER_ENDPOINT_INVALID", "presenter endpoint returned invalid JSON") from exc
    if response.get("status") != "completed" or not isinstance(response.get("content"), str):
        error = response.get("error") or {}
        raise TerminalPresenterError(
            str(error.get("code") or "PRESENTER_FAILED"),
            str(error.get("message") or "presenter invocation failed"),
        )
    return response["content"]
