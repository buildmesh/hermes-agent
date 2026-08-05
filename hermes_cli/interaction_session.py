"""Profile-scoped interaction-session declaration and turn endpoint client."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


CAPABILITY = "specialist_interaction_session.v1"
ENDPOINT_RELATIVE_PATH = Path("state/persistent-runtime/interaction-session-endpoint.json")
PROTOCOL = "hermes.interaction_session_endpoint.v1"
MAX_STATE_BYTES = 4096
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SESSION_ID_RE = re.compile(r"^sctx_[A-Za-z0-9._:-]{1,200}$")
_REFERENCE_KEYWORDS = {"$ref", "$dynamicRef", "$recursiveRef"}
_OBJECT_KEYWORDS = {"properties", "patternProperties", "dependentSchemas", "propertyNames", "unevaluatedProperties"}


class InteractionSessionError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class InteractionContext:
    context_type: str
    version: str
    description: str
    schema_path: Path
    schema_sha256: str
    schema: dict[str, Any]
    consumers: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class InteractionSessionDeclaration:
    profile: str
    profile_version: str
    installed_profile_sha256: str
    contexts: dict[str, InteractionContext]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _validate_canonical_value(value: Any) -> None:
    if value is None or isinstance(value, (str, bool)):
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeError as exc:
                raise InteractionSessionError("SESSION_SCHEMA_MISMATCH", "interaction-session state is not valid Unicode") from exc
        return
    if isinstance(value, float):
        raise InteractionSessionError("SESSION_SCHEMA_MISMATCH", "interaction-session state cannot contain floats")
    if isinstance(value, int):
        if abs(value) > 9007199254740991:
            raise InteractionSessionError("SESSION_SCHEMA_MISMATCH", "interaction-session integer is outside the portable range")
        return
    if isinstance(value, list):
        for item in value:
            _validate_canonical_value(item)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise InteractionSessionError("SESSION_SCHEMA_MISMATCH", "interaction-session object keys must be strings")
        for key, item in value.items():
            _validate_canonical_value(key)
            _validate_canonical_value(item)
        return
    raise InteractionSessionError("SESSION_SCHEMA_MISMATCH", f"unsupported interaction-session value: {type(value).__name__}")


def canonical_session_state(state: Any) -> bytes:
    if not isinstance(state, dict):
        raise InteractionSessionError("SESSION_SCHEMA_MISMATCH", "interaction-session state must be an object")
    _validate_canonical_value(state)
    try:
        raw = _canonical(state)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise InteractionSessionError("SESSION_SCHEMA_MISMATCH", "interaction-session state is not canonical JSON") from exc
    if len(raw) > MAX_STATE_BYTES:
        raise InteractionSessionError("SESSION_SCHEMA_MISMATCH", "interaction-session state exceeds 4096 canonical UTF-8 bytes")
    return raw


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", f"required regular file is unavailable: {path.name}")
    try:
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = item
            return result

        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", f"invalid JSON file: {path.name}") from exc
    if not isinstance(value, dict):
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", f"JSON file must be an object: {path.name}")
    return value


def _validate_packaged_schema(value: Any, filename: str, label: str) -> None:
    try:
        from jsonschema import Draft202012Validator
        schema = _read_json(Path(__file__).resolve().parent / "schemas" / filename)
        Draft202012Validator.check_schema(schema)
        errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda error: list(error.path))
    except InteractionSessionError:
        raise
    except Exception as exc:
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", f"{label} schema validation is unavailable") from exc
    if errors:
        location = ".".join(str(part) for part in errors[0].path) or "<root>"
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", f"{label} is invalid at {location}: {errors[0].message}")


def _confined(root: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw.startswith("contract/"):
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", "contract path is invalid")
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", "contract path escapes profile") from exc
    return path


def _validate_self_contained_schema(value: Any) -> None:
    if isinstance(value, float) or (isinstance(value, int) and not isinstance(value, bool) and abs(value) > 9007199254740991):
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", "interaction-session schema contains an unsafe number")
    if isinstance(value, dict):
        schema_type = value.get("type")
        object_applicator = schema_type == "object" or (isinstance(schema_type, list) and "object" in schema_type) or bool(_OBJECT_KEYWORDS & value.keys())
        if object_applicator and value.get("additionalProperties") is not False:
            raise InteractionSessionError("SESSION_CONTRACT_INVALID", "every object in an interaction-session schema must be closed")
        for key, item in value.items():
            if key in _REFERENCE_KEYWORDS and (not isinstance(item, str) or not item.startswith("#")):
                raise InteractionSessionError("SESSION_CONTRACT_INVALID", "interaction-session schema contains an external reference")
            _validate_self_contained_schema(item)
    elif isinstance(value, list):
        for item in value:
            _validate_self_contained_schema(item)


def load_interaction_session_declaration(
    profile_root: Path,
    *,
    supported_consumers: set[tuple[str, str]] | None = None,
) -> InteractionSessionDeclaration | None:
    """Load the manifest-bound declaration; an unbound file is deliberately ignored."""
    root = profile_root.resolve()
    if not (root / "state/installed-distribution.json").is_file():
        return None
    receipt = _read_json(root / "state/installed-distribution.json")
    manifest = receipt.get("manifest")
    profile_section = manifest.get("profile") if isinstance(manifest, dict) else None
    binding = profile_section.get("interaction_session_contract") if isinstance(profile_section, dict) else None
    if binding is None:
        return None
    _validate_packaged_schema(
        receipt,
        "telegram.bridge.specialist_install_receipt.v1.schema.json",
        "installed profile receipt",
    )
    if not isinstance(manifest, dict):
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", "installed distribution manifest is invalid")
    distribution_path = next((root / name for name in ("distribution.json", "distribution.yaml", "distribution.yml") if (root / name).is_file() and not (root / name).is_symlink()), None)
    if distribution_path is None:
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", "installed distribution manifest file is unavailable")
    try:
        if distribution_path.suffix == ".json":
            installed_manifest = _read_json(distribution_path)
        else:
            import yaml
            installed_manifest = yaml.safe_load(distribution_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", "installed distribution manifest file is invalid") from exc
    if installed_manifest != manifest:
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", "installed distribution manifest does not match receipt")
    _validate_packaged_schema(
        manifest,
        "telegram.bridge.specialist_distribution.v1.schema.json",
        "installed distribution manifest",
    )
    if (
        not isinstance(receipt.get("profile"), str)
        or not _ID_RE.fullmatch(receipt["profile"])
        or not isinstance(receipt.get("profile_version"), str)
        or not _VERSION_RE.fullmatch(receipt["profile_version"])
        or profile_section.get("name") != receipt.get("profile")
        or profile_section.get("version") != receipt.get("profile_version")
    ):
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", "installed profile identity is invalid")
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", "interaction-session manifest binding is invalid")
    if not isinstance(binding.get("sha256"), str) or not _DIGEST_RE.fullmatch(binding["sha256"]):
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", "interaction-session declaration digest is invalid")
    declaration_path = _confined(root, binding.get("path"))
    raw = declaration_path.read_bytes() if declaration_path.is_file() and not declaration_path.is_symlink() else b""
    if not raw or _digest(raw) != binding.get("sha256"):
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", "interaction-session declaration digest mismatch")
    declaration = _read_json(declaration_path)
    if set(declaration) != {"schema_version", "contexts"} or declaration.get("schema_version") != "telegram.bridge.specialist_interaction_sessions.v1":
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", "interaction-session declaration shape is invalid")
    contexts_raw = declaration.get("contexts")
    if not isinstance(contexts_raw, list) or not 1 <= len(contexts_raw) <= 16:
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", "interaction-session contexts are invalid")
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise InteractionSessionError("SESSION_CONTRACT_INVALID", "jsonschema is required") from exc
    contexts: dict[str, InteractionContext] = {}
    seen_consumers: set[tuple[str, str]] = set()
    for item in contexts_raw:
        required = {"context_type", "version", "description", "schema", "schema_sha256", "ttl_seconds", "consumers"}
        if not isinstance(item, dict) or set(item) != required or item.get("ttl_seconds") != 1800:
            raise InteractionSessionError("SESSION_CONTRACT_INVALID", "interaction-session context shape is invalid")
        context_type = item.get("context_type")
        if not isinstance(context_type, str) or not _ID_RE.fullmatch(context_type) or context_type in contexts:
            raise InteractionSessionError("SESSION_CONTRACT_INVALID", "interaction-session context identity is invalid")
        if not isinstance(item.get("version"), str) or not _VERSION_RE.fullmatch(item["version"]):
            raise InteractionSessionError("SESSION_CONTRACT_INVALID", "interaction-session context version is invalid")
        if not isinstance(item.get("description"), str) or not 1 <= len(item["description"]) <= 500:
            raise InteractionSessionError("SESSION_CONTRACT_INVALID", "interaction-session context description is invalid")
        if not isinstance(item.get("schema_sha256"), str) or not _DIGEST_RE.fullmatch(item["schema_sha256"]):
            raise InteractionSessionError("SESSION_CONTRACT_INVALID", "interaction-session schema digest is invalid")
        schema_path = _confined(root, item.get("schema"))
        schema_raw = schema_path.read_bytes() if schema_path.is_file() and not schema_path.is_symlink() else b""
        if not schema_raw or _digest(schema_raw) != item.get("schema_sha256"):
            raise InteractionSessionError("SESSION_CONTRACT_INVALID", f"schema digest mismatch for {context_type}")
        schema = _read_json(schema_path)
        _validate_self_contained_schema(schema)
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as exc:
            raise InteractionSessionError("SESSION_CONTRACT_INVALID", f"invalid schema for {context_type}") from exc
        if schema.get("type") != "object" or schema.get("additionalProperties") is not False or "$ref" in schema:
            raise InteractionSessionError("SESSION_CONTRACT_INVALID", f"session schema for {context_type} must be a closed object")
        consumers_raw = item.get("consumers")
        if not isinstance(consumers_raw, list) or len(consumers_raw) > 32:
            raise InteractionSessionError("SESSION_CONTRACT_INVALID", "interaction-session consumers are invalid")
        consumers: list[tuple[str, str]] = []
        for consumer in consumers_raw:
            if not isinstance(consumer, dict) or set(consumer) != {"kind", "workflow_id"}:
                raise InteractionSessionError("SESSION_CONTRACT_INVALID", "interaction-session consumer is invalid")
            pair = (consumer.get("kind"), consumer.get("workflow_id"))
            if pair[0] not in {"terminal_workflow", "terminal_mutation"} or not isinstance(pair[1], str) or not _ID_RE.fullmatch(pair[1]):
                raise InteractionSessionError("SESSION_CONTRACT_INVALID", "interaction-session consumer is invalid")
            if supported_consumers is not None and pair not in supported_consumers:
                raise InteractionSessionError("SESSION_CONSUMER_UNSUPPORTED", f"unsupported interaction-session consumer: {pair[1]}")
            if pair in seen_consumers:
                raise InteractionSessionError("SESSION_CONTRACT_INVALID", f"consumer may accept only one context: {pair[1]}")
            seen_consumers.add(pair)  # type: ignore[arg-type]
            consumers.append(pair)  # type: ignore[arg-type]
        contexts[context_type] = InteractionContext(
            context_type=context_type,
            version=str(item.get("version")),
            description=str(item.get("description")),
            schema_path=schema_path,
            schema_sha256=str(item.get("schema_sha256")),
            schema=schema,
            consumers=tuple(consumers),
        )
    identity = {key: receipt.get(key) for key in ("profile", "profile_version")}
    identity["manifest"] = manifest
    return InteractionSessionDeclaration(
        profile=str(receipt.get("profile")),
        profile_version=str(receipt.get("profile_version")),
        installed_profile_sha256=_digest(_canonical(identity)),
        contexts=contexts,
    )


def validate_turn_control(
    control: Any,
    declaration: InteractionSessionDeclaration,
    *,
    now: Any,
) -> dict[str, Any]:
    """Validate identity, snapshot integrity, expiry, and the declared state schema."""
    required = {"profile", "profile_version", "installed_profile_sha256", "conversation_generation", "active"}
    if not isinstance(control, dict) or not required.issubset(control) or set(control) - (required | {"snapshot"}):
        raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "interaction-session control is invalid")
    for key, expected in (
        ("profile", declaration.profile),
        ("profile_version", declaration.profile_version),
        ("installed_profile_sha256", declaration.installed_profile_sha256),
    ):
        if control.get(key) != expected:
            raise InteractionSessionError("SESSION_PROFILE_MISMATCH", f"interaction-session {key} does not match installed profile")
    if type(control.get("conversation_generation")) is not int or control["conversation_generation"] < 0:
        raise InteractionSessionError("SESSION_GENERATION_MISMATCH", "interaction-session generation is invalid")
    if type(control.get("active")) is not bool:
        raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "interaction-session active flag is invalid")
    if not control["active"]:
        if "snapshot" in control:
            raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "inactive interaction-session control contains a snapshot")
        return control
    snapshot = control.get("snapshot")
    fields = {"session_id", "context_type", "context_version", "schema_sha256", "revision", "state", "state_sha256", "expires_at"}
    if not isinstance(snapshot, dict) or set(snapshot) != fields:
        raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "active interaction-session snapshot is invalid")
    if not isinstance(snapshot.get("session_id"), str) or not _SESSION_ID_RE.fullmatch(snapshot["session_id"]):
        raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "interaction-session ID is invalid")
    context = declaration.contexts.get(snapshot.get("context_type"))
    if context is None or snapshot.get("context_version") != context.version or snapshot.get("schema_sha256") != context.schema_sha256:
        raise InteractionSessionError("SESSION_SCHEMA_MISMATCH", "interaction-session context does not match installed declaration")
    if type(snapshot.get("revision")) is not int or snapshot["revision"] < 1:
        raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "interaction-session revision is invalid")
    try:
        expires = __import__("datetime").datetime.fromisoformat(str(snapshot["expires_at"]).replace("Z", "+00:00"))
    except (ValueError, TypeError) as exc:
        raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "interaction-session expiry is invalid") from exc
    try:
        expired = expires <= now
    except TypeError as exc:
        raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "interaction-session expiry is invalid") from exc
    if expired:
        raise InteractionSessionError("SESSION_CONTEXT_EXPIRED", "interaction-session snapshot has expired")
    state = snapshot.get("state")
    state_raw = canonical_session_state(state)
    if _digest(state_raw) != snapshot.get("state_sha256"):
        raise InteractionSessionError("SESSION_SCHEMA_MISMATCH", "interaction-session state digest is invalid")
    from jsonschema import Draft202012Validator
    if list(Draft202012Validator(context.schema).iter_errors(state)):
        raise InteractionSessionError("SESSION_SCHEMA_MISMATCH", "interaction-session state does not satisfy installed schema")
    return control


def interaction_sessions_available() -> bool:
    """Process-local service gate. It performs no profile filesystem I/O."""
    return os.environ.get("HERMES_INTERACTION_SESSION_ENABLED") == "1"


def interaction_session_description() -> str:
    return "Replace or clear the current specialist interaction session for a later Telegram turn."


@lru_cache(maxsize=8)
def _codex_additional_context_schema_preflight(
    codex_binary: str,
    binary_mtime_ns: int,
) -> tuple[bool, str]:
    del binary_mtime_ns  # cache-key input; the path may be replaced in place.
    try:
        with tempfile.TemporaryDirectory(prefix="hermes-codex-schema-") as temporary:
            completed = subprocess.run(
                [codex_binary, "app-server", "generate-json-schema", "--experimental", "--out", temporary],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20,
                check=False,
            )
            if completed.returncode != 0:
                return False, (completed.stderr or completed.stdout or "schema generation failed")[:500]
            path = Path(temporary) / "v2" / "TurnStartParams.json"
            schema = _read_json(path)
            additional = (schema.get("properties") or {}).get("additionalContext")
            definitions = schema.get("definitions") or {}
            entry = definitions.get("AdditionalContextEntry")
            kind = definitions.get("AdditionalContextKind")
            valid = (
                isinstance(additional, dict)
                and "object" in (additional.get("type") or [])
                and additional.get("additionalProperties") == {"$ref": "#/definitions/AdditionalContextEntry"}
                and isinstance(entry, dict)
                and set(entry.get("required") or []) == {"kind", "value"}
                and (entry.get("properties") or {}).get("value") == {"type": "string"}
                and (entry.get("properties") or {}).get("kind") == {"$ref": "#/definitions/AdditionalContextKind"}
                and isinstance(kind, dict)
                and set(kind.get("enum") or []) == {"untrusted", "application"}
            )
            return (True, "") if valid else (False, "experimental TurnStartParams.additionalContext has an unsupported shape")
    except Exception as exc:
        return False, str(exc)[:500]


def require_codex_additional_context_support() -> None:
    binary = shutil.which("codex")
    if not binary:
        raise InteractionSessionError("SESSION_CAPABILITY_UNAVAILABLE", "codex executable is unavailable for interaction-session projection")
    try:
        mtime = Path(binary).stat().st_mtime_ns
    except OSError as exc:
        raise InteractionSessionError("SESSION_CAPABILITY_UNAVAILABLE", "codex executable cannot be inspected") from exc
    supported, detail = _codex_additional_context_schema_preflight(str(Path(binary).resolve()), mtime)
    if not supported:
        raise InteractionSessionError(
            "SESSION_CAPABILITY_UNAVAILABLE",
            "installed codex app-server does not support experimental turn additionalContext"
            + (f": {detail}" if detail else ""),
        )


def invoke_interaction_session_endpoint(operation: str, context_type: str | None, state: dict[str, Any] | None) -> str:
    root = get_hermes_home().resolve()
    endpoint = _read_json(root / ENDPOINT_RELATIVE_PATH)
    if endpoint.get("schema_version") != PROTOCOL or set(endpoint) != {"schema_version", "socket", "runtime_instance_id", "turn_token"}:
        raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "no eligible interaction-session turn is active")
    socket_path = (root / str(endpoint["socket"])).resolve()
    try:
        socket_path.relative_to(root)
    except ValueError as exc:
        raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "interaction-session endpoint escapes profile") from exc
    request = {"protocol_version": PROTOCOL, "operation": operation, "turn_token": endpoint["turn_token"]}
    if context_type is not None:
        request["context_type"] = context_type
    if state is not None:
        request["state"] = state
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(10)
        client.connect(str(socket_path))
        client.sendall(_canonical(request) + b"\n")
        response = b""
        while not response.endswith(b"\n"):
            chunk = client.recv(65536)
            if not chunk:
                break
            response += chunk
            if len(response) > 65536:
                raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "interaction-session response is oversized")
    try:
        value = json.loads(response, object_pairs_hook=lambda pairs: _pairs_without_duplicates(pairs))
    except (json.JSONDecodeError, ValueError) as exc:
        raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "interaction-session endpoint returned invalid JSON") from exc
    if not isinstance(value, dict) or value.get("protocol_version") != PROTOCOL or value.get("status") not in {"completed", "failed"}:
        raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "interaction-session endpoint returned an invalid response")
    if value["status"] == "completed":
        if set(value) != {"protocol_version", "status", "transition"} or not isinstance(value.get("transition"), dict):
            raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "interaction-session endpoint returned an invalid completed response")
        transition = value["transition"]
        base = {"operation", "event_id", "profile", "profile_version", "installed_profile_sha256", "conversation_generation"}
        prior = {"prior_session_id", "prior_revision"}
        replacement = {"context_type", "context_version", "schema_sha256", "revision", "state", "state_sha256"}
        op = transition.get("operation")
        expected = base | replacement | (prior if "prior_session_id" in transition else set()) if op == "replace" else base | prior
        if op not in {"replace", "clear"} or set(transition) != expected:
            raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "interaction-session endpoint returned an invalid transition")
        if op == "replace":
            state_raw = canonical_session_state(transition.get("state"))
            if transition.get("state_sha256") != _digest(state_raw):
                raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "interaction-session endpoint returned a mismatched state digest")
    elif set(value) != {"protocol_version", "status", "error"} or not isinstance(value.get("error"), dict) or set(value["error"]) != {"code", "message"}:
        raise InteractionSessionError("SESSION_CONTEXT_UNAVAILABLE", "interaction-session endpoint returned an invalid failure response")
    if value.get("status") != "completed":
        error = value.get("error") if isinstance(value.get("error"), dict) else {}
        raise InteractionSessionError(str(error.get("code") or "SESSION_CONTEXT_UNAVAILABLE"), str(error.get("message") or "interaction-session update failed"))
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value
