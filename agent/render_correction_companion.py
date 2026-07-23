"""Isolated one-call Responses runtime for persistent render correction."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_cli.config import load_config
from hermes_cli.runtime_provider import resolve_runtime_provider
from agent.telegram_bridge_render_contract import CANONICAL_RENDER_ENVELOPE_INSTRUCTIONS


class CorrectionCompanionUnavailable(RuntimeError):
    """The configured parent runtime cannot support an isolated correction."""


class CorrectionCompanionTimeout(TimeoutError):
    """The isolated correction process was killed and reaped after timeout."""


def _configured_model(config: dict[str, Any]) -> str:
    model = config.get("model") or {}
    if isinstance(model, str):
        return model.strip()
    if isinstance(model, dict):
        return str(model.get("default") or model.get("model") or "").strip()
    return ""


def _direct_runtime_config(runtime: dict[str, Any], model: str) -> dict[str, str] | None:
    if runtime.get("api_mode") not in {"codex_app_server", "codex_responses"}:
        return None
    if runtime.get("provider") not in {"openai", "openai-codex"}:
        return None
    result = {
        "provider": str(runtime.get("provider") or ""),
        "api_mode": "codex_responses",
        "model": model,
        "base_url": str(runtime.get("base_url") or "").rstrip("/"),
        "api_key": str(runtime.get("api_key") or ""),
    }
    if not result["base_url"] or not result["api_key"]:
        return None
    return result


def _validated_profile_root(value: Any, expected: Path | None = None) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise CorrectionCompanionUnavailable("render correction profile root is missing")
    path = Path(value)
    if not path.is_absolute():
        raise CorrectionCompanionUnavailable("render correction profile root must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CorrectionCompanionUnavailable("render correction profile root is unavailable") from exc
    if resolved != path or not resolved.is_dir():
        raise CorrectionCompanionUnavailable("render correction profile root is not canonical")
    if expected is not None and resolved != expected.resolve(strict=True):
        raise CorrectionCompanionUnavailable("render correction profile root does not match worker profile")
    return resolved


def validate_companion_descriptor(
    descriptor: dict[str, str],
    expected_profile_root: Path | None = None,
) -> dict[str, str]:
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "provider", "api_mode", "model", "profile_root",
    }:
        raise CorrectionCompanionUnavailable("invalid render correction companion descriptor")
    provider = str(descriptor.get("provider") or "").strip()
    model = str(descriptor.get("model") or "").strip()
    if provider not in {"openai", "openai-codex"} or descriptor.get("api_mode") != "codex_responses" or not model:
        raise CorrectionCompanionUnavailable("invalid render correction companion descriptor")
    profile_root = _validated_profile_root(descriptor.get("profile_root"), expected_profile_root)
    return {
        "provider": provider,
        "api_mode": "codex_responses",
        "model": model,
        "profile_root": str(profile_root),
    }


@contextmanager
def _profile_scope(profile_root: Path):
    token = set_hermes_home_override(profile_root)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def resolve_companion_descriptor(profile_root: Path) -> dict[str, str] | None:
    """Probe app-server correction locally and return only non-secret capability data."""
    root = _validated_profile_root(profile_root)
    with _profile_scope(root):
        config = load_config()
        model = _configured_model(config)
        if not model:
            return None
        try:
            runtime = resolve_runtime_provider(target_model=model)
        except Exception:
            return None
        direct_config = _direct_runtime_config(runtime, model)
        if direct_config is None:
            return None
        try:
            from agent.auxiliary_client import CodexAuxiliaryClient  # noqa: F401
            client = _direct_responses_client(direct_config)
            client.close()
        except Exception:
            return None
    return validate_companion_descriptor({
        "provider": direct_config["provider"],
        "api_mode": direct_config["api_mode"],
        "model": direct_config["model"],
        "profile_root": str(root),
    }, root)


def resolve_companion_credentials(descriptor: dict[str, str]) -> dict[str, str]:
    """Resolve and refresh the bearer immediately before one correction."""
    descriptor = validate_companion_descriptor(descriptor)
    model = descriptor["model"]
    with _profile_scope(Path(descriptor["profile_root"])):
        if _configured_model(load_config()) != model:
            raise CorrectionCompanionUnavailable("render correction model no longer matches specialist profile")
        try:
            runtime = resolve_runtime_provider(target_model=model)
        except Exception as exc:
            raise CorrectionCompanionUnavailable("render correction credentials are unavailable") from exc
        direct_config = _direct_runtime_config(runtime, model)
    if direct_config is None or direct_config["provider"] != descriptor["provider"]:
        raise CorrectionCompanionUnavailable("render correction runtime is no longer bound to configured provider")
    return direct_config


def _validated_model_config(config: Any, descriptor: dict[str, str] | None = None) -> dict[str, str]:
    if not isinstance(config, dict) or set(config) != {
        "provider", "api_mode", "model", "base_url", "api_key",
    }:
        raise CorrectionCompanionUnavailable("invalid render correction model config")
    result = {key: str(config.get(key) or "").strip() for key in config}
    if (
        result["provider"] not in {"openai", "openai-codex"}
        or result["api_mode"] != "codex_responses"
        or not all(result[key] for key in ("model", "base_url", "api_key"))
    ):
        raise CorrectionCompanionUnavailable("invalid render correction model config")
    if descriptor is not None and (
        result["provider"] != descriptor["provider"] or result["model"] != descriptor["model"]
    ):
        raise CorrectionCompanionUnavailable("render correction model config changed provider binding")
    return result


def _direct_responses_client(config: dict[str, str]) -> Any:
    from agent.auxiliary_client import (
        CodexAuxiliaryClient,
        _codex_cloudflare_headers,
        _create_openai_client,
    )
    from utils import base_url_host_matches

    headers = None
    if base_url_host_matches(config["base_url"], "chatgpt.com"):
        headers = _codex_cloudflare_headers(config["api_key"])
    kwargs = {"default_headers": headers} if headers else {}
    real_client = _create_openai_client(
        api_key=config["api_key"],
        base_url=config["base_url"],
        **kwargs,
    )
    return CodexAuxiliaryClient(real_client, config["model"])


def _correction_messages(repair: dict[str, Any]) -> list[dict[str, str]]:
    instructions = (
        "Correct only the formatting of a completed Telegram Bridge render. "
        "Do not repeat, continue, verify, or infer any domain work. Return only a JSON array of "
        "telegram.bridge.render_payload.v1 objects. No tools, memory, persistence, fallback, or "
        "workspace context is available. Preserve the candidate's intended supported presentation "
        "and preserve any valid structured blocks. Repair only the reported formatting or schema "
        "errors; never replace tables, lists, or buttons with plain text merely to make the JSON "
        "valid. "
        + CANONICAL_RENDER_ENVELOPE_INSTRUCTIONS
    )
    material = {
        "candidate": repair["candidate"],
        "validation_errors": repair["validation_errors"],
        "target_constraints": repair["target_constraints"],
    }
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": json.dumps(material, ensure_ascii=False, sort_keys=True)},
    ]


def run_companion_model_call(
    config: dict[str, str],
    repair: dict[str, Any],
    *,
    timeout: float,
) -> str:
    """Make exactly one direct Responses call with an empty tool surface."""
    client = _direct_responses_client(config)
    try:
        response = client.chat.completions.create(
            model=config["model"],
            messages=_correction_messages(repair),
            tools=[],
            tool_choice="none",
            timeout=timeout,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content:
            raise CorrectionCompanionUnavailable("correction model returned no render candidate")
        return content
    finally:
        client.close()


def _sanitized_environment() -> dict[str, str]:
    allowed = {
        "PATH", "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    }
    return {name: value for name, value in os.environ.items() if name in allowed}


def resolve_companion_credentials_subprocess(
    descriptor: dict[str, str],
    timeout: float,
) -> dict[str, str]:
    """Resolve and durably refresh profile credentials in a bounded child."""
    descriptor = validate_companion_descriptor(descriptor)
    if timeout <= 0:
        raise CorrectionCompanionTimeout(
            "render correction credential bootstrap exceeded its hard timeout"
        )
    monotonic_deadline = time.monotonic() + timeout
    profile_root = Path(descriptor["profile_root"])
    payload = json.dumps({"descriptor": descriptor})
    with tempfile.TemporaryDirectory(prefix="hermes-render-credentials-") as directory:
        os.chmod(directory, 0o700)
        environment = _sanitized_environment()
        environment["HERMES_HOME"] = str(profile_root)
        process = subprocess.Popen(
            [sys.executable, "-m", "agent.render_correction_companion", "--credentials-child"],
            cwd=directory,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            close_fds=True,
            start_new_session=True,
        )
        try:
            remaining = monotonic_deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired("render correction credentials", timeout)
            stdout, _ = process.communicate(payload, timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            try:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
            finally:
                try:
                    process.communicate()
                except (OSError, ValueError):
                    pass
                finally:
                    try:
                        process.wait()
                    except (ChildProcessError, OSError):
                        pass
            raise CorrectionCompanionTimeout(
                "render correction credential bootstrap exceeded its hard timeout"
            ) from exc
        process.wait()
    if process.returncode != 0:
        raise CorrectionCompanionUnavailable("render correction credential bootstrap failed")
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise CorrectionCompanionUnavailable("render correction credential bootstrap returned invalid output") from exc
    config = response.get("config") if isinstance(response, dict) else None
    if not isinstance(response, dict) or response.get("status") != "completed":
        raise CorrectionCompanionUnavailable("render correction credential bootstrap did not complete")
    return _validated_model_config(config, descriptor)


def run_companion_subprocess(
    config: dict[str, str],
    repair: dict[str, Any],
    timeout: float,
) -> str:
    """Run one stateless model correction in a killable child."""
    config = _validated_model_config(config)
    if timeout <= 0:
        raise CorrectionCompanionTimeout("render correction companion exceeded its hard timeout")
    monotonic_deadline = time.monotonic() + timeout
    payload = json.dumps({
        "config": config,
        "repair": repair,
        "deadline": time.time() + timeout,
    })
    with tempfile.TemporaryDirectory(prefix="hermes-render-correction-") as directory:
        os.chmod(directory, 0o700)
        if time.monotonic() >= monotonic_deadline:
            raise CorrectionCompanionTimeout("render correction companion exceeded its hard timeout")
        process = subprocess.Popen(
            [sys.executable, "-m", "agent.render_correction_companion", "--model-child"],
            cwd=directory,
            env=_sanitized_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            close_fds=True,
            start_new_session=True,
        )
        try:
            remaining = monotonic_deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired("render correction companion", timeout)
            stdout, _ = process.communicate(payload, timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            try:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
            finally:
                try:
                    process.communicate()
                except (OSError, ValueError):
                    pass
                finally:
                    try:
                        process.wait()
                    except (ChildProcessError, OSError):
                        pass
            raise CorrectionCompanionTimeout("render correction companion exceeded its hard timeout") from exc
        process.wait()
    if process.returncode == 124:
        raise CorrectionCompanionTimeout("render correction companion exceeded its hard timeout")
    if process.returncode != 0:
        raise CorrectionCompanionUnavailable("render correction companion failed")
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise CorrectionCompanionUnavailable("render correction companion returned invalid output") from exc
    candidate = response.get("candidate") if isinstance(response, dict) else None
    if not isinstance(response, dict) or response.get("status") != "completed" or not isinstance(candidate, str):
        raise CorrectionCompanionUnavailable("render correction companion did not complete")
    return candidate


def _credentials_child_main() -> int:
    try:
        request = json.load(sys.stdin)
        descriptor = validate_companion_descriptor(request["descriptor"])
        environment_root = _validated_profile_root(os.environ.get("HERMES_HOME"))
        if Path(descriptor["profile_root"]) != environment_root:
            raise CorrectionCompanionUnavailable("credential bootstrap profile environment mismatch")
        config = resolve_companion_credentials(descriptor)
        json.dump({"status": "completed", "config": config}, sys.stdout)
        sys.stdout.flush()
        return 0
    except Exception:
        return 1


def _model_child_main() -> int:
    try:
        request = json.load(sys.stdin)
        deadline = float(request["deadline"])
        remaining = deadline - time.time()
        if remaining <= 0:
            raise CorrectionCompanionTimeout("render correction deadline expired before model launch")
        config = _validated_model_config(request["config"])
        candidate = run_companion_model_call(
            config,
            request["repair"],
            timeout=remaining,
        )
        json.dump({"status": "completed", "candidate": candidate}, sys.stdout)
        sys.stdout.flush()
        return 0
    except CorrectionCompanionTimeout:
        return 124
    except Exception:
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--credentials-child", action="store_true")
    modes.add_argument("--model-child", action="store_true")
    args = parser.parse_args(argv)
    return _credentials_child_main() if args.credentials_child else _model_child_main()


if __name__ == "__main__":
    raise SystemExit(main())
