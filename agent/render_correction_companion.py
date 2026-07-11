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
from pathlib import Path
from typing import Any

from hermes_cli.config import load_config
from hermes_cli.runtime_provider import resolve_runtime_provider


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
    if runtime.get("api_mode") != "codex_app_server":
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


def resolve_companion_descriptor() -> dict[str, str] | None:
    """Probe app-server correction locally and return only non-secret capability data."""
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
    return {
        "provider": direct_config["provider"],
        "api_mode": direct_config["api_mode"],
        "model": direct_config["model"],
    }


def resolve_companion_credentials(descriptor: dict[str, str]) -> dict[str, str]:
    """Resolve and refresh the bearer immediately before one correction."""
    model = str(descriptor.get("model") or "").strip()
    if descriptor.get("api_mode") != "codex_responses" or not model:
        raise CorrectionCompanionUnavailable("invalid render correction companion descriptor")
    try:
        runtime = resolve_runtime_provider(target_model=model)
    except Exception as exc:
        raise CorrectionCompanionUnavailable("render correction credentials are unavailable") from exc
    direct_config = _direct_runtime_config(runtime, model)
    if direct_config is None or direct_config["provider"] != descriptor.get("provider"):
        raise CorrectionCompanionUnavailable("render correction runtime is no longer available")
    return direct_config


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
        "workspace context is available."
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


def run_companion_subprocess(
    descriptor: dict[str, str],
    repair: dict[str, Any],
    timeout: float,
) -> str:
    """Run credential refresh and one correction in a killable child."""
    if timeout <= 0:
        raise CorrectionCompanionTimeout("render correction companion exceeded its hard timeout")
    monotonic_deadline = time.monotonic() + timeout
    payload = json.dumps({
        "descriptor": descriptor,
        "repair": repair,
        "deadline": time.time() + timeout,
    })
    with tempfile.TemporaryDirectory(prefix="hermes-render-correction-") as directory:
        os.chmod(directory, 0o700)
        if time.monotonic() >= monotonic_deadline:
            raise CorrectionCompanionTimeout("render correction companion exceeded its hard timeout")
        process = subprocess.Popen(
            [sys.executable, "-m", "agent.render_correction_companion", "--child"],
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


def _child_main() -> int:
    try:
        request = json.load(sys.stdin)
        deadline = float(request["deadline"])
        remaining = deadline - time.time()
        if remaining <= 0:
            raise CorrectionCompanionTimeout("render correction deadline expired before credential refresh")
        config = resolve_companion_credentials(request["descriptor"])
        remaining = deadline - time.time()
        if remaining <= 0:
            raise CorrectionCompanionTimeout("render correction deadline expired before model launch")
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
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args(argv)
    if not args.child:
        parser.error("the render correction companion is internal-only")
    return _child_main()


if __name__ == "__main__":
    raise SystemExit(main())
