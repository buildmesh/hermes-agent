"""Isolated one-call Responses runtime for persistent render correction."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
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


def resolve_companion_config() -> dict[str, str] | None:
    """Map an app-server parent to Hermes' direct Codex Responses precedent."""
    config = load_config()
    model = _configured_model(config)
    if not model:
        return None
    try:
        runtime = resolve_runtime_provider(target_model=model)
    except Exception:
        return None
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
    try:
        from agent.auxiliary_client import CodexAuxiliaryClient  # noqa: F401
        client = _direct_responses_client(result)
        client.close()
    except Exception:
        return None
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
    config: dict[str, str],
    repair: dict[str, Any],
    timeout: float,
) -> str:
    """Run one correction in a killable child and wait for process quiescence."""
    payload = json.dumps({"config": config, "repair": repair, "timeout": timeout})
    with tempfile.TemporaryDirectory(prefix="hermes-render-correction-") as directory:
        os.chmod(directory, 0o700)
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
            stdout, _ = process.communicate(payload, timeout=max(0.1, timeout))
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                process.kill()
            process.communicate()
            process.wait()
            raise CorrectionCompanionTimeout("render correction companion exceeded its hard timeout") from exc
        process.wait()
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
        candidate = run_companion_model_call(
            request["config"],
            request["repair"],
            timeout=float(request["timeout"]),
        )
        json.dump({"status": "completed", "candidate": candidate}, sys.stdout)
        sys.stdout.flush()
        return 0
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
