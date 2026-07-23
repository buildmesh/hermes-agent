"""Service-gated terminal presenter tool for persistent Telegram specialists."""

from __future__ import annotations

import json
from typing import Any

from hermes_cli.terminal_presenter import (
    TerminalPresenterError,
    invoke_presenter_endpoint,
    presenter_tool_description,
    terminal_presenters_available,
)
from tools.registry import registry


def finalize_telegram_presentation(presenter_id: str, presenter_input: dict[str, Any]) -> str:
    try:
        return invoke_presenter_endpoint(presenter_id, presenter_input)
    except TerminalPresenterError as exc:
        return json.dumps(
            {"error": {"code": exc.code, "message": str(exc)}},
            ensure_ascii=False,
            separators=(",", ":"),
        )


registry.register(
    name="finalize_telegram_presentation",
    toolset="terminal_presenter",
    schema={
        "name": "finalize_telegram_presentation",
        "description": presenter_tool_description(),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["presenter_id", "input"],
            "properties": {
                "presenter_id": {
                    "type": "string",
                    "description": "Stable presenter_id from the installed profile declaration.",
                },
                "input": {
                    "type": "object",
                    "description": "Structured input satisfying the selected presenter's input schema.",
                },
            },
        },
    },
    handler=lambda args, **_kwargs: finalize_telegram_presentation(
        str(args.get("presenter_id") or ""),
        args.get("input") if isinstance(args.get("input"), dict) else {},
    ),
    check_fn=terminal_presenters_available,
    description="Finalize a Telegram response with a declared presenter.",
)
