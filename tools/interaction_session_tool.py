"""Service-gated interaction-session update tool."""

from __future__ import annotations

import json
from typing import Any

from hermes_cli.interaction_session import (
    InteractionSessionError,
    interaction_session_description,
    interaction_sessions_available,
    invoke_interaction_session_endpoint,
)
from tools.registry import registry


def update_interaction_session(operation: str, context_type: str | None = None, state: dict[str, Any] | None = None) -> str:
    try:
        return invoke_interaction_session_endpoint(
            operation,
            context_type if operation == "replace" else None,
            state if operation == "replace" else None,
        )
    except InteractionSessionError as exc:
        return json.dumps({"error": {"code": exc.code, "message": str(exc)}}, ensure_ascii=False, separators=(",", ":"))


registry.register(
    name="update_interaction_session",
    toolset="interaction_session",
    schema={
        "name": "update_interaction_session",
        "description": interaction_session_description(),
        "parameters": {
            "type": "object", "additionalProperties": False,
            "required": ["operation"],
            "properties": {
                "operation": {"enum": ["replace", "clear"]},
                "context_type": {"type": "string"},
                "state": {"type": "object"},
            },
        },
    },
    handler=lambda args, **_kwargs: update_interaction_session(
        str(args.get("operation") or ""),
        args.get("context_type") if isinstance(args.get("context_type"), str) else None,
        args.get("state") if isinstance(args.get("state"), dict) else None,
    ),
    check_fn=interaction_sessions_available,
    description="Update a negotiated specialist interaction session.",
)
