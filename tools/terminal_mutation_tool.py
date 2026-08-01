"""Service-gated profile-owned terminal mutation tool."""

from __future__ import annotations

import json
from typing import Any

from hermes_cli.terminal_mutation import (
    TerminalMutationError,
    invoke_terminal_mutation_endpoint,
    terminal_mutation_description,
    terminal_mutations_available,
)
from tools.registry import registry


def run_terminal_mutation(
    workflow_id: str,
    workflow_input: dict[str, Any],
) -> str:
    try:
        return invoke_terminal_mutation_endpoint(
            workflow_id,
            workflow_input,
        )
    except TerminalMutationError as exc:
        return json.dumps(
            {
                "error": {"code": exc.code, "message": str(exc)},
                "sealed": exc.sealed,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )


registry.register(
    name="run_terminal_mutation",
    toolset="terminal_mutation",
    schema={
        "name": "run_terminal_mutation",
        "description": terminal_mutation_description(),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["workflow_id", "input"],
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "Declared immediate mutation workflow ID.",
                },
                "input": {
                    "type": "object",
                    "description": "Domain input satisfying the workflow schema.",
                },
            },
        },
    },
    handler=lambda args, **_kwargs: run_terminal_mutation(
        str(args.get("workflow_id") or ""),
        args.get("input") if isinstance(args.get("input"), dict) else {},
    ),
    check_fn=terminal_mutations_available,
    description="Run one declared idempotent terminal mutation.",
)
