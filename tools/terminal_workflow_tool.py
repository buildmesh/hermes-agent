"""Service-gated profile-owned terminal workflow tool."""

from __future__ import annotations

import json
from typing import Any

from hermes_cli.terminal_workflow import (
    TerminalWorkflowError,
    invoke_terminal_workflow_endpoint,
    terminal_workflow_description,
    terminal_workflows_available,
)
from tools.registry import registry


def run_terminal_workflow(workflow_id: str, workflow_input: dict[str, Any]) -> str:
    try:
        return invoke_terminal_workflow_endpoint(workflow_id, workflow_input)
    except TerminalWorkflowError as exc:
        return json.dumps(
            {"error": {"code": exc.code, "message": str(exc)}},
            ensure_ascii=False,
            separators=(",", ":"),
        )


registry.register(
    name="run_terminal_workflow",
    toolset="terminal_workflow",
    schema={
        "name": "run_terminal_workflow",
        "description": terminal_workflow_description(),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["workflow_id", "input"],
            "properties": {
                "workflow_id": {
                    "type": "string",
                    "description": "Stable workflow_id from the installed profile declaration.",
                },
                "input": {
                    "type": "object",
                    "description": "Domain input satisfying the selected workflow producer schema.",
                },
            },
        },
    },
    handler=lambda args, **_kwargs: run_terminal_workflow(
        str(args.get("workflow_id") or ""),
        args.get("input") if isinstance(args.get("input"), dict) else {},
    ),
    check_fn=terminal_workflows_available,
    description="Run a declared read-only producer-to-presenter workflow.",
)
