"""One-call, deny-all model turn for presentation-only correction."""

from __future__ import annotations

from typing import Any


class ToolFreeTurnUnavailable(RuntimeError):
    """The active runtime cannot prove that a turn has no executable tools."""


class ToolExecutionDenied(RuntimeError):
    """A model attempted tool execution during a deny-all turn."""


def run_tool_free_conversation(agent: Any, prompt: str) -> dict[str, Any]:
    """Run exactly one model call on ``agent`` with no advertised tools.

    The Codex app-server owns and advertises native tools outside Hermes' tool
    list, so its current protocol cannot satisfy this contract.
    """
    if getattr(agent, "api_mode", None) == "codex_app_server":
        raise ToolFreeTurnUnavailable(
            "codex app-server cannot disable all native tools for one existing-thread turn"
        )
    if not callable(getattr(agent, "run_conversation", None)):
        raise ToolFreeTurnUnavailable("agent has no same-conversation turn API")

    attributes = {
        "tools": [],
        "valid_tool_names": set(),
        "max_iterations": 1,
        "_fallback_chain": [],
        "_budget_grace_call": False,
        "_skip_mcp_refresh": True,
    }
    previous = {name: getattr(agent, name, None) for name in attributes}
    present = {name: hasattr(agent, name) for name in attributes}
    had_executor_override = "_execute_tool_calls" in getattr(agent, "__dict__", {})
    previous_executor = getattr(agent, "_execute_tool_calls", None)

    def deny_all_executor(*_args: Any, **_kwargs: Any) -> None:
        raise ToolExecutionDenied("tool execution denied during render correction")

    try:
        for name, value in attributes.items():
            setattr(agent, name, value)
        agent._execute_tool_calls = deny_all_executor
        result = agent.run_conversation(prompt)
    finally:
        if had_executor_override:
            agent._execute_tool_calls = previous_executor
        else:
            agent.__dict__.pop("_execute_tool_calls", None)
        for name, value in previous.items():
            if present[name]:
                setattr(agent, name, value)
            else:
                agent.__dict__.pop(name, None)

    if not isinstance(result, dict) or result.get("api_calls") != 1:
        raise ToolFreeTurnUnavailable("tool-free correction did not complete in exactly one model call")
    if not result.get("completed", True) or result.get("partial"):
        raise ToolFreeTurnUnavailable(str(result.get("error") or "tool-free correction failed"))
    return result
