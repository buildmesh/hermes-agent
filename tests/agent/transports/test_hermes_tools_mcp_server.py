"""Tests for the hermes-tools-as-MCP server module surface.

We don't run a live MCP session in unit tests — that requires the codex
subprocess + client + an event loop. These tests pin the static
contract: the module imports, the EXPOSED_TOOLS list is sane, and the
build helper assembles a server when the SDK is present.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def _install_test_presenter(root: Path) -> None:
    handler = root / "skills/test/present.py"
    schema = root / "contract/test.input.schema.json"
    handler.parent.mkdir(parents=True)
    schema.parent.mkdir(parents=True)
    handler.write_text("import sys\nsys.stdout.write(sys.stdin.read())\n", encoding="utf-8")
    schema.write_text(
        json.dumps({"type": "object", "additionalProperties": False}),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "telegram.bridge.terminal_presenters.v1",
        "presenters": [{
            "presenter_id": "test-presenter",
            "version": "1.0.0",
            "description": "Render a test payload.",
            "handler": "skills/test/present.py",
            "sha256": "sha256:" + hashlib.sha256(handler.read_bytes()).hexdigest(),
            "input_schema": "contract/test.input.schema.json",
            "input_schema_sha256": "sha256:" + hashlib.sha256(schema.read_bytes()).hexdigest(),
            "command": ["python3", "{handler}"],
            "timeout_seconds": 2,
            "max_output_bytes": 4096,
        }],
    }
    declaration = root / "contract/terminal-presenters.json"
    declaration.parent.mkdir(parents=True, exist_ok=True)
    declaration.write_text(json.dumps(manifest), encoding="utf-8")


def _codex_mcp_tool_names(
    profile_root: Path,
    *,
    allowed_tools: set[str] | None = None,
) -> set[str]:
    script = (
        "import json\n"
        "from agent.transports.hermes_tools_mcp_server import _build_server\n"
        "server = _build_server()\n"
        "print(json.dumps(sorted(tool.name for tool in server._tool_manager.list_tools())))\n"
    )
    env = os.environ.copy()
    env["HERMES_HOME"] = str(profile_root)
    if allowed_tools is not None:
        env["HERMES_MCP_ALLOWED_TOOLS"] = ",".join(sorted(allowed_tools))
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    return set(json.loads(result.stdout.strip().splitlines()[-1]))




class TestModuleSurface:
    def test_process_allowlist_projects_only_requested_tools(
        self, tmp_path: Path
    ):
        _install_test_presenter(tmp_path)
        endpoint = tmp_path / "state/persistent-runtime/presenter-endpoint.json"
        endpoint.parent.mkdir(parents=True)
        endpoint.write_text(
            json.dumps({
                "schema_version": "hermes.terminal_presenter_endpoint.v1",
                "socket": "state/runtime/worker.sock",
                "runtime_instance_id": "runtime_test",
                "turn_token": "present_turn_test",
            }),
            encoding="utf-8",
        )
        assert _codex_mcp_tool_names(
            tmp_path,
            allowed_tools={"finalize_telegram_presentation"},
        ) == {"finalize_telegram_presentation"}

    def test_module_imports_clean(self):
        from agent.transports import hermes_tools_mcp_server as m
        assert callable(m.main)
        assert callable(m._build_server)
        assert isinstance(m.EXPOSED_TOOLS, tuple)
        assert len(m.EXPOSED_TOOLS) > 0

    def test_exposed_tools_are_safe_subset(self):
        """We MUST NOT expose tools codex already has, because codex'
        own builtins are better-integrated with its sandbox + approvals.
        Specifically: no terminal/shell, no read_file/write_file, no
        patch — those are codex's built-in tools."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        forbidden = {
            "terminal", "shell", "read_file", "write_file", "patch",
            "search_files", "process",
        }
        leaked = forbidden & set(EXPOSED_TOOLS)
        assert not leaked, (
            f"these tools must NOT be exposed via the codex callback "
            f"because codex has built-in equivalents: {leaked}"
        )

    def test_expected_hermes_specific_tools_listed(self):
        """The Hermes-specific tools should be present so users on the
        codex runtime keep access to them."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        for required in (
            "web_search",
            "web_extract",
            "browser_navigate",
            "vision_analyze",
            "image_generate",
            "skill_view",
        ):
            assert required in EXPOSED_TOOLS, f"missing {required!r}"

    def test_terminal_presenter_projection_requires_active_negotiated_turn(
        self, tmp_path: Path
    ):
        """Codex sees the finalizer only while the worker publishes a turn token."""
        _install_test_presenter(tmp_path)
        endpoint = tmp_path / "state/persistent-runtime/presenter-endpoint.json"
        endpoint.parent.mkdir(parents=True)
        endpoint.write_text(
            json.dumps({
                "schema_version": "hermes.terminal_presenter_endpoint.v1",
                "socket": "state/runtime/worker.sock",
                "runtime_instance_id": "runtime_test",
            }),
            encoding="utf-8",
        )

        assert "finalize_telegram_presentation" not in _codex_mcp_tool_names(tmp_path)

        endpoint.write_text(
            json.dumps({
                "schema_version": "hermes.terminal_presenter_endpoint.v1",
                "socket": "state/runtime/worker.sock",
                "runtime_instance_id": "runtime_test",
                "turn_token": "present_turn_test",
            }),
            encoding="utf-8",
        )

        assert "finalize_telegram_presentation" in _codex_mcp_tool_names(tmp_path)

    def test_agent_loop_tools_not_exposed(self):
        """delegate_task / memory / session_search / todo require the
        running AIAgent context to dispatch, so a stateless MCP callback
        can't drive them. They must NOT be in EXPOSED_TOOLS."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        for agent_loop_tool in ("delegate_task", "memory", "session_search", "todo"):
            assert agent_loop_tool not in EXPOSED_TOOLS, (
                f"{agent_loop_tool!r} requires the agent loop context "
                "and can't be reached through a stateless MCP callback"
            )

    def test_kanban_worker_tools_exposed(self):
        """Kanban workers run as `hermes chat -q` subprocesses; if they
        come up on the codex_app_server runtime, the worker can do the
        actual work via codex's shell but needs the kanban tools through
        the MCP callback to report back to the kernel. Without these
        tools available, the worker would hang at completion time."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        # Worker handoff tools — every dispatched worker uses at least
        # one of {complete, block, comment} to close out its task.
        for worker_tool in (
            "kanban_complete",
            "kanban_block",
            "kanban_comment",
            "kanban_heartbeat",
        ):
            assert worker_tool in EXPOSED_TOOLS, (
                f"{worker_tool!r} missing from codex callback — kanban "
                "workers on codex_app_server runtime would hang"
            )

    def test_kanban_orchestrator_tools_exposed(self):
        """Orchestrator agents need to dispatch new tasks, query the
        board, and unblock/link tasks. Exposed so an orchestrator on
        codex_app_server can do its job."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        for orch_tool in (
            "kanban_create",
            "kanban_show",
            "kanban_list",
            "kanban_unblock",
            "kanban_link",
        ):
            assert orch_tool in EXPOSED_TOOLS, (
                f"{orch_tool!r} missing from codex callback"
            )


class TestMain:
    def test_main_returns_2_when_mcp_unavailable(self, monkeypatch):
        """When the mcp package isn't installed, main() should exit
        cleanly with code 2 and an install hint, not crash."""
        import agent.transports.hermes_tools_mcp_server as m

        def boom_build(*a, **kw):
            raise ImportError("mcp not installed")

        monkeypatch.setattr(m, "_build_server", boom_build)
        rc = m.main(["--verbose"])
        assert rc == 2

    def test_main_handles_keyboard_interrupt(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class FakeServer:
            def run(self):
                raise KeyboardInterrupt()

        monkeypatch.setattr(m, "_build_server", lambda: FakeServer())
        rc = m.main([])
        assert rc == 0

    def test_main_returns_1_on_runtime_error(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class CrashingServer:
            def run(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(m, "_build_server", lambda: CrashingServer())
        rc = m.main([])
        assert rc == 1
