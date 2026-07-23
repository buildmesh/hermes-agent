"""Tests for Hermes Telegram adapter integration with the Telegram bridge."""

import asyncio
import os
import signal
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra={})
    adapter._bot = MagicMock()
    return adapter


@pytest.fixture(autouse=True)
def _default_process_groups_exit_with_leader(monkeypatch):
    """Most process doubles have no descendants; focused tests override this."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    monkeypatch.setattr(
        TelegramAdapter,
        "_telegram_bridge_process_group_exists",
        staticmethod(lambda _pid: False),
    )


class DeliveryOutcomeUnknown(RuntimeError):
    delivery_outcome_unknown = True


class PersistentRecoveryRequired(RuntimeError):
    """Test double for TBA's terminal persistent-service error contract."""

    code = "TURN_TIMEOUT"
    recovery_agent_id = "myhomestead"
    recovery_runtime_instance_id = "runtime-stuck"


def test_telegram_menu_commands_uses_bridge_registry_when_installed(tmp_path):
    adapter = _make_adapter()
    profile_root = tmp_path / "profiles" / "telegram-bridge"
    hermes_home = tmp_path

    dispatcher = MagicMock()
    dispatcher.telegram_bot_commands.return_value = (
        [
            ("hello", "Hello world plain text"),
            ("config", "Show or update bridge settings"),
        ],
        0,
    )

    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(profile_root, hermes_home))

    commands, hidden_count = adapter._telegram_menu_commands()

    assert commands == [
        ("hello", "Hello world plain text"),
        ("config", "Show or update bridge settings"),
    ]
    assert hidden_count == 0
    from hermes_cli.commands import telegram_menu_max_commands
    dispatcher.telegram_bot_commands.assert_called_once_with(
        profile_root,
        max_commands=telegram_menu_max_commands(),
    )


def test_telegram_bridge_dispatcher_module_is_cached_per_adapter(tmp_path):
    adapter = _make_adapter()
    profile_root = tmp_path / "profiles/telegram-bridge"
    dispatcher_path = profile_root / "bin/telegram_bridge_dispatch.py"
    dispatcher_path.parent.mkdir(parents=True)
    dispatcher_path.write_text("LOAD_MARKER = object()\n", encoding="utf-8")
    adapter._telegram_bridge_paths = MagicMock(return_value=(profile_root, tmp_path))

    first = adapter._load_telegram_bridge_dispatcher()
    second = adapter._load_telegram_bridge_dispatcher()

    assert first is second
    assert first.LOAD_MARKER is second.LOAD_MARKER


@pytest.mark.asyncio
async def test_config_command_routes_through_bridge_and_passes_bridge_config(tmp_path):
    adapter = _make_adapter()
    profile_root = tmp_path / "profiles" / "telegram-bridge"
    hermes_home = tmp_path

    payloads = [{"action": "send", "render": {"text": "max_width: 42"}}]
    bridge_config = {"max_width": 42, "max_width_default": False}
    result = SimpleNamespace(
        handled=True,
        payloads=payloads,
        bridge_config=bridge_config,
        reason="bridge_config",
    )

    dispatcher = MagicMock()
    dispatcher.can_handle_command.return_value = True
    dispatcher.dispatch_command.return_value = result

    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(profile_root, hermes_home))
    adapter._render_telegram_bridge_payloads = AsyncMock()

    update = SimpleNamespace(update_id=77)
    msg = SimpleNamespace(
        text="/config max_width",
        chat_id=123,
        message_id=456,
        from_user=SimpleNamespace(id=789),
        chat=SimpleNamespace(id=123),
    )

    handled = await adapter._maybe_handle_telegram_bridge_command(update, msg)

    assert handled is True
    dispatcher.can_handle_command.assert_called_once_with(
        profile_root,
        "config",
        hermes_known=False,
    )
    dispatcher.dispatch_command.assert_called_once_with(
        profile_root,
        hermes_home,
        text="/config max_width",
        chat_id=123,
        user_id=789,
        telegram_message_id=456,
        update_id=77,
        hermes_known=False,
    )
    adapter._render_telegram_bridge_payloads.assert_awaited_once_with(
        dispatcher,
        payloads,
        bridge_config,
    )


@pytest.mark.asyncio
async def test_bridge_text_forwards_replied_to_message_id(tmp_path):
    adapter = _make_adapter()
    profile_root = tmp_path / "profiles" / "telegram-bridge"
    hermes_home = tmp_path
    result = SimpleNamespace(
        handled=True,
        payloads=[],
        bridge_config={"max_width": 46},
        envelope=None,
        reason="specialist_notification_reply",
    )
    dispatcher = MagicMock()
    dispatcher.dispatch_text.return_value = result
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(profile_root, hermes_home))
    adapter._render_telegram_bridge_payloads = AsyncMock()
    adapter._clean_bot_trigger_text = MagicMock(return_value="show details")

    update = SimpleNamespace(update_id=88)
    msg = SimpleNamespace(
        text="show details",
        chat_id=123,
        message_id=457,
        from_user=SimpleNamespace(id=789),
        chat=SimpleNamespace(id=123),
        reply_to_message=SimpleNamespace(message_id=7001),
    )

    handled = await adapter._maybe_handle_telegram_bridge_text(update, msg)

    assert handled is True
    dispatcher.dispatch_text.assert_called_once_with(
        profile_root,
        hermes_home,
        text="show details",
        chat_id=123,
        user_id=789,
        telegram_message_id=457,
        reply_to_message_id=7001,
        update_id=88,
    )


@pytest.mark.asyncio
async def test_bridge_notification_worker_uses_connected_bot(tmp_path):
    adapter = _make_adapter()
    adapter._telegram_bridge_notification_worker = None
    profile_root = tmp_path / "profiles" / "telegram-bridge"
    hermes_home = tmp_path
    worker = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), health=MagicMock(return_value={"running": True}))
    dispatcher = MagicMock()
    dispatcher.create_notification_worker.return_value = worker
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(profile_root, hermes_home))

    await adapter._start_telegram_bridge_notification_worker()

    dispatcher.create_notification_worker.assert_called_once_with(
        profile_root,
        hermes_home,
        bot=adapter._bot,
        inline_keyboard_button_cls=pytest.importorskip("telegram").InlineKeyboardButton,
        inline_keyboard_markup_cls=pytest.importorskip("telegram").InlineKeyboardMarkup,
        parse_mode_markdown_v2=pytest.importorskip("telegram.constants").ParseMode.MARKDOWN_V2,
    )
    worker.start.assert_awaited_once()
    assert adapter._telegram_bridge_notification_health() == {"running": True}

    await adapter._stop_telegram_bridge_notification_worker()

    worker.stop.assert_awaited_once()
    assert adapter._telegram_bridge_notification_worker is None


@pytest.mark.asyncio
async def test_bridge_persistent_worker_lifecycle(tmp_path, monkeypatch):
    adapter = _make_adapter()
    adapter._telegram_bridge_persistent_workers = {}
    profile_root = tmp_path / "profiles/telegram-bridge"
    specialist = tmp_path / "profiles/hello-world-specialist"
    socket_path = tmp_path / "hello.sock"
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    dispatcher = MagicMock()
    dispatcher.persistent_service_specs.return_value = [{
        "agent_id": "hello_world",
        "service_id": "hello-world-agent",
        "profile_root": str(specialist),
        "socket_path": str(socket_path),
        "socket_relative": "state/runtime/hello.sock",
        "startup_timeout_seconds": 2,
        "turn_timeout_seconds": 30,
        "reset_timeout_seconds": 45,
        "queue_limit_per_conversation": 4,
        "idle_conversation_seconds": 3600,
    }]
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(profile_root, tmp_path))

    class Process:
        returncode = None

        def __init__(self):
            self.terminated = False

        def terminate(self):
            self.terminated = True
            self.returncode = 0

        async def wait(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = Process()
    ready_sockets = []
    create = AsyncMock(
        side_effect=_create_process_with_ready_socket(process, socket_path, ready_sockets),
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    monkeypatch.setenv("HERMES_HOME", str(profile_root))
    monkeypatch.setenv("TERMINAL_CWD", str(profile_root))

    await adapter._start_telegram_bridge_persistent_workers()

    assert adapter._telegram_bridge_persistent_workers == {"hello_world": process}
    assert create.await_args.args[:3] == (
        sys.executable,
        "-m",
        "hermes_cli.persistent_specialist_worker",
    )
    assert create.await_args.kwargs["cwd"] == str(specialist)
    assert create.await_args.kwargs["env"]["HERMES_HOME"] == str(specialist)
    assert create.await_args.kwargs["env"]["TERMINAL_CWD"] == str(specialist)
    assert os.environ["HERMES_HOME"] == str(profile_root)
    assert os.environ["TERMINAL_CWD"] == str(profile_root)

    await adapter._stop_telegram_bridge_persistent_workers()

    assert process.terminated is True
    assert adapter._telegram_bridge_persistent_workers == {}
    for ready_socket in ready_sockets:
        ready_socket.close()


def _persistent_spec(tmp_path, agent_id):
    """Build one validated persistent-service spec for lifecycle tests."""
    profile_root = tmp_path / "profiles" / agent_id
    socket_path = tmp_path / f"{agent_id}.sock"
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.touch()
    return {
        "agent_id": agent_id,
        "service_id": f"{agent_id}-agent",
        "profile_root": str(profile_root),
        "socket_path": str(socket_path),
        "socket_relative": "state/runtime/worker.sock",
        "startup_timeout_seconds": 2,
        "turn_timeout_seconds": 30,
        "reset_timeout_seconds": 45,
        "queue_limit_per_conversation": 4,
        "idle_conversation_seconds": 3600,
    }


class _PersistentProcess:
    """Controllable asyncio subprocess double for lifecycle tests."""

    def __init__(self, pid, *, wait_timeouts=0):
        self.pid = pid
        self.returncode = None
        self.wait_timeouts = wait_timeouts
        self.wait_calls = 0

    async def wait(self):
        self.wait_calls += 1
        if self.wait_calls <= self.wait_timeouts:
            raise asyncio.TimeoutError
        self.returncode = 0
        return self.returncode


def _create_process_with_ready_socket(process, socket_path, opened_sockets):
    """Return a subprocess factory that creates the worker's Unix socket."""
    async def create(*_args, **_kwargs):
        socket_path.unlink(missing_ok=True)
        ready_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        ready_socket.bind(str(socket_path))
        opened_sockets.append(ready_socket)
        return process

    return create


async def _await_recovery(adapter, agent_id="myhomestead"):
    """Await an adapter-owned recovery task captured before its done callback."""
    task = adapter._telegram_bridge_persistent_worker_recovery_tasks[agent_id]
    await task


@pytest.mark.asyncio
async def test_terminal_persistent_error_replaces_only_affected_worker(tmp_path, monkeypatch):
    adapter = _make_adapter()
    stuck = _PersistentProcess(1101)
    unrelated = _PersistentProcess(1102)
    replacement = _PersistentProcess(1103)
    adapter._telegram_bridge_persistent_workers = {
        "myhomestead": stuck,
        "healthmesh": unrelated,
    }
    adapter._telegram_bridge_persistent_worker_specs = {
        "myhomestead": _persistent_spec(tmp_path, "myhomestead"),
        "healthmesh": _persistent_spec(tmp_path, "healthmesh"),
    }
    signals = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    ready_sockets = []
    create = AsyncMock(
        side_effect=_create_process_with_ready_socket(
            replacement,
            Path(adapter._telegram_bridge_persistent_worker_specs["myhomestead"]["socket_path"]),
            ready_sockets,
        ),
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    recovered = await adapter._recover_telegram_bridge_persistent_worker(PersistentRecoveryRequired())
    await _await_recovery(adapter)

    assert recovered is True
    assert signals == [(stuck.pid, signal.SIGTERM)]
    assert adapter._telegram_bridge_persistent_workers == {
        "myhomestead": replacement,
        "healthmesh": unrelated,
    }
    assert create.await_count == 1
    assert create.await_args.kwargs["start_new_session"] is True
    for ready_socket in ready_sockets:
        ready_socket.close()


@pytest.mark.asyncio
async def test_spawn_waits_for_child_to_replace_preexisting_stale_socket(tmp_path, monkeypatch):
    adapter = _make_adapter()
    spec = _persistent_spec(tmp_path, "myhomestead")
    socket_path = tmp_path / "worker.sock"
    spec["socket_path"] = str(socket_path)
    stale_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale_socket.bind(str(socket_path))
    stale_stat = socket_path.stat()
    stale_identity = (stale_stat.st_dev, stale_stat.st_ino, stale_stat.st_ctime_ns)
    process = _PersistentProcess(1151)
    replacement_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

    async def create_replacing_child(*_args, **_kwargs):
        async def replace_socket():
            await asyncio.sleep(0.02)
            stale_socket.close()
            socket_path.unlink()
            replacement_socket.bind(str(socket_path))

        asyncio.create_task(replace_socket())
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_replacing_child)
    try:
        spawned = await adapter._spawn_telegram_bridge_persistent_worker(spec)
        replacement_stat = socket_path.stat()
        replacement_identity = (
            replacement_stat.st_dev,
            replacement_stat.st_ino,
            replacement_stat.st_ctime_ns,
        )
    finally:
        stale_socket.close()
        replacement_socket.close()

    assert spawned is process
    assert replacement_identity != stale_identity


@pytest.mark.asyncio
async def test_spawn_never_accepts_preexisting_regular_file_as_socket(tmp_path, monkeypatch):
    from plugins.platforms.telegram import adapter as telegram_adapter

    adapter = _make_adapter()
    spec = _persistent_spec(tmp_path, "myhomestead")
    spec["startup_timeout_seconds"] = 0.01
    socket_path = Path(spec["socket_path"])
    socket_path.write_text("preserve me", encoding="utf-8")
    process = _PersistentProcess(1171)
    signals = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    monkeypatch.setattr(telegram_adapter, "_PERSISTENT_WORKER_STOP_TIMEOUT", 0.01)

    with pytest.raises(TimeoutError, match="startup timed out"):
        await adapter._spawn_telegram_bridge_persistent_worker(spec)

    assert socket_path.read_text(encoding="utf-8") == "preserve me"
    assert signals == [(process.pid, signal.SIGTERM)]


@pytest.mark.asyncio
async def test_cancelled_spawn_reaps_child_created_during_cancellation(tmp_path, monkeypatch):
    adapter = _make_adapter()
    spec = _persistent_spec(tmp_path, "myhomestead")
    process = _PersistentProcess(1181)
    create_started = asyncio.Event()
    release_create = asyncio.Event()
    signals = []

    async def delayed_create(*_args, **_kwargs):
        create_started.set()
        await release_create.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_create)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    spawn = asyncio.create_task(adapter._spawn_telegram_bridge_persistent_worker(spec))
    await create_started.wait()
    spawn.cancel()
    release_create.set()

    with pytest.raises(asyncio.CancelledError):
        await spawn

    assert signals == [(process.pid, signal.SIGTERM)]
    assert process.returncode == 0


@pytest.mark.asyncio
async def test_cancelled_spawn_preserves_cancellation_when_cleanup_fails(tmp_path, monkeypatch):
    adapter = _make_adapter()
    spec = _persistent_spec(tmp_path, "myhomestead")
    process = _PersistentProcess(1182)
    child_created = asyncio.Event()

    async def create_without_socket(*_args, **_kwargs):
        child_created.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_without_socket)
    adapter._terminate_telegram_bridge_persistent_worker = AsyncMock(
        side_effect=RuntimeError("cleanup failed")
    )
    spawn = asyncio.create_task(adapter._spawn_telegram_bridge_persistent_worker(spec))
    await child_created.wait()
    await asyncio.sleep(0)
    spawn.cancel()

    with pytest.raises(asyncio.CancelledError):
        await spawn


@pytest.mark.asyncio
async def test_duplicate_terminal_signal_does_not_replace_worker_twice(tmp_path, monkeypatch):
    adapter = _make_adapter()
    stuck = _PersistentProcess(1201)
    replacement = _PersistentProcess(1202)
    adapter._telegram_bridge_persistent_workers = {"myhomestead": stuck}
    adapter._telegram_bridge_persistent_worker_specs = {
        "myhomestead": _persistent_spec(tmp_path, "myhomestead"),
    }
    monkeypatch.setattr(os, "killpg", lambda _pid, _sig: None)
    ready_sockets = []
    create = AsyncMock(
        side_effect=_create_process_with_ready_socket(
            replacement,
            Path(adapter._telegram_bridge_persistent_worker_specs["myhomestead"]["socket_path"]),
            ready_sockets,
        ),
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    error = PersistentRecoveryRequired()

    assert await adapter._recover_telegram_bridge_persistent_worker(error) is True
    assert await adapter._recover_telegram_bridge_persistent_worker(error) is True
    await _await_recovery(adapter)

    assert create.await_count == 1
    assert adapter._telegram_bridge_persistent_workers["myhomestead"] is replacement
    for ready_socket in ready_sockets:
        ready_socket.close()


@pytest.mark.asyncio
async def test_nonterminal_or_incomplete_recovery_signal_is_ignored():
    adapter = _make_adapter()
    adapter._telegram_bridge_persistent_workers = {"myhomestead": _PersistentProcess(1251)}
    unsupported = PersistentRecoveryRequired()
    unsupported.code = "DEADLINE_EXPIRED"
    incomplete = PersistentRecoveryRequired()
    incomplete.recovery_runtime_instance_id = ""

    assert await adapter._recover_telegram_bridge_persistent_worker(unsupported) is False
    assert await adapter._recover_telegram_bridge_persistent_worker(incomplete) is False

    assert adapter._telegram_bridge_persistent_workers["myhomestead"].pid == 1251


@pytest.mark.asyncio
async def test_persistent_recovery_force_kills_uninterruptible_worker_group(tmp_path, monkeypatch):
    adapter = _make_adapter()
    stuck = _PersistentProcess(1301, wait_timeouts=1)
    replacement = _PersistentProcess(1302)
    adapter._telegram_bridge_persistent_workers = {"myhomestead": stuck}
    adapter._telegram_bridge_persistent_worker_specs = {
        "myhomestead": _persistent_spec(tmp_path, "myhomestead"),
    }
    signals = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    ready_sockets = []
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        AsyncMock(
            side_effect=_create_process_with_ready_socket(
                replacement,
                Path(adapter._telegram_bridge_persistent_worker_specs["myhomestead"]["socket_path"]),
                ready_sockets,
            ),
        ),
    )

    assert await adapter._recover_telegram_bridge_persistent_worker(PersistentRecoveryRequired()) is True
    await _await_recovery(adapter)

    assert signals == [(stuck.pid, signal.SIGTERM), (stuck.pid, signal.SIGKILL)]
    assert adapter._telegram_bridge_persistent_workers["myhomestead"] is replacement
    for ready_socket in ready_sockets:
        ready_socket.close()


@pytest.mark.asyncio
async def test_persistent_recovery_fails_when_worker_cannot_be_reaped_after_sigkill(
    tmp_path,
    monkeypatch,
    caplog,
):
    from plugins.platforms.telegram import adapter as telegram_adapter

    adapter = _make_adapter()

    class UnreapableProcess(_PersistentProcess):
        """Process double whose forced reap never completes."""

        async def wait(self):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise asyncio.TimeoutError
            await asyncio.Event().wait()

    stuck = UnreapableProcess(1351)
    adapter._telegram_bridge_persistent_workers = {"myhomestead": stuck}
    adapter._telegram_bridge_persistent_worker_specs = {
        "myhomestead": _persistent_spec(tmp_path, "myhomestead"),
    }
    signals = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(telegram_adapter, "_PERSISTENT_WORKER_KILL_TIMEOUT", 0.01, raising=False)
    create = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    with pytest.raises(TimeoutError, match="could not be reaped"):
        await adapter._terminate_telegram_bridge_persistent_worker("myhomestead", stuck)

    assert signals == [(stuck.pid, signal.SIGTERM), (stuck.pid, signal.SIGKILL)]
    assert create.await_count == 0
    assert "could not be reaped after SIGKILL" in caplog.text


@pytest.mark.asyncio
async def test_termination_does_not_signal_recycled_group_for_already_dead_leader(monkeypatch):
    adapter = _make_adapter()
    process = _PersistentProcess(1361)
    process.returncode = 0
    signals = []
    group_exists = MagicMock(return_value=True)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(
        adapter,
        "_telegram_bridge_process_group_exists",
        group_exists,
    )

    forced = await adapter._terminate_telegram_bridge_persistent_worker(
        "myhomestead", process
    )

    assert forced is False
    assert signals == []
    group_exists.assert_not_called()


@pytest.mark.asyncio
async def test_termination_kills_descendants_after_signaled_leader_exits(monkeypatch):
    adapter = _make_adapter()
    process = _PersistentProcess(1362)
    group_alive = True
    signals = []

    def killpg(pid, sig):
        nonlocal group_alive
        signals.append((pid, sig))
        if sig == signal.SIGKILL:
            group_alive = False

    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(
        adapter,
        "_telegram_bridge_process_group_exists",
        lambda _pid: group_alive,
    )

    forced = await adapter._terminate_telegram_bridge_persistent_worker(
        "myhomestead", process
    )

    assert forced is True
    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]


@pytest.mark.asyncio
async def test_termination_preserves_cancellation_when_cleanup_fails(monkeypatch):
    adapter = _make_adapter()
    process = _PersistentProcess(1363)
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def failing_cleanup(_agent_id, _process):
        cleanup_started.set()
        await release_cleanup.wait()
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(
        adapter,
        "_terminate_telegram_bridge_persistent_worker_impl",
        failing_cleanup,
    )
    termination = asyncio.create_task(
        adapter._terminate_telegram_bridge_persistent_worker("myhomestead", process)
    )
    await cleanup_started.wait()
    termination.cancel()
    release_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await termination


@pytest.mark.asyncio
async def test_shutdown_winning_lifecycle_lock_prevents_recovery_replacement(tmp_path, monkeypatch):
    adapter = _make_adapter()
    stuck = _PersistentProcess(1371)
    adapter._telegram_bridge_persistent_workers = {"myhomestead": stuck}
    adapter._telegram_bridge_persistent_worker_specs = {
        "myhomestead": _persistent_spec(tmp_path, "myhomestead"),
    }
    adapter._telegram_bridge_persistent_workers_shutdown = False
    signals = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    create = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    lifecycle_lock = adapter._get_telegram_bridge_persistent_worker_lifecycle_lock()
    await lifecycle_lock.acquire()
    shutdown_task = asyncio.create_task(adapter._stop_telegram_bridge_persistent_workers())
    await asyncio.sleep(0)
    recovery_task = asyncio.create_task(
        adapter._recover_telegram_bridge_persistent_worker(PersistentRecoveryRequired())
    )
    await asyncio.sleep(0)
    lifecycle_lock.release()

    await asyncio.gather(shutdown_task, recovery_task)

    assert signals == [(stuck.pid, signal.SIGTERM)]
    assert create.await_count == 0
    assert adapter._telegram_bridge_persistent_workers == {}


@pytest.mark.asyncio
async def test_stop_signals_other_workers_when_one_cannot_be_reaped(monkeypatch, caplog):
    from plugins.platforms.telegram import adapter as telegram_adapter

    adapter = _make_adapter()

    class UnreapableProcess(_PersistentProcess):
        """Process double that remains stuck after both termination signals."""

        async def wait(self):
            await asyncio.Event().wait()

    stuck = UnreapableProcess(1381)
    healthy = _PersistentProcess(1382)
    adapter._telegram_bridge_persistent_workers = {
        "myhomestead": stuck,
        "healthmesh": healthy,
    }
    signals = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(telegram_adapter, "_PERSISTENT_WORKER_STOP_TIMEOUT", 0.01)
    monkeypatch.setattr(telegram_adapter, "_PERSISTENT_WORKER_KILL_TIMEOUT", 0.01)

    await asyncio.wait_for(adapter._stop_telegram_bridge_persistent_workers(), timeout=0.1)

    assert (stuck.pid, signal.SIGTERM) in signals
    assert (stuck.pid, signal.SIGKILL) in signals
    assert (healthy.pid, signal.SIGTERM) in signals
    assert healthy.returncode == 0
    assert adapter._telegram_bridge_persistent_workers == {}
    assert "Persistent specialist worker shutdown failed" in caplog.text


@pytest.mark.asyncio
async def test_persistent_recovery_retries_failed_replacement_without_blocking_handler(
    tmp_path, monkeypatch, caplog
):
    from plugins.platforms.telegram import adapter as telegram_adapter

    adapter = _make_adapter()
    stuck = _PersistentProcess(1401)
    replacement = _PersistentProcess(1402)
    adapter._telegram_bridge_persistent_workers = {"myhomestead": stuck}
    adapter._telegram_bridge_persistent_worker_specs = {
        "myhomestead": _persistent_spec(tmp_path, "myhomestead"),
    }
    monkeypatch.setattr(os, "killpg", lambda _pid, _sig: None)
    monkeypatch.setattr(telegram_adapter, "_PERSISTENT_WORKER_RETRY_INITIAL_DELAY", 0)
    ready_sockets = []
    ready_create = _create_process_with_ready_socket(
        replacement,
        Path(adapter._telegram_bridge_persistent_worker_specs["myhomestead"]["socket_path"]),
        ready_sockets,
    )
    attempts = 0

    async def create(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("replacement unavailable")
        return await ready_create(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    assert await asyncio.wait_for(
        adapter._recover_telegram_bridge_persistent_worker(PersistentRecoveryRequired()),
        timeout=0.1,
    ) is True
    await _await_recovery(adapter)

    assert attempts == 2
    assert adapter._telegram_bridge_persistent_workers["myhomestead"] is replacement
    assert "recovery failed; retrying" in caplog.text
    for ready_socket in ready_sockets:
        ready_socket.close()


@pytest.mark.asyncio
async def test_shutdown_cancels_recovery_and_reaps_unpublished_replacement(
    tmp_path, monkeypatch
):
    adapter = _make_adapter()
    stuck = _PersistentProcess(1411)
    replacement = _PersistentProcess(1412)
    adapter._telegram_bridge_persistent_workers = {"myhomestead": stuck}
    adapter._telegram_bridge_persistent_worker_specs = {
        "myhomestead": _persistent_spec(tmp_path, "myhomestead"),
    }
    replacement_created = asyncio.Event()
    signals = []

    async def create_without_socket(*_args, **_kwargs):
        replacement_created.set()
        return replacement

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_without_socket)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    await adapter._recover_telegram_bridge_persistent_worker(PersistentRecoveryRequired())
    await replacement_created.wait()
    await adapter._stop_telegram_bridge_persistent_workers()

    assert (stuck.pid, signal.SIGTERM) in signals
    assert (replacement.pid, signal.SIGTERM) in signals
    assert adapter._telegram_bridge_persistent_workers == {}
    assert adapter._telegram_bridge_persistent_worker_recovery_tasks == {}


@pytest.mark.asyncio
async def test_recovery_does_not_sleep_after_spawn_failure_publishes_shutdown(
    tmp_path, monkeypatch
):
    adapter = _make_adapter()
    stuck = _PersistentProcess(1413)
    adapter._telegram_bridge_persistent_workers = {"myhomestead": stuck}
    adapter._telegram_bridge_persistent_worker_specs = {
        "myhomestead": _persistent_spec(tmp_path, "myhomestead"),
    }
    monkeypatch.setattr(os, "killpg", lambda _pid, _sig: None)

    async def fail_during_shutdown(*_args, **_kwargs):
        adapter._telegram_bridge_persistent_workers_shutdown = True
        raise RuntimeError("shutdown won")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_during_shutdown)

    await adapter._recover_telegram_bridge_persistent_worker(PersistentRecoveryRequired())
    await asyncio.wait_for(_await_recovery(adapter), timeout=0.1)

    assert adapter._telegram_bridge_persistent_workers == {}


@pytest.mark.asyncio
async def test_dispatch_snapshot_prevents_delayed_old_runtime_from_killing_replacement(
    tmp_path, monkeypatch
):
    adapter = _make_adapter()
    old_process = _PersistentProcess(1421)
    replacement = _PersistentProcess(1422)
    adapter._telegram_bridge_persistent_workers = {"myhomestead": replacement}
    adapter._telegram_bridge_persistent_worker_specs = {
        "myhomestead": _persistent_spec(tmp_path, "myhomestead"),
    }
    signals = []
    create = AsyncMock()
    monkeypatch.setattr(os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    assert await adapter._recover_telegram_bridge_persistent_worker(
        PersistentRecoveryRequired(),
        {"myhomestead": old_process},
    ) is True

    assert signals == []
    assert create.await_count == 0
    assert adapter._telegram_bridge_persistent_workers["myhomestead"] is replacement


@pytest.mark.asyncio
async def test_bridge_text_consumes_terminal_recovery_signal(tmp_path):
    adapter = _make_adapter()
    adapter._bot.username = "bridge_bot"
    adapter.send = AsyncMock()
    adapter._recover_telegram_bridge_persistent_worker = AsyncMock(return_value=True)
    worker = _PersistentProcess(1501)
    adapter._telegram_bridge_persistent_workers = {"myhomestead": worker}
    dispatcher = MagicMock()
    error = PersistentRecoveryRequired()
    dispatcher.dispatch_text.side_effect = error
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(tmp_path / "bridge", tmp_path))
    msg = SimpleNamespace(
        text="check the property",
        chat_id=123,
        message_id=456,
        from_user=SimpleNamespace(id=789),
        chat=SimpleNamespace(id=123),
        reply_to_message=None,
    )

    assert await adapter._maybe_handle_telegram_bridge_text(SimpleNamespace(update_id=1), msg) is True

    adapter._recover_telegram_bridge_persistent_worker.assert_awaited_once_with(
        error,
        {"myhomestead": worker},
    )
    adapter.send.assert_awaited_once_with(
        "123",
        "I couldn't complete that request just now. Please try again.",
    )


@pytest.mark.asyncio
async def test_bridge_command_does_not_repair_or_send_after_ambiguous_delivery(tmp_path):
    adapter = _make_adapter()
    result = SimpleNamespace(
        handled=True,
        payloads=[{"action": "send", "message_id": "msg_command"}],
        bridge_config={},
        envelope={"event_id": "evt_command"},
        reason="specialist",
    )
    dispatcher = MagicMock()
    dispatcher.can_handle_command.return_value = True
    dispatcher.dispatch_command.return_value = result
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(tmp_path / "bridge", tmp_path))
    adapter._render_telegram_bridge_payloads = AsyncMock(side_effect=DeliveryOutcomeUnknown())
    adapter._rerender_telegram_bridge_with_repair = AsyncMock()
    adapter.send = AsyncMock()
    msg = SimpleNamespace(
        text="/hello",
        chat_id=123,
        message_id=456,
        from_user=SimpleNamespace(id=789),
        chat=SimpleNamespace(id=123),
    )

    assert await adapter._maybe_handle_telegram_bridge_command(SimpleNamespace(update_id=1), msg)
    adapter._rerender_telegram_bridge_with_repair.assert_not_awaited()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_bridge_text_does_not_repair_or_send_after_ambiguous_delivery(tmp_path):
    adapter = _make_adapter()
    result = SimpleNamespace(
        handled=True,
        payloads=[{"action": "send", "message_id": "msg_text"}],
        bridge_config={},
        envelope={"event_id": "evt_text"},
        reason="specialist_text",
    )
    dispatcher = MagicMock(dispatch_text=MagicMock(return_value=result))
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(tmp_path / "bridge", tmp_path))
    adapter._clean_bot_trigger_text = MagicMock(return_value="hello")
    adapter._render_telegram_bridge_payloads = AsyncMock(side_effect=DeliveryOutcomeUnknown())
    adapter._rerender_telegram_bridge_with_repair = AsyncMock()
    adapter.send = AsyncMock()
    msg = SimpleNamespace(
        text="hello",
        chat_id=123,
        message_id=456,
        from_user=SimpleNamespace(id=789),
        chat=SimpleNamespace(id=123),
        reply_to_message=None,
    )

    assert await adapter._maybe_handle_telegram_bridge_text(SimpleNamespace(update_id=2), msg)
    adapter._rerender_telegram_bridge_with_repair.assert_not_awaited()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_bridge_callback_does_not_repair_or_answer_after_ambiguous_delivery(tmp_path):
    adapter = _make_adapter()
    result = SimpleNamespace(
        handled=True,
        payloads=[{"action": "answer_callback", "message_id": "msg_callback"}],
        bridge_config={},
        envelope={"event_id": "evt_callback"},
        reason="specialist_callback",
    )
    dispatcher = MagicMock(dispatch_callback=MagicMock(return_value=result))
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(tmp_path / "bridge", tmp_path))
    adapter._render_telegram_bridge_payloads = AsyncMock(side_effect=DeliveryOutcomeUnknown())
    adapter._rerender_telegram_bridge_with_repair = AsyncMock()
    query = SimpleNamespace(
        id="cbq_1",
        data="hello.action",
        from_user=SimpleNamespace(id=789),
        message=SimpleNamespace(chat_id=123, message_id=456, text="button"),
        answer=AsyncMock(),
    )

    assert await adapter._maybe_handle_telegram_bridge_callback(SimpleNamespace(update_id=3), query)
    adapter._rerender_telegram_bridge_with_repair.assert_not_awaited()
    query.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_bridge_command_exception_sends_safe_user_message(tmp_path):
    adapter = _make_adapter()
    dispatcher = MagicMock()
    dispatcher.can_handle_command.return_value = True
    dispatcher.dispatch_command.side_effect = RuntimeError("sensitive command failure")
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(tmp_path / "bridge", tmp_path))
    adapter.send = AsyncMock()
    msg = SimpleNamespace(
        text="/hello",
        chat_id=123,
        message_id=456,
        from_user=SimpleNamespace(id=789),
        chat=SimpleNamespace(id=123),
    )

    assert await adapter._maybe_handle_telegram_bridge_command(SimpleNamespace(update_id=4), msg)
    adapter.send.assert_awaited_once_with(
        "123",
        "I couldn't complete that request just now. Please try again.",
    )


@pytest.mark.asyncio
async def test_bridge_text_exception_sends_safe_user_message(tmp_path):
    adapter = _make_adapter()
    dispatcher = MagicMock()
    dispatcher.dispatch_text.side_effect = RuntimeError("sensitive text failure")
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(tmp_path / "bridge", tmp_path))
    adapter._clean_bot_trigger_text = MagicMock(return_value="hello")
    adapter.send = AsyncMock()
    msg = SimpleNamespace(
        text="hello",
        chat_id=123,
        message_id=456,
        from_user=SimpleNamespace(id=789),
        chat=SimpleNamespace(id=123),
        reply_to_message=None,
    )

    assert await adapter._maybe_handle_telegram_bridge_text(SimpleNamespace(update_id=5), msg)
    adapter.send.assert_awaited_once_with(
        "123",
        "I couldn't complete that request just now. Please try again.",
    )


@pytest.mark.asyncio
async def test_bridge_callback_exception_alerts_with_safe_user_message(tmp_path):
    adapter = _make_adapter()
    dispatcher = MagicMock()
    dispatcher.dispatch_callback.side_effect = RuntimeError("sensitive callback failure")
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(tmp_path / "bridge", tmp_path))
    query = SimpleNamespace(
        id="cbq_2",
        data="hello.action",
        from_user=SimpleNamespace(id=789),
        message=SimpleNamespace(chat_id=123, message_id=456, text="button"),
        answer=AsyncMock(),
    )

    assert await adapter._maybe_handle_telegram_bridge_callback(SimpleNamespace(update_id=6), query)
    query.answer.assert_awaited_once_with(
        text="I couldn't complete that action. Please try again.",
        show_alert=True,
    )


@pytest.mark.asyncio
async def test_bridge_acknowledged_callback_dispatch_exception_sends_safe_chat_message(tmp_path):
    adapter = _make_adapter()
    dispatcher = MagicMock()
    dispatcher.is_fast_bridge_callback.return_value = True
    dispatcher.fast_bridge_callback_answer.return_value = "Starting with fresh context…"
    dispatcher.dispatch_callback.side_effect = RuntimeError("sensitive callback failure")
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(tmp_path / "bridge", tmp_path))
    adapter.send = AsyncMock()
    query = SimpleNamespace(
        id="cbq_acknowledged_failure",
        data="tba_ctx.myhomestead.token.f",
        from_user=SimpleNamespace(id=789),
        message=SimpleNamespace(chat_id=123, message_id=456, text="choice"),
        answer=AsyncMock(),
    )

    assert await adapter._maybe_handle_telegram_bridge_callback(
        SimpleNamespace(update_id=61),
        query,
    )

    query.answer.assert_awaited_once_with(text="Starting with fresh context…")
    adapter.send.assert_awaited_once_with(
        "123",
        "I couldn't complete that request just now. Please try again.",
    )


@pytest.mark.asyncio
async def test_bridge_command_failed_render_retry_does_not_send_generic_message(tmp_path):
    """A failed render repair is not followed by a second user-visible message."""
    adapter = _make_adapter()
    result = SimpleNamespace(
        handled=True,
        payloads=[{"action": "send", "message_id": "msg_command"}],
        bridge_config={},
        envelope={"event_id": "evt_command"},
        reason="specialist",
    )
    dispatcher = MagicMock()
    dispatcher.can_handle_command.return_value = True
    dispatcher.dispatch_command.return_value = result
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(tmp_path / "bridge", tmp_path))
    adapter._render_telegram_bridge_payloads = AsyncMock(side_effect=RuntimeError("render failed"))
    adapter._rerender_telegram_bridge_with_repair = AsyncMock(return_value=False)
    adapter.send = AsyncMock()
    msg = SimpleNamespace(
        text="/hello",
        chat_id=123,
        message_id=456,
        from_user=SimpleNamespace(id=789),
        chat=SimpleNamespace(id=123),
    )

    assert await adapter._maybe_handle_telegram_bridge_command(SimpleNamespace(update_id=7), msg)
    adapter._rerender_telegram_bridge_with_repair.assert_awaited_once()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_bridge_text_failed_render_retry_does_not_send_generic_message(tmp_path):
    """A failed render repair is not followed by a second user-visible message."""
    adapter = _make_adapter()
    result = SimpleNamespace(
        handled=True,
        payloads=[{"action": "send", "message_id": "msg_text"}],
        bridge_config={},
        envelope={"event_id": "evt_text"},
        reason="specialist_text",
    )
    dispatcher = MagicMock(dispatch_text=MagicMock(return_value=result))
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(tmp_path / "bridge", tmp_path))
    adapter._clean_bot_trigger_text = MagicMock(return_value="hello")
    adapter._render_telegram_bridge_payloads = AsyncMock(side_effect=RuntimeError("render failed"))
    adapter._rerender_telegram_bridge_with_repair = AsyncMock(return_value=False)
    adapter.send = AsyncMock()
    msg = SimpleNamespace(
        text="hello",
        chat_id=123,
        message_id=456,
        from_user=SimpleNamespace(id=789),
        chat=SimpleNamespace(id=123),
        reply_to_message=None,
    )

    assert await adapter._maybe_handle_telegram_bridge_text(SimpleNamespace(update_id=8), msg)
    adapter._rerender_telegram_bridge_with_repair.assert_awaited_once()
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_bridge_callback_failed_render_retry_answers_without_alert(tmp_path):
    """A failed render repair dismisses the callback spinner without an alert."""
    adapter = _make_adapter()
    result = SimpleNamespace(
        handled=True,
        payloads=[{"action": "send", "message_id": "msg_callback"}],
        bridge_config={},
        envelope={"event_id": "evt_callback"},
        reason="specialist_callback",
    )
    dispatcher = MagicMock(dispatch_callback=MagicMock(return_value=result))
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(tmp_path / "bridge", tmp_path))
    adapter._render_telegram_bridge_payloads = AsyncMock(side_effect=RuntimeError("render failed"))
    adapter._rerender_telegram_bridge_with_repair = AsyncMock(return_value=False)
    query = SimpleNamespace(
        id="cbq_3",
        data="hello.action",
        from_user=SimpleNamespace(id=789),
        message=SimpleNamespace(chat_id=123, message_id=456, text="button"),
        answer=AsyncMock(),
    )

    assert await adapter._maybe_handle_telegram_bridge_callback(SimpleNamespace(update_id=9), query)
    adapter._rerender_telegram_bridge_with_repair.assert_awaited_once()
    query.answer.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_bridge_command_records_successful_delivery(tmp_path):
    adapter = _make_adapter()
    delivery_results = [{"action": "send", "status": "sent", "telegram_message_ids": [7001]}]
    result = SimpleNamespace(
        handled=True,
        payloads=[{"action": "send", "message_id": "msg_command"}],
        bridge_config={},
        envelope={"event_id": "evt_command"},
        reason="specialist",
    )
    dispatcher = MagicMock()
    dispatcher.can_handle_command.return_value = True
    dispatcher.dispatch_command.return_value = result
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    profile_root = tmp_path / "bridge"
    adapter._telegram_bridge_paths = MagicMock(return_value=(profile_root, tmp_path))
    adapter._render_telegram_bridge_payloads = AsyncMock(return_value=delivery_results)
    msg = SimpleNamespace(
        text="/hello",
        chat_id=123,
        message_id=456,
        from_user=SimpleNamespace(id=789),
        chat=SimpleNamespace(id=123),
    )

    assert await adapter._maybe_handle_telegram_bridge_command(SimpleNamespace(update_id=10), msg)

    dispatcher.record_specialist_delivery.assert_called_once_with(
        profile_root,
        result.envelope,
        delivery_results,
        dispatch_reason="specialist",
        rendered_payloads=result.payloads,
    )


@pytest.mark.asyncio
async def test_bridge_callback_fast_acknowledges_before_dispatch(tmp_path):
    adapter = _make_adapter()
    result = SimpleNamespace(
        handled=True,
        payloads=[{"action": "send", "message_id": "msg_boundary"}],
        bridge_config={},
        envelope={"event_id": "evt_boundary"},
        reason="conversation_boundary_continue",
    )
    query = SimpleNamespace(
        id="cbq_boundary",
        data="tba_ctx.myhomestead.token.c",
        from_user=SimpleNamespace(id=789),
        message=SimpleNamespace(chat_id=123, message_id=456, text="choice"),
        answer=AsyncMock(),
    )

    def dispatch_callback(*_args, **kwargs):
        assert query.answer.await_count == 1
        assert kwargs["callback_answered"] is True
        return result

    dispatcher = MagicMock()
    dispatcher.is_fast_bridge_callback.side_effect = lambda data: data.startswith("tba_ctx.")
    dispatcher.fast_bridge_callback_answer.return_value = "Continuing your previous conversation…"
    dispatcher.dispatch_callback.side_effect = dispatch_callback
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    adapter._telegram_bridge_paths = MagicMock(return_value=(tmp_path / "bridge", tmp_path))
    adapter._render_telegram_bridge_payloads = AsyncMock(return_value=[])

    assert await adapter._maybe_handle_telegram_bridge_callback(SimpleNamespace(update_id=11), query)

    query.answer.assert_awaited_once_with(text="Continuing your previous conversation…")


@pytest.mark.asyncio
async def test_bridge_callback_records_repaired_delivery(tmp_path):
    adapter = _make_adapter()
    repaired_payloads = [{"action": "send", "message_id": "msg_repaired"}]
    repaired_results = [{
        "bridge_message_id": "msg_repaired",
        "action": "send",
        "status": "sent",
        "telegram_message_ids": [7002],
    }]
    result = SimpleNamespace(
        handled=True,
        payloads=[{"action": "send", "message_id": "msg_boundary"}],
        bridge_config={},
        envelope={"event_id": "evt_boundary"},
        reason="conversation_boundary_fresh",
    )
    dispatcher = MagicMock(dispatch_callback=MagicMock(return_value=result))
    dispatcher.is_fast_bridge_callback.return_value = True
    dispatcher.fast_bridge_callback_answer.return_value = "Starting with fresh context…"
    adapter._load_telegram_bridge_dispatcher = MagicMock(return_value=dispatcher)
    profile_root = tmp_path / "bridge"
    adapter._telegram_bridge_paths = MagicMock(return_value=(profile_root, tmp_path))
    adapter._render_telegram_bridge_payloads = AsyncMock(side_effect=RuntimeError("render failed"))
    adapter._rerender_telegram_bridge_with_repair = AsyncMock(
        return_value=(repaired_payloads, repaired_results)
    )
    query = SimpleNamespace(
        id="cbq_repaired",
        data="tba_ctx.myhomestead.token.f",
        from_user=SimpleNamespace(id=789),
        message=SimpleNamespace(chat_id=123, message_id=456, text="choice"),
        answer=AsyncMock(),
    )

    assert await adapter._maybe_handle_telegram_bridge_callback(SimpleNamespace(update_id=12), query)

    dispatcher.record_specialist_delivery.assert_called_once_with(
        profile_root,
        result.envelope,
        repaired_results,
        dispatch_reason="conversation_boundary_fresh",
        rendered_payloads=repaired_payloads,
    )
    query.answer.assert_awaited_once_with(text="Starting with fresh context…")
