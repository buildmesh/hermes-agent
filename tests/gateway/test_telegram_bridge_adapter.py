"""Tests for Hermes Telegram adapter integration with the Telegram bridge."""

import asyncio
import os
import sys
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


class DeliveryOutcomeUnknown(RuntimeError):
    delivery_outcome_unknown = True


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
    socket_path = specialist / "state/runtime/hello.sock"
    socket_path.parent.mkdir(parents=True)
    socket_path.touch()

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
    create = AsyncMock(return_value=process)
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
