"""Tests for Hermes Telegram adapter integration with the Telegram bridge."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig


def _make_adapter():
    from gateway.platforms.telegram import TelegramAdapter

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra={})
    adapter._bot = MagicMock()
    return adapter


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
    dispatcher.telegram_bot_commands.assert_called_once_with(profile_root, max_commands=30)


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
