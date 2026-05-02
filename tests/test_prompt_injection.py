from __future__ import annotations

import pytest

from astrbot.api.platform import MessageType

from astrbot_plugin_astrbot_enhance_mode.main import Main
from astrbot_plugin_astrbot_enhance_mode.plugin_config import (
    ActiveReplyConfig,
    GroupFeatureEnhancementConfig,
    GroupHistoryEnhancementConfig,
    PluginConfig,
)
from astrbot_plugin_astrbot_enhance_mode.runtime_state import RuntimeState


class _DummyMessageObj:
    message_id = ""


class _DummyEvent:
    unified_msg_origin = "origin-1"
    message_obj = _DummyMessageObj()
    message_str = ""

    def __init__(self) -> None:
        self._extras: dict[str, object] = {}

    def get_message_type(self) -> MessageType:
        return MessageType.GROUP_MESSAGE

    def get_messages(self) -> list[object]:
        return []

    def get_extra(self, key: str, default: object = None) -> object:
        return self._extras.get(key, default)


class _DummyRequest:
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.contexts: list[object] = ["old-context"]


def _build_plugin() -> Main:
    plugin = Main.__new__(Main)
    plugin.runtime = RuntimeState()
    cfg = PluginConfig(
        group_features=GroupFeatureEnhancementConfig(react_mode_enable=True),
        group_history=GroupHistoryEnhancementConfig(enable=True),
        active_reply=ActiveReplyConfig(unified_context_messages=5),
    )
    plugin._cfg = lambda: cfg
    return plugin


@pytest.mark.asyncio
async def test_passive_injection_uses_request_prompt_for_empty_plugin_event() -> None:
    plugin = _build_plugin()
    event = _DummyEvent()
    req = _DummyRequest("Alice 戳了你一下，请用一句话回复。")
    plugin.runtime.session_chats[event.unified_msg_origin].extend(
        [
            "[Bob/1/12:00:00] #msg1: 刚才大家在聊午饭",
            "[Alice/2/12:00:01] #msg2: 有没有人想喝奶茶",
        ]
    )

    await plugin.inject_group_context(event, req)

    assert "Now, a new message is coming:" in req.prompt
    assert "Alice 戳了你一下，请用一句话回复。" in req.prompt
    assert "[Empty]" not in req.prompt
    assert req.contexts == []


@pytest.mark.asyncio
async def test_passive_injection_backfills_history_for_empty_plugin_event() -> None:
    plugin = _build_plugin()
    event = _DummyEvent()
    req = _DummyRequest("Alice 戳了你一下，请用一句话回复。")
    called = {"backfill": False}

    async def backfill_group_history(
        backfill_event: _DummyEvent,
        _cfg: PluginConfig,
        target_count: int,
    ) -> None:
        called["backfill"] = True
        assert backfill_event is event
        assert target_count == 5
        plugin.runtime.session_chats[event.unified_msg_origin].extend(
            [
                "[Bob/1/12:00:00] #msg1: 刚才大家在聊午饭",
                "[Alice/2/12:00:01] #msg2: 有没有人想喝奶茶",
            ]
        )

    plugin._backfill_group_history = backfill_group_history

    await plugin.inject_group_context(event, req)

    assert called["backfill"] is True
    assert "刚才大家在聊午饭" in req.prompt
    assert "有没有人想喝奶茶" in req.prompt
    assert "(no recent group chat history)" not in req.prompt
