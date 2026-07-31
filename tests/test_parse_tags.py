from __future__ import annotations

import pytest

from astrbot.api.message_components import At, Plain
from astrbot.api.platform import MessageType

from astrbot_plugin_astrbot_enhance_mode.main import Main
from astrbot_plugin_astrbot_enhance_mode.plugin_config import (
    GroupFeatureEnhancementConfig,
    PluginConfig,
)
from astrbot_plugin_astrbot_enhance_mode.runtime_state import RuntimeState


class _DummyResult:
    def __init__(self, chain: list[object]) -> None:
        self.chain = chain


class _DummyDecoratingEvent:
    unified_msg_origin = "origin-dedupe"

    def __init__(
        self,
        chain: list[object],
        active_reply: bool,
        active_mode: str = "",
    ) -> None:
        self.result = _DummyResult(chain)
        self._extras = {
            "_enhance_active_reply_triggered": active_reply,
            "_enhance_active_reply_mode": active_mode,
        }

    def get_result(self) -> _DummyResult:
        return self.result

    def get_message_type(self) -> MessageType:
        return MessageType.GROUP_MESSAGE

    def get_extra(self, key: str, default: object = None) -> object:
        return self._extras.get(key, default)

    def set_extra(self, key: str, value: object) -> None:
        self._extras[key] = value


def _build_plugin(*, refuse_enable: bool = True) -> Main:
    plugin = Main.__new__(Main)
    plugin.runtime = RuntimeState()
    plugin._cfg = lambda: PluginConfig(
        group_features=GroupFeatureEnhancementConfig(
            mention_parse=True,
            refuse_enable=refuse_enable,
        )
    )
    return plugin


def _duplicated_chain() -> list[object]:
    return [
        At(qq="572325141"),
        Plain(text=" 哟，还没戳够？"),
        At(qq="572325141"),
        Plain(text=" 哟，还没戳够？"),
    ]


@pytest.mark.asyncio
async def test_parse_tags_collapses_duplicate_chain_for_active_reply() -> None:
    plugin = _build_plugin()
    event = _DummyDecoratingEvent(_duplicated_chain(), active_reply=True)

    await plugin.parse_tags(event)

    assert len(event.result.chain) == 2
    assert isinstance(event.result.chain[0], At)
    assert isinstance(event.result.chain[1], Plain)


@pytest.mark.asyncio
async def test_parse_tags_keeps_duplicate_chain_for_non_active_reply() -> None:
    plugin = _build_plugin()
    event = _DummyDecoratingEvent(_duplicated_chain(), active_reply=False)

    await plugin.parse_tags(event)

    assert len(event.result.chain) == 4


@pytest.mark.asyncio
async def test_single_pass_refuse_is_silenced_even_if_global_toggle_is_off() -> None:
    plugin = _build_plugin(refuse_enable=False)
    event = _DummyDecoratingEvent(
        [Plain(text="<refuse/>")],
        active_reply=True,
        active_mode="single_pass",
    )
    plugin.runtime.active_reply_pending[event.unified_msg_origin] = 1.0

    await plugin.parse_tags(event)

    assert event.result.chain == []
    assert event.get_extra("_enhance_refused_reply") is True
    assert event.unified_msg_origin not in plugin.runtime.active_reply_pending
