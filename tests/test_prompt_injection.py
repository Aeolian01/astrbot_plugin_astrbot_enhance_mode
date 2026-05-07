from __future__ import annotations

import time

import pytest

from astrbot.api.platform import MessageType

import astrbot_plugin_astrbot_enhance_mode.main as main_module
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

    def set_extra(self, key: str, value: object) -> None:
        self._extras[key] = value


class _DummyRequest:
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.contexts: list[object] = ["old-context"]


class _DummyFace:
    id = 462


class _DummyImage:
    def __init__(self, *, url: str = "", file: str = "") -> None:
        self.url = url
        self.file = file


class Poke:
    pass


class _DummyPokeMessageObj:
    message_id = "poke-1"
    message = [Poke()]
    raw_message = {"user_id": "10001", "target_id": "99999"}

    class Sender:
        nickname = "Alice"
        user_id = "10001"

    sender = Sender()


class _DummyPokeEvent(_DummyEvent):
    unified_msg_origin = "origin-poke"
    message_obj = _DummyPokeMessageObj()
    message_str = ""

    def get_messages(self) -> list[object]:
        return self.message_obj.message

    def get_sender_id(self) -> str:
        return "10001"

    def get_sender_name(self) -> str:
        return "Alice"


class _DummyFaceMessageObj:
    message_id = "face-1"
    message = [_DummyFace()]

    class Sender:
        nickname = "Alice"

    sender = Sender()


class _DummyFaceEvent:
    unified_msg_origin = "origin-face"
    message_obj = _DummyFaceMessageObj()
    message_str = ""

    def get_message_type(self) -> MessageType:
        return MessageType.GROUP_MESSAGE

    def get_messages(self) -> list[object]:
        return self.message_obj.message

    def get_sender_id(self) -> str:
        return "10001"


class _DummyForwardMessageObj:
    message_id = "fwd-1"
    message: list[object] = []

    class Sender:
        nickname = "Alice"

    sender = Sender()


class _DummyForwardEvent(_DummyEvent):
    unified_msg_origin = "origin-forward"
    message_obj = _DummyForwardMessageObj()
    message_str = "[json]"

    def get_messages(self) -> list[object]:
        return []

    def get_sender_id(self) -> str:
        return "20001"


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


async def _consume_group_message(result: object) -> None:
    if hasattr(result, "__aiter__"):
        async for _ in result:  # type: ignore[attr-defined]
            pass
        return
    await result  # type: ignore[misc]


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
async def test_passive_injection_prefers_request_prompt_for_poke_event() -> None:
    plugin = _build_plugin()
    event = _DummyPokeEvent()
    req = _DummyRequest("Alice 戳了你一下，请用一句话回复。")

    await plugin.inject_group_context(event, req)

    assert "Now, a new message is coming:" in req.prompt
    assert "Alice 戳了你一下，请用一句话回复。" in req.prompt
    assert "[戳一戳: Alice/10001 -> 99999]" not in req.prompt
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

@pytest.mark.asyncio
async def test_model_choice_parses_forward_context_before_stack_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _build_plugin()
    cfg = PluginConfig(
        group_features=GroupFeatureEnhancementConfig(react_mode_enable=True),
        group_history=GroupHistoryEnhancementConfig(enable=True),
        active_reply=ActiveReplyConfig(
            enable=True,
            mode="model_choice",
            model_stack_size=1,
            model_history_messages=0,
        ),
    )
    event = _DummyForwardEvent()
    expanded = "expanded current forward text"
    prompts: list[str] = []

    plugin.runtime.session_chats[event.unified_msg_origin].append(
        "[Alice/20001/12:00:00] #msgfwd-1: [json]"
    )

    async def parse_current(parse_event: _DummyForwardEvent) -> str:
        assert parse_event is event
        parse_event.set_extra(main_module.FORWARD_CONTEXT_TEXT_KEY, expanded)
        parse_event.set_extra(main_module.FORWARD_CONTEXT_FOUND_KEY, True)
        parse_event.set_extra(main_module.FORWARD_CONTEXT_IDS_KEY, ["res-1"])
        return expanded

    class FakeProvider:
        async def text_chat(self, *, prompt: str, **_kwargs: object) -> object:
            prompts.append(prompt)

            class Response:
                completion_text = "SKIP"

            return Response()

    async def resolve_persona_mask(_event: _DummyForwardEvent) -> tuple[str, str]:
        return "default", "persona"

    monkeypatch.setattr(
        main_module,
        "_get_forward_context_api",
        lambda: {"parse_current_message": parse_current},
    )
    plugin._resolve_model_choice_provider = lambda _event, _cfg: FakeProvider()
    plugin._resolve_persona_mask = resolve_persona_mask

    assert await plugin._need_active_reply_model_choice(event, cfg) is False

    assert prompts
    assert expanded in prompts[0]
    assert "[json]" not in prompts[0]
    assert plugin.runtime.session_chats[event.unified_msg_origin] == [
        f"[Alice/20001/12:00:00] #msgfwd-1: {expanded}"
    ]


@pytest.mark.asyncio
async def test_group_message_with_only_face_component_is_not_skipped() -> None:
    plugin = _build_plugin()
    event = _DummyFaceEvent()
    called = {"record": False, "active": False}

    async def record_message(
        record_event: _DummyFaceEvent,
        _cfg: PluginConfig,
    ) -> None:
        called["record"] = True
        assert record_event is event

    async def need_active_reply(
        active_event: _DummyFaceEvent,
        _cfg: PluginConfig,
    ) -> bool:
        called["active"] = True
        assert active_event is event
        return False

    plugin._record_message = record_message
    plugin._need_active_reply = need_active_reply

    await _consume_group_message(plugin.on_group_message(event))

    assert called == {"record": True, "active": True}
    body, image_urls, image_cache_sources = plugin._format_event_message_body(event)
    assert body == "[QQ表情: id=462, 含义=未知]"
    assert image_urls == []
    assert image_cache_sources == []


def test_poke_message_body_includes_sender_and_target() -> None:
    plugin = _build_plugin()
    event = _DummyPokeEvent()

    body, image_urls, image_cache_sources = plugin._format_event_message_body(event)

    assert body == "[戳一戳: Alice/10001 -> 99999]"
    assert Main._is_action_only_text(body)
    assert image_urls == []
    assert image_cache_sources == []


@pytest.mark.asyncio
async def test_group_message_skips_active_reply_when_pending_but_records_history() -> None:
    plugin = _build_plugin()
    event = _DummyFaceEvent()
    called = {"record": False, "active": False}
    plugin.runtime.active_reply_pending[event.unified_msg_origin] = time.monotonic()

    async def record_message(
        record_event: _DummyFaceEvent,
        _cfg: PluginConfig,
    ) -> None:
        called["record"] = True
        assert record_event is event

    async def need_active_reply(
        _active_event: _DummyFaceEvent,
        _cfg: PluginConfig,
    ) -> bool:
        called["active"] = True
        raise AssertionError("_need_active_reply should not be called while pending")

    plugin._record_message = record_message
    plugin._need_active_reply = need_active_reply

    await _consume_group_message(plugin.on_group_message(event))

    assert called == {"record": True, "active": False}
    assert event.unified_msg_origin in plugin.runtime.active_reply_pending


def test_active_reply_pending_expires() -> None:
    plugin = _build_plugin()
    origin = "expired-origin"
    plugin.runtime.active_reply_pending[origin] = (
        time.monotonic() - main_module.ACTIVE_REPLY_PENDING_TTL_SEC - 1
    )

    assert plugin._has_active_reply_pending(origin) is False
    assert origin not in plugin.runtime.active_reply_pending


@pytest.mark.asyncio
async def test_group_message_clears_pending_when_active_reply_not_needed() -> None:
    plugin = _build_plugin()
    event = _DummyFaceEvent()
    called = {"record": False, "active": False}

    async def record_message(
        record_event: _DummyFaceEvent,
        _cfg: PluginConfig,
    ) -> None:
        called["record"] = True
        assert record_event is event

    async def need_active_reply(
        active_event: _DummyFaceEvent,
        _cfg: PluginConfig,
    ) -> bool:
        called["active"] = True
        assert active_event is event
        return False

    plugin._record_message = record_message
    plugin._need_active_reply = need_active_reply

    await _consume_group_message(plugin.on_group_message(event))

    assert called == {"record": True, "active": True}
    assert event.unified_msg_origin not in plugin.runtime.active_reply_pending


@pytest.mark.asyncio
async def test_after_message_sent_clears_active_reply_pending() -> None:
    plugin = _build_plugin()
    event = _DummyEvent()
    event.set_extra("_enhance_active_reply_triggered", True)
    plugin.runtime.active_reply_pending[event.unified_msg_origin] = time.monotonic()

    await plugin.after_message_sent(event)

    assert event.unified_msg_origin not in plugin.runtime.active_reply_pending


def test_image_message_uses_file_as_shared_caption_cache_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _build_plugin()
    monkeypatch.setattr(main_module, "Image", _DummyImage)

    class ImageEvent(_DummyEvent):
        def get_messages(self) -> list[object]:
            return [
                _DummyImage(
                    url="https://example.com/download?id=abc",
                    file="D486F2DB1F7B6087234AC5C1050723C6.png",
                )
            ]

    body, image_urls, image_cache_sources = plugin._format_event_message_body(
        ImageEvent()
    )

    assert body == "[Image]"
    assert image_urls == ["https://example.com/download?id=abc"]
    assert image_cache_sources == ["D486F2DB1F7B6087234AC5C1050723C6.png"]


def test_image_caption_sources_include_fileid_and_normalized_url() -> None:
    sources = Main._image_caption_sources(
        "https://example.com/get_image?fileid=abc&rkey=temp&token=secret&size=large",
        "D486F2DB1F7B6087234AC5C1050723C6.png",
    )

    assert sources == [
        "D486F2DB1F7B6087234AC5C1050723C6.png",
        "https://example.com/get_image?fileid=abc&rkey=temp&token=secret&size=large",
        "fileid:abc",
        "https://example.com/get_image?fileid=abc&size=large",
    ]


@pytest.mark.asyncio
async def test_shared_caption_cache_reads_alias_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _build_plugin()
    sources = Main._image_caption_sources(
        "https://example.com/get_image?fileid=abc&rkey=temp",
        "image.png",
    )

    async def get_cached(source_or_sources: object) -> str:
        assert source_or_sources == sources
        return "缓存图片描述"

    monkeypatch.setattr(
        main_module,
        "_get_forward_context_api",
        lambda: {"get_cached_image_caption": get_cached},
    )

    caption = await plugin._read_shared_image_caption_cache(sources)

    assert caption == "缓存图片描述"
