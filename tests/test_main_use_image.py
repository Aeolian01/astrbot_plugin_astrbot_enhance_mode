from __future__ import annotations

import json
import types

from mcp import types as mcp_types
import pytest

import astrbot_plugin_astrbot_enhance_mode.main as main_module
from astrbot_plugin_astrbot_enhance_mode.main import Main
from astrbot_plugin_astrbot_enhance_mode.plugin_config import (
    GlobalSettingsConfig,
    GroupFeatureEnhancementConfig,
    GroupHistoryEnhancementConfig,
    PluginConfig,
    WebSearchConfig,
)
from astrbot_plugin_astrbot_enhance_mode.runtime_state import RuntimeState


class _DummyEvent:
    def __init__(self, origin: str) -> None:
        self.unified_msg_origin = origin


def _build_plugin(*, image_caption: bool) -> tuple[Main, _DummyEvent]:
    plugin = Main.__new__(Main)
    plugin.runtime = RuntimeState()
    plugin._image_caption_inflight = {}
    cfg = PluginConfig(
        group_history=GroupHistoryEnhancementConfig(enable=True, image_caption=image_caption),
        group_features=GroupFeatureEnhancementConfig(react_mode_enable=True),
        global_settings=GlobalSettingsConfig(),
    )
    plugin._cfg = lambda: cfg
    return plugin, _DummyEvent("origin-1")


def _payload_from_results(
    results: list[mcp_types.CallToolResult],
) -> dict[str, object]:
    return json.loads(results[-1].content[0].text)


def _forward_context_api(
    *,
    build_image_caption_sources=None,  # noqa: ANN001
    get_cached_image_caption=None,  # noqa: ANN001
    get_cached_image_message=None,  # noqa: ANN001
    get_or_create_image_caption=None,  # noqa: ANN001
    parse_history_message=None,  # noqa: ANN001
) -> dict[str, object]:
    async def empty_caption(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return ""

    async def empty_message(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return {}

    async def empty_parse(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return ""

    return {
        "build_image_caption_sources": build_image_caption_sources
        or main_module._fallback_build_image_caption_sources,
        "get_cached_image_caption": get_cached_image_caption or empty_caption,
        "get_cached_image_message": get_cached_image_message or empty_message,
        "get_or_create_image_caption": get_or_create_image_caption or empty_caption,
        "parse_history_message": parse_history_message or empty_parse,
    }


def test_forward_context_api_retries_after_initial_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build_image_caption_sources(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return []

    async def get_cached_image_caption(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return ""

    async def get_cached_image_message(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return {}

    async def get_or_create_image_caption(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return ""

    async def parse_history_message(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return ""

    fake_module = types.SimpleNamespace(
        build_image_caption_sources=build_image_caption_sources,
        get_cached_image_caption=get_cached_image_caption,
        get_cached_image_message=get_cached_image_message,
        get_or_create_image_caption=get_or_create_image_caption,
        parse_history_message=parse_history_message,
    )
    calls = {"count": 0}

    def import_module(name: str) -> object:
        assert name == "astrbot_plugin_forward_context"
        calls["count"] += 1
        if calls["count"] == 1:
            raise ModuleNotFoundError(name)
        return fake_module

    monkeypatch.setattr(main_module, "_FORWARD_CONTEXT_API", None)
    monkeypatch.setattr(main_module, "_FORWARD_CONTEXT_API_LAST_ERROR_LOG_AT", 0.0)
    monkeypatch.setattr(main_module, "_find_loaded_forward_context_api", lambda: None)
    monkeypatch.setattr(main_module.importlib, "import_module", import_module)

    assert main_module._get_forward_context_api() is None
    api = main_module._get_forward_context_api()

    assert api is not None
    assert api["build_image_caption_sources"] is build_image_caption_sources
    assert main_module._get_forward_context_api() is api
    assert calls["count"] == 2


def test_forward_context_api_uses_loaded_public_api_when_import_path_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build_image_caption_sources(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return []

    async def get_cached_image_caption(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return ""

    async def get_cached_image_message(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return {}

    async def get_or_create_image_caption(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return ""

    async def parse_history_message(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return ""

    loaded_module = types.SimpleNamespace(
        __file__="/opt/astrbot/plugins/astrbot_plugin_forward_context/public_api.py",
        build_image_caption_sources=build_image_caption_sources,
        get_cached_image_caption=get_cached_image_caption,
        get_cached_image_message=get_cached_image_message,
        get_or_create_image_caption=get_or_create_image_caption,
        parse_history_message=parse_history_message,
    )

    def import_module(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(main_module, "_FORWARD_CONTEXT_API", None)
    monkeypatch.setattr(main_module, "_FORWARD_CONTEXT_API_LAST_ERROR_LOG_AT", 0.0)
    monkeypatch.setattr(main_module.importlib, "import_module", import_module)
    for module_name in list(main_module.sys.modules):
        if "astrbot_plugin_forward_context" in module_name:
            monkeypatch.delitem(main_module.sys.modules, module_name, raising=False)
    monkeypatch.setitem(
        main_module.sys.modules,
        "astrbot.core.plugins.local.forward_context.public_api",
        loaded_module,
    )

    api = main_module._get_forward_context_api()

    assert api is not None
    assert api["get_cached_image_caption"] is get_cached_image_caption
    assert main_module._get_forward_context_api() is api


@pytest.mark.asyncio
async def test_use_image_attach_only_works_without_caption_enabled() -> None:
    plugin, event = _build_plugin(image_caption=False)

    async def should_not_be_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("_get_image_caption should not be called in attach-only mode")

    async def resolve_local_path(_image_ref: str) -> str:
        return "/tmp/fake-image.png"

    plugin._get_image_caption = should_not_be_called
    plugin._resolve_image_ref_to_local_path = resolve_local_path
    plugin._encode_image_file = lambda _path: ("ZmFrZQ==", "image/png")

    plugin.runtime.image_message_registry[event.unified_msg_origin]["123"] = {
        "urls": ["https://example.com/image.png"],
        "captions": {},
    }

    results = []
    async for item in plugin.use_image(
        event=event,
        message_id="123",
        image_index=1,
        attach_to_model=True,
        write_to_history=False,
        prompt="ignored",
    ):
        results.append(item)

    assert len(results) == 2
    image_content = results[0].content[0]
    assert isinstance(image_content, mcp_types.ImageContent)
    assert image_content.mimeType == "image/png"
    assert image_content.data == "ZmFrZQ=="
    payload = _payload_from_results(results)
    assert payload["success"] is True
    assert payload["attach_requested"] is True
    assert payload["attach_success"] is True
    assert payload["write_to_history_requested"] is False


@pytest.mark.asyncio
async def test_use_image_can_use_forward_context_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, event = _build_plugin(image_caption=False)

    async def get_cached_image_message(origin: str, message_id: str) -> dict[str, object]:
        assert origin == "origin-1"
        assert message_id == "123"
        return {
            "urls": ["https://example.com/restored.png"],
            "cache_sources": ["fileid:restored"],
            "captions": {},
            "updated_at": 1000,
        }

    async def resolve_local_path(_image_ref: str) -> str:
        return "/tmp/restored-image.png"

    monkeypatch.setattr(
        main_module,
        "_get_forward_context_api",
        lambda: _forward_context_api(
            get_cached_image_message=get_cached_image_message,
        ),
    )
    plugin._resolve_image_ref_to_local_path = resolve_local_path
    plugin._encode_image_file = lambda _path: ("cmVzdG9yZWQ=", "image/png")

    results = []
    async for item in plugin.use_image(
        event=event,
        message_id="123",
        image_index=1,
        attach_to_model=True,
        write_to_history=False,
    ):
        results.append(item)

    assert len(results) == 2
    assert isinstance(results[0].content[0], mcp_types.ImageContent)
    payload = _payload_from_results(results)
    assert payload["success"] is True
    assert payload["attach_success"] is True
    assert plugin.runtime.image_message_registry["origin-1"]["123"]["urls"] == [
        "https://example.com/restored.png"
    ]


@pytest.mark.asyncio
async def test_use_image_default_mode_attaches_and_writes_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, event = _build_plugin(image_caption=True)
    applied: dict[str, object] = {}

    async def get_cached_caption(_sources: object) -> str:
        return ""

    async def get_or_create_caption(*args, **kwargs):  # noqa: ANN002, ANN003
        return "A test caption"

    async def should_not_be_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("_get_image_caption should not be called")

    async def resolve_local_path(_image_ref: str) -> str:
        return "/tmp/fake-image.png"

    def apply_caption_to_history(**kwargs) -> bool:  # noqa: ANN003
        applied.update(kwargs)
        return True

    monkeypatch.setattr(
        main_module,
        "_get_forward_context_api",
        lambda: _forward_context_api(
            get_cached_image_caption=get_cached_caption,
            get_or_create_image_caption=get_or_create_caption,
        ),
    )

    plugin._get_image_caption = should_not_be_called
    plugin._resolve_image_ref_to_local_path = resolve_local_path
    plugin._encode_image_file = lambda _path: ("ZmFrZQ==", "image/png")
    plugin._apply_image_caption_to_history = apply_caption_to_history

    plugin.runtime.image_message_registry[event.unified_msg_origin]["123"] = {
        "urls": ["https://example.com/image.png"],
        "captions": {},
    }

    results = []
    async for item in plugin.use_image(event=event, message_id="123", image_index=1):
        results.append(item)

    assert len(results) == 2
    assert isinstance(results[0].content[0], mcp_types.ImageContent)
    payload = _payload_from_results(results)
    assert payload["success"] is True
    assert payload["attach_success"] is True
    assert payload["write_to_history_success"] is True
    assert payload["description_cached"] is False
    assert (
        plugin.runtime.image_message_registry[event.unified_msg_origin]["123"]["captions"][0]
        == "A test caption"
    )
    assert applied["message_id"] == "123"
    assert applied["image_index"] == 0
    assert applied["caption"] == "A test caption"


@pytest.mark.asyncio
async def test_use_image_delegates_caption_creation_to_forward_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, event = _build_plugin(image_caption=True)
    called: dict[str, object] = {}

    async def get_cached_caption(_sources: object) -> str:
        return ""

    async def get_or_create_caption(event_arg, image_url: str, **kwargs):  # noqa: ANN001, ANN003
        called["event"] = event_arg
        called["image_url"] = image_url
        called.update(kwargs)
        return "Forward caption"

    async def should_not_be_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("_get_image_caption should not be called after forward hit")

    monkeypatch.setattr(
        main_module,
        "_get_forward_context_api",
        lambda: _forward_context_api(
            get_cached_image_caption=get_cached_caption,
            get_or_create_image_caption=get_or_create_caption,
        ),
    )

    plugin._get_image_caption = should_not_be_called
    plugin._apply_image_caption_to_history = lambda **_kwargs: True
    plugin.runtime.image_message_registry[event.unified_msg_origin]["123"] = {
        "urls": ["https://example.com/image.png"],
        "cache_sources": ["fileid:delegated"],
        "captions": {},
    }

    results = []
    async for item in plugin.use_image(
        event=event,
        message_id="123",
        image_index=1,
        attach_to_model=False,
        write_to_history=True,
    ):
        results.append(item)

    assert _payload_from_results(results)["success"] is True
    assert called["event"] is event
    assert called["image_url"] == "https://example.com/image.png"
    assert called["cache_source"] == "fileid:delegated"
    assert (
        plugin.runtime.image_message_registry[event.unified_msg_origin]["123"]["captions"][0]
        == "Forward caption"
    )


@pytest.mark.asyncio
async def test_use_image_history_only_mode_does_not_attach_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, event = _build_plugin(image_caption=True)
    applied = {"count": 0}

    async def get_cached_caption(_sources: object) -> str:
        return ""

    async def get_or_create_caption(*args, **kwargs):  # noqa: ANN002, ANN003
        return "History only caption"

    async def should_not_be_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("_get_image_caption should not be called")

    async def should_not_resolve(_image_ref: str) -> str:
        raise AssertionError("_resolve_image_ref_to_local_path should not be called")

    def apply_caption_to_history(**kwargs) -> bool:  # noqa: ANN003
        _ = kwargs
        applied["count"] += 1
        return True

    monkeypatch.setattr(
        main_module,
        "_get_forward_context_api",
        lambda: _forward_context_api(
            get_cached_image_caption=get_cached_caption,
            get_or_create_image_caption=get_or_create_caption,
        ),
    )

    plugin._get_image_caption = should_not_be_called
    plugin._resolve_image_ref_to_local_path = should_not_resolve
    plugin._apply_image_caption_to_history = apply_caption_to_history

    plugin.runtime.image_message_registry[event.unified_msg_origin]["123"] = {
        "urls": ["https://example.com/image.png"],
        "captions": {},
    }

    results = []
    async for item in plugin.use_image(
        event=event,
        message_id="123",
        image_index=1,
        attach_to_model=False,
        write_to_history=True,
    ):
        results.append(item)

    assert len(results) == 1
    payload = _payload_from_results(results)
    assert payload["success"] is True
    assert payload["attach_requested"] is False
    assert payload["write_to_history_success"] is True
    assert applied["count"] == 1


@pytest.mark.asyncio
async def test_use_image_does_not_fallback_to_local_caption_when_forward_context_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, event = _build_plugin(image_caption=True)

    async def get_cached_caption(_sources: object) -> str:
        return ""

    async def get_or_create_caption(*args, **kwargs):  # noqa: ANN002, ANN003
        return ""

    async def should_not_be_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("_get_image_caption should not be called on forward miss")

    monkeypatch.setattr(
        main_module,
        "_get_forward_context_api",
        lambda: _forward_context_api(
            get_cached_image_caption=get_cached_caption,
            get_or_create_image_caption=get_or_create_caption,
        ),
    )

    plugin._get_image_caption = should_not_be_called
    plugin.runtime.image_message_registry[event.unified_msg_origin]["123"] = {
        "urls": ["https://example.com/image.png"],
        "captions": {},
    }

    results = []
    async for item in plugin.use_image(
        event=event,
        message_id="123",
        image_index=1,
        attach_to_model=False,
        write_to_history=True,
    ):
        results.append(item)

    assert len(results) == 1
    payload = _payload_from_results(results)
    assert payload["success"] is False
    assert payload["write_to_history_success"] is False
    assert payload["write_to_history_error"] == "Image description is empty."


@pytest.mark.asyncio
async def test_caption_image_with_cache_can_recover_after_resolver_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, event = _build_plugin(image_caption=True)
    cfg = plugin._cfg()
    available = {"value": False}

    async def get_or_create_caption(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args, kwargs
        return "Lazy caption"

    async def should_not_be_called(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args, kwargs
        raise AssertionError("_get_image_caption should not be called")

    monkeypatch.setattr(
        main_module,
        "_get_forward_context_api",
        lambda: (
            _forward_context_api(get_or_create_image_caption=get_or_create_caption)
            if available["value"]
            else None
        ),
    )
    plugin._get_image_caption = should_not_be_called

    first_caption = await plugin._caption_image_with_cache(
        event,
        cfg,
        image_url="https://example.com/image.png",
    )
    available["value"] = True
    second_caption = await plugin._caption_image_with_cache(
        event,
        cfg,
        image_url="https://example.com/image.png",
    )

    assert first_caption == ""
    assert second_caption == "Lazy caption"


@pytest.mark.asyncio
async def test_caption_image_with_cache_uses_shared_cache_without_creating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, event = _build_plugin(image_caption=True)
    cfg = plugin._cfg()

    async def get_cached_caption(_sources: object) -> str:
        return "Cached caption"

    async def should_not_create(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args, kwargs
        raise AssertionError("get_or_create_image_caption should not be called")

    monkeypatch.setattr(
        main_module,
        "_get_forward_context_api",
        lambda: _forward_context_api(
            get_cached_image_caption=get_cached_caption,
            get_or_create_image_caption=should_not_create,
        ),
    )

    caption = await plugin._caption_image_with_cache(
        event,
        cfg,
        image_url="https://example.com/image.png",
    )

    assert caption == "Cached caption"


@pytest.mark.asyncio
async def test_caption_image_with_cache_skips_when_forward_context_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, event = _build_plugin(image_caption=True)
    cfg = plugin._cfg()

    async def should_not_be_called(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args, kwargs
        raise AssertionError("_get_image_caption should not be called")

    monkeypatch.setattr(main_module, "_get_forward_context_api", lambda: None)
    plugin._get_image_caption = should_not_be_called

    caption = await plugin._caption_image_with_cache(
        event,
        cfg,
        image_url="https://example.com/image.png",
    )

    assert caption == ""


@pytest.mark.asyncio
async def test_use_image_rejects_both_modes_disabled() -> None:
    plugin, event = _build_plugin(image_caption=True)

    results = []
    async for item in plugin.use_image(
        event=event,
        message_id="123",
        image_index=1,
        attach_to_model=False,
        write_to_history=False,
    ):
        results.append(item)

    assert len(results) == 1
    assert (
        results[0].content[0].text
        == "Invalid mode: `attach_to_model` and `write_to_history` cannot both be false."
    )


@pytest.mark.asyncio
async def test_use_image_returns_not_found_when_message_id_is_missing() -> None:
    plugin, event = _build_plugin(image_caption=True)

    results = []
    async for item in plugin.use_image(event=event, message_id="not-exist", image_index=1):
        results.append(item)

    assert len(results) == 1
    assert "not found in current runtime history" in results[0].content[0].text


@pytest.mark.asyncio
async def test_use_image_returns_error_when_image_index_out_of_range() -> None:
    plugin, event = _build_plugin(image_caption=True)
    plugin.runtime.image_message_registry[event.unified_msg_origin]["123"] = {
        "urls": ["https://example.com/image.png"],
        "captions": {},
    }

    results = []
    async for item in plugin.use_image(event=event, message_id="123", image_index=2):
        results.append(item)

    assert len(results) == 1
    assert "`image_index` out of range" in results[0].content[0].text


@pytest.mark.asyncio
async def test_use_image_history_only_fails_when_caption_disabled_and_not_cached() -> None:
    plugin, event = _build_plugin(image_caption=False)
    plugin.runtime.image_message_registry[event.unified_msg_origin]["123"] = {
        "urls": ["https://example.com/image.png"],
        "captions": {},
    }

    results = []
    async for item in plugin.use_image(
        event=event,
        message_id="123",
        image_index=1,
        attach_to_model=False,
        write_to_history=True,
    ):
        results.append(item)

    assert len(results) == 1
    assert (
        results[0].content[0].text
        == "Image caption is disabled in enhance mode config."
    )


class _Video:
    def __init__(self, url: str = "", file: str = "") -> None:
        self.url = url
        self.file = file


class _Sender:
    nickname = "Alice"


class _MessageObj:
    sender = _Sender()
    message_id = "123"


class _RecordEvent(_DummyEvent):
    message_obj = _MessageObj()

    def get_messages(self) -> list[object]:
        return [_Video(url="https://example.com/video.mp4", file="video.mp4")]

    def get_sender_id(self) -> str:
        return "10001"

    def is_admin(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_record_message_registers_video_source() -> None:
    plugin, _event = _build_plugin(image_caption=False)
    event = _RecordEvent("origin-1")

    await plugin._record_message(event, plugin._cfg())

    assert plugin.runtime.session_chats[event.unified_msg_origin][-1].endswith(
        "#msg123: [Video]"
    )
    assert plugin.runtime.video_message_registry[event.unified_msg_origin]["123"][
        "urls"
    ] == ["https://example.com/video.mp4"]


@pytest.mark.asyncio
async def test_use_video_captions_and_writes_history() -> None:
    plugin, event = _build_plugin(image_caption=False)
    plugin.runtime.session_chats[event.unified_msg_origin].append(
        "[Alice/10001/12:00:00] #msg123: [Video]"
    )
    plugin.runtime.video_message_registry[event.unified_msg_origin]["123"] = {
        "urls": ["https://example.com/video.mp4"],
        "captions": {},
    }

    async def caption_video(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return "A useful video caption"

    plugin._caption_video_with_gemini = caption_video

    results = []
    async for item in plugin.use_video(event=event, message_id="123", video_index=1):
        results.append(item)

    payload = _payload_from_results(results)
    assert payload["success"] is True
    assert payload["result_type"] == "native_video_analysis"
    assert payload["description"] == "A useful video caption"
    assert payload["native_video_result"] == "A useful video caption"
    assert payload["native_video_used"] is True
    assert payload["native_video_question_specific"] is False
    assert payload["write_to_history_success"] is True
    assert plugin.runtime.session_chats[event.unified_msg_origin][-1].endswith(
        "#msg123: [Video: A useful video caption]"
    )


@pytest.mark.asyncio
async def test_video_caption_helper_uses_configured_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, event = _build_plugin(image_caption=False)
    cfg = PluginConfig(
        group_history=GroupHistoryEnhancementConfig(enable=True),
        group_features=GroupFeatureEnhancementConfig(react_mode_enable=True),
        global_settings=GlobalSettingsConfig(),
        web_search=WebSearchConfig(proxy_url="http://sing-box:7890"),
    )
    provider = object()
    plugin._resolve_video_caption_providers = lambda *_args, **_kwargs: [
        ("p1", provider)
    ]

    async def caption_video(provider_arg, video_url, prompt, **kwargs):  # noqa: ANN001
        assert provider_arg is provider
        assert video_url == "https://example.com/video.mp4"
        assert prompt
        assert kwargs["proxy_url"] == "http://sing-box:7890"
        return "A useful video caption"

    monkeypatch.setattr(
        main_module,
        "caption_video_with_gemini_provider",
        caption_video,
    )

    caption = await plugin._caption_video_with_gemini(
        event,
        cfg,
        "https://example.com/video.mp4",
    )

    assert caption == "A useful video caption"


class _VideoQuestionEvent(_DummyEvent):
    def get_messages(self) -> list[object]:
        return [main_module.Plain("解读一下这个视频是什么意思")]


@pytest.mark.asyncio
async def test_video_caption_helper_infers_current_video_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, _event = _build_plugin(image_caption=False)
    event = _VideoQuestionEvent("origin-1")
    cfg = PluginConfig(
        group_history=GroupHistoryEnhancementConfig(enable=True),
        group_features=GroupFeatureEnhancementConfig(react_mode_enable=True),
        global_settings=GlobalSettingsConfig(),
    )
    provider = object()
    captured: dict[str, str] = {}
    plugin._resolve_video_caption_providers = lambda *_args, **_kwargs: [
        ("p1", provider)
    ]

    async def caption_video(_provider, _video_url, prompt, **_kwargs):  # noqa: ANN001
        captured["prompt"] = prompt
        return "native answer"

    monkeypatch.setattr(
        main_module,
        "caption_video_with_gemini_provider",
        caption_video,
    )

    caption = await plugin._caption_video_with_gemini(
        event,
        cfg,
        "https://example.com/video.mp4",
    )

    assert caption == "native answer"
    assert "用户问题：解读一下这个视频是什么意思" in captured["prompt"]
    assert "不要只说泛泛的画面描述" in captured["prompt"]


@pytest.mark.asyncio
async def test_use_video_bypasses_generic_cache_for_specific_prompt() -> None:
    plugin, event = _build_plugin(image_caption=False)
    plugin.runtime.session_chats[event.unified_msg_origin].append(
        "[Alice/10001/12:00:00] #msg123: [Video: generic caption]"
    )
    plugin.runtime.video_message_registry[event.unified_msg_origin]["123"] = {
        "urls": ["https://example.com/video.mp4"],
        "captions": {0: "generic caption"},
    }
    captured: dict[str, str] = {}

    async def caption_video(*_args, **kwargs):  # noqa: ANN002, ANN003
        captured["prompt"] = kwargs["prompt"]
        return "specific native answer"

    plugin._caption_video_with_gemini = caption_video

    results = []
    async for item in plugin.use_video(
        event=event,
        message_id="123",
        video_index=1,
        prompt="这个视频里的人在做什么？",
    ):
        results.append(item)

    payload = _payload_from_results(results)
    assert payload["success"] is True
    assert payload["native_video_result"] == "specific native answer"
    assert payload["native_video_question_specific"] is True
    assert payload["description_cached"] is False
    assert captured["prompt"] == "这个视频里的人在做什么？"
    assert plugin.runtime.video_message_registry[event.unified_msg_origin]["123"][
        "captions"
    ][0] == "generic caption"


@pytest.mark.asyncio
async def test_video_caption_falls_back_to_next_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin, event = _build_plugin(image_caption=False)
    cfg = PluginConfig(
        group_history=GroupHistoryEnhancementConfig(enable=True),
        group_features=GroupFeatureEnhancementConfig(react_mode_enable=True),
        global_settings=GlobalSettingsConfig(),
    )
    provider_1 = object()
    provider_2 = object()
    plugin._resolve_video_caption_providers = lambda *_args, **_kwargs: [
        ("p1", provider_1),
        ("p2", provider_2),
    ]
    calls = []

    async def caption_video(provider_arg, *_args, **_kwargs):  # noqa: ANN001, ANN002, ANN003
        calls.append(provider_arg)
        if provider_arg is provider_1:
            raise main_module.GeminiVideoError("Gemini video caption failed: 503")
        return "fallback caption"

    monkeypatch.setattr(
        main_module,
        "caption_video_with_gemini_provider",
        caption_video,
    )

    caption = await plugin._caption_video_with_gemini(
        event,
        cfg,
        "https://example.com/video.mp4",
    )

    assert caption == "fallback caption"
    assert calls == [provider_1, provider_2]


@pytest.mark.asyncio
async def test_use_video_returns_error_when_video_index_out_of_range() -> None:
    plugin, event = _build_plugin(image_caption=False)
    plugin.runtime.video_message_registry[event.unified_msg_origin]["123"] = {
        "urls": ["https://example.com/video.mp4"],
        "captions": {},
    }

    results = []
    async for item in plugin.use_video(event=event, message_id="123", video_index=2):
        results.append(item)

    assert len(results) == 1
    assert "`video_index` out of range" in results[0].content[0].text


@pytest.mark.asyncio
async def test_use_video_reports_caption_provider_error() -> None:
    plugin, event = _build_plugin(image_caption=False)
    plugin.runtime.video_message_registry[event.unified_msg_origin]["123"] = {
        "urls": ["https://example.com/video.mp4"],
        "captions": {},
    }

    async def caption_video(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("Provider is not Gemini")

    plugin._caption_video_with_gemini = caption_video

    results = []
    async for item in plugin.use_video(
        event=event,
        message_id="123",
        video_index=1,
    ):
        results.append(item)

    payload = _payload_from_results(results)
    assert payload["success"] is False
    assert "Provider is not Gemini" in payload["error"]
