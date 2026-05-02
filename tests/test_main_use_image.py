from __future__ import annotations

import json

from mcp import types as mcp_types
import pytest

import astrbot_plugin_astrbot_enhance_mode.main as main_module
from astrbot_plugin_astrbot_enhance_mode.main import Main
from astrbot_plugin_astrbot_enhance_mode.plugin_config import (
    GlobalSettingsConfig,
    GroupFeatureEnhancementConfig,
    GroupHistoryEnhancementConfig,
    PluginConfig,
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
        "forward_get_cached_image_message",
        get_cached_image_message,
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

    monkeypatch.setattr(main_module, "FORWARD_CONTEXT_AVAILABLE", True)
    monkeypatch.setattr(main_module, "forward_get_cached_image_caption", get_cached_caption)
    monkeypatch.setattr(
        main_module,
        "forward_get_or_create_image_caption",
        get_or_create_caption,
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

    monkeypatch.setattr(main_module, "FORWARD_CONTEXT_AVAILABLE", True)
    monkeypatch.setattr(main_module, "forward_get_cached_image_caption", get_cached_caption)
    monkeypatch.setattr(
        main_module,
        "forward_get_or_create_image_caption",
        get_or_create_caption,
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

    monkeypatch.setattr(main_module, "FORWARD_CONTEXT_AVAILABLE", True)
    monkeypatch.setattr(main_module, "forward_get_cached_image_caption", get_cached_caption)
    monkeypatch.setattr(
        main_module,
        "forward_get_or_create_image_caption",
        get_or_create_caption,
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

    monkeypatch.setattr(main_module, "FORWARD_CONTEXT_AVAILABLE", True)
    monkeypatch.setattr(main_module, "forward_get_cached_image_caption", get_cached_caption)
    monkeypatch.setattr(
        main_module,
        "forward_get_or_create_image_caption",
        get_or_create_caption,
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
