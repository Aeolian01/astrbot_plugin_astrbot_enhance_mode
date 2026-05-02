from __future__ import annotations

from typing import Any

import pytest

from astrbot_plugin_astrbot_enhance_mode import main as main_module
from astrbot_plugin_astrbot_enhance_mode.main import Main
from astrbot_plugin_astrbot_enhance_mode.plugin_config import PluginConfig, WebSearchConfig


class _DummyEvent:
    def __init__(self, origin: str = "origin-test") -> None:
        self.unified_msg_origin = origin


class _FakeProvider:
    def __init__(self) -> None:
        self.id = "provider-test"
        self.provider_config = {
            "api_base": "https://example.com/v1",
            "model": "grok-4-1-fast-reasoning",
            "key": ["sk-test"],
            "custom_headers": {"X-Test": "1", "Authorization": "override"},
            "custom_extra_body": {
                "tools": [{"type": "web_search"}],
                "tool_choice": "auto",
                "messages": [{"role": "user", "content": "bad"}],
            },
        }

    def get_model(self) -> str:
        return "grok-4-1-fast-reasoning"

    def get_current_key(self) -> str:
        return "sk-test"

    def get_keys(self) -> list[str]:
        return ["sk-test"]


class _FakeGeminiProvider:
    def __init__(self) -> None:
        self.id = "provider-gemini"
        self.provider_config = {
            "api_base": "https://generativelanguage.googleapis.com",
            "model": "models/gemini-2.5-flash",
            "key": ["gemini-key"],
            "custom_headers": {"X-Test": "1", "x-goog-api-key": "override"},
            "custom_extra_body": {
                "tools": [{"bad": {}}],
                "contents": [],
                "systemInstruction": {"parts": [{"text": "bad"}]},
                "safetySettings": [],
            },
        }

    def get_model(self) -> str:
        return "models/gemini-2.5-flash"

    def get_current_key(self) -> str:
        return "gemini-key"

    def get_keys(self) -> list[str]:
        return ["gemini-key"]


class _FakeResponse:
    def __init__(
        self, status: int, text: str, headers: dict[str, str] | None = None
    ) -> None:
        self.status = status
        self._text = text
        self.headers = headers or {"Content-Type": "application/json"}

    async def text(self) -> str:
        return self._text

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.last_post_kwargs: dict[str, Any] = {}

    def post(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.last_post_kwargs = {"args": args, "kwargs": kwargs}
        return self._response

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False


def _build_plugin() -> Main:
    plugin = Main.__new__(Main)
    return plugin


def test_build_web_search_http_requests_uses_provider_config() -> None:
    plugin = _build_plugin()
    provider = _FakeProvider()
    cfg = PluginConfig(
        web_search=WebSearchConfig(
            enable=True,
            provider_id="provider-test",
            system_prompt="system prompt",
            request_mode="responses",
        )
    )

    request_specs, provider_label = plugin._build_web_search_http_requests(
        provider=provider,
        query="latest world model vla",
        cfg=cfg,
    )

    assert len(request_specs) == 1
    assert request_specs[0]["mode"] == "responses"
    assert request_specs[0]["url"] == "https://example.com/v1/responses"
    assert request_specs[0]["headers"]["Authorization"] == "Bearer sk-test"
    assert request_specs[0]["headers"]["X-Test"] == "1"
    assert request_specs[0]["body"]["model"] == "grok-4-1-fast-reasoning"
    assert request_specs[0]["body"]["input"] == "latest world model vla"
    assert request_specs[0]["body"]["tools"] == [{"type": "web_search"}]
    assert request_specs[0]["body"]["tool_choice"] == "auto"

    assert provider_label == "provider-test"


def test_build_web_search_http_requests_uses_gemini_mode() -> None:
    plugin = _build_plugin()
    provider = _FakeGeminiProvider()
    cfg = PluginConfig(
        web_search=WebSearchConfig(
            enable=True,
            provider_id="provider-gemini",
            system_prompt="system prompt",
            request_mode="gemini",
        )
    )

    request_specs, provider_label = plugin._build_web_search_http_requests(
        provider=provider,
        query="latest world model vla",
        cfg=cfg,
    )

    assert len(request_specs) == 1
    request_spec = request_specs[0]
    assert request_spec["mode"] == "gemini"
    assert (
        request_spec["url"]
        == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    )
    assert request_spec["headers"]["x-goog-api-key"] == "gemini-key"
    assert request_spec["headers"]["X-Test"] == "1"
    assert "Authorization" not in request_spec["headers"]
    assert request_spec["body"]["tools"] == [{"google_search": {}}]
    assert request_spec["body"]["contents"] == [
        {"role": "user", "parts": [{"text": "latest world model vla"}]}
    ]
    assert request_spec["body"]["system_instruction"]["parts"][0]["text"] == "system prompt"
    assert request_spec["body"]["generationConfig"]["temperature"] == 0.2
    assert request_spec["body"]["safetySettings"] == []
    assert provider_label == "provider-gemini"


def test_build_web_search_http_requests_auto_tries_gemini_first() -> None:
    plugin = _build_plugin()
    provider = _FakeProvider()
    cfg = PluginConfig(
        web_search=WebSearchConfig(
            enable=True,
            provider_id="provider-test",
            system_prompt="system prompt",
        )
    )

    request_specs, _provider_label = plugin._build_web_search_http_requests(
        provider=provider,
        query="latest world model vla",
        cfg=cfg,
    )

    assert [spec["mode"] for spec in request_specs] == [
        "gemini",
        "responses",
        "chat_completions",
    ]


@pytest.mark.asyncio
async def test_run_web_search_success_with_json_message(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _build_plugin()
    event = _DummyEvent()
    cfg = PluginConfig(
        web_search=WebSearchConfig(
            enable=True,
            provider_id="provider-test",
            system_prompt="sys",
            timeout_sec=10,
        )
    )

    plugin._resolve_web_search_provider = lambda _cfg: object()
    plugin._build_web_search_http_requests = (
        lambda provider, query, cfg: (
            [
                {
                    "mode": "responses",
                    "url": "https://example.com/v1/responses",
                    "headers": {
                        "Authorization": "Bearer sk-test",
                        "Content-Type": "application/json",
                    },
                    "body": {"model": "grok", "input": query},
                }
            ],
            "provider-test",
        )
    )

    raw_json = (
        '{"output":[{"type":"message","content":[{"type":"output_text","text":"DreamZero is a recent VLA direction",'
        '"annotations":[{"type":"url_citation","url":"https://example.com/dreamzero","title":"DreamZero"}]}]}],'
        '"usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30}}'
    )
    fake_response = _FakeResponse(status=200, text=raw_json)
    fake_session = _FakeSession(fake_response)
    monkeypatch.setattr(
        main_module.aiohttp,
        "ClientSession",
        lambda *args, **kwargs: fake_session,
    )

    result = await plugin._run_web_search(event, "latest world model vla", cfg)
    assert result["ok"] is True
    assert "DreamZero" in str(result["content"])
    assert isinstance(result["sources"], list)
    assert result["usage"]["total_tokens"] == 30


@pytest.mark.asyncio
async def test_run_web_search_success_with_gemini_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _build_plugin()
    event = _DummyEvent()
    cfg = PluginConfig(
        web_search=WebSearchConfig(
            enable=True,
            provider_id="provider-gemini",
            system_prompt="sys",
            timeout_sec=10,
        )
    )

    plugin._resolve_web_search_provider = lambda _cfg: object()
    plugin._build_web_search_http_requests = (
        lambda provider, query, cfg: (
            [
                {
                    "mode": "gemini",
                    "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent",
                    "headers": {
                        "x-goog-api-key": "gemini-key",
                        "Content-Type": "application/json",
                    },
                    "body": {"contents": [{"role": "user", "parts": [{"text": query}]}]},
                }
            ],
            "provider-gemini",
        )
    )

    raw_json = (
        '{"candidates":[{"content":{"parts":[{"text":"DreamZero is a recent VLA direction"}]},'
        '"groundingMetadata":{"groundingChunks":[{"web":{"uri":"https://example.com/dreamzero",'
        '"title":"DreamZero"}}]}}],'
        '"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"totalTokenCount":30}}'
    )
    fake_response = _FakeResponse(status=200, text=raw_json)
    fake_session = _FakeSession(fake_response)
    monkeypatch.setattr(
        main_module.aiohttp,
        "ClientSession",
        lambda *args, **kwargs: fake_session,
    )

    result = await plugin._run_web_search(event, "latest world model vla", cfg)
    assert result["ok"] is True
    assert "DreamZero" in str(result["content"])
    assert result["sources"] == [
        {"url": "https://example.com/dreamzero", "title": "DreamZero", "snippet": ""}
    ]
    assert result["usage"]["total_tokens"] == 30


@pytest.mark.asyncio
async def test_run_web_search_http_error_returns_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _build_plugin()
    event = _DummyEvent()
    cfg = PluginConfig(
        web_search=WebSearchConfig(
            enable=True,
            provider_id="provider-test",
            system_prompt="sys",
            timeout_sec=10,
        )
    )

    plugin._resolve_web_search_provider = lambda _cfg: object()
    plugin._build_web_search_http_requests = (
        lambda provider, query, cfg: (
            [
                {
                    "mode": "responses",
                    "url": "https://example.com/v1/responses",
                    "headers": {
                        "Authorization": "Bearer sk-test",
                        "Content-Type": "application/json",
                    },
                    "body": {"model": "grok", "input": query},
                }
            ],
            "provider-test",
        )
    )

    fake_response = _FakeResponse(status=422, text='{"error":"openai_error"}')
    fake_session = _FakeSession(fake_response)
    monkeypatch.setattr(
        main_module.aiohttp,
        "ClientSession",
        lambda *args, **kwargs: fake_session,
    )

    result = await plugin._run_web_search(event, "latest world model vla", cfg)
    assert result["ok"] is False
    assert "HTTP 422" in str(result["error"])
    assert "openai_error" in str(result["raw"])
