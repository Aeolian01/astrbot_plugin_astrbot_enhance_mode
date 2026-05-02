from __future__ import annotations

from typing import Any

import pytest

from astrbot_plugin_astrbot_enhance_mode import main as main_module
from astrbot_plugin_astrbot_enhance_mode.main import Main
from astrbot_plugin_astrbot_enhance_mode.plugin_config import PluginConfig, WebSearchConfig


class _DummyEvent:
    def __init__(self, origin: str = "origin-test") -> None:
        self.unified_msg_origin = origin


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


class _FakeSessionFactory:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session
        self.last_init_kwargs: dict[str, Any] = {}

    def __call__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.last_init_kwargs = {"args": args, "kwargs": kwargs}
        return self._session


def _build_plugin() -> Main:
    plugin = Main.__new__(Main)
    return plugin


@pytest.mark.asyncio
async def test_run_web_search_uses_proxy_url_and_trust_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _build_plugin()
    event = _DummyEvent()
    cfg = PluginConfig(
        web_search=WebSearchConfig(
            enable=True,
            provider_id="provider-test",
            system_prompt="sys",
            timeout_sec=10,
            proxy_url="http://sing-box:7890",
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
        '{"output":[{"type":"message","content":[{"type":"output_text","text":"DreamZero is a recent VLA direction"}]}],'
        '"usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30}}'
    )
    fake_response = _FakeResponse(status=200, text=raw_json)
    fake_session = _FakeSession(fake_response)
    fake_session_factory = _FakeSessionFactory(fake_session)
    monkeypatch.setattr(
        main_module.aiohttp,
        "ClientSession",
        fake_session_factory,
    )

    result = await plugin._run_web_search(event, "latest world model vla", cfg)

    assert result["ok"] is True
    assert fake_session_factory.last_init_kwargs["kwargs"]["trust_env"] is True
    assert (
        fake_session.last_post_kwargs["kwargs"]["proxy"]
        == "http://sing-box:7890"
    )
