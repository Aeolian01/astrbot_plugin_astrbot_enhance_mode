from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


PLUGINS_ROOT = Path(__file__).resolve().parents[2]
_PARENTS = Path(__file__).resolve().parents
PROJECT_ROOT = _PARENTS[4] if len(_PARENTS) > 4 else PLUGINS_ROOT

for path in (PROJECT_ROOT, PLUGINS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


if importlib.util.find_spec("mcp") is None:
    mcp_mod = types.ModuleType("mcp")
    mcp_types_mod = types.ModuleType("mcp.types")

    class TextContent:
        def __init__(self, *, type: str = "text", text: str = "") -> None:
            self.type = type
            self.text = text

    class ImageContent:
        def __init__(
            self, *, type: str = "image", data: str = "", mimeType: str = ""
        ) -> None:
            self.type = type
            self.data = data
            self.mimeType = mimeType

    class CallToolResult:
        def __init__(self, *, content: list[object] | None = None) -> None:
            self.content = content or []

    mcp_types_mod.TextContent = TextContent
    mcp_types_mod.ImageContent = ImageContent
    mcp_types_mod.CallToolResult = CallToolResult
    mcp_mod.types = mcp_types_mod
    sys.modules.update({"mcp": mcp_mod, "mcp.types": mcp_types_mod})


if importlib.util.find_spec("aiohttp") is None:
    aiohttp_mod = types.ModuleType("aiohttp")

    class ClientTimeout:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class ClientSession:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

    aiohttp_mod.ClientTimeout = ClientTimeout
    aiohttp_mod.ClientSession = ClientSession
    sys.modules["aiohttp"] = aiohttp_mod


if importlib.util.find_spec("uvicorn") is None:
    uvicorn_mod = types.ModuleType("uvicorn")

    class Config:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class Server:
        def __init__(self, config=None) -> None:
            self.config = config
            self.should_exit = False

        async def serve(self) -> None:
            return None

    uvicorn_mod.Config = Config
    uvicorn_mod.Server = Server
    sys.modules["uvicorn"] = uvicorn_mod


if importlib.util.find_spec("fastapi") is None:
    fastapi_mod = types.ModuleType("fastapi")
    fastapi_middleware_mod = types.ModuleType("fastapi.middleware")
    fastapi_cors_mod = types.ModuleType("fastapi.middleware.cors")
    fastapi_responses_mod = types.ModuleType("fastapi.responses")
    fastapi_staticfiles_mod = types.ModuleType("fastapi.staticfiles")
    fastapi_testclient_mod = types.ModuleType("fastapi.testclient")

    def Depends(dep=None):
        return dep

    class HTTPException(Exception):
        def __init__(self, status_code: int = 500, detail: str = "") -> None:
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class Request:
        def __init__(self, *, query_params=None, client_host: str = "testclient") -> None:
            self.query_params = query_params or {}
            self.client = types.SimpleNamespace(host=client_host)

    class _Status:
        HTTP_400_BAD_REQUEST = 400
        HTTP_401_UNAUTHORIZED = 401
        HTTP_403_FORBIDDEN = 403
        HTTP_404_NOT_FOUND = 404
        HTTP_429_TOO_MANY_REQUESTS = 429

    class FastAPI:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs
            self.routes = {}

        def add_middleware(self, *args, **kwargs) -> None:
            return None

        def mount(self, *args, **kwargs) -> None:
            return None

        def get(self, *args, **kwargs):
            path = args[0] if args else ""

            def deco(func):
                self.routes[("GET", path)] = func
                return func

            return deco

        def post(self, *args, **kwargs):
            path = args[0] if args else ""

            def deco(func):
                self.routes[("POST", path)] = func
                return func

            return deco

        def delete(self, *args, **kwargs):
            path = args[0] if args else ""

            def deco(func):
                self.routes[("DELETE", path)] = func
                return func

            return deco

    class CORSMiddleware:
        pass

    class HTMLResponse:
        def __init__(self, content: str = "", *args, **kwargs) -> None:
            self.content = content

    class StaticFiles:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    class _TestResponse:
        def __init__(self, status_code: int, payload: object) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self):
            return self._payload

    class TestClient:
        __test__ = False

        def __init__(self, app: FastAPI) -> None:
            self.app = app

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def _run(self, method: str, path: str, *, json=None, headers=None):
            import asyncio
            from urllib.parse import parse_qs, urlsplit

            split = urlsplit(path)
            route = self.app.routes.get((method, split.path))
            if route is None:
                return _TestResponse(404, {"detail": "Not found"})
            query = {
                key: values[-1] if values else ""
                for key, values in parse_qs(split.query).items()
            }
            request = Request(query_params=query)
            try:
                if method == "POST" and split.path == "/api/login":
                    result = route(request, json or {})
                elif method == "GET":
                    result = route(request)
                else:
                    result = route()
                if hasattr(result, "__await__"):
                    result = asyncio.run(result)
                return _TestResponse(200, result)
            except HTTPException as e:
                return _TestResponse(e.status_code, {"detail": e.detail})

        def post(self, path: str, *, json=None, headers=None):
            return self._run("POST", path, json=json, headers=headers)

        def get(self, path: str, *, headers=None):
            return self._run("GET", path, headers=headers)

    fastapi_mod.Depends = Depends
    fastapi_mod.FastAPI = FastAPI
    fastapi_mod.HTTPException = HTTPException
    fastapi_mod.Request = Request
    fastapi_mod.status = _Status
    fastapi_cors_mod.CORSMiddleware = CORSMiddleware
    fastapi_responses_mod.HTMLResponse = HTMLResponse
    fastapi_staticfiles_mod.StaticFiles = StaticFiles
    fastapi_testclient_mod.TestClient = TestClient
    sys.modules.update(
        {
            "fastapi": fastapi_mod,
            "fastapi.middleware": fastapi_middleware_mod,
            "fastapi.middleware.cors": fastapi_cors_mod,
            "fastapi.responses": fastapi_responses_mod,
            "fastapi.staticfiles": fastapi_staticfiles_mod,
            "fastapi.testclient": fastapi_testclient_mod,
        }
    )


if importlib.util.find_spec("astrbot") is None:
    astrbot_mod = types.ModuleType("astrbot")
    api_mod = types.ModuleType("astrbot.api")
    event_mod = types.ModuleType("astrbot.api.event")
    event_filter_mod = types.ModuleType("astrbot.api.event.filter")
    message_components_mod = types.ModuleType("astrbot.api.message_components")
    platform_mod = types.ModuleType("astrbot.api.platform")
    provider_mod = types.ModuleType("astrbot.api.provider")
    star_mod = types.ModuleType("astrbot.api.star")
    core_mod = types.ModuleType("astrbot.core")
    agent_mod = types.ModuleType("astrbot.core.agent")
    agent_message_mod = types.ModuleType("astrbot.core.agent.message")
    core_provider_mod = types.ModuleType("astrbot.core.provider")
    core_provider_provider_mod = types.ModuleType("astrbot.core.provider.provider")
    utils_mod = types.ModuleType("astrbot.core.utils")
    astrbot_path_mod = types.ModuleType("astrbot.core.utils.astrbot_path")
    io_mod = types.ModuleType("astrbot.core.utils.io")

    class _Logger:
        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def exception(self, *args, **kwargs):
            pass

    def _decorator(*_args, **_kwargs):
        def deco(func):
            return func

        return deco

    class _CommandGroup:
        def __call__(self, func):
            func.command = lambda *_args, **_kwargs: _decorator(*_args, **_kwargs)
            return func

    class _Filter:
        class EventMessageType:
            ALL = "ALL"

        class PlatformAdapterType:
            ALL = "ALL"

        event_message_type = staticmethod(_decorator)
        platform_adapter_type = staticmethod(_decorator)
        on_astrbot_loaded = staticmethod(_decorator)
        on_llm_request = staticmethod(_decorator)
        on_decorating_result = staticmethod(_decorator)
        on_llm_response = staticmethod(_decorator)
        after_message_sent = staticmethod(_decorator)

        @staticmethod
        def command_group(*_args, **_kwargs):
            return _CommandGroup()

    class PermissionType:
        ADMIN = "admin"

    def permission_type(*_args, **_kwargs):
        return _decorator(*_args, **_kwargs)

    def llm_tool(*_args, **_kwargs):
        return _decorator(*_args, **_kwargs)

    class _SP:
        async def get_async(self, *args, **kwargs):
            return kwargs.get("default")

    class Star:
        def __init__(self, context=None, config=None) -> None:
            self.context = context
            self.config = config

    class Context:
        pass

    class AstrMessageEvent:
        pass

    class MessageEventResult:
        pass

    class MessageType:
        GROUP_MESSAGE = "GROUP_MESSAGE"
        FRIEND_MESSAGE = "FRIEND_MESSAGE"

    class Plain:
        def __init__(self, text: str = "") -> None:
            self.text = text

    class At:
        def __init__(self, qq: str = "", name: str = "") -> None:
            self.qq = qq
            self.name = name or qq

    class Image:
        def __init__(self, url: str = "", file: str = "") -> None:
            self.url = url
            self.file = file

    class Video:
        def __init__(
            self,
            url: str = "",
            file: str = "",
            video_url: str = "",
        ) -> None:
            self.url = url
            self.file = file
            self.video_url = video_url

    class Reply:
        def __init__(
            self,
            id: str = "",
            sender_nickname: str = "",
            message_str: str = "",
        ) -> None:
            self.id = id
            self.sender_nickname = sender_nickname
            self.message_str = message_str

    class Provider:
        pass

    class ProviderRequest:
        def __init__(self) -> None:
            self.prompt = ""
            self.contexts = []
            self.extra_user_content_parts = []

    class LLMResponse:
        pass

    class EmbeddingProvider:
        pass

    class TextPart:
        def __init__(self, text: str = "") -> None:
            self.text = text

    def get_astrbot_data_path() -> str:
        return str(PROJECT_ROOT / ".test-data")

    async def download_image_by_url(url: str) -> str:
        return url

    api_mod.logger = _Logger()
    api_mod.llm_tool = llm_tool
    api_mod.sp = _SP()
    api_mod.star = star_mod
    event_mod.AstrMessageEvent = AstrMessageEvent
    event_mod.MessageEventResult = MessageEventResult
    event_mod.filter = _Filter
    event_filter_mod.PermissionType = PermissionType
    event_filter_mod.permission_type = permission_type
    message_components_mod.At = At
    message_components_mod.Image = Image
    message_components_mod.Plain = Plain
    message_components_mod.Reply = Reply
    message_components_mod.Video = Video
    platform_mod.MessageType = MessageType
    provider_mod.LLMResponse = LLMResponse
    provider_mod.Provider = Provider
    provider_mod.ProviderRequest = ProviderRequest
    star_mod.Context = Context
    star_mod.Star = Star
    agent_message_mod.TextPart = TextPart
    core_provider_provider_mod.EmbeddingProvider = EmbeddingProvider
    astrbot_path_mod.get_astrbot_data_path = get_astrbot_data_path
    io_mod.download_image_by_url = download_image_by_url

    sys.modules.update(
        {
            "astrbot": astrbot_mod,
            "astrbot.api": api_mod,
            "astrbot.api.event": event_mod,
            "astrbot.api.event.filter": event_filter_mod,
            "astrbot.api.message_components": message_components_mod,
            "astrbot.api.platform": platform_mod,
            "astrbot.api.provider": provider_mod,
            "astrbot.api.star": star_mod,
            "astrbot.core": core_mod,
            "astrbot.core.agent": agent_mod,
            "astrbot.core.agent.message": agent_message_mod,
            "astrbot.core.provider": core_provider_mod,
            "astrbot.core.provider.provider": core_provider_provider_mod,
            "astrbot.core.utils": utils_mod,
            "astrbot.core.utils.astrbot_path": astrbot_path_mod,
            "astrbot.core.utils.io": io_mod,
        }
    )
