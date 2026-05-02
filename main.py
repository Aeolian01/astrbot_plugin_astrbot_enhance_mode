import asyncio
import base64
import datetime
import json
import mimetypes
import random
import re
import time
import traceback
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp
from mcp import types as mcp_types

from astrbot.api import llm_tool, logger, sp, star
from astrbot.api.event import AstrMessageEvent, MessageEventResult, filter
from astrbot.api.event.filter import PermissionType, permission_type
from astrbot.api.message_components import At, Image, Plain, Reply
from astrbot.api.platform import MessageType
from astrbot.api.provider import LLMResponse, Provider, ProviderRequest
from astrbot.core.agent.message import TextPart
from astrbot.core.provider.provider import EmbeddingProvider
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from astrbot.core.utils.io import download_image_by_url

try:
    from astrbot_plugin_forward_context import (
        get_cached_image_caption as forward_get_cached_image_caption,
        parse_history_message as forward_parse_history_message,
        set_cached_image_caption as forward_set_cached_image_caption,
    )
except Exception:

    async def forward_get_cached_image_caption(source: str) -> str:
        return ""

    async def forward_parse_history_message(event: Any, message: Any) -> str:
        return ""

    async def forward_set_cached_image_caption(source: str, caption: str) -> None:
        return None

from .ban_control import BanStore, parse_duration_seconds
from .memory_rag_store import MemoryRAGStore
from .plugin_config import PluginConfig, parse_plugin_config
from .runtime_state import RuntimeState
from .tag_utils import (
    bounded_chat_history_text,
    build_interaction_instructions,
    chain_has_refuse_tag,
    clean_response_text_for_history,
    dedupe_repeated_result_chain,
    has_refuse_tag,
    transform_result_chain,
)
from .webui import RAGWebUIServer

IMAGE_MARKER_PATTERN = re.compile(r"\[Image(?:: [^\]]*)?\]")
MSG_ID_PATTERN = re.compile(r"#msg([^:]+):")
FORWARD_CONTEXT_TEXT_KEY = "_forward_context_text"
FORWARD_CONTEXT_FOUND_KEY = "_forward_context_found"
FORWARD_CONTEXT_IDS_KEY = "_forward_context_ids"
ENHANCE_ACTIVE_REPLY_PROMPT_KEY = "_enhance_active_reply_prompt"
ACTIVE_MESSAGE_TEXT_LIMIT = 4000


class Main(star.Star):
    def __init__(self, context: star.Context, config: dict | None = None) -> None:
        super().__init__(context, config)
        self.context = context
        self.config = config or {}
        self.runtime = RuntimeState()
        self._image_caption_inflight: dict[str, asyncio.Task[str]] = {}
        self._display_timezone = self._resolve_config_timezone()
        plugin_data_dir = (
            Path(get_astrbot_data_path())
            / "plugin_data"
            / "astrbot_plugin_astrbot_enhance_mode"
        )
        self.ban_store = BanStore(plugin_data_dir / "ban_list.db")
        self.memory_rag_store: MemoryRAGStore | None = None
        self.rag_webui_server: RAGWebUIServer | None = None
        try:
            self.memory_rag_store = MemoryRAGStore(
                plugin_data_dir / "memory_rag.db",
                display_timezone=self._display_timezone,
            )
        except Exception as e:
            logger.error(f"enhance-mode | 初始化记忆 RAG 存储失败: {e}", exc_info=True)
        logger.info(
            "enhance-mode | plugin initialized | data_dir=%s memory_rag_store_ready=%s timezone=%s",
            plugin_data_dir,
            self.memory_rag_store is not None,
            self._display_timezone,
        )

    def _cfg(self) -> PluginConfig:
        return parse_plugin_config(self.config)

    def _touch_origin(self, origin: str, cfg: PluginConfig) -> None:
        self.runtime.touch_origin(origin, cfg.global_settings.lru_cache.max_origins)

    @staticmethod
    def _get_extra_value(
        event: AstrMessageEvent, key: str, default: Any = None
    ) -> Any:
        try:
            getter = getattr(event, "get_extra", None)
            if callable(getter):
                try:
                    return getter(key, default)
                except TypeError:
                    value = getter(key)
                    return default if value is None else value
        except Exception:
            pass

        extra = getattr(event, "extra", None)
        if isinstance(extra, dict):
            return extra.get(key, default)
        return default

    @staticmethod
    def _is_truthy_extra(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _looks_forward_text(text: str) -> bool:
        normalized = str(text or "").strip().replace(" ", "")
        if not normalized:
            return False
        lowered = normalized.lower()
        return (
            "[cq:forward" in lowered
            or "[cq:json" in lowered
            or "[componenttype.json]" in lowered
            or "[forward]" in lowered
            or "[json]" in lowered
            or "[jsonshare]" in lowered
            or "[转发消息]" in normalized
            or "[合并转发]" in normalized
        )

    def _event_looks_forward_like(self, event: AstrMessageEvent) -> bool:
        candidates: list[str] = []
        candidates.append(str(getattr(event, "message_str", "") or ""))

        msg_obj = getattr(event, "message_obj", None)
        if msg_obj is not None:
            for attr_name in ("message_str", "raw_message"):
                value = getattr(msg_obj, attr_name, "")
                if value:
                    candidates.append(str(value))

            message_chain = getattr(msg_obj, "message", None)
            if message_chain:
                for comp in message_chain:
                    comp_name = type(comp).__name__.lower()
                    if "forward" in comp_name or "json" in comp_name:
                        return True
                    candidates.append(type(comp).__name__)
                    candidates.append(str(comp))

        if any(self._looks_forward_text(candidate) for candidate in candidates):
            return True
        return False

    def _get_forward_context_text(self, event: AstrMessageEvent) -> str:
        text = str(
            self._get_extra_value(event, FORWARD_CONTEXT_TEXT_KEY, "") or ""
        ).strip()
        if not text:
            return ""

        found = self._get_extra_value(event, FORWARD_CONTEXT_FOUND_KEY, False)
        ids = self._get_extra_value(event, FORWARD_CONTEXT_IDS_KEY, [])
        if (
            self._is_truthy_extra(found)
            or bool(ids)
            or self._event_looks_forward_like(event)
            or self._looks_forward_text(text)
        ):
            return text
        return ""

    @staticmethod
    def _truncate_active_message_text(
        text: str, limit: int = ACTIVE_MESSAGE_TEXT_LIMIT
    ) -> str:
        clean_text = str(text or "").strip()
        if len(clean_text) <= limit:
            return clean_text
        omitted = len(clean_text) - limit
        return f"{clean_text[:limit]}... [truncated {omitted} chars]"

    @staticmethod
    def _extract_history_line_body(line: str) -> str:
        text = str(line or "").strip()
        matched = MSG_ID_PATTERN.search(text)
        if matched:
            return text[matched.end() :].strip()
        return text

    def _find_history_line_by_message_id(self, origin: str, message_id: str) -> str:
        normalized_msg_id = self._normalize_message_id(message_id)
        if not normalized_msg_id:
            return ""

        for line in reversed(self.runtime.session_chats.get(origin) or []):
            line_msg_id = self._normalize_message_id(
                self._extract_message_id_from_history_line(line)
            )
            if line_msg_id == normalized_msg_id:
                return line
        return ""

    def _format_event_message_body(
        self, event: AstrMessageEvent
    ) -> tuple[str, list[str], list[str]]:
        forward_context_text = self._get_forward_context_text(event)
        if forward_context_text:
            return forward_context_text, [], []

        parts: list[str] = []
        image_urls: list[str] = []
        image_cache_sources: list[str] = []
        for comp in event.get_messages():
            if isinstance(comp, Reply):
                quote_nick = comp.sender_nickname or "Unknown"
                quote_text = (comp.message_str or "").strip() or "..."
                quote_id = self._normalize_message_id(getattr(comp, "id", ""))
                if quote_id:
                    parts.append(
                        f" [Quote #msg{quote_id} {quote_nick}: {quote_text}]"
                    )
                else:
                    parts.append(f" [Quote {quote_nick}: {quote_text}]")
            elif isinstance(comp, Plain):
                parts.append(f" {comp.text}")
            elif isinstance(comp, Image):
                image_url = str(comp.url or comp.file or "").strip()
                image_urls.append(image_url)
                image_cache_sources.append(image_url)
                parts.append(" [Image]")
            elif isinstance(comp, At):
                parts.append(f" [At: {comp.name}]")
            else:
                fallback = self._format_unknown_message_component(comp)
                if fallback:
                    parts.append(fallback)

        return "".join(parts).strip(), image_urls, image_cache_sources

    @staticmethod
    def _format_unknown_message_component(comp: Any) -> str:
        comp_name = type(comp).__name__
        normalized_name = comp_name.strip().lower()
        if normalized_name in {"face", "emoji", "mface"}:
            face_id = ""
            for attr in ("id", "face_id", "emoji_id"):
                face_id = str(getattr(comp, attr, "") or "").strip()
                if face_id:
                    break

            meaning = ""
            for attr in ("summary", "name", "text", "emoji_name"):
                meaning = str(getattr(comp, attr, "") or "").strip()
                if meaning:
                    break

            id_part = face_id or "未知"
            meaning_part = meaning or "未知"
            return f" [QQ表情: id={id_part}, 含义={meaning_part}]"

        labels = {
            "record": "语音消息",
            "video": "视频消息",
            "file": "文件",
            "json": "JSON消息",
            "xml": "XML消息",
            "location": "位置消息",
            "share": "分享消息",
            "dice": "骰子",
            "rps": "猜拳",
            "poke": "戳一戳",
            "forward": "转发消息",
        }
        label = labels.get(normalized_name)
        if not label:
            return ""

        for attr in ("id", "face_id", "emoji_id", "name", "text", "file", "url"):
            value = getattr(comp, attr, None)
            clean = str(value or "").strip()
            if clean:
                return f" [{label}: {clean}]"
        return f" [{label}]"

    async def _build_active_message_text(
        self, event: AstrMessageEvent, cfg: PluginConfig
    ) -> str:
        text, _ = await self._resolve_active_current_message_text(event, cfg)
        return text

    def _resolve_config_timezone(self) -> str:
        base_cfg = self.context.get_config()
        if not isinstance(base_cfg, dict):
            return "Asia/Shanghai"
        timezone_name = str(base_cfg.get("timezone") or "").strip()
        return timezone_name or "Asia/Shanghai"

    def _resolve_tzinfo(self) -> datetime.tzinfo:
        timezone_name = self._resolve_config_timezone()
        try:
            return ZoneInfo(timezone_name)
        except Exception:
            try:
                return ZoneInfo("Asia/Shanghai")
            except Exception:
                return datetime.timezone.utc

    def _format_timestamp_iso(self, timestamp: float | int) -> str:
        if self.memory_rag_store is not None:
            return self.memory_rag_store.format_timestamp_iso(timestamp)
        return datetime.datetime.fromtimestamp(
            float(timestamp), tz=self._resolve_tzinfo()
        ).isoformat()

    @staticmethod
    def _provider_label(provider: object | None) -> str:
        if provider is None:
            return "none"
        provider_id = getattr(provider, "provider_id", None) or getattr(
            provider, "id", None
        )
        if provider_id:
            return str(provider_id)
        model = getattr(provider, "model", None)
        cls_name = type(provider).__name__
        return f"{cls_name}({model})" if model else cls_name

    @staticmethod
    def _history_recording_enabled(cfg: PluginConfig) -> bool:
        return cfg.group_history_enabled or cfg.active_reply_enabled

    @staticmethod
    def _normalize_message_id(raw: str | int | None) -> str:
        text = str(raw or "").strip()
        if text.startswith("#msg"):
            text = text[4:]
        if text.endswith(":"):
            text = text[:-1]
        return text.strip()

    @staticmethod
    def _extract_message_id_from_history_line(line: str) -> str:
        matched = MSG_ID_PATTERN.search(str(line or ""))
        if not matched:
            return ""
        return str(matched.group(1) or "").strip()

    @staticmethod
    def _numeric_if_digits(value: str) -> str | int:
        text = str(value or "").strip()
        return int(text) if text.isdigit() else text

    @staticmethod
    def _replace_image_marker_at_index(
        line: str, image_index: int, caption: str
    ) -> tuple[str, bool]:
        if image_index < 0:
            return line, False
        matches = list(IMAGE_MARKER_PATTERN.finditer(line))
        if image_index >= len(matches):
            return line, False

        target = matches[image_index]
        safe_caption = str(caption or "").strip().replace("]", ")")
        replacement = f"[Image: {safe_caption}]"
        new_line = line[: target.start()] + replacement + line[target.end() :]
        return new_line, new_line != line

    def _apply_image_caption_to_history(
        self,
        origin: str,
        message_id: str,
        image_index: int,
        caption: str,
    ) -> bool:
        chats = self.runtime.session_chats.get(origin)
        if not chats:
            return False

        target_marker = f"#msg{message_id}:"
        for idx, line in enumerate(chats):
            if target_marker not in line:
                continue
            replaced_line, changed = self._replace_image_marker_at_index(
                line, image_index, caption
            )
            if not changed:
                return False
            chats[idx] = replaced_line
            return True
        return False

    def _replace_history_line_by_message_id(
        self, origin: str, message_id: str, new_line: str
    ) -> bool:
        chats = self.runtime.session_chats.get(origin)
        if not chats or not message_id:
            return False

        target_marker = f"#msg{message_id}:"
        for idx, line in enumerate(chats):
            if target_marker not in line:
                continue
            if line == new_line:
                return False
            chats[idx] = new_line
            return True
        return False

    @staticmethod
    def _image_caption_sources(image_url: str, cache_source: str = "") -> list[str]:
        sources: list[str] = []
        for candidate in (cache_source, image_url):
            clean = str(candidate or "").strip()
            if clean and clean not in sources:
                sources.append(clean)
        return sources

    @staticmethod
    def _is_captionable_image_ref(image_ref: str) -> bool:
        clean = str(image_ref or "").strip()
        if not clean:
            return False
        if clean.startswith(("http://", "https://", "file://")):
            return True
        try:
            return Path(clean).exists()
        except Exception:
            return False

    async def _read_shared_image_caption_cache(self, sources: list[str]) -> str:
        for source in sources:
            try:
                caption = await forward_get_cached_image_caption(source)
            except Exception as e:
                logger.debug(
                    "enhance-mode | shared image caption cache read failed | source=%s error=%s",
                    source,
                    e,
                )
                continue
            caption = str(caption or "").strip()
            if caption:
                logger.debug(
                    "enhance-mode | shared image caption cache hit | source=%s",
                    source,
                )
                return caption
        return ""

    async def _write_shared_image_caption_cache(
        self, sources: list[str], caption: str
    ) -> None:
        clean_caption = str(caption or "").strip()
        if not clean_caption:
            return
        for source in sources:
            try:
                await forward_set_cached_image_caption(source, clean_caption)
            except Exception as e:
                logger.debug(
                    "enhance-mode | shared image caption cache write failed | source=%s error=%s",
                    source,
                    e,
                )

    async def _caption_image_with_cache(
        self,
        event: AstrMessageEvent,
        cfg: PluginConfig,
        image_url: str,
        cache_source: str = "",
        prompt: str = "",
    ) -> str:
        if not cfg.group_history.image_caption:
            return ""

        clean_image_url = str(image_url or "").strip()
        sources = self._image_caption_sources(clean_image_url, cache_source)
        cached = await self._read_shared_image_caption_cache(sources)
        if cached:
            return cached

        if not self._is_captionable_image_ref(clean_image_url):
            return ""

        inflight_key = sources[0] if sources else clean_image_url
        running_task = self._image_caption_inflight.get(inflight_key)
        if running_task is not None and not running_task.done():
            return await running_task

        async def caption_task() -> str:
            cached_after_wait = await self._read_shared_image_caption_cache(sources)
            if cached_after_wait:
                return cached_after_wait

            final_prompt = str(prompt or "").strip() or cfg.group_history.image_caption_prompt
            caption = await self._get_image_caption(
                image_url=clean_image_url,
                provider_id=cfg.group_history.image_caption_provider_id,
                prompt=final_prompt,
                timeout_sec=cfg.global_settings.timeouts.image_caption_sec,
            )
            caption = str(caption or "").strip()
            if caption:
                await self._write_shared_image_caption_cache(sources, caption)
            return caption

        task = asyncio.create_task(caption_task())
        self._image_caption_inflight[inflight_key] = task
        try:
            return await task
        finally:
            if self._image_caption_inflight.get(inflight_key) is task:
                self._image_caption_inflight.pop(inflight_key, None)

    async def _resolve_image_captions_for_context_lines(
        self,
        event: AstrMessageEvent,
        cfg: PluginConfig,
        lines: list[str],
    ) -> list[str]:
        if not lines or not cfg.group_history.image_caption:
            return lines

        origin = event.unified_msg_origin
        message_registry = self.runtime.image_message_registry.get(origin, {})
        if not message_registry:
            return lines

        resolved_lines: list[str] = []
        resolved_count = 0
        failed_count = 0
        for line in lines:
            current_line = line
            message_id = self._extract_message_id_from_history_line(line)
            message_entry = message_registry.get(message_id) if message_id else None
            if not isinstance(message_entry, dict):
                resolved_lines.append(current_line)
                continue

            urls_raw = message_entry.get("urls")
            if not isinstance(urls_raw, list) or not urls_raw:
                resolved_lines.append(current_line)
                continue

            captions_raw = message_entry.get("captions")
            if isinstance(captions_raw, dict):
                captions_map = captions_raw
            else:
                captions_map = {}
                message_entry["captions"] = captions_map
            cache_sources_raw = message_entry.get("cache_sources")
            cache_sources = cache_sources_raw if isinstance(cache_sources_raw, list) else []

            matches = list(IMAGE_MARKER_PATTERN.finditer(current_line))
            for image_idx, marker in enumerate(matches):
                if marker.group(0).startswith("[Image:"):
                    continue
                if image_idx >= len(urls_raw):
                    continue

                cached_caption = captions_map.get(image_idx) or captions_map.get(
                    str(image_idx)
                )
                caption = (
                    cached_caption.strip()
                    if isinstance(cached_caption, str) and cached_caption.strip()
                    else ""
                )
                cache_source = (
                    str(cache_sources[image_idx] or "").strip()
                    if image_idx < len(cache_sources)
                    else ""
                )
                if not caption:
                    image_url = str(urls_raw[image_idx] or "").strip()
                    try:
                        caption = await self._caption_image_with_cache(
                            event,
                            cfg,
                            image_url=image_url,
                            cache_source=cache_source,
                        )
                    except Exception as e:
                        failed_count += 1
                        logger.warning(
                            "enhance-mode | image caption inject failed | "
                            "origin=%s msg_id=%s image_index=%s error=%s",
                            origin,
                            message_id,
                            image_idx + 1,
                            e,
                        )
                        continue
                    caption = str(caption or "").strip()
                    if not caption:
                        continue
                    captions_map[image_idx] = caption
                else:
                    await self._write_shared_image_caption_cache(
                        self._image_caption_sources(
                            str(urls_raw[image_idx] or "").strip(),
                            cache_source,
                        ),
                        caption,
                    )

                replaced_line, changed = self._replace_image_marker_at_index(
                    current_line,
                    image_idx,
                    caption,
                )
                if changed:
                    current_line = replaced_line
                    resolved_count += 1

            if current_line != line:
                self._replace_history_line_by_message_id(
                    origin,
                    message_id,
                    current_line,
                )
            resolved_lines.append(current_line)

        if resolved_count or failed_count:
            logger.info(
                "enhance-mode | image captions resolved for injection | "
                "origin=%s resolved=%s failed=%s",
                origin,
                resolved_count,
                failed_count,
            )
        return resolved_lines

    async def _resolve_image_ref_to_local_path(self, image_ref: str) -> str:
        clean_ref = str(image_ref or "").strip()
        if not clean_ref:
            return ""

        if clean_ref.startswith("file://"):
            clean_ref = clean_ref[7:]

        candidate = Path(clean_ref)
        if candidate.exists() and candidate.is_file():
            return str(candidate)

        if clean_ref.startswith("http://") or clean_ref.startswith("https://"):
            downloaded = await download_image_by_url(clean_ref)
            return str(downloaded or "")

        return ""

    @staticmethod
    def _encode_image_file(image_path: str) -> tuple[str, str]:
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        raw_bytes = path.read_bytes()
        if not raw_bytes:
            raise ValueError(f"Image file is empty: {image_path}")

        mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        if not mime_type.startswith("image/"):
            mime_type = "image/jpeg"
        encoded = base64.b64encode(raw_bytes).decode("ascii")
        return encoded, mime_type

    @staticmethod
    def _format_duration(seconds: int) -> str:
        seconds = max(0, int(seconds))
        days, rem = divmod(seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs = divmod(rem, 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if secs or not parts:
            parts.append(f"{secs}s")
        return " ".join(parts)

    @staticmethod
    def _ban_scope_id(event: AstrMessageEvent) -> str:
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return ""
        group_id = str(event.get_group_id() or "").strip()
        if not group_id:
            return ""
        platform_id = (
            str(event.get_platform_id() or "").strip()
            or str(event.get_platform_name() or "").strip()
        )
        return f"{platform_id}:{group_id}" if platform_id else group_id

    def _get_admin_sid_set(self) -> set[str]:
        base_cfg = self.context.get_config()
        if not isinstance(base_cfg, dict):
            return set()
        admins = base_cfg.get("admins_id", [])
        if not isinstance(admins, list):
            return set()
        return {str(sid).strip() for sid in admins if str(sid).strip()}

    @staticmethod
    def _parse_role_ids(raw: str) -> list[str]:
        text = str(raw or "").strip()
        if not text:
            return []

        parsed_ids: list[str] = []
        if text.startswith("["):
            try:
                payload = json.loads(text)
                if isinstance(payload, list):
                    parsed_ids = [str(item).strip() for item in payload]
            except json.JSONDecodeError:
                parsed_ids = []

        if not parsed_ids:
            normalized = text.replace("\n", ",").replace(";", ",").replace(" ", ",")
            parsed_ids = [token.strip() for token in normalized.split(",")]

        deduped: list[str] = []
        seen: set[str] = set()
        for role_id in parsed_ids:
            if not role_id or role_id in seen:
                continue
            seen.add(role_id)
            deduped.append(role_id)
        return deduped

    def _parse_optional_timestamp(self, raw: str) -> float | None:
        text = str(raw or "").strip()
        if not text:
            return None

        try:
            numeric = float(text)
            # Heuristic: treat millisecond timestamps as ms.
            if numeric > 1e12:
                numeric /= 1000.0
            return numeric
        except (TypeError, ValueError):
            pass

        tzinfo = self._resolve_tzinfo()
        normalized = text.replace("Z", "+00:00")
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.datetime.strptime(normalized, fmt).replace(tzinfo=tzinfo)
                return dt.timestamp()
            except ValueError:
                continue

        try:
            dt = datetime.datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tzinfo)
            return dt.timestamp()
        except ValueError:
            return None

    @staticmethod
    def _parse_extra_metadata(raw: str) -> dict:
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
            return {"raw_metadata": payload}
        except json.JSONDecodeError:
            return {"raw_metadata": text}

    @staticmethod
    def _normalize_sort_order(raw: str) -> str:
        return "asc" if str(raw or "").strip().lower() == "asc" else "desc"

    @staticmethod
    def _normalize_sort_by(raw: str) -> str:
        return "time" if str(raw or "").strip().lower() == "time" else "relevance"

    def _resolve_memory_scope(
        self,
        event: AstrMessageEvent,
        group_scope: str,
        group_id: str,
        platform_id: str,
    ) -> tuple[str, str, str]:
        explicit_scope = str(group_scope or "").strip()
        explicit_group_id = str(group_id or "").strip()
        explicit_platform_id = str(platform_id or "").strip()

        event_group_id = (
            str(event.get_group_id() or "").strip()
            if event.get_message_type() == MessageType.GROUP_MESSAGE
            else ""
        )
        event_platform_id = str(
            event.get_platform_id() or event.get_platform_name() or ""
        ).strip()

        final_group_id = explicit_group_id or event_group_id
        final_platform_id = explicit_platform_id or event_platform_id

        if explicit_scope:
            final_scope = explicit_scope
        elif final_group_id:
            final_scope = (
                f"{final_platform_id}:{final_group_id}"
                if final_platform_id
                else final_group_id
            )
        else:
            final_scope = ""

        return final_scope, final_group_id, final_platform_id

    def _resolve_embedding_provider(
        self, cfg: PluginConfig
    ) -> EmbeddingProvider | None:
        provider_id = str(cfg.memory_rag.embedding_provider_id or "").strip()
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if provider and isinstance(provider, EmbeddingProvider):
                logger.debug(
                    "enhance-mode | using configured embedding provider | provider=%s",
                    self._provider_label(provider),
                )
                return provider
            logger.warning(
                f"enhance-mode | 配置的 embedding_provider_id 无效或类型不匹配: {provider_id}"
            )

        all_embedding_providers = self.context.get_all_embedding_providers()
        if all_embedding_providers:
            logger.debug(
                "enhance-mode | using fallback embedding provider | provider=%s",
                self._provider_label(all_embedding_providers[0]),
            )
            return all_embedding_providers[0]
        return None

    def _check_memory_rag_ready(self) -> tuple[bool, str]:
        cfg = self._cfg()
        if not cfg.memory_rag.enable:
            return False, "Memory RAG is disabled in enhance mode config."
        if not self.memory_rag_store:
            return False, "Memory RAG store is not initialized."
        return True, ""

    def _memory_rag_webui_url(self, cfg: PluginConfig) -> str:
        host = str(cfg.memory_rag_webui.host or "127.0.0.1").strip() or "127.0.0.1"
        port = int(cfg.memory_rag_webui.port)
        if host in {"0.0.0.0", "::"}:
            host = "127.0.0.1"
        return f"http://{host}:{port}"

    async def _start_memory_rag_webui(self) -> None:
        cfg = self._cfg()
        webui_cfg = cfg.memory_rag_webui
        if not webui_cfg.enable:
            return
        if not self.memory_rag_store:
            logger.warning(
                "enhance-mode | memory_rag_webui enabled but memory_rag_store is unavailable"
            )
            return
        if self.rag_webui_server is not None:
            return

        try:
            self.rag_webui_server = RAGWebUIServer(
                store=self.memory_rag_store,
                config={
                    "host": webui_cfg.host,
                    "port": webui_cfg.port,
                    "access_password": webui_cfg.access_password,
                    "session_timeout": webui_cfg.session_timeout,
                },
                plugin_version="0.2.4",
            )
            await self.rag_webui_server.start()
            logger.info(
                "enhance-mode | Memory RAG WebUI started at %s",
                self._memory_rag_webui_url(cfg),
            )
            if self.rag_webui_server.password_generated:
                logger.info(
                    "enhance-mode | Memory RAG WebUI generated password: %s",
                    self.rag_webui_server.access_password,
                )
        except Exception as e:
            logger.error(
                f"enhance-mode | 启动 Memory RAG WebUI 失败: {e}", exc_info=True
            )
            self.rag_webui_server = None

    async def _stop_memory_rag_webui(self) -> None:
        if self.rag_webui_server is None:
            return
        try:
            await self.rag_webui_server.stop()
        except Exception as e:
            logger.warning(
                f"enhance-mode | 停止 Memory RAG WebUI 失败: {e}", exc_info=True
            )
        finally:
            self.rag_webui_server = None

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self) -> None:
        cfg = self._cfg()
        self._display_timezone = self._resolve_config_timezone()
        if self.memory_rag_store is not None:
            self.memory_rag_store.set_display_timezone(self._display_timezone)
        logger.info(
            "enhance-mode | loaded | react_mode=%s group_history=%s active_reply=%s web_search=%s memory_rag=%s webui=%s lru_max_origins=%s timezone=%s",
            cfg.group_features.react_mode_enable,
            cfg.group_history_enabled,
            cfg.active_reply_enabled,
            cfg.web_search.enable,
            cfg.memory_rag.enable,
            cfg.memory_rag_webui.enable,
            cfg.global_settings.lru_cache.max_origins,
            self._display_timezone,
        )
        await self._start_memory_rag_webui()

    def _allow_active_reply(self, event: AstrMessageEvent, cfg: PluginConfig) -> bool:
        ar = cfg.active_reply
        if not cfg.active_reply_enabled:
            return False
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return False
        if event.is_at_or_wake_command:
            return False
        if ar.whitelist and (
            event.unified_msg_origin not in ar.whitelist
            and (event.get_group_id() and event.get_group_id() not in ar.whitelist)
        ):
            return False
        return True

    async def _resolve_persona_mask(self, event: AstrMessageEvent) -> tuple[str, str]:
        persona_id = ""
        try:
            session_service_config = await sp.get_async(
                scope="umo",
                scope_id=event.unified_msg_origin,
                key="session_service_config",
                default={},
            )
            if isinstance(session_service_config, dict):
                persona_id = str(session_service_config.get("persona_id") or "").strip()
        except Exception as e:
            logger.debug(f"enhance-mode | 获取 session persona 失败: {e}")

        if not persona_id:
            try:
                curr_cid = (
                    await self.context.conversation_manager.get_curr_conversation_id(
                        event.unified_msg_origin,
                    )
                )
                if curr_cid:
                    conv = await self.context.conversation_manager.get_conversation(
                        event.unified_msg_origin,
                        curr_cid,
                    )
                    if conv and conv.persona_id:
                        persona_id = str(conv.persona_id).strip()
            except Exception as e:
                logger.debug(f"enhance-mode | 获取 conversation persona 失败: {e}")

        if not persona_id:
            cfg = self.context.get_config(umo=event.unified_msg_origin)
            persona_id = str(
                cfg.get("provider_settings", {}).get("default_personality") or ""
            ).strip()

        if persona_id == "[%None]":
            return "none", "No persona mask."

        persona = None
        if persona_id:
            try:
                persona = next(
                    (
                        p
                        for p in self.context.persona_manager.personas_v3
                        if p.get("name") == persona_id
                    ),
                    None,
                )
            except Exception:
                persona = None

        if not persona:
            try:
                persona = await self.context.persona_manager.get_default_persona_v3(
                    event.unified_msg_origin
                )
            except Exception:
                persona = {"name": "default", "prompt": ""}

        persona_name = str(persona.get("name") or "default")
        persona_prompt = str(persona.get("prompt") or "").strip()
        if not persona_prompt:
            persona_prompt = "You are a helpful and friendly assistant."
        return persona_name, persona_prompt

    def _resolve_model_choice_provider(
        self, event: AstrMessageEvent, cfg: PluginConfig
    ) -> Provider | None:
        provider_id = str(cfg.active_reply.model_choice_provider_id or "").strip()
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if provider and isinstance(provider, Provider):
                logger.debug(
                    "enhance-mode | model_choice provider from config | provider=%s",
                    self._provider_label(provider),
                )
                return provider
            logger.warning(
                "enhance-mode | 配置的 model_choice_provider_id 无效或类型不匹配: %s",
                provider_id,
            )

        provider = self.context.get_using_provider(event.unified_msg_origin)
        if provider and isinstance(provider, Provider):
            logger.debug(
                "enhance-mode | model_choice provider fallback to current | provider=%s",
                self._provider_label(provider),
            )
            return provider
        return None

    @staticmethod
    def _format_history_time(value: Any) -> str:
        if isinstance(value, datetime.datetime):
            return value.strftime("%H:%M:%S")
        if isinstance(value, (int, float)):
            try:
                return datetime.datetime.fromtimestamp(float(value)).strftime("%H:%M:%S")
            except Exception:
                pass
        return datetime.datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def _extract_text_from_history_content(content: Any) -> str:
        def part_text(part: Any) -> str:
            if isinstance(part, str):
                return part
            if not isinstance(part, dict):
                return ""

            part_type = str(part.get("type") or part.get("role") or "").lower()
            data = part.get("data") if isinstance(part.get("data"), dict) else part

            text = (
                data.get("text")
                or data.get("content")
                or data.get("message")
                or part.get("text")
                or part.get("content")
            )
            if isinstance(text, str) and text.strip():
                return text

            if part_type in {"image", "image_url", "pic", "picture"}:
                return " [Image]"
            if part_type in {"at", "mention"}:
                target = data.get("name") or data.get("qq") or data.get("id") or ""
                return f" [At: {target}]" if target else " [At]"
            if part_type in {"reply", "quote"}:
                target = data.get("id") or data.get("message_id") or ""
                return f" [Quote #msg{target}]" if target else " [Quote]"
            if part_type in {"file", "record", "video", "face", "forward"}:
                return f" [{part_type.title()}]"
            return ""

        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(part_text(part) for part in content).strip()
        if not isinstance(content, dict):
            return str(content or "").strip()

        for key in ("message_str", "raw_message", "text", "content"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        message = content.get("message")
        if isinstance(message, list):
            return "".join(part_text(part) for part in message).strip()
        if isinstance(message, str):
            return message.strip()

        return ""

    @staticmethod
    def _extract_image_sources_from_history_content(content: Any) -> list[dict[str, str]]:
        def pick_first(data: dict[str, Any], keys: tuple[str, ...]) -> str:
            for key in keys:
                value = data.get(key)
                if value is None:
                    continue
                clean = str(value or "").strip()
                if clean:
                    return clean
            return ""

        def part_image_source(part: Any) -> dict[str, str] | None:
            if not isinstance(part, dict):
                return None

            part_type = str(part.get("type") or part.get("role") or "").lower()
            data = part.get("data") if isinstance(part.get("data"), dict) else part
            if part_type not in {"image", "image_url", "pic", "picture"}:
                return None

            image_url = pick_first(
                data,
                (
                    "url",
                    "image_url",
                    "src",
                    "download_url",
                    "origin_url",
                    "file",
                    "path",
                ),
            )
            fileid = pick_first(data, ("fileid", "file_id"))
            cache_source = f"fileid:{fileid}" if fileid else ""
            if not cache_source:
                cache_source = pick_first(
                    data,
                    ("file", "url", "image_url", "src", "path", "download_url"),
                )
            if not image_url and not cache_source:
                return None
            return {
                "url": image_url or cache_source,
                "cache_source": cache_source or image_url,
            }

        if isinstance(content, list):
            return [
                source
                for source in (part_image_source(part) for part in content)
                if source is not None
            ]
        if isinstance(content, dict):
            message = content.get("message")
            if isinstance(message, list):
                return [
                    source
                    for source in (part_image_source(part) for part in message)
                    if source is not None
                ]
            source = part_image_source(content)
            return [source] if source is not None else []
        return []

    @staticmethod
    def _history_content_has_forward_context_payload(content: Any) -> bool:
        def check(value: Any) -> bool:
            if isinstance(value, str):
                lowered = value.lower()
                return (
                    "[cq:forward" in lowered
                    or "[cq:json" in lowered
                    or "[componenttype.json]" in lowered
                    or "[json]" in lowered
                    or "[jsonshare]" in lowered
                    or "m_resid=" in lowered
                )
            if isinstance(value, list):
                return any(check(item) for item in value)
            if not isinstance(value, dict):
                return False

            part_type = str(value.get("type") or value.get("role") or "").lower()
            if part_type in {"forward", "node", "nodes", "json"}:
                return True
            if isinstance(value.get("multiForwardMsgElement"), dict):
                return True
            if isinstance(value.get("arkElement"), dict):
                return True
            if isinstance(value.get("bytesData"), str):
                return True
            data = value.get("data")
            if isinstance(data, dict) and check(data):
                return True
            for key in (
                "message",
                "content",
                "raw_message",
                "elements",
                "records",
                "json",
                "raw",
            ):
                if check(value.get(key)):
                    return True
            return False

        return check(content)

    async def _parse_adapter_history_rich_text(
        self, event: AstrMessageEvent, message: Any, raw_parts: Any
    ) -> str:
        if not self._history_content_has_forward_context_payload(
            message
        ) and not self._history_content_has_forward_context_payload(raw_parts):
            return ""
        try:
            text = await forward_parse_history_message(event, message)
        except Exception as e:
            logger.debug(
                "enhance-mode | forward-context history parse failed | error=%s",
                e,
            )
            return ""
        text = str(text or "").strip()
        if text and text not in {"[Forward]", "[Forward: depth limit]"}:
            return text
        return ""

    def _register_history_image_sources(
        self,
        origin: str,
        message_id: str,
        image_urls: list[str],
        cache_sources: list[str] | None = None,
    ) -> None:
        normalized_msg_id = self._normalize_message_id(message_id)
        if not normalized_msg_id or not image_urls:
            return

        self._touch_origin(origin, self._cfg())
        registry = self.runtime.image_message_registry[origin]
        message_entry = registry.get(normalized_msg_id)
        if not isinstance(message_entry, dict):
            message_entry = {}
            registry[normalized_msg_id] = message_entry

        urls = message_entry.get("urls")
        if not isinstance(urls, list):
            urls = []
            message_entry["urls"] = urls

        stored_cache_sources = message_entry.get("cache_sources")
        if not isinstance(stored_cache_sources, list):
            stored_cache_sources = [str(url or "").strip() for url in urls]
            message_entry["cache_sources"] = stored_cache_sources

        captions = message_entry.get("captions")
        if not isinstance(captions, dict):
            message_entry["captions"] = {}

        existing_keys = {
            str(stored_cache_sources[idx] if idx < len(stored_cache_sources) else "")
            or str(urls[idx] if idx < len(urls) else "")
            for idx in range(max(len(urls), len(stored_cache_sources)))
        }
        cache_sources = cache_sources or []
        for idx, image_url in enumerate(image_urls):
            clean_url = str(image_url or "").strip()
            cache_source = (
                str(cache_sources[idx] or "").strip()
                if idx < len(cache_sources)
                else ""
            )
            key = cache_source or clean_url
            if not key or key in existing_keys:
                continue
            urls.append(clean_url or key)
            stored_cache_sources.append(cache_source or clean_url)
            existing_keys.add(key)

    async def _format_adapter_history_entry(
        self, event: AstrMessageEvent, message: Any
    ) -> dict[str, Any]:
        if not isinstance(message, dict):
            return {}

        raw_parts = message.get("message")
        if raw_parts is None:
            raw_parts = message.get("raw_message") or message.get("message_str") or ""

        text = await self._parse_adapter_history_rich_text(
            event,
            message,
            raw_parts,
        )
        if not text:
            text = self._extract_text_from_history_content(raw_parts)
        if not text:
            return {}

        sender = message.get("sender") if isinstance(message.get("sender"), dict) else {}
        sender_id = str(sender.get("user_id") or sender.get("id") or "").strip()
        sender_name = str(
            sender.get("nickname")
            or sender.get("card")
            or sender.get("name")
            or sender_id
            or "Unknown"
        ).strip()
        message_id = self._normalize_message_id(
            message.get("message_id") or message.get("id") or message.get("real_id") or ""
        )
        timestamp = self._format_history_time(message.get("time") or message.get("timestamp"))
        image_sources = self._extract_image_sources_from_history_content(raw_parts)
        return {
            "line": f"[{sender_name}/{sender_id}/{timestamp}] #msg{message_id}: {text}",
            "message_id": message_id,
            "image_urls": [source["url"] for source in image_sources],
            "cache_sources": [source["cache_source"] for source in image_sources],
        }

    @staticmethod
    def _history_message_sort_key(message: Any) -> tuple[int, int, int]:
        if not isinstance(message, dict):
            return (0, 0, 0)

        def to_int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        return (
            to_int(message.get("time") or message.get("timestamp")),
            to_int(message.get("message_seq") or message.get("real_seq")),
            to_int(message.get("message_id") or message.get("real_id") or message.get("id")),
        )

    @staticmethod
    def _raw_event_value(event: AstrMessageEvent, key: str) -> Any:
        raw_event = getattr(getattr(event, "message_obj", None), "raw_message", None)
        if isinstance(raw_event, dict):
            return raw_event.get(key)
        try:
            getter = getattr(raw_event, "get", None)
            if callable(getter):
                return getter(key)
        except Exception:
            pass
        return getattr(raw_event, key, None)

    @staticmethod
    def _onebot_data(result: Any) -> Any:
        if isinstance(result, dict) and "data" in result:
            return result.get("data")
        return result

    async def _call_onebot_action(
        self, event: AstrMessageEvent, action: str, **params: Any
    ) -> Any:
        bot = getattr(event, "bot", None)
        if bot is None:
            return None

        api = getattr(bot, "api", None)
        for target in (api, bot):
            if target is None:
                continue
            call_action = getattr(target, "call_action", None)
            if callable(call_action):
                try:
                    return await call_action(action, **params)
                except TypeError:
                    try:
                        return await call_action(action, params)
                    except Exception:
                        pass
                except Exception as e:
                    logger.debug(
                        "enhance-mode | onebot call_action failed | action=%s params=%s error=%s",
                        action,
                        params,
                        e,
                    )
            direct = getattr(target, action, None)
            if callable(direct):
                try:
                    return await direct(**params)
                except Exception as e:
                    logger.debug(
                        "enhance-mode | onebot direct api failed | action=%s params=%s error=%s",
                        action,
                        params,
                        e,
                    )
        return None

    @staticmethod
    def _extract_adapter_history_messages(data: Any) -> list[Any]:
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            return []
        for key in ("messages", "records", "message", "data"):
            value = data.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                nested = Main._extract_adapter_history_messages(value)
                if nested:
                    return nested
        return []

    async def _fetch_adapter_group_history(
        self, event: AstrMessageEvent, limit: int
    ) -> list[Any]:
        group_id = str(event.get_group_id() or "").strip()
        if not group_id or limit <= 0:
            return []

        raw_payload = self._raw_event_value(event, "raw")
        current_seq_candidates = (
            ("real_seq", self._raw_event_value(event, "real_seq")),
            (
                "raw.msgSeq",
                raw_payload.get("msgSeq") if isinstance(raw_payload, dict) else None,
            ),
            ("msgSeq", self._raw_event_value(event, "msgSeq")),
            ("message_seq", self._raw_event_value(event, "message_seq")),
        )
        current_seq_source = ""
        current_seq = None
        for source, value in current_seq_candidates:
            clean_value = str(value or "").strip()
            if not clean_value:
                continue
            current_seq_source = source
            current_seq = clean_value
            break

        group_value = self._numeric_if_digits(group_id)
        candidates: list[tuple[dict[str, Any], str]] = [
            ({"group_id": group_value, "count": limit}, "none"),
            ({"group_id": group_value}, "none"),
        ]
        if current_seq:
            seq_value = self._numeric_if_digits(str(current_seq))
            candidates.extend(
                [
                    (
                        {
                            "group_id": group_value,
                            "message_seq": seq_value,
                            "count": limit,
                        },
                        current_seq_source,
                    ),
                    (
                        {
                            "group_id": group_value,
                            "message_seq": seq_value,
                        },
                        current_seq_source,
                    ),
                ]
            )

        best_messages: list[Any] = []
        for params, seq_source in candidates:
            result = await self._call_onebot_action(
                event,
                "get_group_msg_history",
                **params,
            )
            messages = self._extract_adapter_history_messages(self._onebot_data(result))
            if messages:
                if len(messages) > len(best_messages):
                    best_messages = messages
                logger.debug(
                    "enhance-mode | adapter history fetched | origin=%s count=%s seq_source=%s params=%s",
                    event.unified_msg_origin,
                    len(messages),
                    seq_source or "none",
                    params,
                )
                if len(messages) >= min(limit, 2):
                    return messages
        return best_messages

    async def _backfill_group_history(
        self,
        event: AstrMessageEvent,
        cfg: PluginConfig,
        target_count: int,
    ) -> None:
        if target_count <= 0 or event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        origin = event.unified_msg_origin
        fetch_limit = min(
            cfg.group_history.max_messages,
            max(target_count * 2, target_count + 5, 10),
        )
        try:
            raw_messages = await self._fetch_adapter_group_history(event, fetch_limit)
        except Exception as e:
            logger.debug(
                "enhance-mode | adapter history backfill failed | origin=%s error=%s",
                origin,
                e,
            )
            return
        if not raw_messages:
            return

        entries: list[dict[str, Any]] = []
        for message in sorted(raw_messages, key=self._history_message_sort_key):
            try:
                entry = await self._format_adapter_history_entry(event, message)
            except Exception as e:
                logger.debug(
                    "enhance-mode | adapter history format failed | origin=%s error=%s",
                    origin,
                    e,
                )
                continue
            if entry.get("line"):
                entries.append(entry)
        if not entries:
            return

        self._touch_origin(origin, cfg)
        chats = self.runtime.session_chats[origin]
        existing_ids = {
            self._normalize_message_id(self._extract_message_id_from_history_line(line))
            for line in chats
            if self._extract_message_id_from_history_line(line)
        }
        existing_lines = set(chats)
        new_lines: list[str] = []
        for entry in entries:
            line = str(entry.get("line") or "")
            message_id = self._normalize_message_id(entry.get("message_id") or "")
            if not line:
                continue
            if (message_id and message_id in existing_ids) or line in existing_lines:
                continue
            new_lines.append(line)
            existing_lines.add(line)
            if message_id:
                existing_ids.add(message_id)
                self._register_history_image_sources(
                    origin,
                    message_id,
                    list(entry.get("image_urls") or []),
                    list(entry.get("cache_sources") or []),
                )

        if not new_lines:
            return
        chats[:0] = new_lines
        if len(chats) > cfg.group_history.max_messages:
            del chats[: len(chats) - cfg.group_history.max_messages]
        logger.debug(
            "enhance-mode | adapter history backfilled | origin=%s added=%s size=%s",
            origin,
            len(new_lines),
            len(chats),
        )

    async def _get_group_context_lines(
        self,
        event: AstrMessageEvent,
        cfg: PluginConfig,
        exclude_current: bool,
        limit: int | None = None,
        exclude_message_ids: set[str] | None = None,
        backfill_target: int = 0,
    ) -> list[str]:
        origin = event.unified_msg_origin
        normalized_excluded_ids = {
            self._normalize_message_id(message_id)
            for message_id in (exclude_message_ids or set())
            if self._normalize_message_id(message_id)
        }
        current_msg_id = self._normalize_message_id(
            getattr(event.message_obj, "message_id", "")
        )
        if exclude_current and current_msg_id:
            normalized_excluded_ids.add(current_msg_id)

        effective_limit = cfg.group_history.max_messages if limit is None else max(0, limit)
        if backfill_target > 0 and effective_limit > 0:
            cached_available = [
                line
                for line in (self.runtime.session_chats.get(origin) or [])
                if line
                and self._normalize_message_id(
                    self._extract_message_id_from_history_line(line)
                )
                not in normalized_excluded_ids
            ]
            if len(cached_available) < min(backfill_target, effective_limit):
                await self._backfill_group_history(
                    event,
                    cfg,
                    min(backfill_target, effective_limit),
                )

        cached_lines = list(self.runtime.session_chats.get(origin) or [])
        prior_lines = []
        for line in cached_lines:
            if not line:
                continue
            line_msg_id = self._normalize_message_id(
                self._extract_message_id_from_history_line(line)
            )
            if line_msg_id and line_msg_id in normalized_excluded_ids:
                continue
            prior_lines.append(line)
        if exclude_current and not current_msg_id and prior_lines:
            prior_lines = prior_lines[:-1]
        return prior_lines[-effective_limit:] if effective_limit > 0 else []

    async def _resolve_active_current_message_text(
        self, event: AstrMessageEvent, cfg: PluginConfig
    ) -> tuple[str, str]:
        forward_context_text = self._get_forward_context_text(event)
        if forward_context_text:
            return (
                self._truncate_active_message_text(forward_context_text),
                "forward_context",
            )

        origin = event.unified_msg_origin
        current_msg_id = self._normalize_message_id(
            getattr(event.message_obj, "message_id", "")
        )
        if self._history_recording_enabled(cfg) and current_msg_id:
            current_line = self._find_history_line_by_message_id(origin, current_msg_id)
            if current_line:
                resolved_lines = await self._resolve_image_captions_for_context_lines(
                    event,
                    cfg,
                    [current_line],
                )
                resolved_line = resolved_lines[0] if resolved_lines else current_line
                current_body = self._extract_history_line_body(resolved_line)
                if current_body:
                    return (
                        self._truncate_active_message_text(current_body),
                        "session_chats",
                    )

        body, image_urls, image_cache_sources = self._format_event_message_body(event)
        if body and image_urls and cfg.group_history.image_caption:
            for image_idx, image_url in enumerate(image_urls):
                cache_source = (
                    image_cache_sources[image_idx]
                    if image_idx < len(image_cache_sources)
                    else ""
                )
                try:
                    caption = await self._caption_image_with_cache(
                        event,
                        cfg,
                        image_url=image_url,
                        cache_source=cache_source,
                    )
                except Exception as e:
                    logger.debug(
                        "enhance-mode | active message image caption failed | "
                        "origin=%s msg_id=%s image_index=%s error=%s",
                        origin,
                        current_msg_id,
                        image_idx + 1,
                        e,
                    )
                    continue
                if not caption:
                    continue
                body, _ = self._replace_image_marker_at_index(
                    body,
                    image_idx,
                    caption,
                )

        if body:
            return self._truncate_active_message_text(body), "event_message_chain"

        fallback = (event.message_str or "").strip() or "[Empty]"
        fallback_source = (
            "event.message_str" if (event.message_str or "").strip() else "empty"
        )
        return self._truncate_active_message_text(fallback), fallback_source

    async def _collect_active_reply_context(
        self,
        event: AstrMessageEvent,
        cfg: PluginConfig,
        current_message_text: str = "",
        current_message_source: str = "",
        history_limit: int | None = None,
        exclude_current: bool = True,
        exclude_message_ids: set[str] | None = None,
        backfill_target: int = 0,
    ) -> dict[str, Any]:
        origin = event.unified_msg_origin
        if not current_message_text:
            current_message_text, current_message_source = (
                await self._resolve_active_current_message_text(event, cfg)
            )

        recent_history_lines: list[str] = []
        effective_history_limit = (
            cfg.active_reply.unified_context_messages
            if history_limit is None
            else max(0, history_limit)
        )
        if self._history_recording_enabled(cfg) and effective_history_limit > 0:
            recent_history_lines = await self._get_group_context_lines(
                event,
                cfg,
                exclude_current=exclude_current,
                limit=effective_history_limit,
                exclude_message_ids=exclude_message_ids,
                backfill_target=backfill_target,
            )
            recent_history_lines = await self._resolve_image_captions_for_context_lines(
                event,
                cfg,
                recent_history_lines,
            )

        logger.info(
            "enhance-mode | context collected | origin=%s current_source=%s current_len=%s history_count=%s history_limit=%s",
            origin,
            current_message_source or "unknown",
            len(current_message_text),
            len(recent_history_lines),
            effective_history_limit,
        )
        return {
            "origin": origin,
            "current_message_text": current_message_text,
            "current_message_source": current_message_source or "unknown",
            "recent_history_lines": recent_history_lines,
        }

    def _build_active_reply_interaction_instructions(
        self, cfg: PluginConfig
    ) -> str:
        interaction_instructions = build_interaction_instructions(
            cfg.group_features.mention_parse,
            cfg.group_history.include_sender_id,
            cfg.group_features.refuse_enable,
        )
        interaction_instructions += (
            "\nIf a history message contains `[Image]` and visual details are necessary, "
            "you may call `enhance_use_image(message_id, image_index, attach_to_model, write_to_history, prompt)`. "
            "By default it does both: attach image to this run context and write description back into chat history. "
            "Set `attach_to_model=false` for history-only. "
            "Set `write_to_history=false` for attach-only."
        )
        if cfg.web_search.enable:
            interaction_instructions += (
                "\nWhen real-time facts or uncertain external information are needed, "
                "you may call `grok_web_search(query)`."
            )
        return interaction_instructions

    def _build_model_choice_prompt(
        self,
        cfg: PluginConfig,
        messages: list[str],
        persona_name: str,
        persona_mask: str,
        history_context_lines: list[str],
    ) -> str:
        prompt_tmpl = cfg.active_reply.model_choice_prompt
        injected_history_count = len(history_context_lines)
        history_context = (
            bounded_chat_history_text(history_context_lines)
            if history_context_lines
            else "(disabled or no additional history)"
        )

        try:
            return prompt_tmpl.format(
                stack_size=len(messages),
                messages="\n".join(messages),
                history_count=injected_history_count,
                history_context=history_context,
                persona_name=persona_name,
                persona_mask=persona_mask,
            )
        except Exception:
            return (
                f"{prompt_tmpl}\n\n"
                f"人格面具({persona_name}):\n{persona_mask}\n\n"
                f"最近消息:\n{chr(10).join(messages)}\n\n"
                f"额外历史上下文({injected_history_count}):\n{history_context}\n\n"
                "请仅输出 REPLY 或 SKIP。"
            )

    def _build_active_reply_prompt(
        self,
        cfg: PluginConfig,
        context: dict[str, Any],
        active_mode: str,
    ) -> str:
        history_lines = context.get("recent_history_lines", [])
        current_message_text = (
            str(context.get("current_message_text") or "[Empty]").strip() or "[Empty]"
        )
        history_text = (
            bounded_chat_history_text(history_lines)
            if history_lines
            else "(no recent group chat history)"
        )
        interaction_instructions = self._build_active_reply_interaction_instructions(cfg)
        history_prefix = (
            "You are now in a chatroom. The recent chat history is as follows:\n"
            f"{history_text}\n\n"
        )

        if str(active_mode or "").strip():
            return (
                f"{history_prefix}"
                "You decided to actively join this conversation because some recent messages are worth replying to.\n"
                "The latest message is:\n"
                f"{current_message_text}\n\n"
                "Choose the message(s) you want to respond to from the recent chat history and the latest message, "
                "and compose a natural reply. Quote the message you choose in most cases.\n"
                "Only output your response and do not output any other information. "
                "You MUST use the SAME language as the chatroom is using."
                f"{interaction_instructions}"
            )

        return (
            f"{history_prefix}"
            "Now, a new message is coming:\n"
            f"{current_message_text}\n\n"
            "Please react to it. Your entire output is your reply to this message. "
            "Quote the message which is coming in most cases. "
            "Only output your response and do not output any other information. "
            "You MUST use the SAME language as the chatroom is using."
            f"{interaction_instructions}"
        )

    async def _new_active_reply_conversation(
        self,
        event: AstrMessageEvent,
        new_conversation: Any,
    ) -> str | None:
        origin = event.unified_msg_origin
        cid = await new_conversation(origin)
        logger.info(
            "enhance-mode | 已自动创建空会话 | origin=%s conversation_id=%s",
            origin,
            cid,
        )
        return cid

    async def _ensure_active_reply_conversation(
        self, event: AstrMessageEvent, cfg: PluginConfig
    ) -> tuple[str, Any] | tuple[None, None]:
        origin = event.unified_msg_origin
        conv_mgr = self.context.conversation_manager

        curr_cid = await conv_mgr.get_curr_conversation_id(origin)
        if not curr_cid:
            if not cfg.active_reply.auto_create_conversation:
                logger.error(
                    "enhance-mode | 当前未处于对话状态，无法主动回复，"
                    "请使用 /switch 或 /new 创建一个会话。"
                )
                return None, None
            new_conversation = getattr(conv_mgr, "new_conversation", None)
            if not callable(new_conversation):
                logger.error(
                    "enhance-mode | 当前未处于对话状态，且当前 AstrBot 版本不支持自动创建会话"
                )
                return None, None

            curr_cid = await self._new_active_reply_conversation(
                event,
                new_conversation,
            )

        try:
            conv = await conv_mgr.get_conversation(
                origin,
                curr_cid,
                create_if_not_exists=True,
            )
        except TypeError as e:
            if "create_if_not_exists" not in str(e):
                raise
            conv = await conv_mgr.get_conversation(origin, curr_cid)

        if conv:
            return curr_cid, conv

        new_conversation = getattr(conv_mgr, "new_conversation", None)
        if not cfg.active_reply.auto_create_conversation:
            logger.error("enhance-mode | 未找到对话，无法主动回复")
            return None, None
        if not callable(new_conversation):
            logger.error("enhance-mode | 未找到对话，且无法自动创建新会话")
            return None, None

        curr_cid = await self._new_active_reply_conversation(
            event,
            new_conversation,
        )
        try:
            conv = await conv_mgr.get_conversation(
                origin,
                curr_cid,
                create_if_not_exists=True,
            )
        except TypeError as e:
            if "create_if_not_exists" not in str(e):
                raise
            conv = await conv_mgr.get_conversation(origin, curr_cid)

        if not conv:
            logger.error("enhance-mode | 自动创建会话后仍未找到对话，无法主动回复")
            return None, None

        logger.info(
            "enhance-mode | 当前会话对象缺失，已自动创建新会话 | "
            "origin=%s conversation_id=%s",
            origin,
            curr_cid,
        )
        return curr_cid, conv

    @staticmethod
    def _provider_chat_id(provider: Provider) -> str:
        try:
            meta = provider.meta()
            meta_id = getattr(meta, "id", None)
            if meta_id:
                return str(meta_id)
        except Exception:
            pass
        provider_id = getattr(provider, "provider_id", None) or getattr(
            provider, "id", None
        )
        return str(provider_id or "").strip()

    def _resolve_web_search_provider(self, cfg: PluginConfig) -> Provider | None:
        provider_id = str(cfg.web_search.provider_id or "").strip()
        if not provider_id:
            return None
        provider = self.context.get_provider_by_id(provider_id)
        if provider and isinstance(provider, Provider):
            return provider
        logger.warning(
            "enhance-mode | 配置的 web_search.provider_id 无效或类型不匹配: %s",
            provider_id,
        )
        return None

    @staticmethod
    def _normalize_api_base_url(raw_base_url: str) -> str:
        base_url = str(raw_base_url or "").strip().rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        return base_url

    @staticmethod
    def _extract_provider_api_key(provider: Provider) -> str:
        get_current_key = getattr(provider, "get_current_key", None)
        if callable(get_current_key):
            try:
                key = str(get_current_key() or "").strip()
                if key:
                    return key
            except Exception:
                pass

        keys: list[Any] = []
        get_keys = getattr(provider, "get_keys", None)
        if callable(get_keys):
            try:
                fetched = get_keys()
                if isinstance(fetched, list):
                    keys = fetched
                elif isinstance(fetched, str):
                    keys = [fetched]
            except Exception:
                keys = []

        if not keys:
            raw_keys = provider.provider_config.get("key", [])
            if isinstance(raw_keys, list):
                keys = raw_keys
            elif isinstance(raw_keys, str):
                keys = [raw_keys]

        for item in keys:
            key = str(item or "").strip()
            if key:
                return key
        return ""

    @staticmethod
    def _parse_sse_chat_completion(raw_text: str) -> dict[str, Any] | None:
        chunks: list[dict[str, Any]] = []
        for line in str(raw_text or "").splitlines():
            line = line.strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk, dict):
                chunks.append(chunk)

        if not chunks:
            return None

        merged_content = ""
        model_name = ""
        usage_info: dict[str, Any] = {}
        for chunk in chunks:
            if not model_name:
                model_name = str(chunk.get("model") or "")
            chunk_usage = chunk.get("usage")
            if isinstance(chunk_usage, dict):
                usage_info = chunk_usage
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice0 = choices[0]
            if not isinstance(choice0, dict):
                continue
            delta = choice0.get("delta")
            if isinstance(delta, dict):
                delta_content = delta.get("content")
                if isinstance(delta_content, str):
                    merged_content += delta_content

        return {
            "choices": [{"message": {"content": merged_content}}],
            "model": model_name,
            "usage": usage_info,
        }

    @staticmethod
    def _extract_chat_completion_text(data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        choice0 = choices[0]
        if not isinstance(choice0, dict):
            return ""
        message = choice0.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def _extract_usage_tokens(data: dict[str, Any]) -> dict[str, int]:
        usage_raw = data.get("usage")
        if not isinstance(usage_raw, dict):
            return {}
        prompt_tokens = int(
            usage_raw.get("prompt_tokens")
            or usage_raw.get("input_tokens")
            or usage_raw.get("input")
            or 0
        )
        completion_tokens = int(
            usage_raw.get("completion_tokens")
            or usage_raw.get("output_tokens")
            or usage_raw.get("output")
            or 0
        )
        total_tokens = int(
            usage_raw.get("total_tokens")
            or usage_raw.get("total")
            or (prompt_tokens + completion_tokens)
        )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _join_base_with_path(base_url: str, path: str) -> str:
        cleaned_path = str(path or "").strip()
        if cleaned_path.startswith("http://") or cleaned_path.startswith("https://"):
            return cleaned_path
        if not cleaned_path.startswith("/"):
            cleaned_path = f"/{cleaned_path}"
        return f"{base_url.rstrip('/')}{cleaned_path}"

    @staticmethod
    def _extract_responses_text_and_sources(
        data: dict[str, Any],
    ) -> tuple[str, list[dict[str, str]]]:
        text_parts: list[str] = []
        source_map: dict[str, dict[str, str]] = {}

        def push_source(url: str, title: str = "", snippet: str = "") -> None:
            clean_url = str(url or "").strip()
            if not clean_url:
                return
            if clean_url not in source_map:
                source_map[clean_url] = {
                    "url": clean_url,
                    "title": str(title or "").strip(),
                    "snippet": str(snippet or "").strip(),
                }
                return
            if not source_map[clean_url]["title"] and title:
                source_map[clean_url]["title"] = str(title).strip()
            if not source_map[clean_url]["snippet"] and snippet:
                source_map[clean_url]["snippet"] = str(snippet).strip()

        output = data.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "")
                if item_type == "message":
                    content = item.get("content")
                    if not isinstance(content, list):
                        continue
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        part_type = str(part.get("type") or "")
                        if part_type not in {"output_text", "text"}:
                            continue
                        part_text = part.get("text") or part.get("content")
                        if isinstance(part_text, str) and part_text.strip():
                            text_parts.append(part_text)
                        annotations = part.get("annotations")
                        if not isinstance(annotations, list):
                            continue
                        for annotation in annotations:
                            if not isinstance(annotation, dict):
                                continue
                            if str(annotation.get("type") or "") not in {
                                "url_citation",
                                "citation",
                            }:
                                continue
                            push_source(
                                url=str(
                                    annotation.get("url")
                                    or annotation.get("source_url")
                                    or ""
                                ),
                                title=str(annotation.get("title") or ""),
                                snippet=str(annotation.get("snippet") or ""),
                            )
                    continue
                if item_type == "web_search_call":
                    action = item.get("action")
                    if not isinstance(action, dict):
                        continue
                    action_sources = action.get("sources")
                    normalized = Main._normalize_web_search_sources(action_sources)
                    for source in normalized:
                        push_source(
                            url=source.get("url", ""),
                            title=source.get("title", ""),
                            snippet=source.get("snippet", ""),
                        )

        merged_text = "\n".join(part for part in text_parts if part.strip()).strip()
        if not merged_text:
            merged_text = str(data.get("output_text") or "").strip()
        return merged_text, list(source_map.values())

    def _build_web_search_http_requests(
        self, provider: Provider, query: str, cfg: PluginConfig
    ) -> tuple[list[dict[str, Any]], str]:
        provider_label = self._provider_chat_id(provider) or self._provider_label(
            provider
        )
        provider_cfg = (
            provider.provider_config
            if isinstance(provider.provider_config, dict)
            else {}
        )

        api_base_cfg = str(cfg.web_search.base_url_override or "").strip()
        api_base = self._normalize_api_base_url(
            api_base_cfg or str(provider_cfg.get("api_base") or "")
        )
        if not api_base:
            raise ValueError(
                f"Provider `{provider_label}` missing `api_base`, cannot run web search."
            )

        api_key = self._extract_provider_api_key(provider)
        if not api_key:
            raise ValueError(
                f"Provider `{provider_label}` missing API key, cannot run web search."
            )

        model = str(provider.get_model() or provider_cfg.get("model") or "").strip()
        custom_extra_body = provider_cfg.get("custom_extra_body", {})

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        custom_headers = provider_cfg.get("custom_headers", {})
        if isinstance(custom_headers, dict):
            protected_headers = {"authorization", "content-type"}
            for key, value in custom_headers.items():
                if str(key).lower() in protected_headers:
                    continue
                headers[str(key)] = str(value)

        request_mode = str(cfg.web_search.request_mode or "auto").strip().lower()
        modes: list[str]
        if request_mode in {"responses", "chat_completions"}:
            modes = [request_mode]
        else:
            modes = ["responses", "chat_completions"]

        requests: list[dict[str, Any]] = []
        for mode in modes:
            if mode == "responses":
                body = {
                    "input": query,
                    "instructions": cfg.web_search.system_prompt,
                    "temperature": 0.2,
                    "tools": [{"type": "web_search"}],
                    "tool_choice": "auto",
                }
                if model:
                    body["model"] = model
                if isinstance(custom_extra_body, dict):
                    protected_keys = {"model", "input", "instructions"}
                    for key, value in custom_extra_body.items():
                        if str(key) in protected_keys:
                            continue
                        body[str(key)] = value
                requests.append(
                    {
                        "mode": "responses",
                        "url": self._join_base_with_path(api_base, "/v1/responses"),
                        "headers": headers,
                        "body": body,
                    }
                )
                continue

            body = {
                "messages": [
                    {"role": "system", "content": cfg.web_search.system_prompt},
                    {"role": "user", "content": query},
                ],
                "temperature": 0.2,
                "stream": False,
            }
            if model:
                body["model"] = model
            if isinstance(custom_extra_body, dict):
                protected_keys = {"model", "messages", "stream"}
                for key, value in custom_extra_body.items():
                    if str(key) in protected_keys:
                        continue
                    body[str(key)] = value
            requests.append(
                {
                    "mode": "chat_completions",
                    "url": self._join_base_with_path(api_base, "/v1/chat/completions"),
                    "headers": headers,
                    "body": body,
                }
            )

        return requests, provider_label

    @staticmethod
    def _try_parse_web_search_json(text: str) -> dict | None:
        clean_text = str(text or "").strip()
        if not clean_text:
            return None

        if clean_text.startswith("{") and clean_text.endswith("}"):
            try:
                parsed = json.loads(clean_text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        for matched in re.findall(code_block_pattern, clean_text):
            try:
                parsed = json.loads(matched.strip())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        decoder = json.JSONDecoder()
        start_idx = 0
        remaining_attempts = 10
        while start_idx < len(clean_text) and remaining_attempts > 0:
            brace_pos = clean_text.find("{", start_idx)
            if brace_pos == -1:
                break
            try:
                parsed, end_idx = decoder.raw_decode(clean_text, idx=brace_pos)
                if isinstance(parsed, dict) and (
                    "content" in parsed or "sources" in parsed
                ):
                    return parsed
                start_idx = end_idx
            except json.JSONDecodeError:
                start_idx = brace_pos + 1
            remaining_attempts -= 1
        return None

    @staticmethod
    def _normalize_web_search_sources(raw_sources: object) -> list[dict[str, str]]:
        from urllib.parse import urlparse

        normalized: list[dict[str, str]] = []
        if not isinstance(raw_sources, list):
            return normalized
        for item in raw_sources:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            try:
                parsed_url = urlparse(url)
                if parsed_url.scheme not in {"http", "https"}:
                    continue
                if len(url) > 2048 or any(ord(ch) < 32 for ch in url):
                    continue
            except Exception:
                continue
            normalized.append(
                {
                    "url": url,
                    "title": str(item.get("title") or "").strip(),
                    "snippet": str(item.get("snippet") or "").strip(),
                }
            )
        return normalized

    @staticmethod
    def _extract_web_search_sources_from_text(text: str) -> list[dict[str, str]]:
        from urllib.parse import urlparse

        url_pattern = r"https://[^\s)\]}>\"']+|http://[^\s)\]}>\"']+"
        seen: set[str] = set()
        out: list[dict[str, str]] = []
        for match in re.finditer(url_pattern, str(text or "")):
            url = match.group().rstrip(".,;:!?\"'")
            if not url or url in seen:
                continue
            try:
                parsed_url = urlparse(url)
                if parsed_url.scheme not in {"http", "https"}:
                    continue
                if len(url) > 2048 or any(ord(ch) < 32 for ch in url):
                    continue
            except Exception:
                continue
            seen.add(url)
            out.append({"url": url, "title": "", "snippet": ""})
        return out

    async def _run_web_search(
        self,
        event: AstrMessageEvent,
        query: str,
        cfg: PluginConfig,
    ) -> dict[str, object]:
        provider = self._resolve_web_search_provider(cfg)
        if not provider:
            return {
                "ok": False,
                "error": (
                    "Web search provider is not configured or invalid. "
                    "Please set `web_search.provider_id`."
                ),
            }
        try:
            request_specs, provider_label = self._build_web_search_http_requests(
                provider, query, cfg
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}
        if not request_specs:
            return {"ok": False, "error": "No web search request spec could be built."}

        start_ts = time.perf_counter()
        logger.info(
            "enhance-mode | web_search start(direct) | origin=%s provider=%s query_len=%s",
            event.unified_msg_origin,
            provider_label,
            len(query),
        )

        parsed_data: dict[str, Any] | None = None
        text = ""
        usage: dict[str, int] = {}
        sources_from_endpoint: list[dict[str, str]] = []
        last_error: dict[str, object] = {"ok": False, "error": "Web search failed."}

        try:
            timeout = aiohttp.ClientTimeout(total=cfg.web_search.timeout_sec)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for request_spec in request_specs:
                    mode = str(request_spec.get("mode") or "chat_completions")
                    request_url = str(request_spec.get("url") or "")
                    headers = request_spec.get("headers")
                    body = request_spec.get("body")
                    if not request_url or not isinstance(headers, dict):
                        continue

                    logger.info(
                        "enhance-mode | web_search request | origin=%s provider=%s mode=%s url=%s",
                        event.unified_msg_origin,
                        provider_label,
                        mode,
                        request_url,
                    )

                    async with session.post(
                        request_url,
                        json=body,
                        headers=headers,
                    ) as resp:
                        raw_text = await resp.text()
                        if resp.status != 200:
                            logger.warning(
                                "enhance-mode | web_search http failed | origin=%s provider=%s mode=%s status=%s body=%s",
                                event.unified_msg_origin,
                                provider_label,
                                mode,
                                resp.status,
                                raw_text[:800],
                            )
                            last_error = {
                                "ok": False,
                                "error": f"Web search HTTP {resp.status} ({mode})",
                                "raw": raw_text[:2000] if raw_text else "",
                            }
                            continue

                        parsed_data = None
                        content_type = resp.headers.get("Content-Type", "")
                        if mode == "chat_completions" and (
                            "text/event-stream" in content_type
                            or raw_text.strip().startswith("data:")
                        ):
                            parsed_data = self._parse_sse_chat_completion(raw_text)
                        else:
                            try:
                                decoded = json.loads(raw_text)
                                if isinstance(decoded, dict):
                                    parsed_data = decoded
                            except json.JSONDecodeError:
                                parsed_data = None

                        if parsed_data is None:
                            last_error = {
                                "ok": False,
                                "error": (
                                    f"Web search response parsing failed for mode={mode}."
                                ),
                                "raw": raw_text[:2000] if raw_text else "",
                            }
                            continue

                        usage = self._extract_usage_tokens(parsed_data)
                        if mode == "responses":
                            text, sources_from_endpoint = (
                                self._extract_responses_text_and_sources(parsed_data)
                            )
                        else:
                            text = self._extract_chat_completion_text(
                                parsed_data
                            ).strip()
                            sources_from_endpoint = []

                        if not text:
                            last_error = {
                                "ok": False,
                                "error": (
                                    "Provider returned empty response for web search."
                                ),
                                "raw": json.dumps(parsed_data, ensure_ascii=False)[
                                    :2000
                                ],
                            }
                            continue
                        break
        except asyncio.TimeoutError:
            return {
                "ok": False,
                "error": (
                    "Web search timeout. "
                    f"Current timeout={cfg.web_search.timeout_sec:.1f}s."
                ),
            }
        except Exception as e:
            logger.exception(
                "enhance-mode | web_search direct call failed | origin=%s provider=%s error=%s",
                event.unified_msg_origin,
                provider_label,
                e,
            )
            return {"ok": False, "error": f"Web search provider call failed: {e}"}

        if not text:
            last_error["elapsed_ms"] = (time.perf_counter() - start_ts) * 1000
            last_error["usage"] = usage
            return last_error

        elapsed_ms = (time.perf_counter() - start_ts) * 1000
        merged_sources: list[dict[str, str]] = []
        seen_urls: set[str] = set()

        def push_sources(src_list: list[dict[str, str]]) -> None:
            for source in src_list:
                if not isinstance(source, dict):
                    continue
                url = str(source.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                merged_sources.append(
                    {
                        "url": url,
                        "title": str(source.get("title") or "").strip(),
                        "snippet": str(source.get("snippet") or "").strip(),
                    }
                )

        parsed = self._try_parse_web_search_json(text)
        if parsed is not None:
            content = str(parsed.get("content") or "").strip()
            if not content:
                content = text
            push_sources(self._normalize_web_search_sources(parsed.get("sources")))
            push_sources(sources_from_endpoint)
            if not merged_sources:
                push_sources(self._extract_web_search_sources_from_text(content))
            logger.info(
                "enhance-mode | web_search done(direct) | origin=%s provider=%s elapsed_ms=%.1f sources=%s",
                event.unified_msg_origin,
                provider_label,
                elapsed_ms,
                len(merged_sources),
            )
            return {
                "ok": True,
                "content": content,
                "sources": merged_sources,
                "elapsed_ms": elapsed_ms,
                "usage": usage,
            }

        push_sources(sources_from_endpoint)
        if not merged_sources:
            push_sources(self._extract_web_search_sources_from_text(text))
        logger.info(
            "enhance-mode | web_search done(non_json direct) | origin=%s provider=%s elapsed_ms=%.1f sources=%s",
            event.unified_msg_origin,
            provider_label,
            elapsed_ms,
            len(merged_sources),
        )
        return {
            "ok": True,
            "content": text,
            "sources": merged_sources,
            "elapsed_ms": elapsed_ms,
            "usage": usage,
            "raw": text,
        }

    def _format_web_search_tool_result(
        self, result: dict[str, object], cfg: PluginConfig
    ) -> str:
        if not bool(result.get("ok")):
            error = str(result.get("error") or "Unknown error")
            raw = str(result.get("raw") or "").strip()
            if raw:
                return f"Web search failed: {error}\n{raw}"
            return f"Web search failed: {error}"

        content = str(result.get("content") or "").strip()
        sources_raw = result.get("sources")
        sources = (
            sources_raw
            if isinstance(sources_raw, list)
            else self._extract_web_search_sources_from_text(content)
        )

        lines = [f"搜索结果:\n{content}"]
        if cfg.web_search.show_sources and sources:
            max_sources = cfg.web_search.max_sources
            if max_sources > 0:
                sources = sources[:max_sources]
            lines.append("\n参考来源:")
            for idx, source in enumerate(sources, start=1):
                if not isinstance(source, dict):
                    continue
                url = str(source.get("url") or "").strip()
                title = str(source.get("title") or "").strip()
                snippet = str(source.get("snippet") or "").strip()
                if title:
                    lines.append(f"  {idx}. {title}")
                    lines.append(f"     {url}")
                else:
                    lines.append(f"  {idx}. {url}")
                if snippet:
                    lines.append(f"     {snippet}")

        lines.append("\n[提示: 请基于以上搜索结果直接回答用户，不要输出 Markdown。]")
        return "\n".join(lines)

    async def _judge_model_choice(
        self,
        event: AstrMessageEvent,
        cfg: PluginConfig,
        origin: str,
        messages: list[str],
        trigger_reason: str,
    ) -> bool:
        ar = cfg.active_reply
        history_limit = (
            min(ar.model_history_messages, ar.unified_context_messages)
            if ar.model_history_messages > 0
            else 0
        )
        stack_message_ids = {
            self._normalize_message_id(self._extract_message_id_from_history_line(line))
            for line in messages
            if self._extract_message_id_from_history_line(line)
        }
        choice_context = await self._collect_active_reply_context(
            event,
            cfg,
            history_limit=history_limit,
            exclude_message_ids=stack_message_ids,
            backfill_target=history_limit,
        )
        history_context_lines = list(choice_context.get("recent_history_lines") or [])
        injected_history_count = len(history_context_lines)

        logger.info(
            "enhance-mode | model_choice | 开始判定 | "
            f"origin={origin} trigger={trigger_reason} stack_size={len(messages)} "
            f"history={len(history_context_lines)} history_limit={history_limit} "
            f"injected_history={injected_history_count}"
        )

        provider = self._resolve_model_choice_provider(event, cfg)
        if not provider:
            logger.error("enhance-mode | 未找到可用提供商，无法执行模型选择触发")
            return False

        persona_name, persona_mask = await self._resolve_persona_mask(event)
        judge_prompt = self._build_model_choice_prompt(
            cfg,
            messages,
            persona_name,
            persona_mask,
            history_context_lines,
        )

        try:
            judge_resp = await asyncio.wait_for(
                provider.text_chat(
                    prompt=judge_prompt,
                    session_id=uuid.uuid4().hex,
                    persist=False,
                ),
                timeout=cfg.global_settings.timeouts.model_choice_sec,
            )
        except asyncio.TimeoutError:
            logger.error("enhance-mode | 模型选择触发判定超时")
            return False
        except Exception as e:
            logger.error(f"enhance-mode | 模型选择触发判定失败: {e}")
            return False

        decision_raw = (judge_resp.completion_text or "").strip().upper()
        decision = decision_raw.split()[0] if decision_raw else ""
        if decision.startswith("REPLY"):
            logger.info(
                "enhance-mode | model_choice | 判定通过(REPLY) | "
                f"origin={origin} trigger={trigger_reason} persona={persona_name}"
            )
            return True
        if decision and not decision.startswith("SKIP"):
            logger.info(
                "enhance-mode | model_choice | 判定拒绝(非标准输出按 SKIP) | "
                f"origin={origin} trigger={trigger_reason} output={decision_raw}"
            )
            return False
        logger.info(
            "enhance-mode | model_choice | 判定拒绝(SKIP) | "
            f"origin={origin} trigger={trigger_reason} persona={persona_name}"
        )
        return False

    async def _need_active_reply_model_choice(
        self, event: AstrMessageEvent, cfg: PluginConfig
    ) -> bool:
        origin = event.unified_msg_origin
        self._touch_origin(origin, cfg)

        ar = cfg.active_reply
        text = await self._build_active_message_text(event, cfg)
        text = self._truncate_active_message_text(text)
        nickname = event.message_obj.sender.nickname
        sender_id = event.get_sender_id()
        stack = self.runtime.active_reply_stacks[origin]
        current_msg_id = self._normalize_message_id(
            getattr(event.message_obj, "message_id", "")
        )

        stack_line = self._find_history_line_by_message_id(origin, current_msg_id)
        if not stack_line:
            msg_marker = f" #msg{current_msg_id}:" if current_msg_id else ":"
            stack_line = f"[{nickname}/{sender_id}]{msg_marker} {text}"
        stack.append(stack_line)
        if len(stack) > ar.model_stack_size:
            del stack[:-ar.model_stack_size]

        logger.info(
            "enhance-mode | model_choice | 栈填充 | "
            f"origin={origin} progress={len(stack)}/{ar.model_stack_size} "
            f"sender={sender_id}"
        )

        if len(stack) < ar.model_stack_size:
            return False

        messages = stack[-ar.model_stack_size :]
        stack.clear()
        return await self._judge_model_choice(
            event,
            cfg,
            origin,
            messages,
            trigger_reason="stack_full",
        )

    async def _get_image_caption(
        self,
        image_url: str,
        provider_id: str,
        prompt: str,
        timeout_sec: float,
    ) -> str:
        if not provider_id:
            provider = self.context.get_using_provider()
        else:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider:
                raise Exception(f"没有找到 ID 为 {provider_id} 的提供商")

        if not isinstance(provider, Provider):
            raise Exception(f"提供商类型错误({type(provider)})，无法获取图片描述")

        start_ts = time.perf_counter()
        logger.debug(
            "enhance-mode | image_caption start | provider=%s timeout=%.1fs",
            self._provider_label(provider),
            timeout_sec,
        )
        response = await asyncio.wait_for(
            provider.text_chat(
                prompt=prompt,
                session_id=uuid.uuid4().hex,
                image_urls=[image_url],
                persist=False,
            ),
            timeout=timeout_sec,
        )
        elapsed_ms = (time.perf_counter() - start_ts) * 1000
        logger.debug(
            "enhance-mode | image_caption done | provider=%s elapsed_ms=%.1f caption_len=%s",
            self._provider_label(provider),
            elapsed_ms,
            len(response.completion_text or ""),
        )
        return response.completion_text

    async def _need_active_reply(
        self, event: AstrMessageEvent, cfg: PluginConfig
    ) -> bool:
        if not self._allow_active_reply(event, cfg):
            return False

        ar = cfg.active_reply
        if ar.mode == "model_choice":
            return await self._need_active_reply_model_choice(event, cfg)
        sample = random.random()
        decision = sample < ar.possibility
        logger.debug(
            "enhance-mode | active_reply probability | origin=%s sample=%.4f threshold=%.4f decision=%s",
            event.unified_msg_origin,
            sample,
            ar.possibility,
            decision,
        )
        return decision

    @filter.on_llm_request()
    async def inject_role(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        cfg = self._cfg()
        if not cfg.group_features.role_display:
            return

        base_cfg = self.context.get_config(umo=event.unified_msg_origin)
        if not base_cfg.get("identifier"):
            return

        role = "admin" if event.is_admin() else "member"
        role_line = f", Role: {role}"

        for part in req.extra_user_content_parts:
            if isinstance(part, TextPart) and "<system_reminder>" in part.text:
                if "Nickname: " in part.text and role_line not in part.text:
                    nickname_idx = part.text.index("Nickname: ")
                    rest = part.text[nickname_idx:]
                    newline_idx = rest.find("\n")
                    if newline_idx != -1:
                        insert_pos = nickname_idx + newline_idx
                    else:
                        close_idx = part.text.find("</system_reminder>")
                        insert_pos = close_idx if close_idx != -1 else len(part.text)
                    part.text = (
                        part.text[:insert_pos] + role_line + part.text[insert_pos:]
                    )
                return

        reminder = f"<system_reminder>Role: {role}</system_reminder>"
        req.extra_user_content_parts.append(TextPart(text=reminder))

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL)
    @filter.event_message_type(filter.EventMessageType.ALL, priority=9999)
    async def guard_banned_user(self, event: AstrMessageEvent) -> None:
        cfg = self._cfg()
        if not cfg.group_features.ban_control_enable:
            return
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        scope_id = self._ban_scope_id(event)
        if not scope_id:
            return

        released_count = self.ban_store.cleanup_expired(scope_id=scope_id)
        if released_count > 0:
            logger.info(
                "enhance-mode | 自动解封过期用户数量 | "
                f"count={released_count} scope={scope_id}"
            )

        sender_id = str(event.get_sender_id() or "").strip()
        if not sender_id:
            return
        if (
            not cfg.group_features.ban_allow_admin
            and sender_id in self._get_admin_sid_set()
        ):
            active_ban = self.ban_store.get_active_ban(
                scope_id=scope_id, user_id=sender_id
            )
            if active_ban:
                logger.info(
                    "enhance-mode | 管理员保护生效，命中封禁但按 bypass 放行(不修改数据库) | "
                    f"user_id={sender_id} remaining={self._format_duration(active_ban.remaining_seconds)} "
                    f"scope={scope_id} origin={event.unified_msg_origin}"
                )
            return

        active_ban = self.ban_store.get_active_ban(scope_id=scope_id, user_id=sender_id)
        if not active_ban:
            return

        logger.info(
            "enhance-mode | 命中封禁名单，已拦截消息 | "
            f"user_id={sender_id} remaining={self._format_duration(active_ban.remaining_seconds)} "
            f"scope={scope_id} origin={event.unified_msg_origin}"
        )
        event.stop_event()

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL)
    async def on_group_message(self, event: AstrMessageEvent):
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        cfg = self._cfg()
        if not cfg.group_history_enabled and not cfg.active_reply_enabled:
            return

        forward_context_text = self._get_forward_context_text(event)
        event_body, _, _ = self._format_event_message_body(event)
        has_content = bool(event_body)
        if not has_content and not forward_context_text:
            return

        if self._history_recording_enabled(cfg):
            try:
                await self._record_message(event, cfg)
            except Exception as e:
                logger.error(f"enhance-mode | record message error: {e}")

        need_active = await self._need_active_reply(event, cfg)

        if need_active:
            provider = self.context.get_using_provider(event.unified_msg_origin)
            if not provider:
                logger.error("enhance-mode | 未找到任何 LLM 提供商，无法主动回复")
                return
            try:
                logger.info(
                    "enhance-mode | active_reply triggered | origin=%s mode=%s provider=%s",
                    event.unified_msg_origin,
                    cfg.active_reply.mode,
                    self._provider_label(provider),
                )
                if hasattr(event, "set_extra"):
                    event.set_extra("_enhance_active_reply_triggered", True)
                    event.set_extra("_enhance_active_reply_mode", cfg.active_reply.mode)

                active_context = await self._collect_active_reply_context(
                    event,
                    cfg,
                    backfill_target=cfg.active_reply.unified_context_messages,
                )
                active_prompt = self._build_active_reply_prompt(
                    cfg,
                    active_context,
                    cfg.active_reply.mode,
                )
                if hasattr(event, "set_extra"):
                    event.set_extra(ENHANCE_ACTIVE_REPLY_PROMPT_KEY, active_prompt)

                session_curr_cid, conv = await self._ensure_active_reply_conversation(
                    event,
                    cfg,
                )
                if not session_curr_cid or not conv:
                    return

                yield event.request_llm(
                    prompt=active_prompt,
                    session_id=event.session_id,
                    conversation=conv,
                )
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(f"enhance-mode | 主动回复失败: {e}")

    async def _record_message(self, event: AstrMessageEvent, cfg: PluginConfig) -> None:
        history_cfg = cfg.group_history

        datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        nickname = event.message_obj.sender.nickname
        msg_id = event.message_obj.message_id
        normalized_msg_id = self._normalize_message_id(msg_id)
        message_body, image_urls, image_cache_sources = self._format_event_message_body(
            event
        )

        if history_cfg.include_sender_id and history_cfg.include_role_tag:
            sender_id = event.get_sender_id()
            role_tag = "(admin)" if event.is_admin() else "(member)"
            header = f"[{nickname}/{sender_id}/{datetime_str}]{role_tag} #msg{msg_id}:"
        elif history_cfg.include_sender_id:
            sender_id = event.get_sender_id()
            header = f"[{nickname}/{sender_id}/{datetime_str}] #msg{msg_id}:"
        elif history_cfg.include_role_tag:
            role_tag = "(admin)" if event.is_admin() else "(member)"
            header = f"[{nickname}/{datetime_str}]{role_tag} #msg{msg_id}:"
        else:
            header = f"[{nickname}/{datetime_str}] #msg{msg_id}:"

        parts = [header]
        if message_body:
            parts.append(f" {message_body}")

        final_message = "".join(parts)
        logger.debug(f"enhance-mode | {event.unified_msg_origin} | {final_message}")

        self._touch_origin(event.unified_msg_origin, cfg)
        chats = self.runtime.session_chats[event.unified_msg_origin]
        chats.append(final_message)
        if len(chats) > history_cfg.max_messages:
            removed_line = chats.pop(0)
            removed_msg_id = self._extract_message_id_from_history_line(removed_line)
            if removed_msg_id:
                self.runtime.image_message_registry[event.unified_msg_origin].pop(
                    removed_msg_id, None
                )
        if normalized_msg_id and image_urls:
            self.runtime.image_message_registry[event.unified_msg_origin][
                normalized_msg_id
            ] = {
                "urls": image_urls,
                "cache_sources": image_cache_sources,
                "captions": {},
            }
            logger.debug(
                "enhance-mode | image message registered | origin=%s msg_id=%s image_count=%s deferred_caption=%s",
                event.unified_msg_origin,
                normalized_msg_id,
                len(image_urls),
                history_cfg.image_caption,
            )
        logger.debug(
            "enhance-mode | group history updated | origin=%s size=%s",
            event.unified_msg_origin,
            len(chats),
        )

    @filter.on_llm_request()
    async def inject_group_context(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        cfg = self._cfg()
        if not self._history_recording_enabled(cfg):
            return

        is_react_group = (
            cfg.group_features.react_mode_enable
            and event.get_message_type() == MessageType.GROUP_MESSAGE
        )
        if not is_react_group:
            return

        is_active_triggered = event.get_extra(
            "_enhance_active_reply_triggered", False
        )
        active_mode = event.get_extra("_enhance_active_reply_mode", "")

        if is_active_triggered:
            active_prompt = str(
                self._get_extra_value(event, ENHANCE_ACTIVE_REPLY_PROMPT_KEY, "")
                or req.prompt
                or ""
            )
            if not active_prompt:
                active_context = await self._collect_active_reply_context(
                    event,
                    cfg,
                    backfill_target=cfg.active_reply.unified_context_messages,
                )
                active_prompt = self._build_active_reply_prompt(
                    cfg,
                    active_context,
                    str(active_mode or cfg.active_reply.mode),
                )
            req.prompt = active_prompt
            req.contexts = []
            logger.info(
                "enhance-mode | active_reply prompt injected | origin=%s mode=%s prompt_len=%s",
                event.unified_msg_origin,
                active_mode or "unknown",
                len(active_prompt),
            )
            return

        current_message_text, current_message_source = (
            await self._resolve_active_current_message_text(event, cfg)
        )
        request_prompt = str(getattr(req, "prompt", "") or "").strip()
        if (
            current_message_source == "empty"
            and request_prompt
            and request_prompt != "[Empty]"
        ):
            current_message_text = request_prompt
            current_message_source = "provider_request.prompt"

        passive_context = await self._collect_active_reply_context(
            event,
            cfg,
            current_message_text=current_message_text,
            current_message_source=current_message_source,
            exclude_current=current_message_source != "provider_request.prompt",
            backfill_target=cfg.active_reply.unified_context_messages,
        )
        req.prompt = self._build_active_reply_prompt(
            cfg,
            passive_context,
            active_mode="",
        )
        req.contexts = []
        logger.info(
            "enhance-mode | passive prompt injected | origin=%s prompt_len=%s",
            event.unified_msg_origin,
            len(req.prompt or ""),
        )

    @filter.on_decorating_result()
    async def parse_tags(self, event: AstrMessageEvent) -> None:
        result = event.get_result()
        if not result or not result.chain:
            return

        cfg = self._cfg()

        # 全局拦截：模型输出 <refuse/> 时直接清空结果链，阻止后续发送到平台。
        if cfg.group_features.refuse_enable and chain_has_refuse_tag(result.chain):
            logger.info(
                "enhance-mode | 检测到 <refuse/>，已取消发送 | "
                f"origin={event.unified_msg_origin}"
            )
            if hasattr(event, "set_extra"):
                event.set_extra("_enhance_refused_reply", True)
            result.chain = []
            return

        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        transformed = transform_result_chain(
            result.chain,
            parse_mention=cfg.group_features.mention_parse,
        )
        if transformed is not None:
            result.chain = transformed

        active_reply_triggered = self._is_truthy_extra(
            self._get_extra_value(event, "_enhance_active_reply_triggered", False)
        )
        deduped = (
            dedupe_repeated_result_chain(result.chain)
            if active_reply_triggered
            else None
        )
        if deduped is not None:
            logger.info(
                "enhance-mode | active duplicated reply chain collapsed | origin=%s parts=%s",
                event.unified_msg_origin,
                len(result.chain),
            )
            result.chain = deduped

    @filter.on_llm_response()
    async def record_bot_response(
        self, event: AstrMessageEvent, resp: LLMResponse
    ) -> None:
        cfg = self._cfg()
        if not self._history_recording_enabled(cfg):
            return
        if event.unified_msg_origin not in self.runtime.session_chats:
            return
        if not resp.completion_text:
            return

        if cfg.group_features.refuse_enable and has_refuse_tag(resp.completion_text):
            logger.info(
                "enhance-mode | 检测到 <refuse/>，跳过机器人回复历史记录 | "
                f"origin={event.unified_msg_origin}"
            )
            return

        datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        text = clean_response_text_for_history(resp.completion_text)
        if not text:
            return
        final_message = f"[You/{datetime_str}]: {text}"

        logger.debug(
            f"enhance-mode | recorded AI response: "
            f"{event.unified_msg_origin} | {final_message}"
        )

        self._touch_origin(event.unified_msg_origin, cfg)
        chats = self.runtime.session_chats[event.unified_msg_origin]
        chats.append(final_message)
        if len(chats) > cfg.group_history.max_messages:
            removed_line = chats.pop(0)
            removed_msg_id = self._extract_message_id_from_history_line(removed_line)
            if removed_msg_id:
                self.runtime.image_message_registry[event.unified_msg_origin].pop(
                    removed_msg_id, None
                )
        logger.debug(
            "enhance-mode | bot response recorded | origin=%s size=%s",
            event.unified_msg_origin,
            len(chats),
        )

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent) -> None:
        clean_session = event.get_extra("_clean_ltm_session", False)
        if not clean_session:
            return
        self.runtime.cleanup_origin(event.unified_msg_origin)
        logger.info(
            "enhance-mode | runtime session cache cleaned | origin=%s",
            event.unified_msg_origin,
        )

    @llm_tool(name="grok_web_search")
    async def grok_web_search(self, event: AstrMessageEvent, query: str) -> str:
        """Search live web information with configured provider and return plain text.

        Args:
            query(string): Required. Search query text.
        """
        cfg = self._cfg()
        if not cfg.web_search.enable:
            return "Web search tool is disabled in enhance mode config."

        clean_query = str(query or "").strip()
        if not clean_query:
            return "Invalid `query`: empty."

        result = await self._run_web_search(event, clean_query, cfg)
        return self._format_web_search_tool_result(result, cfg)

    @llm_tool(name="enhance_get_ban_list_status")
    async def get_ban_list_status(
        self,
        event: AstrMessageEvent,
        user_id: str = "",
        max_results: int = 20,
    ) -> str:
        """Get ban-list status maintained by enhance mode.

        Args:
            user_id(string): Optional. Target user ID to query. Empty means list active bans.
            max_results(number): Optional. Maximum number of active bans to return when user_id is empty.
        """
        cfg = self._cfg()
        if not cfg.group_features.ban_control_enable:
            return "Ban control is disabled in enhance mode config."

        scope_id = self._ban_scope_id(event)
        if not scope_id:
            return (
                "Ban list is group-scoped. "
                "Please call this tool in current group chat context."
            )

        released_count = self.ban_store.cleanup_expired(scope_id=scope_id)
        target_user_id = str(user_id or "").strip().lstrip("@")
        if (
            target_user_id
            and not cfg.group_features.ban_allow_admin
            and target_user_id in self._get_admin_sid_set()
        ):
            return (
                f"User `{target_user_id}` is AstrBot admin and protected from ban "
                f"(scope={scope_id}, auto released this round: {released_count})."
            )

        if target_user_id:
            active_ban = self.ban_store.get_active_ban(
                scope_id=scope_id, user_id=target_user_id
            )
            if not active_ban:
                return (
                    f"User `{target_user_id}` is not banned now. "
                    f"(scope={scope_id}, auto released this round: {released_count})"
                )

            expires_at_text = datetime.datetime.fromtimestamp(
                active_ban.expires_at
            ).strftime("%Y-%m-%d %H:%M:%S")
            return (
                f"User `{target_user_id}` is currently banned.\n"
                f"- Remaining: {self._format_duration(active_ban.remaining_seconds)}\n"
                f"- Expires at: {expires_at_text}\n"
                f"- Scope: {scope_id}\n"
                f"- Auto released this round: {released_count}"
            )

        try:
            parsed_max_results = int(max_results)
        except (TypeError, ValueError):
            parsed_max_results = 20
        limit = max(1, min(parsed_max_results, 200))
        records = self.ban_store.list_active_bans(scope_id=scope_id, limit=limit)
        if not records:
            return (
                f"No active bans in scope `{scope_id}`. "
                f"(auto released this round: {released_count})"
            )

        lines = [
            f"Active bans in scope `{scope_id}` ({len(records)} shown, limit={limit}). "
            f"Auto released this round: {released_count}",
        ]
        for idx, record in enumerate(records, start=1):
            expire_text = datetime.datetime.fromtimestamp(record.expires_at).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            lines.append(
                f"{idx}. user_id={record.user_id}, remaining={self._format_duration(record.remaining_seconds)}, "
                f"expires_at={expire_text}"
            )
        return "\n".join(lines)

    @llm_tool(name="enhance_ban_user")
    async def ban_user(
        self,
        event: AstrMessageEvent,
        user_id: str,
        duration: str = "10m",
    ) -> str:
        """Temporarily ban a user in current group scope.

        Args:
            user_id(string): Required. Target user ID to ban.
            duration(string): Optional. Ban duration like 60s, 10m, 2h, 1d. Default is 10m.
        """
        cfg = self._cfg()
        if not cfg.group_features.ban_control_enable:
            return "Ban control is disabled in enhance mode config."

        scope_id = self._ban_scope_id(event)
        if not scope_id:
            return (
                "Ban action is group-scoped. "
                "Please call this tool in current group chat context."
            )

        target_user_id = str(user_id or "").strip().lstrip("@")
        if not target_user_id:
            return "Invalid `user_id`: empty."

        duration_seconds = parse_duration_seconds(duration)
        if duration_seconds is None:
            return "Invalid `duration`. Use formats like `60s`, `10m`, `2h`, `1d`."

        if (
            not cfg.group_features.ban_allow_admin
            and target_user_id in self._get_admin_sid_set()
        ):
            return (
                f"User `{target_user_id}` is AstrBot admin and protected from ban "
                f"(scope={scope_id})."
            )

        final_duration = max(
            1,
            min(duration_seconds, cfg.group_features.ban_max_duration_sec),
        )
        expires_at = self.ban_store.ban_user(
            scope_id=scope_id,
            user_id=target_user_id,
            duration_seconds=final_duration,
            source_origin=event.unified_msg_origin,
        )
        expire_time = datetime.datetime.fromtimestamp(expires_at).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        logger.info(
            "enhance-mode | ban applied | scope=%s user_id=%s duration=%s origin=%s",
            scope_id,
            target_user_id,
            self._format_duration(final_duration),
            event.unified_msg_origin,
        )
        return (
            f"Banned user `{target_user_id}` in scope `{scope_id}`.\n"
            f"- Duration: {self._format_duration(final_duration)}\n"
            f"- Expires at: {expire_time}"
        )

    @llm_tool(name="enhance_unban_user")
    async def unban_user(self, event: AstrMessageEvent, user_id: str) -> str:
        """Unban a user in current group scope.

        Args:
            user_id(string): Required. Target user ID to unban.
        """
        cfg = self._cfg()
        if not cfg.group_features.ban_control_enable:
            return "Ban control is disabled in enhance mode config."

        scope_id = self._ban_scope_id(event)
        if not scope_id:
            return (
                "Unban action is group-scoped. "
                "Please call this tool in current group chat context."
            )

        target_user_id = str(user_id or "").strip().lstrip("@")
        if not target_user_id:
            return "Invalid `user_id`: empty."

        removed = self.ban_store.unban_user(scope_id=scope_id, user_id=target_user_id)
        if not removed:
            return f"User `{target_user_id}` is not banned in scope `{scope_id}`."
        logger.info(
            "enhance-mode | unban applied | scope=%s user_id=%s origin=%s",
            scope_id,
            target_user_id,
            event.unified_msg_origin,
        )
        return f"Unbanned user `{target_user_id}` in scope `{scope_id}`."

    @staticmethod
    def _make_text_tool_result(text: str) -> mcp_types.CallToolResult:
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=str(text or ""))]
        )

    @llm_tool(name="enhance_use_image")
    async def use_image(
        self,
        event: AstrMessageEvent,
        message_id: str,
        image_index: int = 1,
        attach_to_model: bool = True,
        write_to_history: bool = True,
        prompt: str = "",
    ) -> AsyncGenerator[mcp_types.CallToolResult, None]:
        """Use one image from runtime history for multimodal input and/or history backfill.

        Args:
            message_id(string): Required. Message ID in history header, for example `123456`.
            image_index(number): Optional. One-based index of image in that message. Default is 1.
            attach_to_model(bool): Optional. Attach image bytes to this run context. Default is true.
            write_to_history(bool): Optional. Replace `[Image]` with description in history. Default is true.
            prompt(string): Optional. Override image caption prompt for this call.
        """
        cfg = self._cfg()
        if not cfg.group_history_enabled:
            yield self._make_text_tool_result(
                "Group history enhancement is disabled in enhance mode config."
            )
            return

        normalized_message_id = self._normalize_message_id(message_id)
        if not normalized_message_id:
            yield self._make_text_tool_result("Invalid `message_id`: empty.")
            return

        try:
            index_number = int(image_index)
        except (TypeError, ValueError):
            yield self._make_text_tool_result(
                "Invalid `image_index`: must be a positive integer."
            )
            return
        if index_number <= 0:
            yield self._make_text_tool_result("Invalid `image_index`: must be >= 1.")
            return
        image_idx = index_number - 1

        attach_requested = bool(attach_to_model)
        history_requested = bool(write_to_history)
        if not attach_requested and not history_requested:
            yield self._make_text_tool_result(
                "Invalid mode: `attach_to_model` and `write_to_history` cannot both be false."
            )
            return

        origin = event.unified_msg_origin
        message_registry = self.runtime.image_message_registry.get(origin, {})
        message_entry = message_registry.get(normalized_message_id)
        if not isinstance(message_entry, dict):
            yield self._make_text_tool_result(
                f"Image message `{normalized_message_id}` not found in current runtime history. "
                "Try a newer message ID from current chat context."
            )
            return

        urls_raw = message_entry.get("urls")
        if not isinstance(urls_raw, list) or not urls_raw:
            yield self._make_text_tool_result(
                f"No image records found for message `{normalized_message_id}`."
            )
            return
        if image_idx >= len(urls_raw):
            yield self._make_text_tool_result(
                f"`image_index` out of range. message `{normalized_message_id}` has "
                f"{len(urls_raw)} image(s)."
            )
            return

        image_url = str(urls_raw[image_idx] or "").strip()
        if not image_url:
            yield self._make_text_tool_result(
                f"Image URL is unavailable for message `{normalized_message_id}` "
                f"at index {index_number}."
            )
            return
        cache_sources_raw = message_entry.get("cache_sources")
        cache_source = (
            str(cache_sources_raw[image_idx] or "").strip()
            if isinstance(cache_sources_raw, list) and image_idx < len(cache_sources_raw)
            else image_url
        )

        captions_raw = message_entry.get("captions")
        if isinstance(captions_raw, dict):
            captions_map = captions_raw
        else:
            captions_map = {}
            message_entry["captions"] = captions_map

        caption = ""
        caption_cached = False
        cached_caption = captions_map.get(image_idx)
        if isinstance(cached_caption, str) and cached_caption.strip():
            caption = cached_caption.strip()
            caption_cached = True
            await self._write_shared_image_caption_cache(
                self._image_caption_sources(image_url, cache_source),
                caption,
            )
        elif history_requested:
            try:
                if not cfg.group_history.image_caption:
                    yield self._make_text_tool_result(
                        "Image caption is disabled in enhance mode config."
                    )
                    return
                caption = await self._caption_image_with_cache(
                    event,
                    cfg,
                    image_url=image_url,
                    cache_source=cache_source,
                    prompt=prompt,
                )
                caption = str(caption or "").strip()
                if caption:
                    captions_map[image_idx] = caption
            except Exception as e:
                logger.exception(
                    "enhance-mode | use_image caption failed | origin=%s msg_id=%s image_index=%s error=%s",
                    origin,
                    normalized_message_id,
                    index_number,
                    e,
                )
                yield self._make_text_tool_result(
                    f"Failed to get image description: {e}"
                )
                return

        attach_success = False
        attach_error = ""
        if attach_requested:
            resolved_paths_raw = message_entry.get("resolved_paths")
            if not isinstance(resolved_paths_raw, list) or len(
                resolved_paths_raw
            ) != len(urls_raw):
                resolved_paths_raw = [""] * len(urls_raw)
                message_entry["resolved_paths"] = resolved_paths_raw

            selected_ref = str(
                resolved_paths_raw[image_idx] or urls_raw[image_idx] or ""
            ).strip()
            if not selected_ref:
                attach_error = (
                    f"Image reference is unavailable for message `{normalized_message_id}` "
                    f"at index {index_number}."
                )
            else:
                try:
                    local_path = await self._resolve_image_ref_to_local_path(
                        selected_ref
                    )
                    if not local_path:
                        attach_error = (
                            f"Failed to resolve image path for message `{normalized_message_id}` "
                            f"index {index_number}."
                        )
                    else:
                        resolved_paths_raw[image_idx] = local_path
                        image_b64, mime_type = self._encode_image_file(local_path)
                        yield mcp_types.CallToolResult(
                            content=[
                                mcp_types.ImageContent(
                                    type="image",
                                    data=image_b64,
                                    mimeType=mime_type,
                                )
                            ]
                        )
                        attach_success = True
                        logger.info(
                            "enhance-mode | use_image attach success | origin=%s msg_id=%s image_index=%s mime=%s",
                            origin,
                            normalized_message_id,
                            index_number,
                            mime_type,
                        )
                except Exception as e:
                    attach_error = str(e)
                    logger.exception(
                        "enhance-mode | use_image attach failed | origin=%s msg_id=%s image_index=%s error=%s",
                        origin,
                        normalized_message_id,
                        index_number,
                        e,
                    )

        history_success = False
        history_error = ""
        if history_requested:
            if not caption:
                history_error = "Image description is empty."
            else:
                history_success = self._apply_image_caption_to_history(
                    origin=origin,
                    message_id=normalized_message_id,
                    image_index=image_idx,
                    caption=caption,
                )
                if history_success:
                    logger.info(
                        "enhance-mode | use_image history write success | origin=%s msg_id=%s image_index=%s",
                        origin,
                        normalized_message_id,
                        index_number,
                    )
                else:
                    history_error = (
                        "Failed to apply image description back to runtime history."
                    )

        if attach_requested and history_requested:
            success = attach_success and history_success
        elif attach_requested:
            success = attach_success
        else:
            success = history_success

        payload: dict[str, object] = {
            "status": "ok" if success else "failed",
            "success": success,
            "origin": origin,
            "message_id": normalized_message_id,
            "image_index": index_number,
            "attach_requested": attach_requested,
            "write_to_history_requested": history_requested,
            "attach_success": attach_success if attach_requested else None,
            "write_to_history_success": history_success if history_requested else None,
            "description_cached": caption_cached,
        }
        if attach_requested and not history_requested:
            payload["description"] = caption
        if attach_error:
            payload["attach_error"] = attach_error
        if history_error:
            payload["write_to_history_error"] = history_error

        logger.info(
            "enhance-mode | use_image done | origin=%s msg_id=%s image_index=%s attach=%s/%s write=%s/%s success=%s",
            origin,
            normalized_message_id,
            index_number,
            attach_success,
            attach_requested,
            history_success,
            history_requested,
            success,
        )
        yield self._make_text_tool_result(json.dumps(payload, ensure_ascii=False))

    @llm_tool(name="enhance_memory_rag_write")
    async def memory_rag_write(
        self,
        event: AstrMessageEvent,
        content: str,
        related_role_ids: str,
        memory_time: str = "",
        group_scope: str = "",
        group_id: str = "",
        platform_id: str = "",
        extra_metadata_json: str = "{}",
    ) -> str:
        """Write a memory record into enhance-mode memory RAG store.

        Args:
            content(string): Required. Memory content text.
            related_role_ids(string): Required. Role IDs in JSON array or comma-separated string.
            memory_time(string): Optional. Memory time as unix timestamp or ISO datetime.
            group_scope(string): Optional. Full group scope, for example `qq:123456`.
            group_id(string): Optional. Group ID, combined with platform_id when group_scope is empty.
            platform_id(string): Optional. Platform ID used with group_id.
            extra_metadata_json(string): Optional. JSON object string for extra metadata.
        """
        ready, reason = self._check_memory_rag_ready()
        if not ready:
            return reason
        if self.memory_rag_store is None:
            return "Memory RAG store is not initialized."

        clean_content = str(content or "").strip()
        if not clean_content:
            return "Invalid `content`: empty."

        role_ids = self._parse_role_ids(related_role_ids)
        if not role_ids:
            return "Invalid `related_role_ids`: at least one role ID is required."

        ts = self._parse_optional_timestamp(memory_time)
        if memory_time and ts is None:
            return (
                "Invalid `memory_time`. Use unix timestamp or ISO datetime "
                "(for example `1735689600` or `2026-01-01 12:00:00`)."
            )
        memory_ts = ts if ts is not None else time.time()

        final_scope, final_group_id, final_platform_id = self._resolve_memory_scope(
            event, group_scope, group_id, platform_id
        )

        cfg = self._cfg()
        embedding_provider = self._resolve_embedding_provider(cfg)
        if not embedding_provider:
            return (
                "No embedding provider is available. "
                "Please configure one in AstrBot provider settings."
            )

        start_ts = time.perf_counter()
        logger.info(
            "enhance-mode | memory_rag_write start | origin=%s roles=%s content_len=%s scope=%s",
            event.unified_msg_origin,
            len(role_ids),
            len(clean_content),
            final_scope,
        )
        try:
            embedding = await embedding_provider.get_embedding(clean_content)
        except Exception as e:
            logger.exception("enhance-mode | memory_rag_write embedding failed: %s", e)
            return f"Failed to generate embedding: {e}"

        metadata = self._parse_extra_metadata(extra_metadata_json)
        try:
            memory_id = self.memory_rag_store.add_memory(
                content=clean_content,
                embedding=embedding,
                role_ids=role_ids,
                memory_time=memory_ts,
                group_scope=final_scope,
                group_id=final_group_id,
                platform_id=final_platform_id,
                extra_metadata=metadata,
            )
        except Exception as e:
            logger.exception("enhance-mode | memory_rag_write store failed: %s", e)
            return f"Failed to write memory: {e}"

        elapsed_ms = (time.perf_counter() - start_ts) * 1000
        logger.info(
            "enhance-mode | memory_rag_write done | memory_id=%s provider=%s embedding_dim=%s elapsed_ms=%.1f",
            memory_id,
            self._provider_label(embedding_provider),
            len(embedding),
            elapsed_ms,
        )

        return json.dumps(
            {
                "status": "ok",
                "memory_id": memory_id,
                "memory_time": memory_ts,
                "memory_time_iso": self._format_timestamp_iso(memory_ts),
                "group_scope": final_scope,
                "group_id": final_group_id,
                "platform_id": final_platform_id,
                "related_role_ids": role_ids,
            },
            ensure_ascii=False,
        )

    @llm_tool(name="enhance_memory_rag_read")
    async def memory_rag_read(
        self,
        event: AstrMessageEvent,
        query: str = "",
        related_role_ids: str = "",
        role_match_mode: str = "any",
        start_time: str = "",
        end_time: str = "",
        group_scope: str = "",
        group_id: str = "",
        platform_id: str = "",
        sort_by: str = "relevance",
        sort_order: str = "desc",
        max_results: int = 10,
        embedding_recall_k: int = 0,
        ignore_group_id: bool = False,
    ) -> str:
        """Read memory records from enhance-mode memory RAG store.

        Args:
            query(string): Optional. Query text for embedding retrieval.
            related_role_ids(string): Optional. Role IDs in JSON array or comma-separated string.
            role_match_mode(string): Optional. `any` or `all`.
            start_time(string): Optional. Start time as unix timestamp or ISO datetime.
            end_time(string): Optional. End time as unix timestamp or ISO datetime.
            group_scope(string): Optional. Full group scope, for example `qq:123456`.
            group_id(string): Optional. Group ID, combined with platform_id when group_scope is empty.
            platform_id(string): Optional. Platform ID used with group_id.
            sort_by(string): Optional. `relevance` or `time`.
            sort_order(string): Optional. `desc` or `asc`.
            max_results(number): Optional. Number of results to return.
            embedding_recall_k(number): Optional. Top-K cutoff for embedding recall before final sorting.
            ignore_group_id(boolean): Optional. When true, do not auto-apply current group filter, allowing cross-group read.
        """
        ready, reason = self._check_memory_rag_ready()
        if not ready:
            return reason
        if self.memory_rag_store is None:
            return "Memory RAG store is not initialized."

        start_time_ts = self._parse_optional_timestamp(start_time)
        end_time_ts = self._parse_optional_timestamp(end_time)
        if start_time and start_time_ts is None:
            return (
                "Invalid `start_time`. Use unix timestamp or ISO datetime "
                "(for example `1735689600` or `2026-01-01 12:00:00`)."
            )
        if end_time and end_time_ts is None:
            return (
                "Invalid `end_time`. Use unix timestamp or ISO datetime "
                "(for example `1735689600` or `2026-01-01 12:00:00`)."
            )
        if (
            start_time_ts is not None
            and end_time_ts is not None
            and start_time_ts > end_time_ts
        ):
            return "Invalid time range: `start_time` cannot be greater than `end_time`."

        normalized_ignore_group = (
            ignore_group_id
            if isinstance(ignore_group_id, bool)
            else str(ignore_group_id or "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if normalized_ignore_group:
            final_scope = str(group_scope or "").strip()
            final_group_id = str(group_id or "").strip()
            final_platform_id = str(platform_id or "").strip()
            if not final_scope and final_group_id:
                final_scope = (
                    f"{final_platform_id}:{final_group_id}"
                    if final_platform_id
                    else final_group_id
                )
        else:
            final_scope, final_group_id, final_platform_id = self._resolve_memory_scope(
                event, group_scope, group_id, platform_id
            )
        normalized_roles = self._parse_role_ids(related_role_ids)
        normalized_mode = (
            "all" if str(role_match_mode or "").strip().lower() == "all" else "any"
        )
        normalized_sort_by = self._normalize_sort_by(sort_by)
        normalized_sort_order = self._normalize_sort_order(sort_order)

        cfg = self._cfg()
        max_allowed = max(1, cfg.memory_rag.max_return_results)
        try:
            requested_max = int(max_results)
        except (TypeError, ValueError):
            requested_max = 10
        if requested_max <= 0:
            final_max_results = max_allowed
        else:
            final_max_results = min(requested_max, max_allowed)

        try:
            parsed_recall_k = int(embedding_recall_k)
        except (TypeError, ValueError):
            parsed_recall_k = 0
        effective_recall_k = (
            parsed_recall_k
            if parsed_recall_k > 0
            else int(cfg.memory_rag.default_recall_k)
        )

        query_embedding: list[float] | None = None
        clean_query = str(query or "").strip()
        read_start_ts = time.perf_counter()
        logger.info(
            "enhance-mode | memory_rag_read start | origin=%s query_len=%s role_count=%s scope=%s max_results=%s sort=%s/%s",
            event.unified_msg_origin,
            len(clean_query),
            len(normalized_roles),
            final_scope,
            final_max_results,
            normalized_sort_by,
            normalized_sort_order,
        )
        if clean_query:
            embedding_provider = self._resolve_embedding_provider(cfg)
            if not embedding_provider:
                return (
                    "No embedding provider is available. "
                    "Please configure one in AstrBot provider settings."
                )
            try:
                query_embedding = await embedding_provider.get_embedding(clean_query)
            except Exception as e:
                logger.exception(
                    "enhance-mode | memory_rag_read embedding failed: %s", e
                )
                return f"Failed to generate query embedding: {e}"

        try:
            records = self.memory_rag_store.search_memories(
                query_embedding=query_embedding,
                embedding_recall_k=effective_recall_k,
                role_ids=normalized_roles,
                role_match_mode=normalized_mode,
                group_scope=final_scope,
                group_id=final_group_id,
                platform_id=final_platform_id,
                start_time=start_time_ts,
                end_time=end_time_ts,
                sort_by=normalized_sort_by,
                sort_order=normalized_sort_order,
                max_results=final_max_results,
            )
        except Exception as e:
            logger.exception("enhance-mode | memory_rag_read search failed: %s", e)
            return f"Failed to read memories: {e}"

        elapsed_ms = (time.perf_counter() - read_start_ts) * 1000
        logger.info(
            "enhance-mode | memory_rag_read done | origin=%s result_count=%s elapsed_ms=%.1f",
            event.unified_msg_origin,
            len(records),
            elapsed_ms,
        )

        return json.dumps(
            {
                "status": "ok",
                "count": len(records),
                "query": clean_query,
                "filters": {
                    "related_role_ids": normalized_roles,
                    "role_match_mode": normalized_mode,
                    "start_time": start_time_ts,
                    "end_time": end_time_ts,
                    "ignore_group_id": normalized_ignore_group,
                    "group_scope": final_scope,
                    "group_id": final_group_id,
                    "platform_id": final_platform_id,
                },
                "sort": {"by": normalized_sort_by, "order": normalized_sort_order},
                "max_results": final_max_results,
                "embedding_recall_k": effective_recall_k if clean_query else None,
                "records": records,
            },
            ensure_ascii=False,
        )

    @filter.command_group("enhance")
    def enhance(self) -> None:
        pass

    @permission_type(PermissionType.ADMIN)
    @enhance.command("rag-webui")
    async def rag_webui(
        self, event: AstrMessageEvent
    ) -> AsyncGenerator[MessageEventResult, None]:
        cfg = self._cfg()
        webui_cfg = cfg.memory_rag_webui
        logger.info(
            "enhance-mode | rag-webui command received | origin=%s webui_enable=%s",
            event.unified_msg_origin,
            webui_cfg.enable,
        )
        if not webui_cfg.enable:
            yield event.plain_result(
                "Memory RAG WebUI is disabled.\n"
                "Please enable `memory_rag_webui.enable` in enhance plugin config."
            )
            return

        if self.rag_webui_server is None:
            await self._start_memory_rag_webui()

        if self.rag_webui_server is None:
            yield event.plain_result(
                "Memory RAG WebUI failed to start.\n"
                "Please check AstrBot logs for error details."
            )
            return

        message = (
            "Enhance Memory RAG WebUI\n\n"
            f"URL: {self._memory_rag_webui_url(cfg)}\n"
            "Functions: login, filters, pagination, detail view, delete memory.\n"
        )
        if self.rag_webui_server.password_generated:
            message += (
                "\nPassword is auto generated for this run. "
                "Please check AstrBot logs for the generated password."
            )
        yield event.plain_result(message)

    async def terminate(self) -> None:
        logger.info("enhance-mode | plugin terminating")
        await self._stop_memory_rag_webui()
        logger.info("enhance-mode | plugin terminated")
