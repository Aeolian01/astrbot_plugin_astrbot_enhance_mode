from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ImageMessageRegistryStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, dict[str, dict[str, object]]]:
        try:
            if not self.path.exists():
                return {}
            with self.path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return {}
        return self._normalize_registry(raw)

    def save(
        self,
        registry: dict[str, dict[str, dict[str, object]]],
        *,
        max_messages_per_origin: int,
        max_origins: int,
    ) -> None:
        data = self._normalize_registry(registry)
        data = self._pruned(
            data,
            max_messages_per_origin=max_messages_per_origin,
            max_origins=max_origins,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(self.path)

    @classmethod
    def _normalize_registry(
        cls, registry: Any
    ) -> dict[str, dict[str, dict[str, object]]]:
        if not isinstance(registry, dict):
            return {}

        normalized: dict[str, dict[str, dict[str, object]]] = {}
        for origin_raw, messages_raw in registry.items():
            origin = "" if origin_raw is None else str(origin_raw).strip()
            if not origin or not isinstance(messages_raw, dict):
                continue

            messages: dict[str, dict[str, object]] = {}
            for message_id_raw, entry_raw in messages_raw.items():
                message_id = (
                    "" if message_id_raw is None else str(message_id_raw).strip()
                )
                entry = cls._normalize_entry(entry_raw)
                if message_id and entry is not None:
                    messages[message_id] = entry

            if messages:
                normalized[origin] = messages
        return normalized

    @staticmethod
    def _normalize_entry(entry: Any) -> dict[str, object] | None:
        if not isinstance(entry, dict):
            return None

        urls_raw = entry.get("urls")
        if not isinstance(urls_raw, list):
            return None
        urls = [str(url or "").strip() for url in urls_raw]
        if not any(urls):
            return None

        cache_sources_raw = entry.get("cache_sources")
        if isinstance(cache_sources_raw, list):
            cache_sources = [str(source or "").strip() for source in cache_sources_raw]
        else:
            cache_sources = list(urls)

        captions: dict[str, str] = {}
        captions_raw = entry.get("captions")
        if isinstance(captions_raw, dict):
            for index_raw, caption_raw in captions_raw.items():
                index = "" if index_raw is None else str(index_raw).strip()
                caption = str(caption_raw or "").strip()
                if index and caption:
                    captions[index] = caption

        try:
            updated_at = float(entry.get("updated_at") or 0)
        except (TypeError, ValueError):
            updated_at = 0
        if updated_at <= 0:
            updated_at = time.time()

        return {
            "urls": urls,
            "cache_sources": cache_sources,
            "captions": captions,
            "updated_at": updated_at,
        }

    @staticmethod
    def _entry_updated_at(entry: dict[str, object]) -> float:
        try:
            return float(entry.get("updated_at") or 0)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _pruned(
        cls,
        registry: dict[str, dict[str, dict[str, object]]],
        *,
        max_messages_per_origin: int,
        max_origins: int,
    ) -> dict[str, dict[str, dict[str, object]]]:
        max_messages_per_origin = max(1, int(max_messages_per_origin))
        max_origins = max(1, int(max_origins))

        pruned: dict[str, dict[str, dict[str, object]]] = {}
        for origin, messages in registry.items():
            sorted_messages = sorted(
                messages.items(),
                key=lambda item: cls._entry_updated_at(item[1]),
                reverse=True,
            )
            kept_messages = dict(sorted_messages[:max_messages_per_origin])
            if kept_messages:
                pruned[origin] = kept_messages

        sorted_origins = sorted(
            pruned.items(),
            key=lambda item: max(
                (cls._entry_updated_at(entry) for entry in item[1].values()),
                default=0,
            ),
            reverse=True,
        )
        return dict(sorted_origins[:max_origins])
