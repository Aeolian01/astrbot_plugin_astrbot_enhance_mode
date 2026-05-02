from __future__ import annotations

import json

from astrbot_plugin_astrbot_enhance_mode.image_registry_store import (
    ImageMessageRegistryStore,
)


def test_image_registry_store_save_and_load_roundtrip(tmp_path) -> None:
    store = ImageMessageRegistryStore(tmp_path / "image_message_registry.json")

    store.save(
        {
            "origin-1": {
                "123": {
                    "urls": ["https://example.com/image.png"],
                    "cache_sources": ["fileid:abc"],
                    "captions": {0: "一张测试图片"},
                    "updated_at": 1000,
                }
            }
        },
        max_messages_per_origin=10,
        max_origins=10,
    )

    assert store.load() == {
        "origin-1": {
            "123": {
                "urls": ["https://example.com/image.png"],
                "cache_sources": ["fileid:abc"],
                "captions": {"0": "一张测试图片"},
                "updated_at": 1000.0,
            }
        }
    }


def test_image_registry_store_bad_json_returns_empty(tmp_path) -> None:
    path = tmp_path / "image_message_registry.json"
    path.write_text("{bad json", encoding="utf-8")

    assert ImageMessageRegistryStore(path).load() == {}


def test_image_registry_store_prunes_by_updated_at(tmp_path) -> None:
    store = ImageMessageRegistryStore(tmp_path / "image_message_registry.json")

    store.save(
        {
            "old-origin": {
                "1": {"urls": ["u1"], "captions": {}, "updated_at": 1},
            },
            "new-origin": {
                "1": {"urls": ["u2"], "captions": {}, "updated_at": 2},
                "2": {"urls": ["u3"], "captions": {}, "updated_at": 3},
            },
        },
        max_messages_per_origin=1,
        max_origins=1,
    )

    data = json.loads((tmp_path / "image_message_registry.json").read_text())
    assert list(data.keys()) == ["new-origin"]
    assert list(data["new-origin"].keys()) == ["2"]
