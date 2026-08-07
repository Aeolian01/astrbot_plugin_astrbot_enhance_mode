from __future__ import annotations

import json
import math
from pathlib import Path

from astrbot_plugin_astrbot_enhance_mode.plugin_config import (
    DEFAULT_MODEL_CHOICE_PROMPT,
    DEFAULT_VIDEO_CAPTION_PROMPT,
    parse_plugin_config,
)


def test_parse_plugin_config_defaults() -> None:
    cfg = parse_plugin_config(None)
    assert cfg.group_history.enable is False
    assert cfg.active_reply.enable is False
    assert cfg.active_reply.mode == "probability"
    assert cfg.active_reply.model_choice_provider_id == ""
    assert cfg.active_reply.unified_context_messages == 20
    assert math.isclose(cfg.active_reply.possibility, 0.1)
    assert cfg.group_history_enabled is False
    assert cfg.active_reply_enabled is False
    assert cfg.web_search.request_mode == "auto"
    assert cfg.web_search.base_url_override == ""
    assert cfg.web_search.proxy_url == ""
    assert cfg.chat_history_tool.enable is False
    assert cfg.chat_history_tool.allowed_group_ids == []
    assert cfg.chat_history_tool.default_limit == 8
    assert cfg.chat_history_tool.max_limit == 20
    assert cfg.chat_history_tool.max_pages_per_turn == 2
    assert cfg.chat_history_tool.max_messages_per_turn == 24
    assert cfg.chat_history_tool.max_output_chars == 8192
    assert math.isclose(cfg.chat_history_tool.cursor_ttl_sec, 60.0)
    assert math.isclose(cfg.chat_history_tool.api_timeout_sec, 3.0)
    assert cfg.group_history.video_caption_prompt == DEFAULT_VIDEO_CAPTION_PROMPT
    assert cfg.group_history.video_caption_provider_ids == []
    assert math.isclose(cfg.global_settings.timeouts.video_caption_sec, 120.0)
    assert cfg.active_reply.pipeline_v2_enable is False
    assert math.isclose(cfg.active_reply.reply_max_age_sec, 15.0)
    assert cfg.active_reply.cancel_on_newer_message is True
    assert cfg.active_reply.preserved_context_rounds == 2
    assert cfg.active_reply.enhancement_prompt_max_chars == 3200
    assert cfg.active_reply.output_gate_enable is True
    assert cfg.active_reply.active_reply_max_chars == 50
    assert cfg.active_reply.unsolicited_image_reply is False
    assert math.isclose(cfg.active_reply.cooldown_sec, 300.0)
    assert cfg.active_reply.bot_density_window == 20
    assert cfg.active_reply.bot_density_max == 3


def test_probability_is_clamped_and_nan_falls_back() -> None:
    cfg_high = parse_plugin_config(
        {
            "group_features": {"react_mode_enable": True},
            "active_reply": {"enable": True, "possibility": 9},
        }
    )
    assert math.isclose(cfg_high.active_reply.possibility, 1.0)
    assert cfg_high.active_reply_enabled is True

    cfg_low = parse_plugin_config({"active_reply": {"possibility": -0.5}})
    assert math.isclose(cfg_low.active_reply.possibility, 0.0)

    cfg_nan = parse_plugin_config({"active_reply": {"possibility": "nan"}})
    assert math.isclose(cfg_nan.active_reply.possibility, 0.1)


def test_active_reply_mode_and_limits_are_normalized() -> None:
    cfg = parse_plugin_config(
        {
            "active_reply": {
                "mode": "something_else",
                "unified_context_messages": -10,
                "model_stack_size": 0,
                "model_history_messages": -99,
                "model_choice_provider_id": "  provider-1  ",
                "whitelist": "a,b, c",
            },
            "global_settings": {
                "lru_cache": {"max_origins": 0},
                "timeouts": {
                    "image_caption_sec": -1,
                    "video_caption_sec": -1,
                    "model_choice_sec": "0",
                },
            },
        }
    )

    assert cfg.active_reply.mode == "probability"
    assert cfg.active_reply.unified_context_messages == 0
    assert cfg.active_reply.model_stack_size == 1
    assert cfg.active_reply.model_history_messages == 0
    assert cfg.active_reply.model_choice_provider_id == "provider-1"
    assert cfg.active_reply.whitelist == ["a", "b", "c"]
    assert cfg.global_settings.lru_cache.max_origins == 1
    assert math.isclose(cfg.global_settings.timeouts.image_caption_sec, 45.0)
    assert math.isclose(cfg.global_settings.timeouts.video_caption_sec, 120.0)
    assert math.isclose(cfg.global_settings.timeouts.model_choice_sec, 45.0)


def test_single_pass_mode_preserves_configurable_message_count() -> None:
    cfg = parse_plugin_config(
        {
            "active_reply": {
                "mode": "single_pass",
                "model_stack_size": 5,
            }
        }
    )

    assert cfg.active_reply.mode == "single_pass"
    assert cfg.active_reply.model_stack_size == 5


def test_pipeline_v2_limits_are_normalized() -> None:
    cfg = parse_plugin_config(
        {
            "active_reply": {
                "pipeline_v2_enable": "true",
                "reply_max_age_sec": -1,
                "cancel_on_newer_message": "false",
                "preserved_context_rounds": 99,
                "enhancement_prompt_max_chars": 20,
                "output_gate_enable": "true",
                "active_reply_max_chars": 0,
                "block_process_leak": "false",
                "require_single_sendable_chain": "false",
                "unsolicited_image_reply": "true",
                "cooldown_sec": -20,
                "bot_density_window": 0,
                "bot_density_max": -1,
            }
        }
    )

    assert cfg.active_reply.pipeline_v2_enable is True
    assert math.isclose(cfg.active_reply.reply_max_age_sec, 15.0)
    assert cfg.active_reply.cancel_on_newer_message is False
    assert cfg.active_reply.preserved_context_rounds == 10
    assert cfg.active_reply.enhancement_prompt_max_chars == 1000
    assert cfg.active_reply.active_reply_max_chars == 1
    assert cfg.active_reply.block_process_leak is False
    assert cfg.active_reply.require_single_sendable_chain is False
    assert cfg.active_reply.unsolicited_image_reply is True
    assert cfg.active_reply.cooldown_sec == 0
    assert cfg.active_reply.bot_density_window == 1
    assert cfg.active_reply.bot_density_max == 0


def test_video_caption_provider_defaults_to_image_provider() -> None:
    cfg = parse_plugin_config(
        {
            "group_history_enhancement": {
                "image_caption_provider_id": "gemini-image",
                "video_caption_prompt": "describe video",
            }
        }
    )

    assert cfg.group_history.video_caption_provider_id == "gemini-image"
    assert cfg.group_history.video_caption_prompt == "describe video"


def test_video_caption_provider_ids_are_normalized() -> None:
    cfg = parse_plugin_config(
        {
            "group_history_enhancement": {
                "image_caption_provider_id": "gemini-image",
                "video_caption_provider_id": "legacy",
                "video_caption_provider_ids": ["p1", "", "p2", "p1"],
            }
        }
    )

    assert cfg.group_history.video_caption_provider_id == "legacy"
    assert cfg.group_history.video_caption_provider_ids == ["p1", "p2"]


def test_web_search_request_mode_and_base_override_are_normalized() -> None:
    cfg = parse_plugin_config(
        {
            "web_search": {
                "request_mode": "GEMINI",
                "base_url_override": "  https://example.com/custom/v1  ",
                "proxy_url": "  http://sing-box:7890  ",
            }
        }
    )
    assert cfg.web_search.request_mode == "gemini"
    assert cfg.web_search.base_url_override == "https://example.com/custom/v1"
    assert cfg.web_search.proxy_url == "http://sing-box:7890"

    cfg_responses = parse_plugin_config({"web_search": {"request_mode": "RESPONSES"}})
    assert cfg_responses.web_search.request_mode == "responses"

    cfg_invalid = parse_plugin_config({"web_search": {"request_mode": "unknown"}})
    assert cfg_invalid.web_search.request_mode == "auto"


def test_chat_history_tool_config_is_bounded_and_whitelisted() -> None:
    cfg = parse_plugin_config(
        {
            "chat_history_tool": {
                "enable": "true",
                "allowed_group_ids": "829586749, 123,829586749",
                "default_limit": 999,
                "max_limit": 999,
                "max_pages_per_turn": 99,
                "max_messages_per_turn": 999,
                "max_output_chars": 10,
                "cursor_ttl_sec": -1,
                "api_timeout_sec": 0,
            }
        }
    )

    assert cfg.chat_history_tool.enable is True
    assert cfg.chat_history_tool.allowed_group_ids == [
        "829586749",
        "123",
        "829586749",
    ]
    assert cfg.chat_history_tool.default_limit == 50
    assert cfg.chat_history_tool.max_limit == 50
    assert cfg.chat_history_tool.max_pages_per_turn == 4
    assert cfg.chat_history_tool.max_messages_per_turn == 100
    assert cfg.chat_history_tool.max_output_chars == 1024
    assert math.isclose(cfg.chat_history_tool.cursor_ttl_sec, 60.0)
    assert math.isclose(cfg.chat_history_tool.api_timeout_sec, 3.0)


def test_schema_model_choice_prompt_matches_default() -> None:
    schema_path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert (
        schema["active_reply"]["items"]["model_choice_prompt"]["default"]
        == DEFAULT_MODEL_CHOICE_PROMPT
    )
    assert schema["chat_history_tool"]["items"]["enable"]["default"] is False


def test_deployed_model_choice_prompt_has_only_three_narrow_reply_categories() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "astrbot_plugin_astrbot_enhance_mode_config.json"
    )
    raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
    prompt = raw["active_reply"]["model_choice_prompt"]

    assert "1. 直接召唤或明确求助" in prompt
    assert "2. 能提供独特且具体的价值" in prompt
    assert "3. 能自然、低风险地加入" in prompt
    assert "除以上三类外，一律输出 SKIP" in prompt
    assert "只能输出 REPLY 或 SKIP" in prompt
    for forbidden in ("{persona_mask}", "政治", "时政", "性别", "女权", "男拳"):
        assert forbidden not in prompt


def test_deployed_active_reply_uses_configurable_single_pass_stack() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "astrbot_plugin_astrbot_enhance_mode_config.json"
    )
    raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
    active_reply = raw["active_reply"]

    assert active_reply["mode"] == "single_pass"
    assert active_reply["model_stack_size"] == 2
    assert active_reply["model_choice_provider_id"] == ""
    assert raw["group_features"]["refuse_enable"] is True


def test_deployed_chat_history_tool_is_scoped_to_target_group() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "astrbot_plugin_astrbot_enhance_mode_config.json"
    )
    raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
    tool = raw["chat_history_tool"]

    assert tool["enable"] is True
    assert tool["allowed_group_ids"] == ["829586749"]
    assert tool["max_pages_per_turn"] == 2
    assert tool["max_messages_per_turn"] == 24


def test_deployed_web_search_prompt_requires_source_and_date_discipline() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "astrbot_plugin_astrbot_enhance_mode_config.json"
    )
    raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
    prompt = raw["web_search"]["system_prompt"]

    assert "actual sources" in prompt
    assert "publication date" in prompt
    assert "event date" in prompt
    assert "insufficient evidence" in prompt
    assert "content" in prompt
    assert "sources" in prompt
