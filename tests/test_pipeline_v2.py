from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from astrbot.api.message_components import Image, Plain
from astrbot.api.platform import MessageType

import astrbot_plugin_astrbot_enhance_mode.main as main_module
from astrbot_plugin_astrbot_enhance_mode.main import Main
from astrbot_plugin_astrbot_enhance_mode.plugin_config import (
    ActiveReplyConfig,
    GroupFeatureEnhancementConfig,
    GroupHistoryEnhancementConfig,
    PluginConfig,
)
from astrbot_plugin_astrbot_enhance_mode.runtime_state import (
    MediaRef,
    PreparedMedia,
    RuntimeState,
)


class _Result:
    def __init__(self, chain: list[object]) -> None:
        self.chain = chain


class _Event:
    unified_msg_origin = "origin-v2"
    session_id = "session-v2"
    is_at_or_wake_command = False

    def __init__(
        self,
        *,
        message_id: str = "m1",
        components: list[object] | None = None,
        message_str: str = "hello",
    ) -> None:
        self.message_obj = SimpleNamespace(
            message_id=message_id,
            message=components or [Plain(text=message_str)],
            sender=SimpleNamespace(nickname="Alice"),
        )
        self.message_str = message_str
        self._extras: dict[str, object] = {}
        self.result = _Result([])

    def get_message_type(self) -> MessageType:
        return MessageType.GROUP_MESSAGE

    def get_messages(self) -> list[object]:
        return list(self.message_obj.message)

    def get_sender_id(self) -> str:
        return "10001"

    def get_group_id(self) -> str:
        return "group-v2"

    def get_extra(self, key: str, default: object = None) -> object:
        return self._extras.get(key, default)

    def set_extra(self, key: str, value: object) -> None:
        self._extras[key] = value

    def get_result(self) -> _Result:
        return self.result


def _cfg(**active_overrides: object) -> PluginConfig:
    active = {
        "enable": True,
        "mode": "single_pass",
        "pipeline_v2_enable": True,
        "model_stack_size": 8,
        "unified_context_messages": 6,
        "unsolicited_image_reply": True,
    }
    active.update(active_overrides)
    return PluginConfig(
        group_features=GroupFeatureEnhancementConfig(react_mode_enable=True),
        group_history=GroupHistoryEnhancementConfig(enable=True),
        active_reply=ActiveReplyConfig(**active),
    )


def _plugin(cfg: PluginConfig | None = None) -> Main:
    plugin = Main.__new__(Main)
    plugin.runtime = RuntimeState()
    selected = cfg or _cfg()
    plugin._cfg = lambda: selected
    return plugin


@pytest.mark.asyncio
async def test_v2_single_pass_uses_only_current_message_and_keeps_legacy_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _plugin()
    event = _Event(
        message_id="target",
        components=[Image(file="https://example.com/target.png")],
        message_str="[Image]",
    )
    monkeypatch.setattr(main_module, "Image", Image)

    async def get_entry(_origin: str, _message_id: str) -> dict[str, object]:
        return {}

    async def prepare(refs: list[MediaRef]) -> list[PreparedMedia]:
        assert [(ref.message_id, ref.image_index) for ref in refs] == [("target", 0)]
        return [
            PreparedMedia(
                refs[0],
                "READY",
                prepared_url="data:image/png;base64,dGFyZ2V0",
            )
        ]

    plugin._get_image_message_entry = get_entry
    plugin._preprocess_single_pass_media_refs = prepare
    # A stale legacy item must not join the v2 target batch.
    plugin.runtime.active_reply_stacks[event.unified_msg_origin].append(
        "[Bob/2] #msgold: [Image]"
    )

    assert await plugin._need_active_reply_single_pass(event, _cfg()) is True
    assert event.get_extra(main_module.ENHANCE_SINGLE_PASS_STACK_KEY) == [
        "[Alice/10001] #msgtarget: [Image]"
    ]
    assert event.get_extra(main_module.ENHANCE_SINGLE_PASS_IMAGE_URLS_KEY) == [
        "data:image/png;base64,dGFyZ2V0"
    ]
    manifest = event.get_extra(main_module.ENHANCE_SINGLE_PASS_MEDIA_MANIFEST_KEY)
    assert manifest == [
        {
            "message_id": "target",
            "image_index": 0,
            "is_target": True,
            "status": "READY",
            "prepared_url": "data:image/png;base64,dGFyZ2V0",
            "error_type": "",
            "ref_label": MediaRef(
                "target", 0, "https://example.com/target.png", is_target=True
            ).ref_label,
        }
    ]


@pytest.mark.asyncio
async def test_v2_pure_image_target_is_cancelled_when_target_media_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _plugin()
    event = _Event(
        message_id="target",
        components=[Image(file="https://example.com/missing.png")],
        message_str="[Image]",
    )
    monkeypatch.setattr(main_module, "Image", Image)

    async def get_entry(_origin: str, _message_id: str) -> dict[str, object]:
        return {}

    async def prepare(refs: list[MediaRef]) -> list[PreparedMedia]:
        return [PreparedMedia(refs[0], "UNAVAILABLE", error_type="FileNotFoundError")]

    plugin._get_image_message_entry = get_entry
    plugin._preprocess_single_pass_media_refs = prepare

    assert await plugin._need_active_reply_single_pass(event, _cfg()) is False
    assert event.get_extra(main_module.ENHANCE_SINGLE_PASS_IMAGE_URLS_KEY) is None


@pytest.mark.asyncio
async def test_current_message_registry_preserves_duplicate_slots_and_order() -> None:
    plugin = _plugin()
    event = _Event(message_id="target", message_str="[Image] [Image] [Image]")

    async def get_entry(_origin: str, message_id: str) -> dict[str, object]:
        assert message_id == "target"
        return {
            "urls": [
                "https://example.com/same.png",
                "https://example.com/same.png",
                "https://example.com/third.png",
            ],
            "cache_sources": ["slot-0", "slot-1", "slot-2"],
        }

    plugin._get_image_message_entry = get_entry
    plugin._format_event_message_body = lambda _event: (
        "[Image] [Image] [Image]",
        ["https://example.com/same.png"],
        ["direct-duplicate"],
    )

    refs = await plugin._collect_single_pass_media_refs(event)

    assert [ref.image_index for ref in refs] == [0, 1, 2]
    assert [ref.url for ref in refs] == [
        "https://example.com/same.png",
        "https://example.com/same.png",
        "https://example.com/third.png",
    ]
    assert [ref.cache_source for ref in refs] == ["slot-0", "slot-1", "slot-2"]


def test_v2_media_manifest_prompt_has_exact_mapping_and_failure_status() -> None:
    plugin = _plugin()
    cfg = _cfg()
    prompt = plugin._build_active_reply_prompt(
        cfg,
        {
            "recent_history_lines": [],
            "current_message_text": "[Alice/1] #msgtarget: 看这个",
            "single_pass_messages": ["[Alice/1] #msgtarget: 看这个"],
            "single_pass_media_manifest": [
                PreparedMedia(
                    MediaRef("target", 0, "", is_target=True),
                    "READY",
                    prepared_url="data:image/png;base64,QQ==",
                ),
                PreparedMedia(
                    MediaRef("target", 1, "", is_target=True),
                    "UNAVAILABLE",
                    error_type="TimeoutError",
                ),
            ],
        },
        "single_pass",
    )

    assert "ATTACHED_IMAGE_1 -> #msgtarget image[0] TARGET" in prompt
    assert "UNAVAILABLE -> #msgtarget image[1] TARGET error=TimeoutError" in prompt
    assert "Never infer the contents of an UNAVAILABLE image" in prompt


def test_failed_media_does_not_compress_later_image_index() -> None:
    plugin = _plugin()
    prompt = plugin._build_active_reply_prompt(
        _cfg(),
        {
            "recent_history_lines": [],
            "current_message_text": "[Alice/1] #msgtarget: 两张图",
            "single_pass_messages": ["[Alice/1] #msgtarget: 两张图"],
            "single_pass_media_manifest": [
                PreparedMedia(
                    MediaRef("target", 0, "https://example.com/bad.png", is_target=True),
                    "UNAVAILABLE",
                    error_type="FileNotFoundError",
                ),
                PreparedMedia(
                    MediaRef("target", 1, "https://example.com/good.png", is_target=True),
                    "READY",
                    prepared_url="data:image/png;base64,QQ==",
                ),
            ],
        },
        "single_pass",
    )

    assert "UNAVAILABLE -> #msgtarget image[0] TARGET" in prompt
    assert "ATTACHED_IMAGE_1 -> #msgtarget image[1] TARGET" in prompt


@pytest.mark.asyncio
async def test_unsolicited_image_gate_runs_before_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(unsolicited_image_reply=False)
    plugin = _plugin(cfg)
    event = _Event(
        message_id="target",
        components=[Image(file="https://example.com/target.png")],
        message_str="[Image]",
    )
    monkeypatch.setattr(main_module, "Image", Image)

    async def get_entry(_origin: str, _message_id: str) -> dict[str, object]:
        return {}

    async def must_not_prepare(_refs: list[MediaRef]) -> list[PreparedMedia]:
        raise AssertionError("unsolicited image gate must run before preprocessing")

    plugin._get_image_message_entry = get_entry
    plugin._preprocess_single_pass_media_refs = must_not_prepare

    assert await plugin._need_active_reply_single_pass(event, cfg) is False


def test_context_trim_preserves_recent_round_objects_tools_and_multimodal() -> None:
    image_content = [
        {"type": "text", "text": "latest"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QQ=="}},
    ]
    contexts = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old-answer"},
        {"role": "user", "content": "tool-round"},
        {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "tool-result"},
        {"role": "assistant", "content": "tool-answer"},
        {"role": "user", "content": image_content},
        {"role": "assistant", "content": "latest-answer"},
        {"role": "user", "content": "duplicate current"},
    ]

    trimmed = Main._trim_provider_contexts(
        contexts,
        max_rounds=2,
        duplicate_user_text="duplicate current",
    )

    assert trimmed[0]["content"] == "tool-round"
    assert trimmed[1]["tool_calls"][0]["id"] == "call-1"
    assert trimmed[2]["role"] == "tool"
    assert trimmed[-2]["content"] is image_content
    assert trimmed[-1]["content"] == "latest-answer"


@pytest.mark.asyncio
async def test_generation_and_ttl_are_checked_post_generation_and_pre_send() -> None:
    cfg = _cfg()
    plugin = _plugin(cfg)
    event = _Event()
    plugin.runtime.bump_generation(event.unified_msg_origin)
    attempt = plugin._mark_active_reply_pending(
        event.unified_msg_origin,
        event=event,
        cfg=cfg,
    )
    assert attempt is not None

    assert plugin._validate_active_reply_attempt(event, cfg, phase="post_generation") == (
        True,
        "",
    )
    plugin.runtime.bump_generation(event.unified_msg_origin)
    assert plugin._validate_active_reply_attempt(event, cfg, phase="pre_send") == (
        False,
        "newer_message",
    )

    fresh = plugin.runtime.begin_active_reply_attempt(
        event.unified_msg_origin,
        target_message_id="m1",
        ttl_sec=15,
        now=time.monotonic() - 20,
    )
    assert fresh is not None
    event.set_extra(main_module.ENHANCE_ACTIVE_REPLY_ATTEMPT_ID_KEY, fresh.attempt_id)
    assert plugin._validate_active_reply_attempt(event, cfg, phase="pre_send") == (
        False,
        "expired",
    )


@pytest.mark.asyncio
async def test_output_gate_blocks_process_leak_long_and_multiple_payloads() -> None:
    cfg = _cfg()
    plugin = _plugin(cfg)

    async def run_gate(chain: list[object]) -> str:
        event = _Event()
        event.set_extra("_enhance_active_reply_triggered", True)
        event.result.chain = chain
        plugin.runtime.bump_generation(event.unified_msg_origin)
        plugin._mark_active_reply_pending(event.unified_msg_origin, event=event, cfg=cfg)
        await plugin.parse_tags(event)
        assert event.result.chain == []
        return str(event.get_extra(main_module.ENHANCE_OUTPUT_GATE_BLOCK_KEY))

    assert await run_gate([Plain(text="I will search that")]) == "process_leak"
    assert await run_gate([Plain(text="我将搜索一下")]) == "process_leak"
    assert await run_gate([Plain(text="长" * 51)]) == "active_reply_too_long"
    assert await run_gate([Plain(text="第一段"), Plain(text="第二段")]) == (
        "multiple_sendable_payloads"
    )
    assert await run_gate([Plain(text="SKIP")]) == "decision_token_leak"


def test_short_c_pragmatic_rules_are_in_active_and_passive_prompts() -> None:
    plugin = _plugin()
    cfg = _cfg()
    context = {
        "recent_history_lines": [],
        "current_message_text": "这是反话吗？",
        "single_pass_messages": ["这是反话吗？"],
    }

    for mode in ("", "single_pass"):
        prompt = plugin._build_active_reply_prompt(cfg, context, mode)
        assert "认真提问、接梗、反问、阴阳、引用还是反串" in prompt
        assert "图片作者立场、讽刺靶子和靶向范围" in prompt
        assert "被纠错先明确承认具体错在哪里" in prompt
        assert "缺图、看不清或没有把握时不猜" in prompt


@pytest.mark.parametrize(
    ("raw_completion", "expected"),
    [
        ({"choices": [{"finish_reason": "length"}]}, "length"),
        (
            SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="max_tokens")]
            ),
            "max_tokens",
        ),
        ({"candidates": [{"finishReason": "MAX_TOKENS"}]}, "max_tokens"),
        (
            {"candidates": [{"finishReason": SimpleNamespace(name="MAX_TOKENS")}]},
            "max_tokens",
        ),
        ({"candidates": [{"finishReason": "FinishReason.MAX_TOKENS"}]}, "max_tokens"),
        ('{"choices":[{"finish_reason":"stop"}]}', "stop"),
    ],
)
def test_finish_reason_reads_real_raw_completion_shapes(
    raw_completion: object,
    expected: str,
) -> None:
    response = SimpleNamespace(
        completion_text="answer",
        raw_completion=raw_completion,
    )

    assert Main._response_finish_reason(response) == expected


@pytest.mark.asyncio
async def test_history_and_cooldown_update_only_after_confirmed_send() -> None:
    cfg = _cfg(cooldown_sec=300)
    plugin = _plugin(cfg)
    event = _Event()
    event.set_extra("_enhance_active_reply_triggered", True)
    event.set_extra("_enhance_active_reply_mode", "single_pass")
    plugin.runtime.session_chats[event.unified_msg_origin].append(
        "[Alice/1] #msgm1: hello"
    )
    plugin.runtime.bump_generation(event.unified_msg_origin)
    plugin._mark_active_reply_pending(event.unified_msg_origin, event=event, cfg=cfg)
    response = SimpleNamespace(completion_text="接住了", finish_reason="stop")

    await plugin.record_bot_response(event, response)

    assert plugin.runtime.session_chats[event.unified_msg_origin] == [
        "[Alice/1] #msgm1: hello"
    ]
    assert event.unified_msg_origin not in plugin.runtime.active_reply_last_sent_at
    assert event.get_extra(main_module.ENHANCE_PENDING_BOT_HISTORY_TEXT_KEY) == "接住了"

    event.result.chain = [Plain(text="接住了")]
    await plugin.parse_tags(event)
    await plugin.after_message_sent(event)

    assert plugin.runtime.session_chats[event.unified_msg_origin][-1].endswith(
        ": 接住了"
    )
    assert event.unified_msg_origin in plugin.runtime.active_reply_last_sent_at
    assert plugin._allow_active_reply(event, cfg) is False


@pytest.mark.asyncio
async def test_pre_send_block_does_not_pollute_history_or_cooldown() -> None:
    cfg = _cfg()
    plugin = _plugin(cfg)
    event = _Event()
    event.set_extra("_enhance_active_reply_triggered", True)
    event.set_extra("_enhance_active_reply_mode", "single_pass")
    plugin.runtime.session_chats[event.unified_msg_origin].append(
        "[Alice/1] #msgm1: hello"
    )
    plugin.runtime.bump_generation(event.unified_msg_origin)
    plugin._mark_active_reply_pending(event.unified_msg_origin, event=event, cfg=cfg)

    await plugin.record_bot_response(
        event,
        SimpleNamespace(completion_text="第一段第二段", finish_reason="stop"),
    )
    event.result.chain = [Plain(text="第一段"), Plain(text="第二段")]
    await plugin.parse_tags(event)
    # Even if a framework erroneously calls after_message_sent for an emptied
    # chain, the block marker prevents success accounting.
    await plugin.after_message_sent(event)

    assert plugin.runtime.session_chats[event.unified_msg_origin] == [
        "[Alice/1] #msgm1: hello"
    ]
    assert event.unified_msg_origin not in plugin.runtime.active_reply_last_sent_at


def test_density_gate_uses_recent_window() -> None:
    cfg = _cfg(bot_density_window=4, bot_density_max=2, cooldown_sec=0)
    plugin = _plugin(cfg)
    event = _Event()
    plugin.runtime.session_chats[event.unified_msg_origin].extend(
        [
            "[Alice/1] #msg1: a",
            "[You/12:00:00]: one",
            "[Bob/2] #msg2: b",
            "[You/12:00:01]: two",
        ]
    )

    assert plugin._allow_active_reply(event, cfg) is False


@pytest.mark.asyncio
async def test_new_group_message_invalidates_pending_attempt_before_pending_skip() -> None:
    cfg = _cfg(cooldown_sec=0, bot_density_max=3)
    plugin = _plugin(cfg)
    first = _Event(message_id="m1", message_str="first")
    plugin.runtime.bump_generation(first.unified_msg_origin)
    attempt = plugin._mark_active_reply_pending(
        first.unified_msg_origin,
        event=first,
        cfg=cfg,
    )
    assert attempt is not None

    second = _Event(message_id="m2", message_str="second")

    async def record_noop(_event: _Event, _cfg: PluginConfig) -> None:
        return None

    plugin._record_message = record_noop
    async for _item in plugin.on_group_message(second):
        raise AssertionError("pending message must not create another LLM request")

    assert plugin.runtime.active_reply_generations[first.unified_msg_origin] == 2
    assert plugin.runtime.validate_active_reply_attempt(
        first.unified_msg_origin,
        attempt.attempt_id,
        target_message_id="m1",
    ) == (False, "newer_message")


@pytest.mark.asyncio
async def test_v2_single_pass_canary_samples_before_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _cfg(possibility=0.05, cooldown_sec=0)
    plugin = _plugin(cfg)
    event = _Event(message_id="canary", message_str="普通群聊")
    calls: list[str] = []

    async def single_pass(_event: _Event, _cfg: PluginConfig) -> bool:
        calls.append("single_pass")
        return True

    plugin._need_active_reply_single_pass = single_pass

    monkeypatch.setattr(main_module.random, "random", lambda: 0.80)
    assert await plugin._need_active_reply(event, cfg) is False
    assert calls == []

    monkeypatch.setattr(main_module.random, "random", lambda: 0.01)
    assert await plugin._need_active_reply(event, cfg) is True
    assert calls == ["single_pass"]
