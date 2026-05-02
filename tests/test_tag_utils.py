from __future__ import annotations

from astrbot.api.message_components import At, Plain

from astrbot_plugin_astrbot_enhance_mode.tag_utils import (
    build_interaction_instructions,
    dedupe_repeated_result_chain,
)


def test_dedupe_repeated_result_chain_collapses_identical_halves() -> None:
    chain = [
        At(qq="572325141"),
        Plain(text=" 哟，还没戳够？"),
        At(qq="572325141"),
        Plain(text=" 哟，还没戳够？"),
    ]

    deduped = dedupe_repeated_result_chain(chain)

    assert deduped == chain[:2]


def test_dedupe_repeated_result_chain_keeps_non_duplicate_chain() -> None:
    chain = [
        At(qq="572325141"),
        Plain(text=" 第一段"),
        At(qq="572325141"),
        Plain(text=" 第二段"),
    ]

    assert dedupe_repeated_result_chain(chain) is None


def test_interaction_instructions_tell_model_not_to_mix_tool_call_and_reply() -> None:
    instructions = build_interaction_instructions(
        mention_parse=True,
        include_sender_id=True,
    )

    assert "If you call any tool" in instructions
    assert "output the final group reply exactly once" in instructions
