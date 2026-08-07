from __future__ import annotations

from astrbot_plugin_astrbot_enhance_mode.runtime_state import (
    ActiveReplyAttempt,
    RuntimeState,
)


def test_touch_origin_evicts_oldest_state() -> None:
    state = RuntimeState()

    state.session_chats["o1"].append("m1")
    state.active_reply_stacks["o1"].append("a1")
    state.active_reply_pending["o1"] = 123.0
    state.active_reply_generations["o1"] = 3
    state.active_reply_attempts["o1"] = ActiveReplyAttempt.create(
        target_message_id="mid", snapshot_generation=3, ttl_sec=15, now=1
    )
    state.active_reply_last_sent_at["o1"] = 100.0
    state.image_message_registry["o1"]["mid"] = {"urls": ["u1"], "captions": {}}
    state.video_message_registry["o1"]["mid"] = {"urls": ["v1"], "captions": {}}
    state.reserve_chat_history_call(
        "o1",
        trigger_message_id="mid",
        requested_limit=8,
        max_calls=2,
        max_messages=24,
        ttl_sec=60,
        now=1,
    )
    state.issue_chat_history_cursor(
        "o1",
        trigger_message_id="mid",
        mode="recent",
        next_before_seq="10",
        page=1,
        returned_count=8,
        seen_message_ids=("1",),
        ttl_sec=60,
        now=1,
    )

    state.touch_origin("o1", max_origins=1)
    state.touch_origin("o2", max_origins=1)

    assert "o1" not in state.session_chats
    assert "o1" not in state.active_reply_stacks
    assert "o1" not in state.active_reply_pending
    assert "o1" not in state.active_reply_generations
    assert "o1" not in state.active_reply_attempts
    assert "o1" not in state.active_reply_last_sent_at
    assert "o1" not in state.image_message_registry
    assert "o1" not in state.video_message_registry
    assert "o1" not in state.chat_history_cursors
    assert "o1" not in state.chat_history_usage
    assert list(state.origin_lru.keys()) == ["o2"]


def test_cleanup_origin_removes_all_runtime_state() -> None:
    state = RuntimeState()
    state.session_chats["origin"].append("msg")
    state.active_reply_stacks["origin"].append("stack")
    state.active_reply_pending["origin"] = 123.0
    state.active_reply_generations["origin"] = 3
    state.active_reply_attempts["origin"] = ActiveReplyAttempt.create(
        target_message_id="1", snapshot_generation=3, ttl_sec=15, now=1
    )
    state.active_reply_last_sent_at["origin"] = 100.0
    state.image_message_registry["origin"]["1"] = {"urls": ["x"], "captions": {}}
    state.video_message_registry["origin"]["1"] = {"urls": ["v"], "captions": {}}
    state.reserve_chat_history_call(
        "origin",
        trigger_message_id="1",
        requested_limit=8,
        max_calls=2,
        max_messages=24,
        ttl_sec=60,
        now=1,
    )
    state.touch_origin("origin", max_origins=10)

    state.cleanup_origin("origin")

    assert "origin" not in state.session_chats
    assert "origin" not in state.active_reply_stacks
    assert "origin" not in state.active_reply_pending
    assert "origin" not in state.active_reply_generations
    assert "origin" not in state.active_reply_attempts
    assert "origin" not in state.active_reply_last_sent_at
    assert "origin" not in state.image_message_registry
    assert "origin" not in state.video_message_registry
    assert "origin" not in state.chat_history_cursors
    assert "origin" not in state.chat_history_usage
    assert "origin" not in state.origin_lru


def test_attempt_generation_ttl_and_compare_clear() -> None:
    state = RuntimeState()
    assert state.bump_generation("origin") == 1
    attempt = state.begin_active_reply_attempt(
        "origin", target_message_id="m1", ttl_sec=15, now=10
    )
    assert attempt is not None

    assert state.validate_active_reply_attempt(
        "origin", attempt.attempt_id, target_message_id="m1", now=20
    ) == (True, "")
    assert state.validate_active_reply_attempt(
        "origin", attempt.attempt_id, target_message_id="m1", now=26
    ) == (False, "expired")

    state.bump_generation("origin")
    assert state.validate_active_reply_attempt(
        "origin", attempt.attempt_id, target_message_id="m1", now=20
    ) == (False, "newer_message")
    assert state.compare_and_clear_active_reply_attempt("origin", "wrong") is False
    assert (
        state.compare_and_clear_active_reply_attempt("origin", attempt.attempt_id)
        is True
    )


def test_chat_history_usage_limits_calls_and_messages_per_trigger() -> None:
    state = RuntimeState()

    assert state.reserve_chat_history_call(
        "origin",
        trigger_message_id="m1",
        requested_limit=20,
        max_calls=2,
        max_messages=24,
        ttl_sec=60,
        now=10,
    ) == (True, 20, "")
    state.record_chat_history_results(
        "origin",
        trigger_message_id="m1",
        count=20,
    )
    assert state.reserve_chat_history_call(
        "origin",
        trigger_message_id="m1",
        requested_limit=20,
        max_calls=2,
        max_messages=24,
        ttl_sec=60,
        now=11,
    ) == (True, 4, "")
    assert state.reserve_chat_history_call(
        "origin",
        trigger_message_id="m1",
        requested_limit=1,
        max_calls=2,
        max_messages=24,
        ttl_sec=60,
        now=12,
    ) == (False, 0, "call_limit")

    assert state.reserve_chat_history_call(
        "origin",
        trigger_message_id="m2",
        requested_limit=8,
        max_calls=2,
        max_messages=24,
        ttl_sec=60,
        now=13,
    ) == (True, 8, "")


def test_chat_history_cursor_is_one_time_scoped_and_expiring() -> None:
    state = RuntimeState()
    token = state.issue_chat_history_cursor(
        "origin",
        trigger_message_id="m1",
        mode="recent",
        next_before_seq="42",
        page=1,
        returned_count=8,
        seen_message_ids=("a", "b"),
        ttl_sec=60,
        now=10,
    )

    cursor, error = state.consume_chat_history_cursor(
        "origin",
        token,
        trigger_message_id="m1",
        now=20,
    )
    assert error == ""
    assert cursor is not None
    assert cursor.next_before_seq == "42"
    assert cursor.seen_message_ids == ("a", "b")
    assert state.consume_chat_history_cursor(
        "origin",
        token,
        trigger_message_id="m1",
        now=21,
    ) == (None, "invalid_cursor")

    expired_token = state.issue_chat_history_cursor(
        "origin",
        trigger_message_id="m1",
        mode="before",
        next_before_seq="30",
        page=1,
        returned_count=4,
        seen_message_ids=(),
        ttl_sec=10,
        now=30,
    )
    assert state.consume_chat_history_cursor(
        "origin",
        expired_token,
        trigger_message_id="m1",
        now=41,
    ) == (None, "cursor_expired")
