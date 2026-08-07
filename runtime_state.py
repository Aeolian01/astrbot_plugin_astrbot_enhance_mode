from __future__ import annotations

import hashlib
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MediaRef:
    """An image reference whose identity survives preprocessing failures."""

    message_id: str
    image_index: int
    url: str
    cache_source: str = ""
    is_target: bool = False

    @property
    def ref_label(self) -> str:
        return hashlib.sha256(str(self.url or "").encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class PreparedMedia:
    """Result of preparing one MediaRef for a provider request."""

    media_ref: MediaRef
    status: str
    prepared_url: str = ""
    error_type: str = ""
    ref_label: str = ""

    def __post_init__(self) -> None:
        if not self.ref_label:
            object.__setattr__(self, "ref_label", self.media_ref.ref_label)

    @property
    def is_available(self) -> bool:
        return self.status == "READY" and bool(self.prepared_url)


@dataclass(frozen=True)
class StackItem:
    """A structured active-reply candidate.

    ``rendered_text`` keeps compatibility with the legacy prompt/history format,
    while the other fields provide stable target and media identity.
    """

    message_id: str
    sender_id: str
    received_at: float
    text: str
    rendered_text: str
    media_refs: tuple[MediaRef, ...] = field(default_factory=tuple)


@dataclass
class ActiveReplyAttempt:
    attempt_id: str
    target_message_id: str
    snapshot_generation: int
    started_at: float
    expires_at: float
    status: str = "pending"

    @classmethod
    def create(
        cls,
        *,
        target_message_id: str,
        snapshot_generation: int,
        ttl_sec: float,
        now: float | None = None,
    ) -> ActiveReplyAttempt:
        started_at = time.monotonic() if now is None else now
        return cls(
            attempt_id=uuid.uuid4().hex,
            target_message_id=str(target_message_id or ""),
            snapshot_generation=max(0, int(snapshot_generation)),
            started_at=started_at,
            expires_at=started_at + max(0.1, float(ttl_sec)),
        )


@dataclass(frozen=True)
class ChatHistoryCursor:
    token: str
    origin: str
    trigger_message_id: str
    mode: str
    next_before_seq: str
    page: int
    returned_count: int
    seen_message_ids: tuple[str, ...]
    expires_at: float

    @classmethod
    def create(
        cls,
        *,
        origin: str,
        trigger_message_id: str,
        mode: str,
        next_before_seq: str,
        page: int,
        returned_count: int,
        seen_message_ids: tuple[str, ...],
        ttl_sec: float,
        now: float | None = None,
    ) -> ChatHistoryCursor:
        current = time.monotonic() if now is None else now
        return cls(
            token=uuid.uuid4().hex,
            origin=str(origin or ""),
            trigger_message_id=str(trigger_message_id or ""),
            mode=str(mode or "recent"),
            next_before_seq=str(next_before_seq or ""),
            page=max(1, int(page)),
            returned_count=max(0, int(returned_count)),
            seen_message_ids=tuple(str(item) for item in seen_message_ids if str(item)),
            expires_at=current + max(1.0, float(ttl_sec)),
        )


@dataclass
class ChatHistoryUsage:
    trigger_message_id: str
    calls: int
    returned_messages: int
    expires_at: float


class RuntimeState:
    def __init__(self) -> None:
        self.session_chats: dict[str, list[str]] = defaultdict(list)
        # Legacy text stacks remain available for model_choice and pipeline-v1.
        self.active_reply_stacks: dict[str, list[str]] = defaultdict(list)
        self.active_reply_stack_items: dict[str, list[StackItem]] = defaultdict(list)
        self.active_reply_pending: dict[str, float] = {}
        self.active_reply_generations: dict[str, int] = defaultdict(int)
        self.active_reply_attempts: dict[str, ActiveReplyAttempt] = {}
        self.active_reply_last_sent_at: dict[str, float] = {}
        self.image_message_registry: dict[str, dict[str, dict[str, object]]] = (
            defaultdict(dict)
        )
        self.video_message_registry: dict[str, dict[str, dict[str, object]]] = (
            defaultdict(dict)
        )
        self.chat_history_cursors: dict[str, dict[str, ChatHistoryCursor]] = (
            defaultdict(dict)
        )
        self.chat_history_usage: dict[str, ChatHistoryUsage] = {}
        self.origin_lru: OrderedDict[str, None] = OrderedDict()

    def reserve_chat_history_call(
        self,
        origin: str,
        *,
        trigger_message_id: str,
        requested_limit: int,
        max_calls: int,
        max_messages: int,
        ttl_sec: float,
        now: float | None = None,
    ) -> tuple[bool, int, str]:
        if not origin or not trigger_message_id:
            return False, 0, "missing_scope"
        current = time.monotonic() if now is None else now
        usage = self.chat_history_usage.get(origin)
        if (
            usage is None
            or usage.trigger_message_id != trigger_message_id
            or current > usage.expires_at
        ):
            usage = ChatHistoryUsage(
                trigger_message_id=trigger_message_id,
                calls=0,
                returned_messages=0,
                expires_at=current + max(1.0, float(ttl_sec)),
            )
            self.chat_history_usage[origin] = usage
        if usage.calls >= max(1, int(max_calls)):
            return False, 0, "call_limit"
        remaining = max(0, int(max_messages) - usage.returned_messages)
        if remaining <= 0:
            return False, 0, "message_limit"
        usage.calls += 1
        return True, min(max(1, int(requested_limit)), remaining), ""

    def record_chat_history_results(
        self,
        origin: str,
        *,
        trigger_message_id: str,
        count: int,
    ) -> None:
        usage = self.chat_history_usage.get(origin)
        if usage is None or usage.trigger_message_id != trigger_message_id:
            return
        usage.returned_messages += max(0, int(count))

    def issue_chat_history_cursor(
        self,
        origin: str,
        *,
        trigger_message_id: str,
        mode: str,
        next_before_seq: str,
        page: int,
        returned_count: int,
        seen_message_ids: tuple[str, ...],
        ttl_sec: float,
        max_per_origin: int = 8,
        now: float | None = None,
    ) -> str:
        if not origin or not trigger_message_id or not next_before_seq:
            return ""
        current = time.monotonic() if now is None else now
        cursors = self.chat_history_cursors[origin]
        for token, cursor in list(cursors.items()):
            if current > cursor.expires_at:
                cursors.pop(token, None)
        cursor = ChatHistoryCursor.create(
            origin=origin,
            trigger_message_id=trigger_message_id,
            mode=mode,
            next_before_seq=next_before_seq,
            page=page,
            returned_count=returned_count,
            seen_message_ids=seen_message_ids,
            ttl_sec=ttl_sec,
            now=current,
        )
        cursors[cursor.token] = cursor
        while len(cursors) > max(1, int(max_per_origin)):
            oldest_token = min(cursors, key=lambda token: cursors[token].expires_at)
            cursors.pop(oldest_token, None)
        return cursor.token

    def consume_chat_history_cursor(
        self,
        origin: str,
        token: str,
        *,
        trigger_message_id: str,
        now: float | None = None,
    ) -> tuple[ChatHistoryCursor | None, str]:
        if not origin or not token:
            return None, "invalid_cursor"
        cursors = self.chat_history_cursors.get(origin)
        if not cursors:
            return None, "invalid_cursor"
        cursor = cursors.pop(str(token), None)
        if cursor is None:
            return None, "invalid_cursor"
        current = time.monotonic() if now is None else now
        if current > cursor.expires_at:
            return None, "cursor_expired"
        if cursor.origin != origin or cursor.trigger_message_id != trigger_message_id:
            return None, "cursor_scope_mismatch"
        return cursor, ""

    def bump_generation(self, origin: str) -> int:
        if not origin:
            return 0
        generation = self.active_reply_generations[origin] + 1
        self.active_reply_generations[origin] = generation
        return generation

    def begin_active_reply_attempt(
        self,
        origin: str,
        *,
        target_message_id: str,
        ttl_sec: float,
        now: float | None = None,
    ) -> ActiveReplyAttempt | None:
        if not origin:
            return None
        attempt = ActiveReplyAttempt.create(
            target_message_id=target_message_id,
            snapshot_generation=self.active_reply_generations[origin],
            ttl_sec=ttl_sec,
            now=now,
        )
        self.active_reply_attempts[origin] = attempt
        self.active_reply_pending[origin] = attempt.started_at
        return attempt

    def validate_active_reply_attempt(
        self,
        origin: str,
        attempt_id: str,
        *,
        target_message_id: str = "",
        cancel_on_newer_message: bool = True,
        now: float | None = None,
    ) -> tuple[bool, str]:
        attempt = self.active_reply_attempts.get(origin)
        if attempt is None:
            return False, "missing_attempt"
        if not attempt_id or attempt.attempt_id != attempt_id:
            return False, "attempt_replaced"
        if target_message_id and attempt.target_message_id != str(target_message_id):
            return False, "target_mismatch"
        if attempt.status != "pending":
            return False, f"status:{attempt.status}"
        current = time.monotonic() if now is None else now
        if current > attempt.expires_at:
            return False, "expired"
        if (
            cancel_on_newer_message
            and self.active_reply_generations[origin] != attempt.snapshot_generation
        ):
            return False, "newer_message"
        return True, ""

    def compare_and_clear_active_reply_attempt(
        self,
        origin: str,
        attempt_id: str = "",
    ) -> bool:
        attempt = self.active_reply_attempts.get(origin)
        if attempt is not None and attempt_id and attempt.attempt_id != attempt_id:
            return False
        if attempt is None and attempt_id:
            return False
        self.active_reply_attempts.pop(origin, None)
        self.active_reply_pending.pop(origin, None)
        return True

    def mark_active_reply_sent(
        self,
        origin: str,
        attempt_id: str = "",
        *,
        now: float | None = None,
    ) -> bool:
        attempt = self.active_reply_attempts.get(origin)
        if attempt is None:
            return False
        if attempt_id and attempt.attempt_id != attempt_id:
            return False
        if attempt.status != "pending":
            return False
        attempt.status = "sent"
        self.active_reply_last_sent_at[origin] = (
            time.monotonic() if now is None else now
        )
        self.active_reply_attempts.pop(origin, None)
        self.active_reply_pending.pop(origin, None)
        return True

    def _evict_origin_state(self, origin: str) -> None:
        self.session_chats.pop(origin, None)
        self.active_reply_stacks.pop(origin, None)
        self.active_reply_stack_items.pop(origin, None)
        self.active_reply_pending.pop(origin, None)
        self.active_reply_generations.pop(origin, None)
        self.active_reply_attempts.pop(origin, None)
        self.active_reply_last_sent_at.pop(origin, None)
        self.image_message_registry.pop(origin, None)
        self.video_message_registry.pop(origin, None)
        self.chat_history_cursors.pop(origin, None)
        self.chat_history_usage.pop(origin, None)

    def touch_origin(self, origin: str, max_origins: int) -> None:
        if not origin:
            return
        self.origin_lru.pop(origin, None)
        self.origin_lru[origin] = None
        while len(self.origin_lru) > max_origins:
            oldest, _ = self.origin_lru.popitem(last=False)
            self._evict_origin_state(oldest)

    def cleanup_origin(self, origin: str) -> None:
        self._evict_origin_state(origin)
        self.origin_lru.pop(origin, None)
