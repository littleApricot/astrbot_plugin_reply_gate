"""Coalesce events before AstrBot's per-session request queue."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


def merge_delays(config: dict) -> tuple[float, float]:
    """Validate configurable quiet and maximum collection delays in seconds."""
    values = []
    for key, default in (("merge_idle_seconds", 1), ("merge_max_wait_seconds", 3)):
        value = config.get(key, default)
        if isinstance(value, bool):
            raise ValueError("invalid_merge_delay")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid_merge_delay") from exc
        if not math.isfinite(value) or not 0 <= value <= 60:
            raise ValueError("invalid_merge_delay")
        values.append(value)
    return values[0], values[1]


@dataclass(eq=False)
class BatchMember:
    event: Any
    batch: MessageBatch
    cleanup: Callable[[], None]
    processed: asyncio.Event = field(default_factory=asyncio.Event)
    cleaned: bool = False

    def clean(self) -> None:
        if not self.cleaned:
            self.cleaned = True
            self.cleanup()


@dataclass(eq=False)
class MessageBatch:
    key: str
    first: float
    last: float
    idle: float
    maximum: float
    members: list[BatchMember] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    owner: BatchMember | None = None
    timer: asyncio.Task | None = None
    past_messages: list[dict] = field(default_factory=list)
    group_duplicates: set[str] = field(default_factory=set)
    mentioned: bool = False
    image_in_past: bool = False

    async def wait_other_handlers(self) -> None:
        """Wait until sibling pipelines have finished their message handlers."""
        for member in self.members:
            if member is not self.owner:
                await member.processed.wait()


class MessageBatcher:
    """Collect independently of all batches already judging or replying.

    AstrBot 4.29's scheduler always invokes event cleanup in its finally block.
    Retain sibling attachments until their own shared request has finished.
    """

    def __init__(self) -> None:
        self._heads: dict[str, MessageBatch] = {}
        self._active: set[MessageBatch] = set()
        self._closed = False

    async def join(self, event, idle: float, maximum: float) -> BatchMember:
        if self._closed:
            raise asyncio.CancelledError
        now = asyncio.get_running_loop().time()
        head = self._heads.get(event.unified_msg_origin)
        if head is None or head.ready.is_set():
            batch = MessageBatch(event.unified_msg_origin, now, now, idle, maximum)
            self._heads[batch.key] = batch
            self._active.add(batch)
        else:
            batch = head
        member = BatchMember(event, batch, event.cleanup_temporary_local_files)
        batch.members.append(member)
        batch.last = now
        batch.changed.set()
        event.cleanup_temporary_local_files = lambda: self._event_finished(member)
        event.set_extra("reply_gate_batch_member", member)
        if batch.timer is None:
            batch.timer = asyncio.create_task(self._settle(batch))
        await batch.ready.wait()
        if self._closed or batch.done.is_set():
            raise asyncio.CancelledError
        return member

    async def _settle(self, batch: MessageBatch) -> None:
        while batch.members:
            delay = min(batch.last + batch.idle, batch.first + batch.maximum)
            delay -= asyncio.get_running_loop().time()
            if delay <= 0:
                break
            batch.changed.clear()
            try:
                await asyncio.wait_for(batch.changed.wait(), timeout=delay)
            except TimeoutError:
                pass
        if batch.members:
            # The latest event carries the most recent platform reply token and
            # naturally remains the final user message of the shared request.
            batch.owner = batch.members[-1]
        batch.ready.set()
        if self._heads.get(batch.key) is batch:
            self._heads.pop(batch.key, None)

    def _event_finished(self, member: BatchMember) -> None:
        batch = member.batch
        member.processed.set()
        if not batch.ready.is_set():
            # An event cancelled while collecting must not become the owner.
            if member in batch.members:
                batch.members.remove(member)
                batch.changed.set()
            member.clean()
            if not batch.members:
                self._finish(batch)
        elif member is batch.owner:
            self._finish(batch)
        elif batch.done.is_set():
            member.clean()

    def _finish(self, batch: MessageBatch) -> None:
        if batch.done.is_set():
            return
        for member in batch.members:
            # A slow sibling handler retains ownership until its own cleanup.
            if member.processed.is_set():
                member.clean()
        batch.done.set()
        batch.ready.set()
        self._active.discard(batch)
        if batch.timer is not None and not batch.timer.done():
            batch.timer.cancel()
        if self._heads.get(batch.key) is batch:
            self._heads.pop(batch.key, None)

    async def close(self) -> None:
        self._closed = True
        batches = set(self._active)
        timers = [b.timer for b in batches if b.timer is not None]
        for batch in batches:
            self._finish(batch)
        if timers:
            await asyncio.gather(*timers, return_exceptions=True)
