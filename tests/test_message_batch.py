"""Exercise collection deadlines, backlog coalescing and cleanup boundaries."""

import asyncio
import math
import unittest
from types import SimpleNamespace

from astrbot_plugin_reply_gate.message_batch import MessageBatcher, merge_delays


def event(key="group"):
    extras = {}
    calls = []
    return SimpleNamespace(
        unified_msg_origin=key,
        set_extra=lambda key, value: extras.__setitem__(key, value),
        cleanup_temporary_local_files=lambda: calls.append("cleaned"),
        extras=extras,
        cleanups=calls,
    )


class BatchingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.batcher = MessageBatcher()

    async def asyncTearDown(self):
        await self.batcher.close()

    async def test_quiet_period_extends_collection_and_latest_event_owns_reply(self):
        first, second = event(), event()
        start = asyncio.get_running_loop().time()
        a = asyncio.create_task(self.batcher.join(first, 0.08, 0.5))
        await asyncio.sleep(0.04)
        b = asyncio.create_task(self.batcher.join(second, 0.08, 0.5))
        one, two = await asyncio.gather(a, b)
        self.assertIs(one.batch, two.batch)
        self.assertIs(one.batch.owner, two)
        self.assertGreaterEqual(asyncio.get_running_loop().time() - start, 0.11)
        first.cleanup_temporary_local_files()
        self.assertTrue(one.processed.is_set())
        self.assertEqual(first.cleanups, [])
        second.cleanup_temporary_local_files()
        self.assertEqual(first.cleanups, ["cleaned"])
        self.assertEqual(second.cleanups, ["cleaned"])
        self.assertEqual(self.batcher._heads, {})

    async def test_continuous_messages_hit_maximum_instead_of_waiting_forever(self):
        events = [event() for _ in range(4)]
        start = asyncio.get_running_loop().time()
        tasks = [asyncio.create_task(self.batcher.join(events[0], 0.5, 0.12))]
        for e in events[1:]:
            await asyncio.sleep(0.025)
            tasks.append(asyncio.create_task(self.batcher.join(e, 0.5, 0.12)))
        members = await asyncio.wait_for(asyncio.gather(*tasks), 0.4)
        elapsed = asyncio.get_running_loop().time() - start
        self.assertGreaterEqual(elapsed, 0.11)
        self.assertLess(elapsed, 0.4)
        self.assertTrue(all(m.batch is members[0].batch for m in members))
        for e in events:
            e.cleanup_temporary_local_files()

    async def test_backlog_keeps_merging_until_previous_reply_finishes(
        self,
    ):
        active = event()
        original = await self.batcher.join(active, 0, 0)
        new_events = [event() for _ in range(5)]
        tasks = []
        for e in new_events:
            tasks.append(asyncio.create_task(self.batcher.join(e, 0.02, 0.04)))
            await asyncio.sleep(0.05)
        self.assertTrue(all(not t.done() for t in tasks))
        self.assertEqual(len(self.batcher._active), 2)
        self.assertFalse(original.batch.done.is_set())
        active.cleanup_temporary_local_files()
        members = await asyncio.wait_for(asyncio.gather(*tasks), 0.3)
        self.assertEqual(len(members[0].batch.members), 5)
        self.assertTrue(all(m.batch is members[0].batch for m in members))
        self.assertIs(members[-1].batch.owner, members[-1])
        for e in new_events:
            e.cleanup_temporary_local_files()
        self.assertTrue(original.batch.done.is_set())
        self.assertEqual(self.batcher._running, {})

    async def test_cancelled_backlog_releases_its_media_and_allows_next_batch(self):
        active, waiting, next_event = event(), event(), event()
        await self.batcher.join(active, 0, 0)
        pending = asyncio.create_task(self.batcher.join(waiting, 0, 0))
        await asyncio.sleep(0)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        waiting.cleanup_temporary_local_files()
        self.assertEqual(waiting.cleanups, ["cleaned"])
        next_task = asyncio.create_task(self.batcher.join(next_event, 0, 0))
        await asyncio.sleep(0)
        self.assertFalse(next_task.done())
        active.cleanup_temporary_local_files()
        await asyncio.wait_for(next_task, 0.2)
        next_event.cleanup_temporary_local_files()
        self.assertEqual(self.batcher._active, set())
        self.assertEqual(self.batcher._running, {})

    async def test_different_sessions_collect_independently(self):
        first, other = event("a"), event("b")
        one = await self.batcher.join(first, 0, 0)
        two = await asyncio.wait_for(self.batcher.join(other, 0, 0), 0.2)
        self.assertIsNot(one.batch, two.batch)
        first.cleanup_temporary_local_files()
        other.cleanup_temporary_local_files()

    async def test_cancelled_collector_does_not_leave_the_session_blocked(self):
        first = event()
        pending = asyncio.create_task(self.batcher.join(first, 1, 3))
        await asyncio.sleep(0)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        first.cleanup_temporary_local_files()
        second = event()
        await asyncio.wait_for(self.batcher.join(second, 0, 0), 0.2)
        second.cleanup_temporary_local_files()
        self.assertEqual(first.cleanups, ["cleaned"])

    async def test_shutdown_wakes_collectors_and_releases_temporary_media(self):
        e = event()
        task = asyncio.create_task(self.batcher.join(e, 1, 3))
        await asyncio.sleep(0)
        await self.batcher.close()
        with self.assertRaises(asyncio.CancelledError):
            await task
        e.cleanup_temporary_local_files()
        self.assertEqual(e.cleanups, ["cleaned"])

    def test_delay_defaults_and_invalid_values(self):
        self.assertEqual(merge_delays({}), (1, 3))
        self.assertEqual(
            merge_delays({"merge_idle_seconds": 2.5, "merge_max_wait_seconds": 8}),
            (2.5, 8),
        )
        for value in [True, None, "bad", -1, 61, math.inf, math.nan]:
            for key in ["merge_idle_seconds", "merge_max_wait_seconds"]:
                with self.subTest(value=value, key=key), self.assertRaises(ValueError):
                    merge_delays({key: value})

    async def test_mention_immediately_claims_collecting_and_ready_unjudged_messages(
        self,
    ):
        ready_event, waiting_event, at_event = event(), event(), event()
        ready = await self.batcher.join(ready_event, 0, 0)
        pending = asyncio.create_task(self.batcher.join(waiting_event, 30, 60))
        await asyncio.sleep(0)
        at = await asyncio.wait_for(
            self.batcher.join(at_event, 30, 60, immediate=True), 0.2
        )
        waiting = await asyncio.wait_for(pending, 0.2)
        self.assertIs(ready.batch, at.batch)
        self.assertIs(waiting.batch, at.batch)
        self.assertEqual(len(at.batch.members), 3)
        self.assertFalse(self.batcher.begin(ready))
        self.assertTrue(self.batcher.begin(at))
        for e in [ready_event, waiting_event, at_event]:
            e.cleanup_temporary_local_files()
            self.assertEqual(e.cleanups, ["cleaned"] if e is at_event else [])
        self.assertEqual(ready_event.cleanups, ["cleaned"])
        self.assertEqual(waiting_event.cleanups, ["cleaned"])

    async def test_mention_does_not_take_messages_that_started_judgment(self):
        previous, at_event = event(), event()
        old = await self.batcher.join(previous, 0, 0)
        self.assertTrue(self.batcher.begin(old))
        task = asyncio.create_task(
            self.batcher.join(at_event, 60, 60, immediate=True)
        )
        await asyncio.sleep(0.02)
        self.assertFalse(task.done())
        previous.cleanup_temporary_local_files()
        at = await asyncio.wait_for(task, 0.2)
        self.assertIsNot(old.batch, at.batch)
        self.assertEqual(at.batch.members, [at])
        at_event.cleanup_temporary_local_files()
        self.assertTrue(old.batch.done.is_set())
