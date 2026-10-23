"""Exercise scoped native locks and out-of-order, real history serialization."""

import asyncio
import copy
import json
import unittest
from types import SimpleNamespace

from astrbot.core.agent.message import Message
from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot_plugin_reply_gate.parallel_runtime import ParallelRuntime


def make_event(text="hello"):
    extras = {}
    return SimpleNamespace(
        unified_msg_origin="group",
        set_extra=lambda key, value: extras.__setitem__(key, value),
        get_extra=lambda key, default=None: extras.get(key, default),
        cleanup_temporary_local_files=lambda: None,
    )


class ParallelRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.saved = [{"role": "user", "content": "FULL_OLD_HISTORY" * 10000}]

        async def read(origin, cid):
            return SimpleNamespace(
                cid=cid, history=json.dumps(self.saved), token_usage=0
            )

        async def write(origin, cid, *, history, **kwargs):
            await asyncio.sleep(0)
            self.saved = copy.deepcopy(history)

        self.context = SimpleNamespace(
            conversation_manager=SimpleNamespace(
                get_conversation=read, update_conversation=write
            )
        )
        self.original_lock = internal.session_lock_manager
        self.runtime = ParallelRuntime(self.context)

    async def asyncTearDown(self):
        self.runtime.close()
        self.assertIs(internal.session_lock_manager, self.original_lock)

    async def test_native_locks_allow_many_batches_but_other_events_still_serialize(
        self,
    ):
        async def group(parallel):
            active = peak = 0

            async def worker():
                nonlocal active, peak
                event = make_event("hello")
                if parallel:
                    self.runtime.enter(event)
                try:
                    async with internal.session_lock_manager.acquire_lock(
                        event.unified_msg_origin
                    ):
                        active += 1
                        peak = max(peak, active)
                        await asyncio.sleep(0.03)
                        active -= 1
                finally:
                    event.cleanup_temporary_local_files()

            await asyncio.gather(*(worker() for _ in range(12)))
            return peak

        self.assertEqual(await group(True), 12)
        self.assertEqual(await group(False), 1)

    async def test_reverse_completion_preserves_users_tools_and_all_old_history(self):
        initial = copy.deepcopy(self.saved)
        ready = [asyncio.Event() for _ in range(4)]
        release = [asyncio.Event() for _ in range(4)]
        requests = []

        async def worker(i):
            event = make_event(f"batch-{i}")
            self.runtime.enter(event)
            try:
                req = ProviderRequest(
                    prompt=f"batch-{i}",
                    contexts=copy.deepcopy(initial),
                    conversation=SimpleNamespace(cid="conv", token_usage=0),
                )
                await self.runtime.prepare(event, req, len(initial))
                requests.append((i, copy.deepcopy(req.contexts)))
                ready[i].set()
                await release[i].wait()
                messages = [Message.model_validate(m) for m in req.contexts]
                messages += [
                    Message.model_validate(await req.assemble_context()),
                    Message(
                        role="assistant",
                        content="checking",
                        tool_calls=[
                            {
                                "id": f"call-{i}",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    ),
                    Message(
                        role="tool", content=f"result-{i}", tool_call_id=f"call-{i}"
                    ),
                    Message(role="assistant", content=f"answer-{i}"),
                ]
                await internal.InternalAgentSubStage._save_to_history(
                    SimpleNamespace(),
                    event,
                    req,
                    LLMResponse(role="assistant", completion_text=f"answer-{i}"),
                    messages,
                    None,
                )
                # Live-mode native pipelines can save a completed turn twice.
                await internal.InternalAgentSubStage._save_to_history(
                    SimpleNamespace(),
                    event,
                    req,
                    LLMResponse(role="assistant", completion_text=f"answer-{i}"),
                    messages,
                    None,
                )
            finally:
                event.cleanup_temporary_local_files()

        tasks = []
        for i in range(4):
            tasks.append(asyncio.create_task(worker(i)))
            await asyncio.wait_for(ready[i].wait(), 1)
        for i, context in requests:
            for previous in range(i):
                self.assertIn(f"batch-{previous}", json.dumps(context))
        for i in reversed(range(4)):
            release[i].set()
            await asyncio.wait_for(tasks[i], 1)
        self.assertEqual(self.saved[:1], initial)
        self.assertEqual(len(self.saved), 1 + 4 + 4 * 3)
        self.assertEqual(
            [m["tool_call_id"] for m in self.saved if m["role"] == "tool"],
            [f"call-{i}" for i in reversed(range(4))],
        )
        for i in range(4):
            self.assertEqual(json.dumps(self.saved).count(f"batch-{i}"), 1)
            self.assertEqual(json.dumps(self.saved).count(f"answer-{i}"), 1)

    async def test_hot_reload_keeps_inflight_bridge_until_last_event_finishes(self):
        event = make_event("old run")
        self.runtime.enter(event)
        bridge = self.runtime.bridge
        self.runtime.close()
        replacement = ParallelRuntime(self.context)

        async def new_run():
            fresh = make_event("new run")
            replacement.enter(fresh)
            self.assertIs(replacement.bridge, bridge)
            fresh.cleanup_temporary_local_files()

        try:
            await asyncio.create_task(new_run())
            replacement.close()
            self.assertIsNot(internal.session_lock_manager, self.original_lock)
            event.cleanup_temporary_local_files()
            self.assertIs(internal.session_lock_manager, self.original_lock)
        finally:
            replacement.close()

    async def test_native_compacted_history_is_kept_without_concurrent_writes(self):
        event = make_event("new input")
        self.runtime.enter(event)
        try:
            req = ProviderRequest(
                prompt="new input",
                contexts=copy.deepcopy(self.saved),
                conversation=SimpleNamespace(cid="conv", token_usage=0),
            )
            await self.runtime.prepare(event, req, len(req.contexts))
            messages = [
                Message(role="system", content="PERSONA"),
                Message(role="user", content="native summary of older context"),
                Message.model_validate(await req.assemble_context()),
                Message(role="assistant", content="reply"),
            ]
            await internal.InternalAgentSubStage._save_to_history(
                SimpleNamespace(),
                event,
                req,
                LLMResponse(role="assistant", completion_text="reply"),
                messages,
                None,
            )
            self.assertEqual(len(self.saved), 3)
            self.assertEqual(
                self.saved[0]["content"], "native summary of older context"
            )
            self.assertEqual(self.saved[-1]["content"], "reply")
        finally:
            event.cleanup_temporary_local_files()
