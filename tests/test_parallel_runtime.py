"""Exercise scoped native locks and out-of-order, real history serialization."""

import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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

    def register_inputs(self, event, messages):
        runner = SimpleNamespace(
            run_context=SimpleNamespace(
                context=SimpleNamespace(event=event), messages=messages
            )
        )
        with patch.object(self.runtime.bridge, "register"):
            internal.register_active_runner(event.unified_msg_origin, runner)

    async def test_native_locks_serialize_batches_and_other_events(
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

        self.assertEqual(await group(True), 1)
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
                messages.append(Message.model_validate(await req.assemble_context()))
                self.register_inputs(event, messages)
                messages += [
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
            self.assertIs(internal.session_lock_manager, self.original_lock)
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
            ]
            self.register_inputs(event, messages)
            messages.append(Message(role="assistant", content="reply"))
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

    async def test_tool_images_survive_concurrent_merges_and_repeated_saves(self):
        initial = copy.deepcopy(self.saved)
        first, later = make_event(), make_event()
        self.runtime.enter(first)
        try:
            req = ProviderRequest(
                prompt="first batch",
                contexts=copy.deepcopy(initial),
                conversation=SimpleNamespace(cid="conv", token_usage=0),
            )
            await self.runtime.prepare(first, req, len(initial))
            messages = [Message.model_validate(m) for m in req.contexts]
            messages.append(Message.model_validate(await req.assemble_context()))
            self.register_inputs(first, messages)
            self.runtime.enter(later)
            req_later = ProviderRequest(
                prompt="later batch",
                contexts=copy.deepcopy(initial),
                conversation=SimpleNamespace(cid="conv", token_usage=0),
            )
            await self.runtime.prepare(later, req_later, len(initial))
            admitted = copy.deepcopy(self.saved)
            for index in range(2):
                messages.extend(
                    [
                        Message(
                            role="assistant",
                            tool_calls=[
                                {
                                    "id": f"view-{index}",
                                    "type": "function",
                                    "function": {
                                        "name": "meme_view",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        ),
                        Message(
                            role="tool",
                            tool_call_id=f"view-{index}",
                            content=f"metadata-{index}",
                        ),
                        Message(
                            role="user",
                            content=[
                                {
                                    "type": "text",
                                    "text": f"[Image from tool meme_view {index}]",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,AA{index}="
                                    },
                                },
                            ],
                        ),
                        Message(role="assistant", content=f"reviewed-{index}"),
                    ]
                )
                for _ in range(2):
                    await internal.InternalAgentSubStage._save_to_history(
                        SimpleNamespace(),
                        first,
                        req,
                        LLMResponse(
                            role="assistant", completion_text=f"reviewed-{index}"
                        ),
                        messages,
                        None,
                    )
                expected_outputs = [
                    m.model_dump() for m in messages[-4 * (index + 1) :]
                ]
                self.assertEqual(self.saved, admitted + expected_outputs)
            self.assertEqual(self.saved[:1], initial)
            third = make_event()
            self.runtime.enter(third)
            try:
                follow_up = ProviderRequest(
                    prompt="what did that picture mean?",
                    contexts=copy.deepcopy(initial),
                    conversation=SimpleNamespace(cid="conv", token_usage=0),
                )
                await self.runtime.prepare(third, follow_up, len(initial))
                self.assertEqual(follow_up.contexts, self.saved[:-1])
                self.assertIn(
                    "data:image/png;base64,AA0=", json.dumps(follow_up.contexts)
                )
                self.assertIn(
                    "data:image/png;base64,AA1=", json.dumps(follow_up.contexts)
                )
            finally:
                third.cleanup_temporary_local_files()
        finally:
            later.cleanup_temporary_local_files()
            first.cleanup_temporary_local_files()

    async def test_compacted_prefix_and_temporary_hints_keep_the_real_input_boundary(
        self,
    ):
        from astrbot.core.agent.message import TextPart

        first, later = make_event(), make_event()
        initial = copy.deepcopy(self.saved)
        self.runtime.enter(first)
        try:
            req = ProviderRequest(
                prompt="first batch",
                contexts=copy.deepcopy(initial),
                conversation=SimpleNamespace(cid="conv", token_usage=0),
            )
            await self.runtime.prepare(first, req, len(initial))
            current = Message.model_validate(await req.assemble_context())
            self.register_inputs(first, [current])
            if isinstance(current.content, str):
                current.content = [TextPart(text=current.content)]
            current.content.append(TextPart(text="temporary hint").mark_as_temp())
            self.runtime.enter(later)
            req_later = ProviderRequest(
                prompt="later batch",
                contexts=copy.deepcopy(initial),
                conversation=SimpleNamespace(cid="conv", token_usage=0),
            )
            await self.runtime.prepare(later, req_later, len(initial))
            admitted = copy.deepcopy(self.saved)
            hidden = Message(role="user", content="internal temporary message")
            hidden._no_save = True
            messages = [
                Message(role="system", content="PERSONA"),
                Message(role="user", content="native summary"),
                current,
                hidden,
                Message(role="user", content="[Image from tool meme_view]"),
                Message(role="assistant", content="reply"),
            ]
            first.set_extra("llm_checkpoint_id", "offline-checkpoint")
            for _ in range(2):
                await internal.InternalAgentSubStage._save_to_history(
                    SimpleNamespace(),
                    first,
                    req,
                    LLMResponse(role="assistant", completion_text="reply"),
                    messages,
                    None,
                )
            self.assertEqual(self.saved[: len(admitted)], admitted)
            self.assertEqual(
                [m["role"] for m in self.saved[len(admitted) :]],
                ["user", "assistant", "_checkpoint"],
            )
            self.assertNotIn("temporary", json.dumps(self.saved))
            self.assertNotIn("native summary", json.dumps(self.saved))
        finally:
            later.cleanup_temporary_local_files()
            first.cleanup_temporary_local_files()

    async def test_native_runner_tool_image_is_saved_and_visible_to_the_next_batch(
        self,
    ):
        from unittest.mock import AsyncMock, MagicMock

        from astrbot.core.agent.hooks import BaseAgentRunHooks
        from astrbot.core.agent.run_context import ContextWrapper
        from astrbot.core.agent.runners.tool_loop_agent_runner import (
            ToolLoopAgentRunner,
        )
        from astrbot.core.agent.tool import FunctionTool, ToolSet
        from astrbot.core.provider.entities import ProviderMeta
        from astrbot.core.provider.provider import Provider
        from mcp.types import CallToolResult, ImageContent, TextContent

        image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jewAAAABJRU5ErkJggg=="
        first, later = make_event(), make_event()
        initial = copy.deepcopy(self.saved)
        self.runtime.enter(first)
        try:
            req = ProviderRequest(
                prompt="first batch",
                system_prompt="PERSONA",
                contexts=copy.deepcopy(initial),
                conversation=SimpleNamespace(cid="conv", token_usage=0),
                func_tool=ToolSet(
                    [
                        FunctionTool(
                            name="meme_view", description="View a meme", parameters={}
                        )
                    ]
                ),
            )
            await self.runtime.prepare(first, req, len(initial))
            provider = MagicMock(spec=Provider)
            provider.provider_config = {
                "modalities": ["text", "image"],
                "max_context_tokens": 128000,
            }
            provider.meta.return_value = ProviderMeta("test", "test", "test", None)
            provider.get_model.return_value = "test"
            provider.text_chat = AsyncMock(
                side_effect=[
                    LLMResponse(
                        role="assistant",
                        tools_call_name=["meme_view"],
                        tools_call_args=[{}],
                        tools_call_ids=["native-view"],
                    ),
                    LLMResponse(role="assistant", completion_text="reviewed image"),
                ]
            )

            async def execute(**kwargs):
                yield CallToolResult(
                    content=[
                        TextContent(type="text", text="meme metadata"),
                        ImageContent(type="image", data=image, mimeType="image/png"),
                    ]
                )

            runner = ToolLoopAgentRunner()
            await runner.reset(
                provider=provider,
                request=req,
                run_context=ContextWrapper(context=SimpleNamespace(event=first)),
                tool_executor=SimpleNamespace(execute=execute),
                agent_hooks=BaseAgentRunHooks(),
            )
            self.register_inputs(first, runner.run_context.messages)
            async for _ in runner.step():
                pass
            self.assertEqual(runner.run_context.messages[-1].role, "user")
            self.assertIn(
                "image_url", json.dumps(runner.run_context.messages[-1].model_dump())
            )
            self.runtime.enter(later)
            req_later = ProviderRequest(
                prompt="later batch",
                contexts=copy.deepcopy(initial),
                conversation=SimpleNamespace(cid="conv", token_usage=0),
            )
            await self.runtime.prepare(later, req_later, len(initial))
            admitted = copy.deepcopy(self.saved)
            async for _ in runner.step():
                pass
            for _ in range(2):
                await internal.InternalAgentSubStage._save_to_history(
                    SimpleNamespace(),
                    first,
                    req,
                    runner.get_final_llm_resp(),
                    runner.run_context.messages,
                    runner.stats,
                )
            outputs = self.saved[len(admitted) :]
            self.assertEqual(
                [m["role"] for m in outputs], ["assistant", "tool", "user", "assistant"]
            )
            self.assertEqual(outputs[1]["tool_call_id"], "native-view")
            self.assertIn("data:image/png;base64," + image, json.dumps(outputs[2]))
            self.assertIn(
                "image_url", json.dumps(provider.text_chat.call_args.kwargs["contexts"])
            )
            third = make_event()
            self.runtime.enter(third)
            try:
                follow_up = ProviderRequest(
                    prompt="what did that picture mean?",
                    contexts=copy.deepcopy(initial),
                    conversation=SimpleNamespace(cid="conv", token_usage=0),
                )
                await self.runtime.prepare(third, follow_up, len(initial))
                self.assertIn(
                    "data:image/png;base64," + image, json.dumps(follow_up.contexts)
                )
                self.assertIn("native-view", json.dumps(follow_up.contexts))
            finally:
                third.cleanup_temporary_local_files()
        finally:
            later.cleanup_temporary_local_files()
            first.cleanup_temporary_local_files()
