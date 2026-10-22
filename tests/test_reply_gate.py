"""Exercise real AstrBot dispatch and runner code with a local decision server."""

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import web
from astrbot.api.message_components import At, Plain
from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.message import ImageURLPart, TextPart
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.pipeline.context import PipelineContext
from astrbot.core.pipeline.context_utils import call_event_hook
from astrbot.core.pipeline.process_stage.stage import ProcessStage, StarRequestSubStage
from astrbot.core.pipeline.waking_check.stage import WakingCheckStage
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.platform.astrbot_message import AstrBotMessage, MessageMember
from astrbot.core.platform.message_type import MessageType
from astrbot.core.platform.platform_metadata import PlatformMetadata
from astrbot.core.provider.entities import LLMResponse, ProviderMeta, ProviderRequest
from astrbot.core.provider.provider import Provider
from astrbot.core.star.star import star_map
from astrbot.core.star.star_handler import EventType, star_handlers_registry

from astrbot_plugin_reply_gate.main import ReplyGate
from astrbot_plugin_reply_gate.shared_context import (
    DEFAULT_INSTRUCTIONS,
    GROUP_BEGIN,
    GROUP_END,
    MODEL,
    ContextLimitError,
    context_cost,
    fit_context,
    parse_probability,
)


def make_event(
    text="这件事怎么看？", mention=False, sender="user-1", session="group-1"
):
    message = AstrBotMessage()
    message.type = MessageType.GROUP_MESSAGE
    message.self_id = "bot-1"
    message.session_id = session
    message.message_id = "test-message"
    message.group_id = session
    message.sender = MessageMember(user_id=sender, nickname="测试用户")
    message.message = ([At(qq="bot-1")] if mention else []) + [Plain(text)]
    message.message_str = text
    return AstrMessageEvent(
        text, message, PlatformMetadata("test", "isolated", "test"), session
    )


class ContextTests(unittest.TestCase):
    def test_larger_text_window_retains_more_history(self):
        history = [{"role": "user", "content": "old text " * 4000}]
        current = {"role": "user", "content": "latest"}
        window, payload = fit_context(
            history, current, "PERSONA", DEFAULT_INSTRUCTIONS, 60000, True
        )
        self.assertEqual(window.history, history)
        self.assertLessEqual(context_cost(payload), 60000)
        with self.assertRaises(ValueError):
            fit_context(history, current, "PERSONA", DEFAULT_INSTRUCTIONS, 65536, True)

    def test_old_media_round_is_removed_from_both_model_contexts(self):
        history = [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "earlier answer"},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "test-image"}}],
            },
            {"role": "assistant", "content": "image answer"},
            {"role": "user", "content": "recent text"},
            {"role": "assistant", "content": "recent answer"},
        ]
        current = {"role": "user", "content": "latest"}
        window, payload = fit_context(
            history, current, "PERSONA", DEFAULT_INSTRUCTIONS, 60000, True
        )
        self.assertEqual(window.dropped_messages, 4)
        self.assertEqual(window.history, history[4:])
        self.assertEqual(payload["state"]["messages"], window.messages())
        self.assertNotIn("images", payload)

    def test_current_media_is_rejected_instead_of_hidden_from_decision(self):
        with self.assertRaisesRegex(ContextLimitError, "unsupported_media"):
            fit_context(
                [],
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "test-image"}}
                    ],
                },
                "PERSONA",
                DEFAULT_INSTRUCTIONS,
                60000,
                True,
            )

    def test_same_model_family_snapshot_is_accepted(self):
        self.assertEqual(
            parse_probability(
                {
                    "model": f"{MODEL}-2026-09-24",
                    "answers": {"should_reply": {"type": "noul", "noul": 0.75}},
                }
            ),
            0.75,
        )

    def test_old_rounds_go_first_and_tools_stay_paired(self):
        history = [
            {"role": "user", "content": "OLD" * 1800},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "t", "content": "old result"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ]
        current = {
            "role": "user",
            "content": [
                {"type": "text", "text": "latest"},
                {"type": "text", "text": "MEMORY: remember my cat", "_no_save": True},
            ],
        }
        window, payload = fit_context(
            history, current, "PERSONA", DEFAULT_INSTRUCTIONS, 4000, True
        )
        self.assertEqual(window.dropped_messages, 4)
        self.assertEqual(len(window.history), 2)
        self.assertEqual(window.system_prompt, "PERSONA")
        self.assertIn("MEMORY: remember my cat", json.dumps(payload))
        self.assertLessEqual(context_cost(payload), 4000)
        self.assertNotIn("_no_save", json.dumps(payload))

    def test_old_group_records_are_trimmed_inside_extra_block(self):
        records = "\n".join(
            f"[person/12:00:{n:02d}]: {'OLD' if n == 0 else 'recent'} {'x' * 450}"
            for n in range(10)
        )
        current = {
            "role": "user",
            "content": [
                {"type": "text", "text": "latest"},
                {
                    "type": "text",
                    "text": "<system_reminder>"
                    + GROUP_BEGIN
                    + records
                    + GROUP_END
                    + "</system_reminder>",
                },
                {"type": "text", "text": "PINNED_MEMORY"},
            ],
        }
        window, payload = fit_context(
            [], current, "PERSONA", DEFAULT_INSTRUCTIONS, 4000, True
        )
        encoded = json.dumps(payload)
        self.assertGreater(window.dropped_group_records, 0)
        self.assertNotIn("12:00:00", encoded)
        self.assertIn("12:00:09", encoded)
        self.assertIn("PINNED_MEMORY", encoded)

    def test_current_text_keeps_recent_tail_after_history_is_exhausted(self):
        current = {
            "role": "user",
            "content": [
                {"type": "text", "text": "OLD" * 4000 + "NEWEST_TAIL"},
                {"type": "text", "text": "PINNED_MEMORY"},
            ],
        }
        window, payload = fit_context(
            [], current, "PERSONA", DEFAULT_INSTRUCTIONS, 4000, True
        )
        self.assertTrue(window.shortened_current)
        self.assertIn("NEWEST_TAIL", json.dumps(payload))
        self.assertIn("PINNED_MEMORY", json.dumps(payload))
        self.assertLessEqual(context_cost(payload), 4000)

    def test_protected_content_is_never_silently_cut(self):
        with self.assertRaises(ContextLimitError):
            fit_context(
                [],
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                "PERSONA" * 5000,
                DEFAULT_INSTRUCTIONS,
                4000,
                True,
            )

    def test_bad_probabilities_and_silent_model_substitution_are_rejected(self):
        for value in [True, "0.9", -1, 2, float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_probability({"answers": {"should_reply": {"noul": value}}})
        with self.assertRaises(ValueError):
            parse_probability(
                {"model": "multilingual", "answers": {"should_reply": {"noul": 0.9}}}
            )


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.reply_calls = []
        self.probability = 0.9
        self.status = 200
        self.histories = {}
        self.response_body = None
        self.delay = 0
        self.extra_parts = [TextPart(text="MEMORY: user likes cats").mark_as_temp()]
        app = web.Application()
        app.router.add_post("/compatible-mode/v1/systemone", self.serve_decision)
        self.server = web.AppRunner(app)
        await self.server.setup()
        site = web.TCPSite(self.server, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.endpoint_patch = patch(
            "astrbot_plugin_reply_gate.main.ENDPOINT",
            f"http://127.0.0.1:{port}/compatible-mode/v1/systemone",
        )
        self.endpoint_patch.start()
        self.config = {
            "enabled": True,
            "bailian_api_key": "test-only-fake-key",
            "context_budget": 4000,
        }
        self.astr_config = copy.deepcopy(DEFAULT_CONFIG)
        self.context = SimpleNamespace(
            get_config=lambda *args, **kwargs: self.astr_config,
            conversation_manager=SimpleNamespace(
                update_conversation=AsyncMock(side_effect=self.save_history)
            ),
        )
        self.plugin = ReplyGate(self.context, self.config)
        self.metadata = star_map[ReplyGate.__module__]
        self.metadata.name = "astrbot_plugin_reply_gate"
        self.metadata.author = "eeo"
        self.old_handlers = star_handlers_registry._handlers
        handlers = []
        for handler in self.old_handlers:
            if handler.handler_module_path == ReplyGate.__module__:
                bound = copy.copy(handler)
                bound.handler = getattr(self.plugin, handler.handler_name)
                handlers.append(bound)
        star_handlers_registry._handlers = handlers
        self.session_patch = patch(
            "astrbot.core.pipeline.waking_check.stage.SessionPluginManager.filter_handlers_by_session",
            AsyncMock(side_effect=lambda event, handlers: handlers),
        )
        self.session_patch.start()
        self.pipeline_context = PipelineContext(
            self.astr_config, SimpleNamespace(context=self.context), "test", None
        )
        self.waking = WakingCheckStage()
        await self.waking.initialize(self.pipeline_context)
        self.waking._umo_auto_name_recorder = SimpleNamespace(
            schedule=lambda event: None
        )
        self.process_stage = ProcessStage()
        self.process_stage.ctx = self.pipeline_context
        self.process_stage.star_request_sub_stage = StarRequestSubStage()
        await self.process_stage.star_request_sub_stage.initialize(
            self.pipeline_context
        )
        self.process_stage.agent_sub_stage = SimpleNamespace(
            process=self.run_reply_stage
        )

    async def asyncTearDown(self):
        await self.plugin.terminate()
        await self.server.cleanup()
        star_handlers_registry._handlers = self.old_handlers
        self.session_patch.stop()
        self.endpoint_patch.stop()

    async def serve_decision(self, request):
        import asyncio

        self.requests.append(
            {
                "body": await request.json(),
                "authorization": request.headers.get("Authorization"),
                "path": request.path_qs,
            }
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        return web.json_response(
            self.response_body
            if self.response_body is not None
            else {
                "model": MODEL,
                "answers": {"should_reply": {"type": "noul", "noul": self.probability}},
            },
            status=self.status,
        )

    async def save_history(self, origin, cid, history, **kwargs):
        self.histories[origin] = copy.deepcopy(history)

    async def run_reply_stage(self, event):
        req = ProviderRequest(
            prompt=event.message_str,
            system_prompt="PERSONA: you are a helpful group member.",
            contexts=copy.deepcopy(self.histories.get(event.unified_msg_origin, [])),
            extra_user_content_parts=copy.deepcopy(self.extra_parts),
            conversation=SimpleNamespace(cid=event.unified_msg_origin, token_usage=0),
        )
        if await call_event_hook(event, EventType.OnLLMRequestEvent, req):
            return
        provider = MagicMock(spec=Provider)
        provider.provider_config = {
            "id": "fake-reply",
            "max_context_tokens": 128000,
            "modalities": ["text", "image"],
        }
        provider.get_model.return_value = "fake-reply"
        provider.meta.return_value = ProviderMeta(
            "fake-reply", "fake-reply", "test", None
        )

        async def record_reply(**kwargs):
            self.reply_calls.append({"contexts": copy.deepcopy(kwargs["contexts"])})
            return LLMResponse(role="assistant", completion_text="这是测试回复")

        provider.text_chat = AsyncMock(side_effect=record_reply)
        runner = ToolLoopAgentRunner()
        await runner.reset(
            provider=provider,
            request=req,
            run_context=ContextWrapper(context=SimpleNamespace(event=event)),
            tool_executor=MagicMock(),
            agent_hooks=BaseAgentRunHooks(),
        )
        async for result in runner.step():
            yield result
        event.call_llm = True

    async def dispatch(self, event):
        await self.waking.process(event)
        if not event.is_stopped():
            async for _ in self.process_stage.process(event):
                pass
        return event

    async def test_mention_bypasses_missing_key_and_the_http_service(self):
        self.config["bailian_api_key"] = ""
        event = await self.dispatch(make_event(mention=True))
        self.assertEqual(len(self.requests), 0)
        self.assertEqual(len(self.reply_calls), 1)
        self.assertEqual(event.get_extra("reply_gate_decision")["source"], "mention")

    async def test_allowed_reply_and_decision_use_same_actual_provider_context(self):
        event = make_event()
        self.histories[event.unified_msg_origin] = [
            {"role": "user", "content": "OLD" * 2000},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "recent"},
            {"role": "assistant", "content": "recent reply"},
        ]
        await self.dispatch(event)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.reply_calls), 1)
        decision = self.requests[0]["body"]
        reply = self.reply_calls[0]
        self.assertEqual(decision["model"], MODEL)
        self.assertEqual(self.requests[0]["authorization"], "test-only-fake-key")
        actual = reply["contexts"]
        self.assertEqual(decision["state"]["messages"], actual)
        self.assertNotIn("OLD", json.dumps(actual))
        self.assertIn("MEMORY", json.dumps(actual))

    async def test_complete_bearer_authorization_is_not_prefixed_twice(self):
        self.config["bailian_api_key"] = "Bearer test-only-fake-key"
        await self.dispatch(make_event())
        self.assertEqual(self.requests[0]["authorization"], "Bearer test-only-fake-key")
        self.assertEqual(len(self.reply_calls), 1)

    async def test_configured_endpoint_is_used(self):
        self.config["bailian_api_url"] = self.endpoint_patch.new + "?configured=1"
        await self.dispatch(make_event())
        self.assertEqual(
            self.requests[0]["path"], "/compatible-mode/v1/systemone?configured=1"
        )
        self.assertEqual(len(self.reply_calls), 1)

    async def test_missing_endpoint_does_not_send_a_request(self):
        self.config["bailian_api_url"] = ""
        event = await self.dispatch(make_event())
        self.assertEqual(self.requests, [])
        self.assertEqual(self.reply_calls, [])
        self.assertEqual(
            event.get_extra("reply_gate_decision")["reason"], "api_url_missing"
        )

    async def test_current_image_keeps_silent_but_mention_can_still_reply(self):
        self.extra_parts.append(
            ImageURLPart(
                image_url=ImageURLPart.ImageURL(url="data:image/png;base64,dGVzdA==")
            )
        )
        event = await self.dispatch(make_event("看这张图"))
        self.assertEqual(self.requests, [])
        self.assertEqual(self.reply_calls, [])
        self.assertEqual(
            event.get_extra("reply_gate_decision")["reason"], "unsupported_media"
        )
        await self.dispatch(make_event("看这张图", mention=True))
        self.assertEqual(self.requests, [])
        self.assertEqual(len(self.reply_calls), 1)

    async def test_text_after_an_unanswered_image_can_be_judged(self):
        self.extra_parts.append(
            ImageURLPart(
                image_url=ImageURLPart.ImageURL(url="data:image/png;base64,dGVzdA==")
            )
        )
        await self.dispatch(make_event("看这张图"))
        self.extra_parts.pop()
        await self.dispatch(make_event("继续聊文字"))
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.reply_calls), 1)
        self.assertEqual(
            self.requests[0]["body"]["state"]["messages"],
            self.reply_calls[0]["contexts"],
        )

    async def test_declined_message_is_visible_to_the_next_decision(self):
        self.probability = 0.1
        first = await self.dispatch(make_event("I mentioned a blue cat"))
        self.assertTrue(first.is_stopped())
        self.assertEqual(len(self.reply_calls), 0)
        self.probability = 0.9
        await self.dispatch(make_event("What colour was it?"))
        self.assertEqual(len(self.requests), 2)
        self.assertIn("blue cat", json.dumps(self.requests[-1]["body"]))
        saved = json.dumps(self.histories[first.unified_msg_origin])
        self.assertNotIn("MEMORY", saved)
        self.assertEqual(len(self.reply_calls), 1)

    async def test_error_does_not_fall_through_to_normal_reply(self):
        self.status = 503
        event = await self.dispatch(make_event())
        self.assertTrue(event.is_stopped())
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.reply_calls), 0)
        await self.dispatch(make_event(mention=True))
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.reply_calls), 1)

    async def test_missing_key_does_not_send_a_request(self):
        self.config["bailian_api_key"] = ""
        await self.dispatch(make_event())
        self.assertEqual(len(self.requests), 0)
        self.assertEqual(len(self.reply_calls), 0)

    async def test_legacy_platform_key_is_never_sent_to_bailian(self):
        self.config.pop("bailian_api_key")
        self.config["api_key"] = "legacy-test-only-fake-key"
        event = await self.dispatch(make_event())
        self.assertEqual(self.requests, [])
        self.assertEqual(self.reply_calls, [])
        self.assertEqual(
            event.get_extra("reply_gate_decision")["reason"], "api_key_missing"
        )

    async def test_invalid_output_and_substituted_model_stay_silent(self):
        self.response_body = {
            "model": "multilingual",
            "answers": {"should_reply": {"noul": 0.99}},
        }
        event = await self.dispatch(make_event())
        self.assertTrue(event.is_stopped())
        self.assertEqual(len(self.reply_calls), 0)

    async def test_histories_are_isolated_by_conversation(self):
        self.probability = 0.1
        await self.dispatch(make_event("private topic", session="group-a"))
        await self.dispatch(make_event("other topic", session="group-b"))
        self.assertNotIn("private topic", json.dumps(self.requests[-1]["body"]))

    async def test_own_messages_do_not_trigger_a_decision(self):
        await self.dispatch(make_event(sender="bot-1"))
        self.assertEqual(len(self.requests), 0)
        self.assertEqual(len(self.reply_calls), 0)
