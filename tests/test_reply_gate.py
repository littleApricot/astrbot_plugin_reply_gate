"""Exercise real AstrBot dispatch and runner code with a local decision server."""

import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from aiohttp import web
from astrbot.api import AstrBotConfig
from astrbot.api.message_components import At, Plain
from astrbot.core.agent.hooks import BaseAgentRunHooks
from astrbot.core.agent.message import ImageURLPart, TextPart
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.config.default import DEFAULT_CONFIG
from astrbot.core.pipeline.context import PipelineContext
from astrbot.core.pipeline.context_utils import call_event_hook
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage,
)
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
from openai import APIStatusError


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

    def test_old_media_round_is_removed_from_decision_window(self):
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
        self.vision_calls = []
        self.vision_decision = True
        self.vision_response = None
        self.prepared_requests = []
        self.request_snapshots = []
        self.probability = 0.9
        self.status = 200
        self.histories = {}
        self.response_body = None
        self.delay = 0
        self.reply_started = asyncio.Event()
        self.reply_release = None
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
            "merge_idle_seconds": 0,
            "merge_max_wait_seconds": 0,
        }
        self.astr_config = copy.deepcopy(DEFAULT_CONFIG)
        self.context = SimpleNamespace(
            get_config=lambda *args, **kwargs: self.astr_config,
            conversation_manager=SimpleNamespace(
                update_conversation=AsyncMock(side_effect=self.save_history)
            ),
        )
        self.vision_provider = MagicMock(spec=Provider)
        self.vision_provider.provider_config = {
            "id": "fake-reply",
            "modalities": ["text", "image"],
        }

        async def record_vision(**kwargs):
            self.vision_calls.append(copy.deepcopy(kwargs))
            return LLMResponse(
                role="assistant",
                completion_text=self.vision_response
                if self.vision_response is not None
                else json.dumps({"probability": float(self.vision_decision)}),
            )

        self.vision_provider.text_chat = AsyncMock(side_effect=record_vision)
        self.context.get_using_provider_async = AsyncMock(
            return_value=self.vision_provider
        )
        self.context.get_provider_by_id = MagicMock(return_value=self.vision_provider)
        self.plugin = ReplyGate(self.context, self.config)
        self.history_stage = InternalAgentSubStage()
        self.history_stage.conv_manager = self.context.conversation_manager
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
        if await call_event_hook(event, EventType.OnWaitingLLMRequestEvent):
            return
        req = ProviderRequest(
            prompt=event.message_str,
            model=event.get_extra("selected_model"),
            system_prompt="PERSONA: you are a helpful group member.",
            contexts=copy.deepcopy(self.histories.get(event.unified_msg_origin, [])),
            extra_user_content_parts=copy.deepcopy(self.extra_parts),
            conversation=SimpleNamespace(
                cid=event.unified_msg_origin, token_usage=1234
            ),
        )
        self.prepared_requests.append(req)
        self.request_snapshots.append(copy.deepcopy(req))
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
            self.reply_started.set()
            if self.reply_release is not None:
                await self.reply_release.wait()
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
        await self.history_stage._save_to_history(
            event,
            req,
            runner.get_final_llm_resp(),
            runner.run_context.messages,
            runner.stats,
        )
        event.call_llm = True

    async def dispatch(self, event):
        try:
            await self.waking.process(event)
            if not event.is_stopped():
                async for _ in self.process_stage.process(event):
                    pass
            return event
        finally:
            event.cleanup_temporary_local_files()

    async def test_mention_bypasses_missing_key_and_the_http_service(self):
        self.config["bailian_api_key"] = ""
        self.config["failure_default_reply"] = False
        event = await self.dispatch(make_event(mention=True))
        self.assertEqual(len(self.requests), 0)
        self.assertEqual(len(self.reply_calls), 1)
        self.assertEqual(event.get_extra("reply_gate_decision")["source"], "mention")

    async def test_prompt_upgrade_preserves_credentials_and_custom_rules(self):
        for original_prompt in ("", "CUSTOM policy\nkeep this wording"):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "plugin.json"
                original = {
                    "decision_instructions": original_prompt,
                    "bailian_api_key": "test-only-fake-key",
                    "bailian_api_url": "https://example.invalid/systemone",
                    "reply_threshold": 0.7,
                    "context_budget": 20000,
                }
                path.write_text(json.dumps(original))
                config = AstrBotConfig(
                    str(path), default_config=copy.deepcopy(original)
                )
                ReplyGate(self.context, config)
                saved = json.loads(path.read_text(encoding="utf-8-sig"))
                self.assertEqual(
                    saved.pop("decision_instructions"),
                    original_prompt or DEFAULT_INSTRUCTIONS,
                )
                self.assertEqual(
                    saved,
                    {k: v for k, v in original.items() if k != "decision_instructions"},
                )

    async def test_failure_policy_upgrade_defaults_open_and_preserves_saved_choice(
        self,
    ):
        schema = json.loads(
            (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text()
        )
        for previous in (None, False, True):
            with (
                self.subTest(previous=previous),
                tempfile.TemporaryDirectory() as folder,
            ):
                path = Path(folder) / "plugin.json"
                original = {
                    "decision_instructions": "MY_RULES",
                    "bailian_api_key": "test-only-fake-key",
                    "bailian_api_url": "https://example.invalid/systemone",
                    "reply_threshold": 0.7,
                    "context_budget": 20000,
                }
                if previous is not None:
                    original["failure_default_reply"] = previous
                path.write_text(json.dumps(original))
                config = AstrBotConfig(str(path), schema=schema)
                ReplyGate(self.context, config)
                saved = json.loads(path.read_text(encoding="utf-8-sig"))
                self.assertIs(
                    saved["failure_default_reply"],
                    True if previous is None else previous,
                )
                self.assertEqual(
                    {key: saved[key] for key in original if key != "reply_threshold"},
                    {
                        key: value
                        for key, value in original.items()
                        if key != "reply_threshold"
                    },
                )
                self.assertNotIn("reply_threshold", saved)
                self.assertEqual(saved["reply_activity"], 0.5)

    async def test_configured_prompt_reaches_text_and_image_judges_only(self):
        custom = "CUSTOM_POLICY: 按当前人格决定是否接话。"
        self.config["decision_instructions"] = custom
        await self.dispatch(make_event("文字消息"))
        question = self.requests[-1]["body"]["questions"]["should_reply"]
        self.assertEqual(question["instructions"], custom)
        self.assertNotIn("回应日常社交互动", json.dumps(question, ensure_ascii=False))
        self.extra_parts.append(
            ImageURLPart(
                image_url=ImageURLPart.ImageURL(url="data:image/png;base64,dGVzdA==")
            )
        )
        await self.dispatch(make_event("图片消息"))
        call = self.vision_calls[-1]
        self.assertIn(custom, call["system_prompt"])
        self.assertIn(custom, call["contexts"][-1]["content"])
        self.assertNotIn(DEFAULT_INSTRUCTIONS, call["system_prompt"])
        for request in self.prepared_requests:
            self.assertNotIn(custom, request.system_prompt)
            self.assertNotIn(custom, str(request.contexts))
        self.assertEqual(self.prepared_requests, self.request_snapshots)

    async def test_mention_without_text_passes_real_empty_message_guard(self):
        self.config["failure_default_reply"] = False
        e = make_event("   ", mention=True)
        e.message_obj.message = [At(qq="bot-1")]
        original_chain = copy.deepcopy(e.get_messages())
        await self.plugin.observe_message(e)
        self.assertTrue(e.message_str.strip())
        self.assertEqual(e.get_messages(), original_chain)
        stage = InternalAgentSubStage()
        stage.streaming_response = False
        stage.show_reasoning = False
        waiting = AsyncMock(return_value=True)
        with (
            patch(
                "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal.try_capture_follow_up",
                return_value=None,
            ),
            patch(
                "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal.call_event_hook",
                waiting,
            ),
        ):
            async for _ in stage.process(e, ""):
                pass
        waiting.assert_awaited_once_with(e, EventType.OnWaitingLLMRequestEvent)
        self.config["bailian_api_key"] = ""
        await self.dispatch(e)
        self.assertEqual(self.requests, [])
        self.assertEqual(len(self.reply_calls), 1)
        self.assertEqual(e.get_extra("reply_gate_decision")["source"], "mention")

    async def test_empty_mentions_of_others_are_not_fabricated(self):
        e = make_event("")
        e.message_obj.message = [At(qq="someone-else")]
        await self.plugin.observe_message(e)
        self.assertEqual(e.message_str, "")

    async def test_allowed_reply_keeps_full_context_and_saved_history(self):
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
        self.assertNotIn("OLD", json.dumps(decision))
        self.assertIn("recent", json.dumps(decision))
        self.assertIn("OLD", json.dumps(actual))
        self.assertIn("MEMORY", json.dumps(actual))
        self.assertEqual(self.prepared_requests[-1], self.request_snapshots[-1])
        saved = self.histories[event.unified_msg_origin]
        self.assertEqual(saved[:4], self.request_snapshots[-1].contexts)
        self.assertEqual(saved[-1]["role"], "assistant")
        self.assertNotIn("MEMORY", json.dumps(saved))

    async def test_decline_and_failed_open_preserve_full_history_and_current_text(self):
        for mode in ("decline", "http_error"):
            with self.subTest(mode=mode):
                self.probability = 0.1
                self.status = 503 if mode == "http_error" else 200
                event = make_event(
                    "CURRENT_START" + "中文" * 3000 + "LATEST", session=mode
                )
                original = [
                    {"role": "user", "content": "OLD" * 2000},
                    {"role": "assistant", "content": "old answer"},
                ]
                self.histories[event.unified_msg_origin] = copy.deepcopy(original)
                await self.dispatch(event)
                self.assertEqual(event.is_stopped(), mode == "decline")
                self.assertEqual(len(self.reply_calls), int(mode == "http_error"))
                saved = self.histories[event.unified_msg_origin]
                self.assertEqual(saved[:2], original)
                self.assertIn(event.message_str, json.dumps(saved, ensure_ascii=False))
                self.assertNotIn(
                    "Earlier part of this message omitted", json.dumps(saved)
                )
                self.assertEqual(self.prepared_requests[-1], self.request_snapshots[-1])

    async def test_group_records_are_trimmed_only_for_decision(self):
        records = "\n".join(
            f"[person/12:00:{n:02d}]: record-{n} {'x' * 450}" for n in range(10)
        )
        self.extra_parts.append(TextPart(text=GROUP_BEGIN + records + GROUP_END))
        event = await self.dispatch(make_event())
        decision = json.dumps(self.requests[-1]["body"])
        self.assertNotIn("record-0", decision)
        self.assertIn("record-9", decision)
        self.assertIn("record-0", json.dumps(self.reply_calls[-1]))
        self.assertIn("record-0", json.dumps(self.histories[event.unified_msg_origin]))
        self.assertEqual(self.prepared_requests[-1], self.request_snapshots[-1])

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
        self.assertEqual(len(self.reply_calls), 1)
        self.assertEqual(
            event.get_extra("reply_gate_decision")["reason"], "api_url_missing"
        )

    async def test_current_image_uses_reply_model_with_full_context_and_no_bailian_key(
        self,
    ):
        self.config["bailian_api_key"] = ""
        original = [
            {"role": "user", "content": "OLD" * 4000},
            {"role": "assistant", "content": "old reply"},
        ]
        self.histories[make_event().unified_msg_origin] = copy.deepcopy(original)
        self.extra_parts.append(
            ImageURLPart(
                image_url=ImageURLPart.ImageURL(url="data:image/png;base64,dGVzdA==")
            )
        )
        event = await self.dispatch(make_event("看这张图"))
        self.assertEqual(self.requests, [])
        self.assertEqual(len(self.reply_calls), 1)
        self.assertEqual(len(self.vision_calls), 1)
        self.assertEqual(self.vision_calls[0]["contexts"][:2], original)
        self.assertIn("data:image/png;base64,", json.dumps(self.vision_calls[0]))
        self.assertIn("MEMORY", json.dumps(self.vision_calls[0]))
        self.assertIsNone(self.vision_calls[0]["func_tool"])
        self.assertEqual(self.prepared_requests[-1], self.request_snapshots[-1])
        self.assertEqual(
            event.get_extra("reply_gate_decision")["source"], "reply_model_vision"
        )
        await self.dispatch(make_event("看这张图", mention=True))
        self.assertEqual(self.requests, [])
        self.assertEqual(len(self.reply_calls), 2)
        self.assertEqual(len(self.vision_calls), 1)

    async def test_image_decision_honors_selected_provider_and_model(self):
        self.extra_parts.append(
            ImageURLPart(
                image_url=ImageURLPart.ImageURL(url="data:image/png;base64,dGVzdA==")
            )
        )
        e = make_event("image")
        e.set_extra("selected_provider", "explicit-provider")
        e.set_extra("selected_model", "explicit-model")
        await self.dispatch(e)
        self.context.get_provider_by_id.assert_called_with("explicit-provider")
        self.assertEqual(self.vision_calls[0]["model"], "explicit-model")

    async def test_image_rejection_stops_but_invalid_output_passes_full_history(
        self,
    ):
        self.extra_parts.append(
            ImageURLPart(
                image_url=ImageURLPart.ImageURL(url="data:image/png;base64,dGVzdA==")
            )
        )
        for value in [False, "invalid"]:
            with self.subTest(value=value):
                self.vision_decision = False
                self.vision_response = "not json" if value == "invalid" else None
                e = make_event("original image message", session=str(value))
                original = [{"role": "user", "content": "OLD" * 3000}]
                self.histories[e.unified_msg_origin] = copy.deepcopy(original)
                await self.dispatch(e)
                self.assertEqual(e.is_stopped(), value is False)
                saved = self.histories[e.unified_msg_origin]
                self.assertEqual(saved[:1], original)
                self.assertIn("image_url", json.dumps(saved))
                self.assertEqual(self.prepared_requests[-1], self.request_snapshots[-1])
        self.assertEqual(self.requests, [])
        self.assertEqual(len(self.reply_calls), 1)

    async def test_text_after_an_unanswered_image_can_be_judged(self):
        self.vision_decision = False
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
        self.assertNotIn("image_url", json.dumps(self.requests[0]["body"]))
        self.assertIn("image_url", json.dumps(self.reply_calls[0]["contexts"]))
        saved = self.histories[make_event().unified_msg_origin]
        self.assertIn("image_url", json.dumps(saved))

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

    async def test_http_error_passes_to_normal_reply_and_mentions_still_bypass(self):
        self.status = 503
        event = await self.dispatch(make_event())
        self.assertFalse(event.is_stopped())
        self.assertEqual(
            event.get_extra("reply_gate_decision")["source"], "error_fallback"
        )
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.reply_calls), 1)
        await self.dispatch(make_event(mention=True))
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.reply_calls), 2)

    async def test_missing_key_does_not_send_a_request(self):
        self.config["bailian_api_key"] = ""
        await self.dispatch(make_event())
        self.assertEqual(len(self.requests), 0)
        self.assertEqual(len(self.reply_calls), 1)

    async def test_legacy_platform_key_is_never_sent_to_bailian(self):
        self.config.pop("bailian_api_key")
        self.config["api_key"] = "legacy-test-only-fake-key"
        event = await self.dispatch(make_event())
        self.assertEqual(self.requests, [])
        self.assertEqual(len(self.reply_calls), 1)
        self.assertEqual(
            event.get_extra("reply_gate_decision")["reason"], "api_key_missing"
        )

    async def test_substituted_model_result_is_not_trusted_and_passes(self):
        self.response_body = {
            "model": "multilingual",
            "answers": {"should_reply": {"noul": 0.99}},
        }
        event = await self.dispatch(make_event())
        self.assertFalse(event.is_stopped())
        self.assertEqual(len(self.reply_calls), 1)
        self.assertEqual(
            event.get_extra("reply_gate_decision")["source"], "error_fallback"
        )

    async def test_text_timeout_passes_original_request_once_and_saves_once(self):
        self.config["timeout_seconds"] = 0.01
        self.delay = 0.1
        e = make_event("timeout current")
        original = [{"role": "user", "content": "OLD" * 3000}]
        self.histories[e.unified_msg_origin] = copy.deepcopy(original)
        await self.dispatch(e)
        self.assertFalse(e.is_stopped())
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.reply_calls), 1)
        result = e.get_extra("reply_gate_decision")
        self.assertEqual(result["source"], "error_fallback")
        self.assertEqual(result["reason"], "TimeoutError")
        self.assertEqual(result["decision_source"], MODEL)
        self.assertEqual(result["timeout_seconds"], 0.01)
        self.assertEqual(self.prepared_requests[-1], self.request_snapshots[-1])
        saved = self.histories[e.unified_msg_origin]
        self.assertEqual(saved[:1], original)
        self.assertEqual(len(saved), 3)
        self.context.conversation_manager.update_conversation.assert_awaited_once()

    async def test_invalid_or_empty_decision_result_passes(self):
        for index, body in enumerate(
            [{}, {"answers": {}}, {"answers": {"should_reply": {"noul": None}}}]
        ):
            with self.subTest(body=body):
                self.response_body = body
                e = await self.dispatch(make_event(session=str(index)))
                self.assertFalse(e.is_stopped())
                self.assertEqual(
                    e.get_extra("reply_gate_decision")["source"], "error_fallback"
                )
        self.assertEqual(len(self.reply_calls), 3)

    async def test_text_failures_follow_policy_and_retain_full_unsent_turns(self):
        for fallback in (False, True):
            for mode, reason in (
                ("http", "decision_http_503"),
                ("empty", "invalid_decision_response"),
                ("timeout", "TimeoutError"),
                ("missing_key", "api_key_missing"),
                ("context", "context_budget_out_of_range"),
            ):
                with self.subTest(fallback=fallback, mode=mode):
                    self.config.update(
                        failure_default_reply=fallback,
                        bailian_api_key="" if mode == "missing_key" else "test-key",
                        timeout_seconds=0.01 if mode == "timeout" else 30,
                        context_budget=1 if mode == "context" else 4000,
                    )
                    self.status = 503 if mode == "http" else 200
                    self.response_body = {} if mode == "empty" else None
                    self.delay = 0.1 if mode == "timeout" else 0
                    event = make_event(
                        "CURRENT_START" + "中文" * 3000 + "LATEST",
                        session=f"text-{fallback}-{mode}",
                    )
                    original = [
                        {"role": "user", "content": "OLD" * 2000},
                        {"role": "assistant", "content": "old answer"},
                    ]
                    self.histories[event.unified_msg_origin] = copy.deepcopy(original)
                    replies_before = len(self.reply_calls)
                    saves_before = self.context.conversation_manager.update_conversation.await_count
                    with self.assertLogs(self.plugin.logger, level="WARNING") as logs:
                        await self.dispatch(event)
                    self.assertEqual(event.is_stopped(), not fallback)
                    result = event.get_extra("reply_gate_decision")
                    self.assertIs(result["reply"], fallback)
                    self.assertEqual(result["source"], "error_fallback")
                    self.assertEqual(result["reason"], reason)
                    self.assertIn(
                        "failed open" if fallback else "failed closed", str(logs.output)
                    )
                    self.assertEqual(
                        len(self.reply_calls) - replies_before, int(fallback)
                    )
                    self.assertEqual(
                        self.prepared_requests[-1], self.request_snapshots[-1]
                    )
                    saved = self.histories[event.unified_msg_origin]
                    self.assertEqual(saved[:2], original)
                    self.assertEqual(len(saved), 4 if fallback else 3)
                    self.assertIn(
                        event.message_str, json.dumps(saved, ensure_ascii=False)
                    )
                    self.assertNotIn("MEMORY", json.dumps(saved))
                    self.assertEqual(
                        self.context.conversation_manager.update_conversation.await_count
                        - saves_before,
                        1,
                    )

    async def test_image_failures_follow_policy_and_retain_full_unsent_images(self):
        self.extra_parts.append(
            ImageURLPart(
                image_url=ImageURLPart.ImageURL(url="data:image/png;base64,dGVzdA==")
            )
        )
        normal_judge = self.vision_provider.text_chat.side_effect
        for fallback in (False, True):
            for mode, reason in (
                ("invalid_json", "invalid_vision_json"),
                ("invalid_schema", "invalid_vision_schema"),
                ("empty", "empty_vision_decision"),
                ("timeout", "TimeoutError"),
                ("http", "provider_http_402"),
            ):
                with self.subTest(fallback=fallback, mode=mode):
                    self.config["failure_default_reply"] = fallback
                    self.vision_response = {
                        "invalid_json": "not json",
                        "invalid_schema": '{"should_reply":"false"}',
                        "empty": "",
                    }.get(mode)
                    self.vision_provider.text_chat.side_effect = (
                        TimeoutError()
                        if mode == "timeout"
                        else APIStatusError(
                            "private-error-body",
                            response=httpx.Response(
                                402,
                                request=httpx.Request(
                                    "POST", "https://example.invalid"
                                ),
                            ),
                            body={"private": "private-error-body"},
                        )
                        if mode == "http"
                        else normal_judge
                    )
                    event = make_event(
                        "original image", session=f"image-{fallback}-{mode}"
                    )
                    original = [{"role": "user", "content": "OLD" * 3000}]
                    self.histories[event.unified_msg_origin] = copy.deepcopy(original)
                    replies_before = len(self.reply_calls)
                    saves_before = self.context.conversation_manager.update_conversation.await_count
                    with self.assertLogs(self.plugin.logger, level="WARNING") as logs:
                        await self.dispatch(event)
                    result = event.get_extra("reply_gate_decision")
                    self.assertEqual(event.is_stopped(), not fallback)
                    self.assertIs(result["reply"], fallback)
                    self.assertEqual(result["source"], "error_fallback")
                    self.assertEqual(result["decision_source"], "reply_model_vision")
                    self.assertEqual(result["reason"], reason)
                    self.assertNotIn("private-error-body", str(logs.output))
                    self.assertEqual(
                        len(self.reply_calls) - replies_before, int(fallback)
                    )
                    self.assertEqual(
                        self.prepared_requests[-1], self.request_snapshots[-1]
                    )
                    saved = self.histories[event.unified_msg_origin]
                    self.assertEqual(saved[:1], original)
                    self.assertIn("image_url", json.dumps(saved))
                    self.assertNotIn("MEMORY", json.dumps(saved))
                    self.assertEqual(len(saved), 3 if fallback else 2)
                    self.assertEqual(
                        self.context.conversation_manager.update_conversation.await_count
                        - saves_before,
                        1,
                    )
        self.assertEqual(self.requests, [])

    async def test_failure_policy_does_not_override_valid_text_or_image_decisions(self):
        for fallback in (False, True):
            for image in (False, True):
                for reply in (False, True):
                    with self.subTest(fallback=fallback, image=image, reply=reply):
                        self.config["failure_default_reply"] = fallback
                        self.extra_parts = []
                        if image:
                            self.extra_parts.append(
                                ImageURLPart(
                                    image_url=ImageURLPart.ImageURL(
                                        url="data:image/png;base64,dGVzdA=="
                                    )
                                )
                            )
                        self.probability = 0.9 if reply else 0.1
                        self.vision_decision = reply
                        event = await self.dispatch(
                            make_event(session=f"valid-{fallback}-{image}-{reply}")
                        )
                        self.assertEqual(event.is_stopped(), not reply)
                        self.assertIs(
                            event.get_extra("reply_gate_decision")["reply"], reply
                        )
                        self.assertNotEqual(
                            event.get_extra("reply_gate_decision")["source"],
                            "error_fallback",
                        )

    async def test_failed_closed_turn_is_visible_after_decision_service_recovers(self):
        self.config["failure_default_reply"] = False
        self.status = 503
        first = await self.dispatch(make_event("I mentioned a blue cat"))
        self.assertTrue(first.is_stopped())
        self.assertEqual(self.reply_calls, [])
        self.status = 200
        await self.dispatch(make_event("What colour was it?"))
        self.assertIn("blue cat", json.dumps(self.requests[-1]["body"]))
        self.assertEqual(len(self.reply_calls), 1)

    async def test_vision_timeout_passes_but_cancellation_still_cancels(self):
        self.extra_parts.append(
            ImageURLPart(
                image_url=ImageURLPart.ImageURL(url="data:image/png;base64,dGVzdA==")
            )
        )
        self.vision_provider.text_chat.side_effect = TimeoutError()
        e = await self.dispatch(make_event("image timeout"))
        self.assertFalse(e.is_stopped())
        self.assertEqual(len(self.reply_calls), 1)
        self.assertEqual(
            e.get_extra("reply_gate_decision")["decision_source"], "reply_model_vision"
        )
        self.assertEqual(self.prepared_requests[-1], self.request_snapshots[-1])
        self.vision_provider.text_chat.side_effect = asyncio.CancelledError()
        cancelled = make_event("cancelled", session="cancelled")
        cancelled.set_extra("reply_gate_candidate", True)
        req = ProviderRequest(
            prompt="cancelled", extra_user_content_parts=copy.deepcopy(self.extra_parts)
        )
        with self.assertRaises(asyncio.CancelledError):
            await self.plugin.select_reply(cancelled, req)
        self.assertTrue(cancelled.is_stopped())
        self.assertEqual(len(self.reply_calls), 1)

    async def test_vision_http_error_logs_only_status_and_passes_original_request(self):
        self.extra_parts.append(
            ImageURLPart(
                image_url=ImageURLPart.ImageURL(url="data:image/png;base64,dGVzdA==")
            )
        )
        self.vision_provider.text_chat.side_effect = APIStatusError(
            "private-provider-error-body",
            response=httpx.Response(
                402, request=httpx.Request("POST", "https://example.invalid")
            ),
            body={"private": "private-provider-error-body"},
        )
        with self.assertLogs(self.plugin.logger, level="WARNING") as logs:
            e = await self.dispatch(make_event("image"))
        self.assertFalse(e.is_stopped())
        self.assertEqual(
            e.get_extra("reply_gate_decision")["reason"], "provider_http_402"
        )
        self.assertNotIn("private-provider-error-body", str(logs.output))
        self.assertEqual(len(self.reply_calls), 1)
        self.assertEqual(self.prepared_requests[-1], self.request_snapshots[-1])

    async def test_histories_are_isolated_by_conversation(self):
        self.probability = 0.1
        await self.dispatch(make_event("private topic", session="group-a"))
        await self.dispatch(make_event("other topic", session="group-b"))
        self.assertNotIn("private topic", json.dumps(self.requests[-1]["body"]))

    async def test_own_messages_do_not_trigger_a_decision(self):
        await self.dispatch(make_event(sender="bot-1"))
        self.assertEqual(len(self.requests), 0)
        self.assertEqual(len(self.reply_calls), 0)

    async def test_activity_accumulation_and_decrease_use_real_request_and_history_path(
        self,
    ):
        self.config["reply_activity"] = 0.6
        outcomes = []
        for score in [0.6, 0.1, 0.6, 0.6]:
            self.probability = score
            event = await self.dispatch(make_event(f"message with score {score}"))
            outcomes.append(event.get_extra("reply_gate_decision"))
        self.assertEqual([d["reply"] for d in outcomes], [False, False, False, True])
        self.assertAlmostEqual(outcomes[0]["accumulation_remaining"], 0.8109302162)
        self.assertEqual(outcomes[1]["accumulation_remaining"], 0)
        self.assertEqual(outcomes[-1]["accumulation_remaining"], 0)
        self.assertEqual(len(self.requests), 4)
        self.assertEqual(len(self.reply_calls), 1)
        self.assertEqual(self.prepared_requests, self.request_snapshots)
        saved = self.histories[event.unified_msg_origin]
        self.assertEqual([m["role"] for m in saved], ["user"] * 4 + ["assistant"])
        self.assertNotIn("MEMORY", json.dumps(saved))
        self.assertNotIn("accumulation", json.dumps(saved))

    async def test_activity_endpoints_skip_both_judges_but_mentions_always_pass(self):
        self.config["bailian_api_key"] = ""
        self.config["failure_default_reply"] = False
        for image in [False, True]:
            self.extra_parts = []
            if image:
                self.extra_parts.append(
                    ImageURLPart(
                        image_url=ImageURLPart.ImageURL(
                            url="data:image/png;base64,dGVzdA=="
                        )
                    )
                )
            for activity in [0, 1]:
                with self.subTest(image=image, activity=activity):
                    self.config["reply_activity"] = activity
                    origin = f"endpoint-{image}-{activity}"
                    event = await self.dispatch(make_event(session=origin))
                    self.assertEqual(event.is_stopped(), activity == 0)
                    self.assertEqual(
                        event.get_extra("reply_gate_decision")["source"], "activity"
                    )
                    saved = self.histories[event.unified_msg_origin]
                    self.assertEqual(len(saved), 1 if activity == 0 else 2)
                    mention = await self.dispatch(
                        make_event(mention=True, session=origin)
                    )
                    self.assertFalse(mention.is_stopped())
                    self.assertEqual(
                        mention.get_extra("reply_gate_decision")["source"], "mention"
                    )
        self.assertEqual(self.requests, [])
        self.assertEqual(self.vision_calls, [])
        self.assertEqual(self.prepared_requests, self.request_snapshots)

    async def test_activity_credit_is_shared_by_group_members_and_isolated_between_groups(
        self,
    ):
        self.config["reply_activity"] = 0.6
        self.probability = 0.6
        for group in ["a", "b"]:
            self.assertTrue(
                (
                    await self.dispatch(make_event(sender="alice", session=group))
                ).is_stopped()
            )
        for group in ["a", "b"]:
            self.assertFalse(
                (
                    await self.dispatch(make_event(sender="bob", session=group))
                ).is_stopped()
            )
        self.assertEqual(len(self.reply_calls), 2)

    async def test_same_event_hook_does_not_judge_accumulate_or_save_twice(self):
        self.config["reply_activity"] = 0.6
        self.probability = 0.6
        event = await self.dispatch(make_event())
        saves = self.context.conversation_manager.update_conversation.await_count
        await self.plugin.select_reply(event, self.prepared_requests[-1])
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(
            self.context.conversation_manager.update_conversation.await_count, saves
        )
        next_event = await self.dispatch(make_event())
        self.assertFalse(next_event.is_stopped())
        self.assertEqual(len(self.reply_calls), 1)

    async def test_mention_and_image_decisions_clear_previous_text_credit(self):
        self.config["reply_activity"] = 0.6
        self.probability = 0.6
        self.assertTrue((await self.dispatch(make_event())).is_stopped())
        self.assertFalse((await self.dispatch(make_event(mention=True))).is_stopped())
        self.assertTrue((await self.dispatch(make_event())).is_stopped())
        self.extra_parts.append(
            ImageURLPart(
                image_url=ImageURLPart.ImageURL(url="data:image/png;base64,dGVzdA==")
            )
        )
        self.vision_decision = False
        self.assertTrue((await self.dispatch(make_event())).is_stopped())
        self.extra_parts = []
        self.assertTrue((await self.dispatch(make_event())).is_stopped())

    async def test_activity_upgrade_preserves_saved_activity_and_ignores_legacy_threshold(
        self,
    ):
        schema = json.loads(
            (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text()
        )
        for activity in [None, 0, 0.6, 1]:
            with (
                self.subTest(activity=activity),
                tempfile.TemporaryDirectory() as folder,
            ):
                path = Path(folder) / "plugin.json"
                original = {
                    "bailian_api_key": "test-only-fake-key",
                    "bailian_api_url": "https://example.invalid/systemone",
                    "decision_instructions": "MY_RULES",
                    "context_budget": 20000,
                    "failure_default_reply": False,
                    "reply_threshold": 0.7,
                }
                if activity is not None:
                    original["reply_activity"] = activity
                path.write_text(json.dumps(original))
                config = AstrBotConfig(str(path), schema=schema)
                ReplyGate(self.context, config)
                saved = json.loads(path.read_text(encoding="utf-8-sig"))
                self.assertEqual(
                    saved["reply_activity"], 0.5 if activity is None else activity
                )
                self.assertNotIn("reply_threshold", saved)
                self.assertEqual(
                    {k: saved[k] for k in original if k != "reply_threshold"},
                    {k: v for k, v in original.items() if k != "reply_threshold"},
                )

    async def test_visual_probability_shares_credit_with_text_and_honors_activity(self):
        image = ImageURLPart(
            image_url=ImageURLPart.ImageURL(url="data:image/png;base64,dGVzdA==")
        )
        self.config["reply_activity"] = 0.6
        self.vision_decision = 0.6
        self.extra_parts = [image]
        first = await self.dispatch(make_event("image"))
        self.assertTrue(first.is_stopped())
        self.assertAlmostEqual(
            first.get_extra("reply_gate_decision")["probability"], 0.6
        )
        self.extra_parts = []
        self.probability = 0.6
        second = await self.dispatch(make_event("text"))
        self.assertFalse(second.is_stopped())
        self.assertEqual(len(self.reply_calls), 1)
        self.assertEqual(self.prepared_requests, self.request_snapshots)
        self.extra_parts = [image]
        for activity, score, expected in [(0.01, 0.9, False), (0.99, 0.1, True)]:
            self.config["reply_activity"] = activity
            self.vision_decision = score
            result = await self.dispatch(make_event("image", session=str(activity)))
            self.assertIs(result.get_extra("reply_gate_decision")["reply"], expected)
            self.assertEqual(
                result.get_extra("reply_gate_decision")["source"], "reply_model_vision"
            )

    async def test_failed_judgments_do_not_count_and_fallback_reply_clears_credit(self):
        self.config["reply_activity"] = 0.6
        self.config["failure_default_reply"] = False
        self.probability = 0.6
        self.assertTrue((await self.dispatch(make_event())).is_stopped())
        self.status = 503
        self.assertTrue((await self.dispatch(make_event())).is_stopped())
        self.status = 200
        self.assertFalse((await self.dispatch(make_event())).is_stopped())
        self.assertTrue((await self.dispatch(make_event())).is_stopped())
        self.status = 503
        self.config["failure_default_reply"] = True
        self.assertFalse((await self.dispatch(make_event())).is_stopped())
        self.status = 200
        self.assertTrue((await self.dispatch(make_event())).is_stopped())

    async def test_zero_activity_stays_silent_even_if_request_assembly_fails(self):
        self.config["reply_activity"] = 0
        self.config["failure_default_reply"] = True
        event = make_event()
        event.set_extra("reply_gate_candidate", True)
        req = ProviderRequest(prompt="text")
        with patch.object(
            type(req),
            "assemble_context",
            AsyncMock(side_effect=ValueError("unsupported_media")),
        ):
            await self.plugin.select_reply(event, req)
        self.assertTrue(event.is_stopped())
        self.assertFalse(event.get_extra("reply_gate_decision")["reply"])
        self.assertEqual(self.requests, [])

    def enable_batching(self, idle=0.04, maximum=0.15):
        self.config.update(merge_idle_seconds=idle, merge_max_wait_seconds=maximum)
        self.astr_config["timezone"] = "Asia/Shanghai"
        manager = self.context.conversation_manager
        manager.get_curr_conversation_id = AsyncMock(side_effect=lambda umo: umo)
        manager.get_conversation = AsyncMock(
            side_effect=lambda umo, cid: SimpleNamespace(
                cid=cid, history=json.dumps(self.histories.get(umo, [])), token_usage=0
            )
        )

    async def test_continuous_messages_share_one_decision_reply_and_individual_history(
        self,
    ):
        self.enable_batching()
        events = [
            make_event("第一句", sender="alice"),
            make_event("第二句", sender="bob"),
            make_event("第三句", sender="alice"),
        ]
        for i, event in enumerate(events):
            event.message_obj.sender.nickname = ["Alice", "Bob", "Alice"][i]
        await asyncio.wait_for(asyncio.gather(*(self.dispatch(e) for e in events)), 2)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.reply_calls), 1)
        decision_text = json.dumps(self.requests[0]["body"], ensure_ascii=False)
        reply_text = json.dumps(self.reply_calls[0], ensure_ascii=False)
        for text in ["第一句", "第二句", "第三句", "Alice", "Bob"]:
            self.assertIn(text, decision_text)
            self.assertIn(text, reply_text)
        self.assertIn("+08:00", decision_text)
        self.assertIn("本轮合并了 3 条", decision_text)
        self.assertIn("本轮合并了 3 条", reply_text)
        saved = self.histories[events[0].unified_msg_origin]
        self.assertEqual([m["role"] for m in saved], ["user"] * 3 + ["assistant"])
        for text in ["第一句", "第二句", "第三句"]:
            self.assertEqual(json.dumps(saved, ensure_ascii=False).count(text), 1)
        self.assertNotIn("MEMORY", json.dumps(saved))
        self.assertNotIn("本轮合并", json.dumps(saved, ensure_ascii=False))

    async def test_one_batch_changes_activity_only_once_and_silence_preserves_all_messages(
        self,
    ):
        self.enable_batching()
        self.config["reply_activity"] = 0.6
        self.probability = 0.6
        events = [make_event(f"part-{i}") for i in range(3)]
        await asyncio.wait_for(asyncio.gather(*(self.dispatch(e) for e in events)), 2)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.reply_calls, [])
        decision = events[-1].get_extra("reply_gate_decision")
        self.assertAlmostEqual(decision["accumulation_remaining"], 0.8109302162)
        saved = self.histories[events[0].unified_msg_origin]
        self.assertEqual([m["role"] for m in saved], ["user"] * 3)
        await self.dispatch(make_event("next batch"))
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(self.reply_calls), 1)

    async def test_busy_messages_form_one_next_batch_and_earlier_mention_bypasses_gate(
        self,
    ):
        self.enable_batching()
        self.reply_release = asyncio.Event()
        first = asyncio.create_task(self.dispatch(make_event("first")))
        await asyncio.wait_for(self.reply_started.wait(), 1)
        events = [
            make_event("mention in middle", mention=True, sender="alice"),
            make_event("new topic", sender="bob"),
            make_event("last detail", sender="alice"),
        ]
        tasks = [asyncio.create_task(self.dispatch(e)) for e in events]
        await asyncio.sleep(0.2)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.reply_calls), 1)
        self.assertFalse(any(t.done() for t in tasks))
        self.reply_release.set()
        await asyncio.wait_for(asyncio.gather(first, *tasks), 2)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.reply_calls), 2)
        self.assertEqual(
            events[-1].get_extra("reply_gate_decision")["source"], "mention"
        )
        self.assertIn("new topic", json.dumps(self.reply_calls[-1]))
        self.assertIn("last detail", json.dumps(self.reply_calls[-1]))
        self.assertEqual(len(self.histories[events[0].unified_msg_origin]), 6)

    async def test_earlier_image_and_quote_remain_visible_in_shared_visual_request(
        self,
    ):
        from astrbot.api.message_components import Image, Reply

        self.enable_batching()
        first = make_event("image from alice", sender="alice")
        first.message_obj.message.append(
            Image.fromBase64(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jewAAAABJRU5ErkJggg=="
            )
        )
        first.message_obj.message.append(
            Reply(
                id="quote",
                message_str="quoted original",
                chain=[Plain("quoted original")],
            )
        )
        second = make_event("bob asks about it", sender="bob")
        await asyncio.wait_for(
            asyncio.gather(self.dispatch(first), self.dispatch(second)), 3
        )
        self.assertEqual(self.requests, [])
        self.assertEqual(len(self.vision_calls), 1)
        self.assertEqual(len(self.reply_calls), 1)
        self.assertIn(
            "最后 2 条用户消息", self.vision_calls[0]["contexts"][-1]["content"]
        )
        for payload in [
            self.vision_calls[0],
            self.reply_calls[0],
            self.histories[first.unified_msg_origin],
        ]:
            text = json.dumps(payload)
            self.assertIn("image_url", text)
            self.assertIn("data:image/png;base64,", text)
            self.assertIn("quoted original", text)
            self.assertIn("bob asks about it", text)

    async def test_saved_merge_settings_survive_upgrade_and_receive_defaults(self):
        schema = json.loads(
            (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text()
        )
        for previous in [{}, {"merge_idle_seconds": 2.5, "merge_max_wait_seconds": 8}]:
            with (
                self.subTest(previous=previous),
                tempfile.TemporaryDirectory() as folder,
            ):
                path = Path(folder) / "plugin.json"
                old = {
                    "reply_activity": 0.35,
                    "bailian_api_key": "test-only-fake-key",
                    **previous,
                }
                path.write_text(json.dumps(old))
                saved = AstrBotConfig(str(path), schema=schema)
                self.assertEqual(
                    saved["merge_idle_seconds"], previous.get("merge_idle_seconds", 1)
                )
                self.assertEqual(
                    saved["merge_max_wait_seconds"],
                    previous.get("merge_max_wait_seconds", 3),
                )
                self.assertEqual(saved["reply_activity"], 0.35)
                self.assertEqual(saved["bailian_api_key"], old["bailian_api_key"])

    async def test_real_scheduler_internal_runner_sender_and_cleanup_coalesce_busy_followups(
        self,
    ):
        from astrbot.core.astr_main_agent import collect_initial_request
        from astrbot.core.pipeline.respond.stage import RespondStage
        from astrbot.core.pipeline.scheduler import PipelineScheduler

        self.enable_batching(idle=0.03, maximum=0.1)
        self.astr_config["provider_settings"]["streaming_response"] = False
        self.astr_config["platform_settings"]["segmented_reply"]["enable"] = False
        internal = InternalAgentSubStage()
        await internal.initialize(self.pipeline_context)
        self.process_stage.agent_sub_stage = SimpleNamespace(
            process=lambda event: internal.process(event, "")
        )
        respond = RespondStage()
        await respond.initialize(self.pipeline_context)
        scheduler = PipelineScheduler(self.pipeline_context)
        scheduler.stages = [self.waking, self.process_stage, respond]
        started, release = asyncio.Event(), asyncio.Event()
        sent = []

        async def build(*, event, plugin_context, config, **kwargs):
            req, _ = await collect_initial_request(event, plugin_context, config)
            req.system_prompt = "PERSONA"
            req.extra_user_content_parts = copy.deepcopy(self.extra_parts)
            provider = MagicMock(spec=Provider)
            provider.provider_config = {
                "id": "test",
                "modalities": ["text", "image"],
                "max_context_tokens": 128000,
            }
            provider.get_model.return_value = "test"
            provider.meta.return_value = ProviderMeta("test", "test", "test", None)

            async def chat(**kwargs):
                self.reply_calls.append({"contexts": copy.deepcopy(kwargs["contexts"])})
                started.set()
                await release.wait()
                return LLMResponse(role="assistant", completion_text="merged reply")

            provider.text_chat = AsyncMock(side_effect=chat)
            runner = ToolLoopAgentRunner()
            reset = runner.reset(
                provider=provider,
                request=req,
                run_context=ContextWrapper(context=SimpleNamespace(event=event)),
                tool_executor=MagicMock(),
                agent_hooks=BaseAgentRunHooks(),
            )
            return SimpleNamespace(
                agent_runner=runner,
                provider_request=req,
                provider=provider,
                reset_coro=reset,
            )

        def incoming(text, mention=False):
            e = make_event(text, mention=mention)
            e.message_obj.message_id = text

            async def send(chain):
                sent.append(chain)
                e._has_send_oper = True

            e.send = AsyncMock(side_effect=send)
            return e

        module = "astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal"
        with (
            patch(module + ".build_main_agent", side_effect=build),
            patch(module + "._record_internal_agent_stats", new=AsyncMock()),
            patch(module + ".Metric.upload", new=AsyncMock()),
            patch.object(
                ToolLoopAgentRunner,
                "follow_up",
                side_effect=AssertionError("unbatched follow-up"),
            ),
        ):
            first = incoming("initial")
            active = asyncio.create_task(scheduler.execute(first))
            await asyncio.wait_for(started.wait(), 2)
            early = incoming("second", mention=True)
            latest = incoming("third")
            pending = [
                asyncio.create_task(scheduler.execute(e)) for e in [early, latest]
            ]
            await asyncio.sleep(0.15)
            self.assertEqual(len(self.reply_calls), 1)
            self.assertEqual(sent, [])
            release.set()
            await asyncio.wait_for(asyncio.gather(active, *pending), 3)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.reply_calls), 2)
        self.assertEqual(len(sent), 2)
        self.assertEqual(latest.get_extra("reply_gate_decision")["source"], "mention")
        saved = self.histories[first.unified_msg_origin]
        self.assertEqual(
            [m["role"] for m in saved],
            ["user", "assistant", "user", "user", "assistant"],
        )
        self.assertEqual(self.plugin._batcher._heads, {})
        # A declined native run never reaches sending hooks; scheduler cleanup
        # must still release the batch so a subsequent mention can run.
        self.probability = 0.1
        silent = incoming("native silent turn")
        with (
            patch(module + ".build_main_agent", side_effect=build),
            patch(module + "._record_internal_agent_stats", new=AsyncMock()),
            patch(module + ".Metric.upload", new=AsyncMock()),
        ):
            await asyncio.wait_for(scheduler.execute(silent), 2)
            self.assertTrue(silent.is_stopped())
            self.assertEqual(self.plugin._batcher._heads, {})
            await asyncio.wait_for(
                scheduler.execute(incoming("mention after silence", True)), 2
            )
        self.assertEqual(len(sent), 3)

    async def test_merged_mention_bypasses_zero_activity_and_missing_key(self):
        self.enable_batching()
        self.config.update(
            reply_activity=0, bailian_api_key="", failure_default_reply=False
        )
        events = [
            make_event("", mention=True),
            make_event("details from another member", sender="bob"),
        ]
        await asyncio.wait_for(asyncio.gather(*(self.dispatch(e) for e in events)), 2)
        self.assertEqual(self.requests, [])
        self.assertEqual(len(self.reply_calls), 1)
        self.assertEqual(
            events[-1].get_extra("reply_gate_decision")["source"], "mention"
        )

    async def test_failed_closed_batch_preserves_all_messages_and_releases_next_batch(
        self,
    ):
        self.enable_batching()
        self.status = 503
        self.config["failure_default_reply"] = False
        events = [make_event("first failed part"), make_event("second failed part")]
        await asyncio.wait_for(asyncio.gather(*(self.dispatch(e) for e in events)), 2)
        self.assertEqual(self.reply_calls, [])
        self.assertEqual(len(self.histories[events[0].unified_msg_origin]), 2)
        self.assertEqual(self.plugin._batcher._heads, {})
        self.status = 200
        await self.dispatch(make_event("recovered"))
        self.assertEqual(len(self.reply_calls), 1)
        self.assertIn("first failed part", json.dumps(self.requests[-1]))
        self.assertIn("second failed part", json.dumps(self.requests[-1]))

    async def test_commands_and_empty_events_do_not_wait_or_take_batch_ownership(self):
        from astrbot.core.star.filter.command import CommandFilter

        self.config.update(merge_idle_seconds=10, merge_max_wait_seconds=30)
        command = make_event("/help")
        command.set_extra(
            "activated_handlers",
            [SimpleNamespace(event_filters=[CommandFilter("help")])],
        )
        await asyncio.wait_for(self.plugin.observe_message(command), 0.2)
        self.assertIsNone(command.get_extra("reply_gate_batch_member"))
        empty = make_event("")
        await asyncio.wait_for(self.plugin.observe_message(empty), 0.2)
        self.assertIsNone(empty.get_extra("reply_gate_batch_member"))
