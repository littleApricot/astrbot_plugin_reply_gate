"""Verify strict decisions and the real OpenAI adapter's SDK request payload."""

import base64
import copy
import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from astrbot_plugin_reply_gate.vision_gate import VISION_POLICY, judge_image
from openai.types.chat import ChatCompletion
from PIL import Image


def image_message():
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "blue").save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": url}}],
    }


def event_and_context(provider):
    event = SimpleNamespace(unified_msg_origin="test-group", get_extra=lambda key: None)
    context = SimpleNamespace(get_using_provider_async=AsyncMock(return_value=provider))
    return event, context


class VisionProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_deepseek_uses_one_typed_result_with_full_context_and_no_execution(
        self,
    ):
        config = {
            "id": "test",
            "type": "openai_chat_completion",
            "model": "deepseek-flash",
            "key": ["test-only-fake-key"],
            "api_base": "https://api.deepseek.com",
            "custom_extra_body": {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
                "seed": 123,
            },
        }
        provider = ProviderOpenAIOfficial(config, {})
        event, context = event_and_context(provider)
        req = ProviderRequest(
            contexts=[{"role": "user", "content": "OLD" * 4000}],
            system_prompt="PERSONA and memory; normal chat uses meme_search.",
        )
        current = image_message()
        original_req, original_current, original_config = copy.deepcopy(
            (req, current, config)
        )
        cases = [
            ("submit_reply_decision", '{"probability":0.6}', 0.6),
            ("submit_reply_decision", '{"probability":0}', 0.0),
            ("meme_search", '{"probability":0.6}', "invalid_vision_decision"),
            (
                "submit_reply_decision",
                '{"probability":true}',
                "invalid_vision_schema",
            ),
            (
                "submit_reply_decision",
                '{"probability":0.6,"extra":1}',
                "invalid_vision_schema",
            ),
        ]
        try:
            for name, arguments, expected in cases:
                response = ChatCompletion(
                    id="test-result",
                    created=0,
                    model="deepseek-flash",
                    object="chat.completion",
                    choices=[
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "decision-1",
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": arguments,
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                )
                create = AsyncMock(return_value=response)
                provider.client.chat.completions.create = create
                with self.subTest(name=name, arguments=arguments):
                    call = judge_image(context, event, req, current, "rule", 5)
                    if isinstance(expected, float):
                        self.assertEqual(await call, expected)
                    else:
                        with self.assertRaisesRegex(ValueError, expected):
                            await call
                    create.assert_awaited_once()
                    payload = create.await_args.kwargs
                    self.assertEqual(
                        [t["function"]["name"] for t in payload["tools"]],
                        ["submit_reply_decision"],
                    )
                    self.assertEqual(
                        payload["tools"][0]["function"]["parameters"]["properties"][
                            "probability"
                        ],
                        {"type": "number", "minimum": 0, "maximum": 1},
                    )
                    self.assertEqual(
                        payload["tool_choice"],
                        {
                            "type": "function",
                            "function": {"name": "submit_reply_decision"},
                        },
                    )
                    self.assertEqual(
                        payload["extra_body"]["thinking"], {"type": "disabled"}
                    )
                    self.assertNotIn("response_format", payload["extra_body"])
                    self.assertNotIn("reasoning_effort", payload["extra_body"])
                    self.assertIn("OLD" * 4000, str(payload["messages"]))
                    self.assertIn("data:image/png;base64,", str(payload["messages"]))
                    self.assertEqual(payload["messages"][-1]["role"], "user")
                    self.assertIn("rule", payload["messages"][-1]["content"])
                    self.assertIn("image_url", str(payload["messages"][-2]))
                    self.assertEqual(provider.provider_config, original_config)
                    self.assertEqual(req, original_req)
                    self.assertEqual(current, original_current)
            formal_response = response.model_copy(deep=True)
            formal_response.choices[0].message.content = "formal reply"
            formal_response.choices[0].message.tool_calls = None
            formal_response.choices[0].finish_reason = "stop"
            create.return_value = formal_response
            await provider.text_chat(prompt="formal reply", func_tool=None)
            self.assertEqual(
                create.await_args.kwargs["extra_body"], config["custom_extra_body"]
            )
        finally:
            await provider.client.close()

    async def test_strict_probability_json_with_optional_complete_fence(self):
        provider = SimpleNamespace(
            provider_config={"modalities": ["image"]}, text_chat=AsyncMock()
        )
        event, context = event_and_context(provider)
        cases = [
            ('{"probability":0.6}', 0.6),
            ('{"probability":0}', 0.0),
            ('```json\n{"probability":0.1}\n```', 0.1),
            ('```json{"probability":1}```', 1.0),
            ("````bad fence", "invalid_vision_json"),
            ("回复一下", "invalid_vision_json"),
            (
                '<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name="meme_search"></｜｜DSML｜｜ invoke></｜｜DSML｜｜ calls>',
                "invalid_vision_json",
            ),
            ('{"probability":"0.6"}', "invalid_vision_schema"),
            ('{"probability":true}', "invalid_vision_schema"),
            ('{"probability":-0.1}', "invalid_vision_schema"),
            ('{"probability":1.1}', "invalid_vision_schema"),
            ('{"probability":NaN}', "invalid_vision_schema"),
            ('{"probability":Infinity}', "invalid_vision_schema"),
            ('{"probability":0.6,"extra":1}', "invalid_vision_schema"),
            ('{"reply":true}', "invalid_vision_schema"),
            ("[]", "invalid_vision_schema"),
            ("", "empty_vision_decision"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                provider.text_chat.return_value = LLMResponse(
                    role="assistant", completion_text=text
                )
                call = judge_image(
                    context, event, ProviderRequest(), image_message(), "rule", 1
                )
                if isinstance(expected, float):
                    self.assertEqual(await call, expected)
                else:
                    with self.assertRaisesRegex(ValueError, expected):
                        await call

    async def test_json_mode_reaches_sdk_without_changing_shared_provider_or_request(
        self,
    ):
        config = {
            "id": "test",
            "type": "openai_chat_completion",
            "model": "deepseek-flash",
            "key": ["test-only-fake-key"],
            "api_base": "https://example.invalid/v1",
            "custom_extra_body": {"seed": 123},
        }
        provider = ProviderOpenAIOfficial(config, {})
        response = ChatCompletion(
            id="test-completion",
            created=0,
            model="deepseek-flash",
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": '{"probability":0.1}',
                    },
                }
            ],
        )
        create = AsyncMock(return_value=response)
        provider.client.chat.completions.create = create
        event, context = event_and_context(provider)
        req = ProviderRequest(
            contexts=[{"role": "user", "content": "OLD" * 4000}],
            system_prompt="PERSONA: answer casually and follow normal tool instructions.",
            model="model-override",
        )
        current = image_message()
        original_req = copy.deepcopy(req)
        original_current = copy.deepcopy(current)
        original_config = copy.deepcopy(config)
        try:
            result = await judge_image(
                context, event, req, current, "reply to greetings", 5
            )
            self.assertEqual(result, 0.1)
            payload = create.await_args.kwargs
            self.assertEqual(
                payload["extra_body"]["response_format"], {"type": "json_object"}
            )
            self.assertEqual(payload["extra_body"]["seed"], 123)
            self.assertEqual(payload["model"], "model-override")
            self.assertNotIn("tools", payload)
            system = payload["messages"][0]["content"]
            self.assertTrue(system.startswith(VISION_POLICY))
            self.assertIn(req.system_prompt, system)
            self.assertIn("<reply_context>", system)
            self.assertIn("OLD" * 4000, str(payload["messages"]))
            self.assertIn("data:image/png;base64,", str(payload["messages"]))
            self.assertEqual(provider.provider_config, original_config)
            self.assertEqual(req, original_req)
            self.assertEqual(current, original_current)
            await provider.text_chat(
                prompt="formal reply",
                contexts=[],
                system_prompt="normal persona",
                func_tool=None,
            )
            self.assertNotIn("response_format", create.await_args.kwargs["extra_body"])
        finally:
            await provider.client.close()
