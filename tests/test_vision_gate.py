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
    async def test_strict_boolean_json_with_optional_complete_fence(self):
        provider = SimpleNamespace(
            provider_config={"modalities": ["image"]}, text_chat=AsyncMock()
        )
        event, context = event_and_context(provider)
        cases = [
            ('{"should_reply":true}', True),
            ('{"should_reply":false}', False),
            ('```json\n{"should_reply":false}\n```', False),
            ('```json{"should_reply":true}```', True),
            ("````bad fence", "invalid_vision_json"),
            ("回复一下", "invalid_vision_json"),
            (
                '<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name="meme_search"></｜｜DSML｜｜ invoke></｜｜DSML｜｜ calls>',
                "invalid_vision_json",
            ),
            ('{"should_reply":"false"}', "invalid_vision_schema"),
            ('{"should_reply":1}', "invalid_vision_schema"),
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
                if isinstance(expected, bool):
                    self.assertIs(await call, expected)
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
                        "content": '{"should_reply":false}',
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
            self.assertIs(result, False)
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
