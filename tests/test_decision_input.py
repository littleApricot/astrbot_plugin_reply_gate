"""Check bounded media projection with existing metadata and full memory."""

import json
import unittest

from astrbot_plugin_reply_gate.decision_input import project_message
from astrbot_plugin_reply_gate.shared_context import context_cost, fit_context


class ProjectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_known_image_metadata_is_used_without_removing_adjacent_text(self):
        original = {
            "role": "user",
            "content": [
                {"type": "text", "text": "原配字"},
                {"type": "image_url", "image_url": {"url": "opaque"}},
            ],
        }

        async def resolver(reference):
            self.assertEqual(reference, "opaque")
            return "[图片；名称：企鹅；描述：被拿捏]"

        value = await project_message(original, resolver)
        self.assertIn("企鹅", json.dumps(value, ensure_ascii=False))
        self.assertIn("原配字", json.dumps(value, ensure_ascii=False))
        self.assertIn("image_url", str(original))
        memory = "完整记忆" * 150
        history = [
            {"role": "user", "content": "最老消息" * 1000},
            {"role": "assistant", "content": "最近回答"},
            value,
        ]
        w, p = fit_context(
            history,
            {"role": "user", "content": [{"type": "text", "text": "新消息"}]},
            memory,
            "判断和输出",
            4000,
            True,
        )
        self.assertEqual(p["state"]["memory"], memory)
        self.assertLessEqual(context_cost(p), 4000)
        self.assertNotIn("最老消息", json.dumps(p, ensure_ascii=False))
        self.assertIn("原配字", json.dumps(p, ensure_ascii=False))
