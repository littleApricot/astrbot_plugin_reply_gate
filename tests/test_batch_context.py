"""Ensure built-in raw group context does not duplicate merged user messages."""

import unittest
from types import SimpleNamespace

from astrbot.core.agent.message import TextPart
from astrbot.core.provider.entities import ProviderRequest
from astrbot_plugin_reply_gate.batch_context import (
    inject_past_messages,
    remove_duplicate_group_records,
)
from astrbot_plugin_reply_gate.message_batch import MessageBatch
from astrbot_plugin_reply_gate.shared_context import GROUP_BEGIN, GROUP_END


def message(record_id, name):
    extras = {"_group_context_record_id": record_id}
    return SimpleNamespace(
        get_extra=lambda key: extras.get(key),
        set_extra=lambda key, value: extras.__setitem__(key, value),
        get_sender_name=lambda: name,
        get_sender_id=lambda: name,
        get_self_id=lambda: "bot",
        get_messages=lambda: [],
        message_obj=SimpleNamespace(timestamp=0),
    )


class BatchContextTests(unittest.TestCase):
    def test_slow_sibling_record_after_owner_is_consumed_and_only_own_duplicates_removed(
        self,
    ):
        first, owner = message("first", "Alice"), message("owner", "Bob")
        batch = MessageBatch("group", 0, 0, 1, 3)
        one, two = SimpleNamespace(event=first), SimpleNamespace(event=owner)
        batch.members = [one, two]
        batch.owner = two
        batch.past_messages = [{"role": "user", "content": "Alice original"}]
        unrelated = "[Other/12:00:00]: keep this"
        owner_raw = "[Bob/12:00:01]: owner original"
        first_raw = "[Alice/12:00:02]: Alice original\nsecond line"
        manager = SimpleNamespace(
            _record_ids={"group": ["other", "owner", "first"]},
            raw_records={"group": [unrelated, owner_raw, first_raw]},
        )
        ctx = SimpleNamespace(
            get_config=lambda **kwargs: {"timezone": "Asia/Shanghai"},
            get_registered_star=lambda name: SimpleNamespace(
                star_cls=SimpleNamespace(group_chat_context=manager)
            ),
        )
        req = ProviderRequest(
            prompt="owner original",
            contexts=[{"role": "assistant", "content": "old reply"}],
        )
        inject_past_messages(ctx, owner, req, batch)
        self.assertEqual(owner.get_extra("_group_context_record_id"), "first")
        self.assertEqual(len(req.contexts), 2)
        # The built-in hook skips the last record and injects preceding records.
        req.extra_user_content_parts = [
            TextPart(
                text="<system_reminder>"
                + GROUP_BEGIN
                + unrelated
                + "\n"
                + owner_raw
                + GROUP_END
                + "</system_reminder>"
            ),
            TextPart(text="PINNED_MEMORY"),
        ]
        remove_duplicate_group_records(req, batch)
        all_text = str(req.extra_user_content_parts)
        self.assertIn("keep this", all_text)
        self.assertNotIn(owner_raw, all_text)
        self.assertIn("PINNED_MEMORY", all_text)
        self.assertEqual(manager._record_ids["group"], ["other", "owner", "first"])
