"""Keep individual messages, speakers and media inside a shared reply turn."""

from __future__ import annotations

import copy
import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from astrbot.core.agent.message import TextPart
from astrbot.core.astr_main_agent import (
    MainAgentBuildConfig,
    _get_quoted_message_parser_settings,
    collect_initial_request,
)
from astrbot.core.message.components import At, Reply
from astrbot.core.utils.quoted_message_parser import extract_quoted_message_text

from .message_batch import MessageBatch
from .shared_context import GROUP_BEGIN, GROUP_END, RECORD_START
from .vision_gate import has_image


def sender_header(event, timezone: str) -> str:
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        tz = datetime.UTC
    timestamp = datetime.datetime.fromtimestamp(event.message_obj.timestamp, tz)
    name = event.get_sender_name() or event.get_sender_id()
    mention = " @机器人" if is_mention(event) else ""
    return f"[{name}({event.get_sender_id()})/{timestamp.isoformat(timespec='seconds')}{mention}] "


def is_mention(event) -> bool:
    return any(
        isinstance(p, At) and str(p.qq) == str(event.get_self_id())
        for p in event.get_messages()
    )


async def prepare_past_messages(context, batch: MessageBatch) -> None:
    """Capture sibling media before deferred event cleanup can release it."""
    await batch.wait_other_handlers()
    batch.mentioned = any(is_mention(m.event) for m in batch.members)
    if len(batch.members) == 1:
        return
    cfg = context.get_config(umo=batch.key)
    settings = cfg.get("provider_settings", {})
    build_cfg = MainAgentBuildConfig(tool_call_timeout=60, provider_settings=settings)
    for member in batch.members:
        if member is batch.owner:
            continue
        event = member.event
        previous = event.get_extra("provider_request")
        try:
            raw, _ = await collect_initial_request(event, context, build_cfg)
            if raw is None:
                raise ValueError("batch_message_unavailable")
            raw.prompt = sender_header(event, cfg.get("timezone", "UTC")) + (
                raw.prompt or ""
            )
            for part in event.get_messages():
                if isinstance(part, Reply):
                    quoted = (
                        await extract_quoted_message_text(
                            event,
                            part,
                            settings=_get_quoted_message_parser_settings(settings),
                        )
                        or part.message_str
                        or "[Empty Text]"
                    )
                    raw.extra_user_content_parts.append(
                        TextPart(text=f"<Quoted Message>\n{quoted}\n</Quoted Message>")
                    )
            message = await raw.assemble_context()
            batch.past_messages.append(message)
            batch.image_in_past |= has_image(message)
        finally:
            event.set_extra("provider_request", previous)


def inject_past_messages(context, event, req, batch: MessageBatch) -> None:
    """Append complete sibling turns without clipping the reply window."""
    req.contexts = copy.deepcopy(req.contexts) + copy.deepcopy(batch.past_messages)
    cfg = context.get_config(umo=batch.key)
    req.prompt = sender_header(event, cfg.get("timezone", "UTC")) + (req.prompt or "")
    req.extra_user_content_parts.append(
        TextPart(
            text=(
                f"本轮合并了 {len(batch.members)} 条连续用户消息。当前消息与紧邻它之前的 "
                f"{len(batch.members) - 1} 条用户消息属于同一批，请综合这一批的内容判断和回复，"
                "不必逐条作答。发送人和时间见各条消息的开头。"
            )
        ).mark_as_temp()
    )
    # Read exact built-in raw records so only duplicate batch messages are
    # removed after its request hook, leaving unrelated group context intact.
    lookup = getattr(context, "get_registered_star", None)
    metadata = lookup("astrbot") if callable(lookup) else None
    manager = getattr(getattr(metadata, "star_cls", None), "group_chat_context", None)
    if manager is not None:
        ids = list(manager._record_ids.get(batch.key, ()))
        records = list(manager.raw_records.get(batch.key, ()))
        own_ids = {m.event.get_extra("_group_context_record_id") for m in batch.members}
        batch.group_duplicates = {
            record.rstrip("\n")
            for key, record in zip(ids, records, strict=False)
            if key in own_ids
        }
        # Slow sibling handlers may record after the owner. Consume the whole
        # batch's raw-record range, keeping later batches and unrelated records.
        last_id = next((key for key in reversed(ids) if key in own_ids), None)
        if last_id is not None:
            event.set_extra("_group_context_record_id", last_id)


def remove_duplicate_group_records(req, batch: MessageBatch) -> None:
    """Remove only built-in group summaries already represented as user turns."""
    if not batch.group_duplicates:
        return
    parts = []
    for original in req.extra_user_content_parts:
        text = (
            original.get("text")
            if isinstance(original, dict)
            else getattr(original, "text", None)
        )
        if (
            not isinstance(text, str)
            or GROUP_BEGIN not in text
            or GROUP_END not in text
        ):
            parts.append(original)
            continue
        prefix, _, rest = text.partition(GROUP_BEGIN)
        body, _, suffix = rest.partition(GROUP_END)
        records = [r.rstrip("\n") for r in RECORD_START.split(body) if r]
        remaining = [r for r in records if r not in batch.group_duplicates]
        if remaining:
            updated = prefix + GROUP_BEGIN + "\n".join(remaining) + GROUP_END + suffix
        else:
            updated = prefix + suffix
            if updated.strip() in {"", "<system_reminder></system_reminder>"}:
                continue
        part = copy.deepcopy(original)
        if isinstance(part, dict):
            part["text"] = updated
        else:
            part.text = updated
        parts.append(part)
    req.extra_user_content_parts = parts
