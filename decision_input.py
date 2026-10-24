"""Project chat data for the text judge without copying tool instructions."""

from __future__ import annotations

import copy
from datetime import datetime
from zoneinfo import ZoneInfo

from astrbot.core.agent.message import bind_checkpoint_messages


async def memory_snapshot(context, event, req):
    """Resolve only the selected persona; the memory plugin refreshes this later."""
    manager = getattr(context, "persona_manager", None)
    if manager is None or req.conversation is None:
        return ""
    cfg = context.get_config(umo=event.unified_msg_origin)
    _, persona, _, _ = await manager.resolve_selected_persona(
        umo=event.unified_msg_origin,
        conversation_persona_id=getattr(req.conversation, "persona_id", None),
        platform_name=event.get_platform_name(),
        provider_settings=cfg.get("provider_settings", {}),
    )
    return str(persona.get("prompt", "")) if persona else ""


def image_resolver(context):
    lookup = getattr(context, "get_registered_star", None)
    star = lookup("astrbot_plugin_meme_library") if callable(lookup) else None
    owner = getattr(star, "star_cls", None)
    resolver = getattr(owner, "decision_image_description", None)
    return resolver if callable(resolver) else None


async def project_message(message, resolver=None):
    """Keep message text/call structure; replace every media part in a copy."""
    value = copy.deepcopy(message)
    value.pop("_no_save", None)
    content = value.get("content")
    if not isinstance(content, list):
        return value
    parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind == "image_url":
            reference = part.get("image_url", {})
            reference = (
                reference.get("url", "") if isinstance(reference, dict) else reference
            )
            description = ""
            if resolver:
                try:
                    description = await resolver(reference)
                except Exception:
                    # Unknown, unavailable or not yet recognized images remain images.
                    description = ""
            parts.append({"type": "text", "text": description or "[图片]"})
        elif kind != "think":
            parts.append(
                {"type": "text", "text": "[媒体：" + str(kind or "未知") + "]"}
            )
    value["content"] = parts
    return value


async def build_input(context, event, req):
    resolver = image_resolver(context)
    history = []
    for message in bind_checkpoint_messages(copy.deepcopy(req.contexts or [])):
        if message.role in ("user", "assistant", "tool"):
            history.append(await project_message(message.model_dump(), resolver))
    # Extra parts can contain catalogs, recommendations and operational prompts.
    # Quoted messages are actual chat data and are the sole text-extra exception.
    detached = copy.copy(req)
    detached.extra_user_content_parts = []
    for part in req.extra_user_content_parts or []:
        raw = part if isinstance(part, dict) else part.model_dump()
        text = raw.get("text", "")
        if raw.get("type") != "text" or text.startswith("<Quoted Message>\n"):
            detached.extra_user_content_parts.append(copy.deepcopy(part))
    current = await project_message(await detached.assemble_context(), resolver)
    if isinstance(current.get("content"), str):
        current["content"] = [{"type": "text", "text": current["content"]}]
    current["sender_id"] = getattr(event, "get_sender_id", lambda: "")()
    current["sender_name"] = getattr(event, "get_sender_name", lambda: "")()
    config = context.get_config(umo=event.unified_msg_origin)
    zone = ZoneInfo(config.get("timezone", "Asia/Shanghai") or "Asia/Shanghai")
    current["current_time"] = datetime.now(zone).strftime("%Y-%m-%d %H:%M:%S")
    return history, current
