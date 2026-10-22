"""Gate ordinary replies with Alibaba Cloud's decision model."""

from __future__ import annotations

import asyncio
import copy
import time
from sys import maxsize

import aiohttp
from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.context.truncator import ContextTruncator
from astrbot.core.agent.message import Message, dump_messages_with_checkpoints

from .shared_context import (
    DEFAULT_INSTRUCTIONS,
    ENDPOINT,
    MODEL,
    ContextLimitError,
    fit_context,
    parse_probability,
)


class ReplyGate(Star):
    """Use one shared request window before generating an ordinary reply."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context, config)
        self.config = config
        self._http: aiohttp.ClientSession | None = None
        self._last_warning: dict[str, float] = {}

    @filter.event_message_type(filter.EventMessageType.ALL, priority=10_000)
    async def observe_message(self, event: AstrMessageEvent) -> None:
        """Allow every incoming user message to reach the request hook.

        Args:
            event: Incoming event, after AstrBot's admission and safety checks.
        """
        if not self.config.get("enabled", True):
            return
        if event.get_sender_id() == event.get_self_id():
            event.stop_event()
            return
        event.set_extra("reply_gate_candidate", True)
        # This also makes the built-in probability responder defer, so it cannot
        # bypass the decision or start a second response for the same message.
        event.is_at_or_wake_command = True

    @filter.on_llm_request(priority=-maxsize)
    async def select_reply(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """Evaluate the fully decorated request before the reply model runs.

        Args:
            event: Current message event.
            req: AstrBot's request, already containing persona and memory.
        """
        if not self.config.get("enabled", True):
            return
        if any(
            isinstance(part, At) and str(part.qq) == str(event.get_self_id())
            for part in event.get_messages()
        ):
            event.set_extra("reply_gate_decision", {"reply": True, "source": "mention"})
            return
        if not event.get_extra("reply_gate_candidate", False):
            return

        current = None
        try:
            current = await req.assemble_context()
            instructions = str(
                self.config.get("decision_instructions", DEFAULT_INSTRUCTIONS)
            ).strip()
            if not instructions:
                instructions = DEFAULT_INSTRUCTIONS
            history = [
                copy.deepcopy(msg)
                for msg in req.contexts
                if isinstance(msg, dict) and msg.get("role")
            ]
            # Apply AstrBot's configured turn limit before deciding, so its
            # runner does not remove another set of messages after the decision.
            compression = (
                self.context.get_config(event.unified_msg_origin)
                .get("agent_runner", {})
                .get("config", {})
                .get("compression", {})
            )
            turns = int(compression.get("max_turns", -1))
            if turns != -1:
                messages = [Message.model_validate(msg) for msg in history + [current]]
                bounded = ContextTruncator().truncate_by_turns(
                    messages,
                    keep_most_recent_turns=turns,
                    drop_turns=int(compression.get("trim_turns", 1)),
                )
                history = [message.model_dump() for message in bounded[:-1]]

            window, payload = fit_context(
                history,
                current,
                req.system_prompt,
                instructions,
                int(self.config.get("context_budget", 60_000)),
                bool(req.prompt and req.prompt.strip()),
            )
            req.contexts = window.history
            req.system_prompt = window.system_prompt
            req.prompt = None
            req.image_urls = []
            req.audio_urls = []
            req.extra_user_content_parts = window.current["content"]
            current = window.current
            if req.conversation:
                req.conversation = copy.copy(req.conversation)
                req.conversation.token_usage = 0

            key = str(self.config.get("bailian_api_key", "")).strip()
            if not key:
                raise ValueError("api_key_missing")
            endpoint = str(self.config.get("bailian_api_url", ENDPOINT)).strip()
            if not endpoint:
                raise ValueError("api_url_missing")
            if self._http is None or self._http.closed:
                self._http = aiohttp.ClientSession()
            async with self._http.post(
                endpoint,
                # Match the Authorization value in the user's console example.
                # A complete value containing "Bearer " is also accepted as-is.
                headers={"Authorization": key},
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=float(self.config.get("timeout_seconds", 8))
                ),
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    # Response bodies may contain user text or credentials.
                    raise ValueError(f"decision_http_{response.status}")
                body = await response.json(content_type=None)
            probability = parse_probability(body)
            threshold = float(self.config.get("reply_threshold", 0.5))
            if not 0 <= threshold <= 1:
                raise ValueError("invalid_reply_threshold")
            should_reply = probability >= threshold
            event.set_extra(
                "reply_gate_decision",
                {
                    "reply": should_reply,
                    "source": MODEL,
                    "probability": probability,
                    "dropped_messages": window.dropped_messages,
                    "dropped_group_records": window.dropped_group_records,
                    "shortened_current": window.shortened_current,
                },
            )
            self.logger.info(
                "Reply gate: reply=%s probability=%.4f dropped_messages=%d dropped_group_records=%d shortened_current=%s",
                should_reply,
                probability,
                window.dropped_messages,
                window.dropped_group_records,
                window.shortened_current,
            )
            if should_reply:
                return
        except asyncio.CancelledError:
            event.stop_event()
            raise
        except Exception as exc:
            # Never log raw exception strings from HTTP or media code: they may
            # include URLs, API keys or private message content.
            reason = (
                str(exc)
                if isinstance(exc, (ValueError, ContextLimitError))
                and str(exc)
                in {
                    "api_key_missing",
                    "api_url_missing",
                    "protected_context_too_large",
                    "unsupported_media",
                    "context_budget_out_of_range",
                    "unexpected_decision_model",
                    "invalid_decision_response",
                    "invalid_decision_probability",
                    "invalid_reply_threshold",
                }
                else (
                    str(exc)
                    if isinstance(exc, ValueError)
                    and str(exc).startswith("decision_http_")
                    else type(exc).__name__
                )
            )
            event.set_extra(
                "reply_gate_decision",
                {"reply": False, "source": "error", "reason": reason},
            )
            now = time.monotonic()
            if now - self._last_warning.get(reason, -60) >= 60:
                self.logger.warning(
                    "Reply gate kept silent: %s. Mentions still bypass the gate.",
                    reason,
                )
                self._last_warning[reason] = now

        event.stop_event()
        if req.conversation and current is not None:
            # AstrBot consumes pending group context while assembling a request.
            # Preserve this unsent user turn so the next decision can still see it.
            try:
                messages = [
                    Message.model_validate(msg)
                    for msg in req.contexts + [current]
                    if msg.get("role") != "system"
                ]
                history = dump_messages_with_checkpoints(messages)
                await self.context.conversation_manager.update_conversation(
                    event.unified_msg_origin,
                    req.conversation.cid,
                    history=history,
                    token_usage=0,
                )
            except Exception as exc:
                self.logger.error(
                    "Reply gate could not retain an unsent turn: %s", type(exc).__name__
                )

    async def terminate(self) -> None:
        """Release the plugin's HTTP connections.

        Returns:
            None.
        """
        if self._http is not None:
            await self._http.close()
