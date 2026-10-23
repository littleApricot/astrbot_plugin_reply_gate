"""Gate ordinary replies with Alibaba Cloud's decision model."""

from __future__ import annotations

import asyncio
import copy
import time
from sys import maxsize

import aiohttp
from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Plain
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import Message, dump_messages_with_checkpoints
from openai import APIStatusError

from .activity_gate import DEFAULT_ACTIVITY, ReplyAccumulator, validate_activity
from .shared_context import (
    DEFAULT_INSTRUCTIONS,
    ENDPOINT,
    MODEL,
    ContextLimitError,
    fit_context,
    parse_probability,
)
from .vision_gate import has_image, judge_image


class ReplyGate(Star):
    """Judge a bounded copy while preserving the full reply request."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context, config)
        self.config = config
        if not str(config.get("decision_instructions") or "").strip():
            # Materialize the editable default instead of hiding it in code.
            config["decision_instructions"] = DEFAULT_INSTRUCTIONS
            if isinstance(config, AstrBotConfig):
                config.save_config()
        self._http: aiohttp.ClientSession | None = None
        self._last_warning: dict[str, float] = {}
        self._accumulator = ReplyAccumulator()

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
        parts = event.get_messages()
        if (
            not (event.message_str or "").strip()
            and any(
                isinstance(part, At) and str(part.qq) == str(event.get_self_id())
                for part in parts
            )
            and all(
                isinstance(part, At)
                or isinstance(part, Plain)
                and not part.text.strip()
                for part in parts
            )
        ):
            # AstrBot skips mention-only messages before constructing a request.
            # Describe the actual interaction so the normal history builder runs.
            event.message_str = "[用户仅 @ 了你，没有附加文字]"
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
        previous = event.get_extra("reply_gate_decision")
        if previous is not None:
            if not previous["reply"]:
                event.stop_event()
            return
        conversation = getattr(req.conversation, "cid", "")
        balance_key = (event.unified_msg_origin, str(conversation))
        if any(
            isinstance(part, At) and str(part.qq) == str(event.get_self_id())
            for part in event.get_messages()
        ):
            self._accumulator.reset(balance_key)
            event.set_extra("reply_gate_decision", {"reply": True, "source": "mention"})
            self.logger.info("Reply gate: reply=True source=mention")
            return
        if not event.get_extra("reply_gate_candidate", False):
            return

        current = None
        started = time.monotonic()
        decision_source = "request_preparation"
        timeout = None
        activity = None
        try:
            activity = validate_activity(
                self.config.get("reply_activity", DEFAULT_ACTIVITY)
            )
            if activity in (0, 1):
                self._accumulator.reset(balance_key)
            if activity == 1:
                event.set_extra(
                    "reply_gate_decision",
                    {"reply": True, "source": "activity", "activity": activity},
                )
                self.logger.info("Reply gate: reply=True source=activity activity=1")
                return
            current = await req.assemble_context()
            instructions = str(self.config.get("decision_instructions") or "").strip()
            if not instructions:
                instructions = DEFAULT_INSTRUCTIONS
            if activity == 0:
                should_reply = False
                event.set_extra(
                    "reply_gate_decision",
                    {"reply": False, "source": "activity", "activity": activity},
                )
                self.logger.info("Reply gate: reply=False source=activity activity=0")
            elif has_image(current):
                decision_source = "reply_model_vision"
                timeout = float(self.config.get("vision_timeout_seconds", 30))
                probability = await judge_image(
                    self.context,
                    event,
                    req,
                    current,
                    instructions,
                    timeout,
                )
                decision = self._accumulator.evaluate(
                    balance_key, probability, activity
                )
                should_reply = decision.reply
                event.set_extra(
                    "reply_gate_decision",
                    {
                        "reply": should_reply,
                        "source": "reply_model_vision",
                        "probability": probability,
                        "activity": activity,
                        "accumulation_before": decision.before,
                        "accumulation_change": decision.change,
                        "accumulation_value": decision.accumulated,
                        "accumulation_remaining": decision.remaining,
                    },
                )
                self.logger.info(
                    "Reply gate: reply=%s source=reply_model_vision full_context=True probability=%.4f activity=%.4f accumulation_before=%.4f accumulation_change=%s accumulation_value=%.4f accumulation_remaining=%.4f",
                    should_reply,
                    probability,
                    activity,
                    decision.before,
                    f"{decision.change:.4f}"
                    if decision.change is not None
                    else "endpoint",
                    decision.accumulated,
                    decision.remaining,
                )
            else:
                decision_source = MODEL
                timeout = float(self.config.get("timeout_seconds", 30))
                history = [
                    copy.deepcopy(msg)
                    for msg in req.contexts
                    if isinstance(msg, dict) and msg.get("role")
                ]
                # Only this detached decision window is trimmed. AstrBot retains
                # ownership of the original request, context limits and persistence.
                window, payload = fit_context(
                    history,
                    current,
                    req.system_prompt,
                    instructions,
                    int(self.config.get("context_budget", 60_000)),
                    bool(req.prompt and req.prompt.strip()),
                )
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
                    timeout=aiohttp.ClientTimeout(total=timeout),
                    allow_redirects=False,
                ) as response:
                    if response.status != 200:
                        # Response bodies may contain user text or credentials.
                        raise ValueError(f"decision_http_{response.status}")
                    body = await response.json(content_type=None)
                probability = parse_probability(body)
                decision = self._accumulator.evaluate(
                    balance_key, probability, activity
                )
                should_reply = decision.reply
                event.set_extra(
                    "reply_gate_decision",
                    {
                        "reply": should_reply,
                        "source": MODEL,
                        "probability": probability,
                        "activity": activity,
                        "accumulation_before": decision.before,
                        "accumulation_change": decision.change,
                        "accumulation_value": decision.accumulated,
                        "accumulation_remaining": decision.remaining,
                        "dropped_messages": window.dropped_messages,
                        "dropped_group_records": window.dropped_group_records,
                        "shortened_current": window.shortened_current,
                    },
                )
                self.logger.info(
                    "Reply gate: reply=%s probability=%.4f activity=%.4f accumulation_before=%.4f accumulation_change=%s accumulation_value=%.4f accumulation_remaining=%.4f decision_dropped_messages=%d decision_dropped_group_records=%d decision_shortened_current=%s",
                    should_reply,
                    probability,
                    activity,
                    decision.before,
                    f"{decision.change:.4f}"
                    if decision.change is not None
                    else "endpoint",
                    decision.accumulated,
                    decision.remaining,
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
                    "invalid_reply_activity",
                    "reply_provider_missing",
                    "reply_provider_has_no_vision",
                    "invalid_vision_decision",
                    "empty_vision_decision",
                    "invalid_vision_json",
                    "invalid_vision_schema",
                }
                else (
                    str(exc)
                    if isinstance(exc, ValueError)
                    and str(exc).startswith("decision_http_")
                    else type(exc).__name__
                )
            )
            if isinstance(exc, APIStatusError):
                reason = f"provider_http_{exc.status_code}"
            fallback_reply = activity != 0 and bool(
                self.config.get("failure_default_reply", True)
            )
            event.set_extra(
                "reply_gate_decision",
                {
                    "reply": fallback_reply,
                    "source": "error_fallback",
                    "decision_source": decision_source,
                    "reason": reason,
                    "elapsed_seconds": time.monotonic() - started,
                    "timeout_seconds": timeout,
                },
            )
            now = time.monotonic()
            warning_key = f"{decision_source}:{reason}:{fallback_reply}"
            if now - self._last_warning.get(warning_key, -60) >= 60:
                self.logger.warning(
                    "Reply gate failed %s: reason=%s source=%s elapsed_seconds=%.3f timeout_seconds=%s fallback_reply=%s.",
                    "open" if fallback_reply else "closed",
                    reason,
                    decision_source,
                    now - started,
                    timeout,
                    fallback_reply,
                )
                self._last_warning[warning_key] = now
            if fallback_reply:
                self._accumulator.reset(balance_key)
                return

        event.stop_event()
        if req.conversation and current is not None:
            # AstrBot consumes pending group context while assembling a request.
            # Preserve this unsent user turn so the next decision can still see it.
            try:
                # Keep existing records (including checkpoints) intact and append
                # the full current turn, excluding temporary persona/memory parts.
                history = copy.deepcopy(req.contexts) + dump_messages_with_checkpoints(
                    [Message.model_validate(current)]
                )
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
