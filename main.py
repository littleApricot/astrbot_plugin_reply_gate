"""Gate ordinary replies with Alibaba Cloud's decision model."""

from __future__ import annotations

import asyncio
import copy
import time
from sys import maxsize

import aiohttp
from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, File, Image, Plain, Record, Reply, Video
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import Message, dump_messages_with_checkpoints
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.command_group import CommandGroupFilter

from .activity_gate import DEFAULT_ACTIVITY, ReplyAccumulator, validate_activity
from .batch_context import (
    inject_past_messages,
    prepare_past_messages,
    remove_duplicate_group_records,
)
from .decision_input import build_input, memory_snapshot
from .message_batch import MessageBatcher, merge_delays
from .parallel_runtime import ParallelRuntime
from .shared_context import (
    DEFAULT_INSTRUCTIONS,
    ENDPOINT,
    MODEL,
    ContextLimitError,
    fit_context,
    parse_probability,
)


class ReplyGate(Star):
    """Judge a bounded copy while preserving the full reply request."""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context, config)
        self.config = config
        previous_instructions = str(config.get("decision_instructions") or "")
        instructions = previous_instructions
        # Replace only known legacy scope clauses; preserve custom policy text.
        for old, new in (
            (
                "判断机器人此刻是否有充分理由回应当前这批新消息。",
                "判断机器人此刻是否有充分理由接话。",
            ),
            (
                "state.messages 是按时间排列的最近历史与新消息，最后 pending_count 条用户消息属于本批待回复消息。",
                "state.messages 是按时间排列的可见聊天记录，包括用户发言和机器人已经说过的话。",
            ),
            (
                "历史和记忆用于理解称呼、关系与话题，不要重新回答历史中的问题。",
                "结合整个可见上下文判断此刻是否值得回应，较早的话题也可以接。是否重复依据机器人实际已经说过的内容判断。",
            ),
            (
                "记忆中的回复约定只在当前消息满足其条件时适用。",
                "记忆中的回复约定按可见对话是否满足其条件判断。",
            ),
        ):
            instructions = instructions.replace(old, new)
        if not instructions.strip():
            instructions = DEFAULT_INSTRUCTIONS
        if instructions != previous_instructions:
            config["decision_instructions"] = instructions
            if isinstance(config, AstrBotConfig):
                config.save_config()
        self._http: aiohttp.ClientSession | None = None
        self._last_warning: dict[str, float] = {}
        self._accumulator = ReplyAccumulator()
        self._batcher = MessageBatcher()
        self._parallel = ParallelRuntime(context)

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
        # Commands keep their normal handlers and never enter the reply batch.
        handlers = event.get_extra("activated_handlers", []) or []
        if any(
            isinstance(f, (CommandFilter, CommandGroupFilter))
            for handler in handlers
            for f in handler.event_filters
        ):
            return
        if event.get_extra("reply_gate_batch_member") is not None:
            return
        if not (event.message_str or "").strip() and not any(
            isinstance(part, (Image, File, Record, Reply, Video)) for part in parts
        ):
            # Match the native empty-message admission rule. An empty event
            # must not become the owner and swallow preceding valid messages.
            return
        try:
            idle, maximum = merge_delays(self.config)
        except ValueError:
            self.logger.warning("回复判断：合并等待时间配置无效，改用 1 至 3 秒")
            idle, maximum = 1.0, 3.0
        member = await self._batcher.join(
            event,
            idle,
            maximum,
            immediate=any(
                isinstance(p, At) and str(p.qq) == str(event.get_self_id())
                for p in parts
            ),
        )
        if member is not member.batch.owner:
            # Other plugins still observe the original message, while this event
            # does not enqueue another LLM turn or enter native follow-up capture.
            event.call_llm = True
            event.set_extra("reply_gate_batch_follower", True)
        else:
            event.set_extra("reply_gate_batch", member.batch)
            self._parallel.enter(event)
            self.logger.info(
                "回复判断合并：消息数=%d，等待秒数=%.3f",
                len(member.batch.members),
                asyncio.get_running_loop().time() - member.batch.first,
            )

    @filter.on_waiting_llm_request(priority=maxsize)
    async def prepare_batch(self, event: AstrMessageEvent) -> None:
        """Capture siblings after their handlers and before the session lock."""
        member = event.get_extra("reply_gate_batch_member")
        if member is not None and not self._batcher.begin(member):
            event.stop_event()
            return
        if event.get_extra("reply_gate_batch_follower", False):
            event.stop_event()
            return
        batch = event.get_extra("reply_gate_batch")
        if batch is None:
            return
        try:
            await prepare_past_messages(self.context, batch)
        except asyncio.CancelledError:
            event.stop_event()
            raise
        except Exception as exc:
            # Do not produce a partial reply that silently omits batch members.
            self.logger.error("待回复消息合并失败：%s", type(exc).__name__)
            event.stop_event()

    @filter.on_llm_request(priority=maxsize)
    async def expand_batch(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """Expose sibling messages to all downstream request decorators."""
        batch = event.get_extra("reply_gate_batch")
        if batch is None:
            return
        if event.get_extra("reply_gate_batch_expanded", False):
            return
        event.set_extra(
            "reply_gate_memory", await memory_snapshot(self.context, event, req)
        )
        base_length = len(req.contexts)
        if len(batch.members) > 1:
            inject_past_messages(self.context, event, req, batch)
        try:
            await self._parallel.prepare(event, req, base_length)
        except asyncio.CancelledError:
            event.stop_event()
            raise
        except Exception as exc:
            self.logger.error("回复判断历史准备失败：%s", type(exc).__name__)
            event.stop_event()
            return
        event.set_extra("reply_gate_batch_expanded", True)

    @filter.on_llm_request(priority=-maxsize)
    async def select_reply(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """Evaluate the fully decorated request before the reply model runs.

        Args:
            event: Current message event.
            req: AstrBot's request, already containing persona and memory.
        """
        if not self.config.get("enabled", True):
            return
        if event.get_extra("reply_gate_batch_follower", False):
            event.stop_event()
            return
        batch = event.get_extra("reply_gate_batch")
        if batch is not None:
            remove_duplicate_group_records(req, batch)
        previous = event.get_extra("reply_gate_decision")
        if previous is not None:
            if not previous["reply"]:
                event.stop_event()
            return
        if batch is not None:
            try:
                # Pin before every admission path, including @ and activity=1.
                await self._parallel.refresh_request(event, req)
            except asyncio.CancelledError:
                event.stop_event()
                raise
            except Exception as exc:
                self.logger.error(
                    "回复判断上下文固定失败：%s；启用 LCM 时请同时更新至 v0.6.7 或更新版本",
                    type(exc).__name__,
                )
                event.stop_event()
                return
        conversation = getattr(req.conversation, "cid", "")
        balance_key = (event.unified_msg_origin, str(conversation))
        if (batch is not None and batch.mentioned) or any(
            isinstance(part, At) and str(part.qq) == str(event.get_self_id())
            for part in event.get_messages()
        ):
            self._accumulator.reset(balance_key)
            event.set_extra("reply_gate_decision", {"reply": True, "source": "mention"})
            self.logger.info("回复判断：回复=True，来源=@提及")
            return
        if not event.get_extra("reply_gate_candidate", False):
            return

        current = None
        full_current = None
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
                self.logger.info("回复判断：回复=True，来源=活跃度，活跃度=1")
                return
            current = await req.assemble_context()
            full_current = copy.deepcopy(current)
            instructions = str(self.config.get("decision_instructions") or "").strip()
            if not instructions:
                instructions = DEFAULT_INSTRUCTIONS
            if activity == 0:
                should_reply = False
                event.set_extra(
                    "reply_gate_decision",
                    {"reply": False, "source": "activity", "activity": activity},
                )
                self.logger.info("回复判断：回复=False，来源=活跃度，活跃度=0")
            else:
                decision_source = MODEL
                timeout = float(self.config.get("timeout_seconds", 30))
                if batch is None:
                    await self._parallel.refresh_request(event, req)
                history, current = await build_input(self.context, event, req)
                memory = event.get_extra("reply_gate_memory", "")
                # Only this detached decision window is trimmed. AstrBot retains
                # ownership of the original request, context limits and persistence.
                window, payload = fit_context(
                    history,
                    current,
                    memory,
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
                    self._http = aiohttp.ClientSession(
                        connector=aiohttp.TCPConnector(limit=0)
                    )
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
                    "回复判断：回复=%s，概率=%.4f，活跃度=%.4f，积累前值=%.4f，积累变化=%s，积累结果=%.4f，积累剩余=%.4f，判断移除消息数=%d，判断移除群记录数=%d，当前消息已截短=%s",
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
                    "unprojected_media",
                    "context_budget_out_of_range",
                    "unexpected_decision_model",
                    "invalid_decision_response",
                    "invalid_decision_probability",
                    "invalid_reply_activity",
                }
                else (
                    str(exc)
                    if isinstance(exc, ValueError)
                    and str(exc).startswith("decision_http_")
                    else type(exc).__name__
                )
            )
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
                    "回复判断失败，%s：原因=%s，来源=%s，耗时秒数=%.3f，超时秒数=%s，回退回复=%s。",
                    "回退为回复" if fallback_reply else "回退为沉默",
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
        parallel = event.get_extra("reply_gate_parallel")
        if (
            req.conversation
            and full_current is not None
            and not (parallel is not None and parallel.admitted is not None)
        ):
            # AstrBot consumes pending group context while assembling a request.
            # Preserve this unsent user turn so the next decision can still see it.
            try:
                # Keep existing records (including checkpoints) intact and append
                # the full current turn, excluding temporary persona/memory parts.
                history = copy.deepcopy(req.contexts) + dump_messages_with_checkpoints(
                    [Message.model_validate(full_current)]
                )
                await self.context.conversation_manager.update_conversation(
                    event.unified_msg_origin,
                    req.conversation.cid,
                    history=history,
                    token_usage=0,
                )
            except Exception as exc:
                self.logger.error(
                    "回复判断未能保存本轮未回复消息：%s", type(exc).__name__
                )

    async def terminate(self) -> None:
        """Release the plugin's HTTP connections.

        Returns:
            None.
        """
        await self._batcher.close()
        self._parallel.close()
        if self._http is not None:
            await self._http.close()
