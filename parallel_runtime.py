"""Maintain batch history while preserving AstrBot's native session locks."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MethodType, SimpleNamespace
from uuid import uuid4

from astrbot.core.agent.message import Message, dump_messages_with_checkpoints
from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
from astrbot.core.pipeline.process_stage.method.agent_sub_stages import internal

from .live_history import bind, without_inputs


@dataclass(eq=False)
class ParallelRun:
    controller: ParallelRuntime
    event: object
    key: str = field(default_factory=lambda: uuid4().hex)
    cid: str = ""
    admitted: list[dict] | None = None
    input_message: Message | None = None
    saved_outputs: list[dict] | None = None
    inputs: list[dict] = field(default_factory=list)
    request: object = None
    frozen_history: list[dict] | None = None


class _Bridge:
    """Keep shared patches scoped to participating events, including reloads."""

    def __init__(self):
        self.current: ContextVar[ParallelRun | None] = ContextVar(
            "reply_gate_run", default=None
        )
        self.owners = set()
        self.runs = set()
        self.history_locks = {}
        self.capture = internal.try_capture_follow_up
        self.register = internal.register_active_runner
        self.unregister = internal.unregister_active_runner
        self.save = internal.InternalAgentSubStage._save_to_history
        self.reset = ToolLoopAgentRunner.reset
        bridge = self
        signature = inspect.signature(self.reset)

        async def reset(runner, *args, **kwargs):
            bound = signature.bind(runner, *args, **kwargs)
            await bridge.reset(runner, *args, **kwargs)
            wrapper = bound.arguments.get("run_context")
            event = getattr(getattr(wrapper, "context", None), "event", None)
            run = event.get_extra("reply_gate_parallel") if event else None
            if run is None or run.request is not bound.arguments.get("request"):
                return
            # LCM owns the same refresh at every step when installed.
            if event.get_extra("lcm_prepared"):
                return
            batch = event.get_extra("reply_gate_batch")
            count = len(batch.members) if batch is not None else 1
            owned = runner.run_context.messages[-count:]
            boundary = len(runner.run_context.messages)
            systems = (
                runner.run_context.messages[:1]
                if runner.run_context.messages
                and runner.run_context.messages[0].role == "system"
                else []
            )
            original_step = runner.step

            async def step(self):
                nonlocal boundary
                tail = runner.run_context.messages[boundary:]
                latest = run.frozen_history
                if latest is None:
                    latest = await run.controller.history(
                        event.unified_msg_origin, run.cid
                    )
                runner.run_context.messages = (
                    systems + bind(without_inputs(latest, run.inputs)) + owned + tail
                )
                boundary = len(runner.run_context.messages) - len(tail)
                async for result in original_step():
                    yield result

            runner.step = MethodType(step, runner)

        reset.__signature__ = signature
        self.reset_wrapper = reset
        ToolLoopAgentRunner.reset = reset

        def runner_key(origin, runner):
            event = getattr(getattr(runner.run_context, "context", None), "event", None)
            run = event.get_extra("reply_gate_parallel") if event is not None else None
            return f"{origin}:reply_gate:{run.key}" if run is not None else origin

        def capture(event):
            if event.get_extra("reply_gate_parallel") is not None:
                return None
            return bridge.capture(event)

        def register(origin, runner):
            event = getattr(getattr(runner.run_context, "context", None), "event", None)
            if event is not None:
                run = event.get_extra("reply_gate_parallel")
                if run is not None and run.input_message is None:
                    # Capture before generation: tools may later append their
                    # own user messages containing images or other results.
                    run.input_message = next(
                        (
                            m
                            for m in reversed(runner.run_context.messages)
                            if m.role == "user"
                        ),
                        None,
                    )
            return bridge.register(runner_key(origin, runner), runner)

        def unregister(origin, runner):
            return bridge.unregister(runner_key(origin, runner), runner)

        async def save(
            stage, event, req, response, messages, stats, user_aborted=False
        ):
            run = event.get_extra("reply_gate_parallel")
            if run is None or run.admitted is None:
                return await bridge.save(
                    stage, event, req, response, messages, stats, user_aborted
                )

            async def update(origin, cid, *, history, **kwargs):
                current = next(
                    (
                        i
                        for i, message in enumerate(messages)
                        if message is run.input_message
                    ),
                    None,
                )
                if current is None:
                    raise ValueError("parallel_history_missing_current_user")
                # Match the native serializer's prefix, including temporary
                # filtering and checkpoints, without inspecting generated roles.
                prefix = []
                skipped_system = False
                for message in messages[: current + 1]:
                    if message.role == "system" and not skipped_system:
                        skipped_system = True
                        continue
                    if message.role in ("assistant", "user") and message._no_save:
                        continue
                    prefix.append(message)
                saved_prefix = dump_messages_with_checkpoints(prefix)
                if history[: len(saved_prefix)] != saved_prefix:
                    raise ValueError("parallel_history_input_prefix_changed")
                await run.controller.finish(run, history, history[len(saved_prefix) :])

            proxy = SimpleNamespace(
                conv_manager=SimpleNamespace(update_conversation=update)
            )
            return await bridge.save(
                proxy, event, req, response, messages, stats, user_aborted
            )

        self.replacements = {
            "try_capture_follow_up": capture,
            "register_active_runner": register,
            "unregister_active_runner": unregister,
        }
        for name, replacement in self.replacements.items():
            setattr(internal, name, replacement)
        self.save_wrapper = save
        internal.InternalAgentSubStage._save_to_history = save

    def restore_if_unused(self):
        if self.owners or self.runs:
            return
        for name, original in (
            ("try_capture_follow_up", self.capture),
            ("register_active_runner", self.register),
            ("unregister_active_runner", self.unregister),
        ):
            if getattr(internal, name) is self.replacements[name]:
                setattr(internal, name, original)
        if internal.InternalAgentSubStage._save_to_history is self.save_wrapper:
            internal.InternalAgentSubStage._save_to_history = self.save
        if ToolLoopAgentRunner.reset is self.reset_wrapper:
            ToolLoopAgentRunner.reset = self.reset
        if getattr(internal, "_reply_gate_parallel_bridge", None) is self:
            delattr(internal, "_reply_gate_parallel_bridge")


class ParallelRuntime:
    """Retain the existing history bridge without bypassing session locks.

    The class and event key stay compatible with batches surviving hot reloads.
    MessageBatcher holds the session through sending and event cleanup.
    """

    def __init__(self, context):
        self.context = context
        self.bridge = None

    def enter(self, event):
        self.bridge = getattr(internal, "_reply_gate_parallel_bridge", None)
        if self.bridge is None:
            self.bridge = _Bridge()
            internal._reply_gate_parallel_bridge = self.bridge
        # An older plugin instance may still own the bridge during hot reload.
        # Restore only its own lock replacement, leaving other plugins intact.
        legacy_locks = getattr(self.bridge, "locks", None)
        if (
            legacy_locks is not None
            and internal.session_lock_manager
            is self.bridge.replacements.get("session_lock_manager")
        ):
            internal.session_lock_manager = legacy_locks
        self.bridge.owners.add(self)
        run = ParallelRun(self, event)
        self.bridge.runs.add(run)
        event.set_extra("reply_gate_parallel", run)
        token = self.bridge.current.set(run)
        task = asyncio.current_task()
        cleanup = event.cleanup_temporary_local_files

        def finish():
            try:
                cleanup()
            finally:
                if run in self.bridge.runs:
                    self.bridge.runs.remove(run)
                    if asyncio.current_task() is task:
                        self.bridge.current.reset(token)
                    self.bridge.restore_if_unused()

        event.cleanup_temporary_local_files = finish

    def lock(self, origin, cid):
        key = (id(self.context.conversation_manager), origin, cid)
        return self.bridge.history_locks.setdefault(key, asyncio.Lock())

    async def history(self, origin, cid):
        conversation = await self.context.conversation_manager.get_conversation(
            origin, cid
        )
        if conversation is None:
            raise ValueError("parallel_conversation_unavailable")
        return json.loads(conversation.history)

    async def prepare(self, event, req, base_length):
        run = event.get_extra("reply_gate_parallel")
        if run is None or run.admitted is not None or req.conversation is None:
            return
        run.cid = str(req.conversation.cid)
        run.request = req
        current = dump_messages_with_checkpoints(
            [Message.model_validate(await req.assemble_context())]
        )
        batch = event.get_extra("reply_gate_batch")
        inputs = (
            dump_messages_with_checkpoints(
                [Message.model_validate(m) for m in batch.past_messages]
            )
            if batch is not None
            else []
        )
        inputs += current
        run.inputs = copy.deepcopy(inputs)
        origin = event.unified_msg_origin
        async with self.lock(origin, run.cid):
            latest = await self.history(origin, run.cid)
            # Preserve decorations after the old history, including this batch's
            # earlier user messages. Newer admitted users are visible immediately.
            if latest != req.contexts[:base_length]:
                req.conversation.token_usage = 0
                req.contexts = copy.deepcopy(latest) + req.contexts[base_length:]
            run.admitted = copy.deepcopy(latest) + inputs
            await self.context.conversation_manager.update_conversation(
                origin, run.cid, history=run.admitted, token_usage=0
            )

    async def refresh_request(self, event, req):
        """Refresh after decorators and before judgment, respecting LCM projection."""
        manager = getattr(self.context, "conversation_manager", None)
        if manager is None:
            return
        observer = getattr(manager, "_lcm_history_bridge", None)
        if observer is not None and getattr(observer, "active", False):
            await observer.refresh_request(event, req)
            state = event.get_extra("lcm_prepared")
            batch = event.get_extra("reply_gate_batch")
            if getattr(batch, "freeze_context", False) and (
                not state or state.get("frozen_base") is None
            ):
                raise ValueError("lcm_batch_snapshot_unavailable")
            return
        run = event.get_extra("reply_gate_parallel")
        if run is None or not run.cid:
            return
        latest = run.frozen_history
        if latest is None:
            latest = await self.history(event.unified_msg_origin, run.cid)
            run.frozen_history = copy.deepcopy(latest)
        batch = event.get_extra("reply_gate_batch")
        req.contexts = without_inputs(latest, run.inputs)
        if batch is not None:
            req.contexts += copy.deepcopy(batch.past_messages)

    async def finish(self, run, branch, outputs):
        origin = run.event.unified_msg_origin
        async with self.lock(origin, run.cid):
            complete_outputs = copy.deepcopy(outputs)
            if run.saved_outputs is not None:
                if outputs[: len(run.saved_outputs)] != run.saved_outputs:
                    raise ValueError("parallel_history_output_changed")
                outputs = outputs[len(run.saved_outputs) :]
                if not outputs:
                    return
            latest = await self.history(origin, run.cid)
            # Persist native context compaction when no other batch has changed
            # history; otherwise append only this run's new output, atomically.
            history = branch if latest == run.admitted else latest + outputs
            await self.context.conversation_manager.update_conversation(
                origin, run.cid, history=history, token_usage=0
            )
            run.saved_outputs = complete_outputs

    def close(self):
        if self.bridge is not None:
            self.bridge.owners.discard(self)
            self.bridge.restore_if_unused()
