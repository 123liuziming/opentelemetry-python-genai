# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patch functions for Qwen-Agent instrumentation.

Wraps key qwen-agent methods to generate OpenTelemetry spans through the shared
``opentelemetry-util-genai`` telemetry handler:

- ``Agent.run()`` -> ``invoke_agent`` spans (``Agent.run_nonstream()`` is not
  wrapped separately; it calls ``self.run()`` internally, so a single
  ``invoke_agent`` span is produced by this wrapper).
- ``BaseChatModel.chat()`` -> ``chat`` (inference) spans.
- ``Agent._call_tool()`` -> ``execute_tool`` spans.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Iterator, Tuple

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    InferenceInvocation,
)

from .utils import (
    apply_usage_to_inference,
    convert_qwen_agent_final_output_messages,
    convert_qwen_messages_to_input_messages,
    convert_qwen_messages_to_output_messages,
    get_provider_name,
    get_tool_definitions,
    parse_tool_args,
)

logger = logging.getLogger(__name__)

# Active invoke_agent invocations, used to roll up nested LLM token usage.
_active_agent_invocations: ContextVar[Tuple[AgentInvocation, ...]] = (
    ContextVar("qwen_active_agent_invocations", default=())
)

# Reentrancy guards to prevent duplicate spans when Agent/BaseChatModel are
# abstract classes and a subclass calls super() (proxy/wrapper scenarios).
_agent_run_instance_stack: ContextVar[Tuple[int, ...]] = ContextVar(
    "qwen_agent_run_instance_stack", default=()
)
_in_chat: ContextVar[bool] = ContextVar("qwen_in_chat", default=False)
_in_call_tool: ContextVar[bool] = ContextVar(
    "qwen_in_call_tool", default=False
)


def _accumulate_llm_usage_on_active_agents(
    invocation: InferenceInvocation,
) -> None:
    """Roll up child LLM token usage onto active ``invoke_agent`` spans.

    The rollup is intentionally transitive: a parent agent records the total
    nested LLM cost of its run, so consumers should not sum agent spans to
    calculate global token usage.
    """
    active_agents = _active_agent_invocations.get()
    if not active_agents:
        return

    for active_agent in active_agents:
        if invocation.input_tokens is not None:
            active_agent.input_tokens = (active_agent.input_tokens or 0) + (
                invocation.input_tokens or 0
            )
        if invocation.output_tokens is not None:
            active_agent.output_tokens = (active_agent.output_tokens or 0) + (
                invocation.output_tokens or 0
            )
        if invocation.cache_read_input_tokens is not None:
            active_agent.cache_read_input_tokens = (
                active_agent.cache_read_input_tokens or 0
            ) + (invocation.cache_read_input_tokens or 0)
        if invocation.cache_creation_input_tokens is not None:
            active_agent.cache_creation_input_tokens = (
                active_agent.cache_creation_input_tokens or 0
            ) + (invocation.cache_creation_input_tokens or 0)


def wrap_agent_run(wrapped, instance, args, kwargs, handler: TelemetryHandler):
    """Wrapper for ``Agent.run()`` to create ``invoke_agent`` spans.

    ``Agent.run()`` is a generator that yields ``List[Message]``. We wrap it to
    create an agent span covering the full execution.
    """
    run_stack = _agent_run_instance_stack.get()
    instance_id = id(instance)
    if instance_id in run_stack:
        yield from wrapped(*args, **kwargs)
        return
    run_token = _agent_run_instance_stack.set(run_stack + (instance_id,))

    messages = args[0] if args else kwargs.get("messages", [])

    provider_name = None
    request_model = None
    if getattr(instance, "llm", None):
        provider_name = get_provider_name(instance.llm)
        request_model = getattr(instance.llm, "model", None)

    agent_name = getattr(instance, "name", None) or type(instance).__name__

    try:
        invocation = handler.invoke_local_agent(
            request_model=request_model,
            agent_name=agent_name,
        )
    except Exception as e:
        logger.debug("Failed to create agent invocation: %s", e)
        _agent_run_instance_stack.reset(run_token)
        yield from wrapped(*args, **kwargs)
        return

    invocation.provider = provider_name
    invocation.agent_description = (
        getattr(instance, "description", None) or None
    )
    if handler.should_capture_content():
        invocation.input_messages = convert_qwen_messages_to_input_messages(
            messages
        )
        system_message = getattr(instance, "system_message", None)
        if system_message:
            from opentelemetry.util.genai.types import Text  # noqa: PLC0415

            invocation.system_instruction = [Text(content=system_message)]

    active_agent_token = _active_agent_invocations.set(
        _active_agent_invocations.get() + (invocation,)
    )

    try:
        last_response = None
        for response in wrapped(*args, **kwargs):
            last_response = response
            yield response

        if last_response and handler.should_capture_content():
            invocation.output_messages = (
                convert_qwen_agent_final_output_messages(last_response)
            )

        invocation.stop()
    except Exception as e:
        invocation.fail(e)
        raise
    finally:
        _active_agent_invocations.reset(active_agent_token)
        _agent_run_instance_stack.reset(run_token)


def wrap_chat_model_chat(
    wrapped, instance, args, kwargs, handler: TelemetryHandler
):
    """Wrapper for ``BaseChatModel.chat()`` to create inference (chat) spans.

    ``chat()`` can return ``List[Message]`` (non-stream) or
    ``Iterator[List[Message]]`` (stream).
    """
    if _in_chat.get():
        return wrapped(*args, **kwargs)
    chat_token = _in_chat.set(True)

    try:
        messages = args[0] if args else kwargs.get("messages", [])
        functions = (
            kwargs.get("functions")
            if len(args) < 2
            else (args[1] if len(args) > 1 else None)
        )
        stream = kwargs.get("stream", True)
        extra_generate_cfg = kwargs.get("extra_generate_cfg")

        provider_name = get_provider_name(instance)
        request_model = getattr(instance, "model", "unknown_model")

        try:
            invocation = handler.inference(
                provider_name,
                request_model=request_model,
            )
        except Exception as e:
            logger.debug("Failed to create inference invocation: %s", e)
            return wrapped(*args, **kwargs)

        if handler.should_capture_content():
            invocation.input_messages = (
                convert_qwen_messages_to_input_messages(messages)
            )
        tool_definitions = get_tool_definitions(functions)
        if tool_definitions:
            invocation.tool_definitions = tool_definitions
        if extra_generate_cfg:
            if extra_generate_cfg.get("max_tokens"):
                invocation.max_tokens = extra_generate_cfg["max_tokens"]
            if extra_generate_cfg.get("temperature"):
                invocation.temperature = extra_generate_cfg["temperature"]
            if extra_generate_cfg.get("top_p"):
                invocation.top_p = extra_generate_cfg["top_p"]

        try:
            result = wrapped(*args, **kwargs)
        except Exception as e:
            invocation.fail(e)
            raise

        if (
            stream
            and hasattr(result, "__iter__")
            and not isinstance(result, list)
        ):
            return _wrap_streaming_llm_response(result, invocation, handler)

        if result:
            _finalize_llm_invocation(invocation, result, handler)
        else:
            invocation.stop()
        return result
    finally:
        _in_chat.reset(chat_token)


def _finalize_llm_invocation(
    invocation: InferenceInvocation,
    result: Any,
    handler: TelemetryHandler,
) -> None:
    """Populate response metadata on a completed inference invocation."""
    apply_usage_to_inference(invocation, result)
    if handler.should_capture_content():
        invocation.output_messages = convert_qwen_messages_to_output_messages(
            result
        )
    invocation.response_model_name = invocation.request_model

    finish_reasons = ["stop"]
    for msg in result:
        fc = (
            msg.function_call
            if hasattr(msg, "function_call")
            else msg.get("function_call")
            if isinstance(msg, dict)
            else None
        )
        if fc:
            finish_reasons = ["tool_calls"]
            break
    invocation.finish_reasons = finish_reasons

    _accumulate_llm_usage_on_active_agents(invocation)
    invocation.stop()


def _wrap_streaming_llm_response(
    response_iter: Iterator,
    invocation: InferenceInvocation,
    handler: TelemetryHandler,
) -> Iterator:
    """Wrap a streaming LLM response iterator to capture output on completion."""
    try:
        last_response = None
        for response in response_iter:
            apply_usage_to_inference(invocation, response)
            last_response = response
            yield response

        if last_response:
            if handler.should_capture_content():
                invocation.output_messages = (
                    convert_qwen_messages_to_output_messages(last_response)
                )
            invocation.response_model_name = invocation.request_model

            finish_reasons = ["stop"]
            for msg in last_response:
                fc = (
                    msg.function_call
                    if hasattr(msg, "function_call")
                    else msg.get("function_call")
                    if isinstance(msg, dict)
                    else None
                )
                if fc:
                    finish_reasons = ["tool_calls"]
                    break
            invocation.finish_reasons = finish_reasons

        _accumulate_llm_usage_on_active_agents(invocation)
        invocation.stop()
    except Exception as e:
        invocation.fail(e)
        raise


def wrap_agent_call_tool(
    wrapped, instance, args, kwargs, handler: TelemetryHandler
):
    """Wrapper for ``Agent._call_tool()`` to create ``execute_tool`` spans.

    ``_call_tool(tool_name, tool_args, **kwargs) -> str | List[ContentItem]``
    """
    if _in_call_tool.get():
        return wrapped(*args, **kwargs)
    tool_guard_token = _in_call_tool.set(True)

    try:
        tool_name = (
            args[0] if args else kwargs.get("tool_name", "unknown_tool")
        )
        tool_args = args[1] if len(args) > 1 else kwargs.get("tool_args", "{}")

        tool_description = None
        if hasattr(instance, "function_map"):
            tool_instance = instance.function_map.get(tool_name)
            if tool_instance is not None:
                tool_description = getattr(tool_instance, "description", None)

        try:
            invocation = handler.tool(
                tool_name,
                tool_type="function",
                tool_description=tool_description,
            )
        except Exception as e:
            logger.debug("Failed to create tool invocation: %s", e)
            return wrapped(*args, **kwargs)

        if invocation.should_capture_content_on_span:
            invocation.arguments = parse_tool_args(tool_args)

        try:
            result = wrapped(*args, **kwargs)
        except Exception as e:
            invocation.fail(e)
            raise

        if invocation.should_capture_content_on_span:
            if isinstance(result, str):
                invocation.tool_result = result
            elif result is not None:
                invocation.tool_result = str(result)

        invocation.stop()
        return result
    finally:
        _in_call_tool.reset(tool_guard_token)
