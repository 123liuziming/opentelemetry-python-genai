# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Data-extraction helpers for Qwen-Agent instrumentation.

These helpers translate Qwen-Agent ``Message`` objects into the
``opentelemetry-util-genai`` message model. Invocations themselves are always
created through the public
:class:`~opentelemetry.util.genai.handler.TelemetryHandler` factory methods
(``inference`` / ``invoke_local_agent`` / ``tool``) in ``patch.py`` — never
constructed directly.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    OutputMessage,
    Text,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
)

logger = logging.getLogger(__name__)


class QwenAgentGenAiProviderName(str, Enum):
    """Provider names not present in the standard OpenTelemetry semantic conventions."""

    DASHSCOPE = "dashscope"


# Map qwen-agent ``model_type`` to a provider name.
_MODEL_TYPE_PROVIDER_MAP = {
    "qwen_dashscope": QwenAgentGenAiProviderName.DASHSCOPE.value,
    "qwenvl_dashscope": QwenAgentGenAiProviderName.DASHSCOPE.value,
    "qwenaudio_dashscope": QwenAgentGenAiProviderName.DASHSCOPE.value,
    "qwenvlo_dashscope": QwenAgentGenAiProviderName.DASHSCOPE.value,
    "oai": GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
    "azure": GenAIAttributes.GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "qwenvl_oai": GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
    "qwenomni_oai": GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
}


def get_provider_name(llm_instance: Any) -> str:
    """Extract a provider name from a qwen-agent LLM instance."""
    model_type = getattr(llm_instance, "model_type", "")
    if model_type in _MODEL_TYPE_PROVIDER_MAP:
        return _MODEL_TYPE_PROVIDER_MAP[model_type]

    class_name = type(llm_instance).__name__.lower()
    if "dashscope" in class_name:
        return QwenAgentGenAiProviderName.DASHSCOPE.value
    if "openai" in class_name or "oai" in class_name:
        return GenAIAttributes.GenAiProviderNameValues.OPENAI.value
    if "azure" in class_name:
        return GenAIAttributes.GenAiProviderNameValues.AZURE_AI_OPENAI.value

    return QwenAgentGenAiProviderName.DASHSCOPE.value


def _extract_content_text(content: Any) -> str:
    """Extract text from a qwen-agent ``Message.content`` field.

    Content can be a ``str`` or a ``List[ContentItem]``.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if hasattr(item, "text") and item.text is not None:
                texts.append(item.text)
            elif hasattr(item, "get_type_and_value"):
                t, v = item.get_type_and_value()
                if t == "text":
                    texts.append(v)
        return "\n".join(texts)
    return str(content) if content else ""


def _field_value(value: Any, *names: str) -> Any:
    """Read the first present field from a mapping or SDK response object."""
    if value is None:
        return None

    for name in names:
        if isinstance(value, dict):
            if name in value:
                return value[name]
            continue

        try:
            attr_value = getattr(value, name)
        except Exception:
            attr_value = None
        if attr_value is not None:
            return attr_value

        get_method = getattr(value, "get", None)
        if callable(get_method):
            try:
                got_value = get_method(name)
            except Exception:
                got_value = None
            if got_value is not None:
                return got_value

    return None


def _int_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_token_values(usage: Any) -> Dict[str, int]:
    if usage is None:
        return {}

    input_tokens = _int_value(
        _field_value(usage, "input_tokens", "prompt_tokens")
    )
    output_tokens = _int_value(
        _field_value(usage, "output_tokens", "completion_tokens")
    )
    cache_read_tokens = _int_value(
        _field_value(usage, "cache_read_input_tokens", "cached_prompt_tokens")
    )
    cache_creation_tokens = _int_value(
        _field_value(usage, "cache_creation_input_tokens")
    )

    for detail_name in ("prompt_tokens_details", "input_tokens_details"):
        details = _field_value(usage, detail_name)
        if details is not None and cache_read_tokens is None:
            cache_read_tokens = _int_value(
                _field_value(details, "cached_tokens")
            )

    values: Dict[str, int] = {}
    if input_tokens is not None:
        values["input_tokens"] = input_tokens
    if output_tokens is not None:
        values["output_tokens"] = output_tokens
    if cache_read_tokens is not None and cache_read_tokens > 0:
        values["cache_read_input_tokens"] = cache_read_tokens
    if cache_creation_tokens is not None and cache_creation_tokens > 0:
        values["cache_creation_input_tokens"] = cache_creation_tokens

    return values


def _usage_score(usage_values: Dict[str, int]) -> int:
    return (usage_values.get("input_tokens") or 0) + (
        usage_values.get("output_tokens") or 0
    )


def _usage_sources(value: Any) -> List[Any]:
    sources = []
    usage = _field_value(value, "usage")
    if usage is not None:
        sources.append(usage)

    extra = _field_value(value, "extra")
    if extra is not None:
        extra_usage = _field_value(extra, "usage", "usage_metadata")
        if extra_usage is not None:
            sources.append(extra_usage)

        service_info = _field_value(extra, "model_service_info")
        if service_info is not None:
            sources.append(service_info)

    service_info = _field_value(value, "model_service_info")
    if service_info is not None:
        sources.append(service_info)

    return sources


def _extract_usage_values(value: Any, depth: int = 0) -> Dict[str, int]:
    """Extract token usage from a qwen-agent Message/extra/model_service_info."""
    if value is None or depth > 4:
        return {}

    best_values: Dict[str, int] = {}
    values = _usage_token_values(value)
    if values:
        best_values = values

    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            item_values = _extract_usage_values(item, depth + 1)
            if _usage_score(item_values) > _usage_score(best_values):
                best_values = item_values
        return best_values

    for source in _usage_sources(value):
        source_values = _extract_usage_values(source, depth + 1)
        if _usage_score(source_values) > _usage_score(best_values):
            best_values = source_values

    return best_values


def apply_usage_to_inference(
    invocation: InferenceInvocation, value: Any
) -> None:
    """Apply qwen-agent token usage metadata to an ``InferenceInvocation``.

    Qwen-Agent stores DashScope responses under
    ``Message.extra["model_service_info"]`` for both streaming and
    non-streaming calls. Streaming chunks can carry cumulative usage, so only
    replace existing values when the candidate usage has at least as many
    observed tokens as the current invocation.
    """
    usage_values = _extract_usage_values(value)
    if not usage_values:
        return

    current_score = (invocation.input_tokens or 0) + (
        invocation.output_tokens or 0
    )
    if current_score and _usage_score(usage_values) < current_score:
        return

    if "input_tokens" in usage_values:
        invocation.input_tokens = usage_values["input_tokens"]
    if "output_tokens" in usage_values:
        invocation.output_tokens = usage_values["output_tokens"]
    if "cache_read_input_tokens" in usage_values:
        invocation.cache_read_input_tokens = usage_values[
            "cache_read_input_tokens"
        ]
    if "cache_creation_input_tokens" in usage_values:
        invocation.cache_creation_input_tokens = usage_values[
            "cache_creation_input_tokens"
        ]


def convert_qwen_messages_to_input_messages(
    messages: Any,
) -> List[InputMessage]:
    """Convert a qwen-agent Message list to GenAI ``InputMessage`` objects."""
    if not messages:
        return []

    if not isinstance(messages, list):
        messages = [messages]

    input_messages = []
    for msg in messages:
        try:
            role = (
                msg.role if hasattr(msg, "role") else msg.get("role", "user")
            )
            content = (
                msg.content
                if hasattr(msg, "content")
                else msg.get("content", "")
            )
            function_call = (
                msg.function_call
                if hasattr(msg, "function_call")
                else msg.get("function_call")
            )
            name = msg.name if hasattr(msg, "name") else msg.get("name")

            parts = []

            if function_call:
                fc_name = (
                    function_call.name
                    if hasattr(function_call, "name")
                    else function_call.get("name", "")
                )
                fc_args = (
                    function_call.arguments
                    if hasattr(function_call, "arguments")
                    else function_call.get("arguments", "{}")
                )
                if isinstance(fc_args, str):
                    try:
                        fc_args = json.loads(fc_args)
                    except (json.JSONDecodeError, ValueError):
                        pass
                parts.append(
                    ToolCallRequest(name=fc_name, arguments=fc_args, id=None)
                )

            # qwen-agent uses role="function" internally, but the DashScope API
            # converts it to role="tool". Handle both.
            if role in ("function", "tool") and content:
                text = _extract_content_text(content)

                tool_call_id: str = ""
                if hasattr(msg, "id"):
                    tool_call_id = getattr(msg, "id", "") or ""
                elif isinstance(msg, dict):
                    tool_call_id = msg.get("id") or ""

                if not tool_call_id:
                    extra = (
                        getattr(msg, "extra", None)
                        if not isinstance(msg, dict)
                        else msg.get("extra")
                    )
                    if extra is not None:
                        if isinstance(extra, dict):
                            tool_call_id = extra.get("function_id") or ""
                        else:
                            tool_call_id = (
                                getattr(extra, "function_id", "") or ""
                            )

                if not tool_call_id:
                    tool_call_id = name or ""

                parts.append(ToolCallResponse(id=tool_call_id, response=text))
            elif content:
                text = _extract_content_text(content)
                if text:
                    parts.append(Text(content=text))

            if parts:
                input_messages.append(InputMessage(role=role, parts=parts))

        except Exception as e:
            logger.debug("Error converting message: %s", e)
            continue

    return input_messages


def convert_qwen_messages_to_output_messages(
    messages: Any,
) -> List[OutputMessage]:
    """Convert qwen-agent response messages to GenAI ``OutputMessage`` objects."""
    if not messages:
        return []

    if not isinstance(messages, list):
        messages = [messages]

    output_messages = []
    for msg in messages:
        try:
            content = (
                msg.content
                if hasattr(msg, "content")
                else msg.get("content", "")
            )
            function_call = (
                msg.function_call
                if hasattr(msg, "function_call")
                else msg.get("function_call")
            )

            parts = []
            finish_reason = "stop"

            if function_call:
                fc_name = (
                    function_call.name
                    if hasattr(function_call, "name")
                    else function_call.get("name", "")
                )
                fc_args = (
                    function_call.arguments
                    if hasattr(function_call, "arguments")
                    else function_call.get("arguments", "{}")
                )
                if isinstance(fc_args, str):
                    try:
                        fc_args = json.loads(fc_args)
                    except (json.JSONDecodeError, ValueError):
                        pass
                parts.append(
                    ToolCallRequest(name=fc_name, arguments=fc_args, id=None)
                )
                finish_reason = "tool_calls"

            if content:
                text = _extract_content_text(content)
                if text:
                    parts.append(Text(content=text))

            if not parts:
                parts.append(Text(content=""))

            output_messages.append(
                OutputMessage(
                    role="assistant",
                    parts=parts,
                    finish_reason=finish_reason,
                )
            )

        except Exception as e:
            logger.debug("Error converting output message: %s", e)
            continue

    return output_messages


def convert_qwen_agent_final_output_messages(
    messages: Any,
) -> List[OutputMessage]:
    """Convert only the final qwen-agent answer to GenAI ``OutputMessage`` objects."""
    if not messages:
        return []

    if not isinstance(messages, list):
        messages = [messages]

    for msg in reversed(messages):
        try:
            role = _field_value(msg, "role") or "assistant"
            function_call = _field_value(msg, "function_call")
            content = _field_value(msg, "content") or ""

            if role in ("function", "tool") or function_call:
                continue

            text = _extract_content_text(content)
            if text:
                return convert_qwen_messages_to_output_messages([msg])
        except Exception as e:
            logger.debug("Error extracting final agent output message: %s", e)
            continue

    logger.debug("No final qwen-agent assistant text output message found")
    return []


def get_tool_definitions(
    functions: Optional[List[Dict]],
) -> Optional[List[ToolDefinition]]:
    """Extract tool definitions as ``FunctionToolDefinition`` objects."""
    if not functions:
        return None

    try:
        tool_defs: List[ToolDefinition] = []
        for func in functions:
            if not isinstance(func, dict):
                continue
            name = func.get("name")
            if not name:
                continue
            tool_defs.append(
                FunctionToolDefinition(
                    name=name,
                    description=func.get("description"),
                    parameters=func.get("parameters"),
                )
            )
        if tool_defs:
            return tool_defs
    except Exception:
        pass

    return None


def parse_tool_args(tool_args: Any) -> Any:
    """Parse qwen-agent tool arguments into a JSON-friendly object."""
    if isinstance(tool_args, str):
        try:
            return json.loads(tool_args)
        except (json.JSONDecodeError, ValueError):
            return {"raw_args": tool_args}
    if isinstance(tool_args, dict):
        return tool_args
    return {}
