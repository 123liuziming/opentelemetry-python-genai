# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for QwenAgentInstrumentor and the conversion helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from qwen_agent.llm.schema import ContentItem, FunctionCall, Message

from opentelemetry.instrumentation.genai.qwen_agent import (
    QwenAgentInstrumentor,
)
from opentelemetry.instrumentation.genai.qwen_agent.utils import (
    convert_qwen_messages_to_input_messages,
    convert_qwen_messages_to_output_messages,
    get_provider_name,
    get_tool_definitions,
)
from opentelemetry.util.genai.types import (
    Text,
    ToolCallRequest,
    ToolCallResponse,
)


class TestQwenAgentInstrumentor:
    """Test the instrumentor lifecycle."""

    def test_instrument_and_uninstrument(
        self, tracer_provider, logger_provider, meter_provider
    ):
        instrumentor = QwenAgentInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            skip_dep_check=True,
        )
        assert instrumentor._handler is not None
        instrumentor.uninstrument()
        assert instrumentor._handler is None

    def test_instrumentation_dependencies(self):
        instrumentor = QwenAgentInstrumentor()
        deps = instrumentor.instrumentation_dependencies()
        assert ("qwen-agent >= 0.0.20",) == tuple(deps)


class TestProviderName:
    """Test provider name detection."""

    def test_dashscope_model_type(self):
        llm = MagicMock()
        llm.model_type = "qwen_dashscope"
        assert get_provider_name(llm) == "dashscope"

    def test_oai_model_type(self):
        llm = MagicMock()
        llm.model_type = "oai"
        assert get_provider_name(llm) == "openai"

    def test_unknown_model_type(self):
        llm = MagicMock()
        llm.model_type = "unknown_custom"
        type(llm).__name__ = "CustomModel"
        assert get_provider_name(llm) == "dashscope"

    def test_class_name_fallback_dashscope(self):
        llm = MagicMock()
        llm.model_type = "custom"
        type(llm).__name__ = "QwenDashScopeChat"
        assert get_provider_name(llm) == "dashscope"


class TestMessageConversion:
    """Test qwen-agent message to GenAI type conversion."""

    def test_convert_simple_user_message(self):
        messages = [Message(role="user", content="Hello")]
        result = convert_qwen_messages_to_input_messages(messages)
        assert len(result) == 1
        assert result[0].role == "user"
        assert len(result[0].parts) == 1
        assert isinstance(result[0].parts[0], Text)
        assert result[0].parts[0].content == "Hello"

    def test_convert_function_call_message(self):
        msg = Message(
            role="assistant",
            content="",
            function_call=FunctionCall(
                name="get_weather",
                arguments='{"city": "Beijing"}',
            ),
        )
        result = convert_qwen_messages_to_output_messages([msg])
        assert len(result) == 1
        assert result[0].finish_reason == "tool_calls"
        tool_calls = [
            p for p in result[0].parts if isinstance(p, ToolCallRequest)
        ]
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == {"city": "Beijing"}

    def test_convert_function_response_message(self):
        msg = Message(
            role="function", name="get_weather", content="Sunny, 25C"
        )
        result = convert_qwen_messages_to_input_messages([msg])
        assert len(result) == 1
        assert result[0].role == "function"
        tool_responses = [
            p for p in result[0].parts if isinstance(p, ToolCallResponse)
        ]
        assert len(tool_responses) == 1
        assert tool_responses[0].response == "Sunny, 25C"

    def test_convert_empty_messages(self):
        assert convert_qwen_messages_to_input_messages([]) == []

    def test_convert_multimodal_content(self):
        msg = Message(
            role="user",
            content=[ContentItem(text="Describe this image")],
        )
        result = convert_qwen_messages_to_input_messages([msg])
        assert len(result) == 1
        assert isinstance(result[0].parts[0], Text)
        assert result[0].parts[0].content == "Describe this image"


class TestToolDefinitions:
    """Test tool definition extraction."""

    def test_get_tool_definitions(self):
        functions = [
            {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        tool_defs = get_tool_definitions(functions)
        assert tool_defs is not None
        assert len(tool_defs) == 1
        assert tool_defs[0].name == "get_weather"
        assert tool_defs[0].description == "Get weather info"

    def test_get_tool_definitions_empty(self):
        assert get_tool_definitions(None) is None
        assert get_tool_definitions([]) is None
