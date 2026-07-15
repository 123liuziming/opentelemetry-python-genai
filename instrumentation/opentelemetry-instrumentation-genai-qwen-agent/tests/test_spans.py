# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for span generation from the qwen-agent instrumentation.

Tests verify that the instrumented methods produce the expected OpenTelemetry
spans with the correct names, kinds, and attributes. The migrated
instrumentation emits only the standard ``gen_ai.operation.name`` (no
``gen_ai.span.kind``) and no longer produces ``react step`` spans.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from qwen_agent.agent import Agent
from qwen_agent.llm.base import BaseChatModel
from qwen_agent.llm.schema import ContentItem, FunctionCall, Message

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import SpanKind, StatusCode

from ._test_helpers import assert_no_removed_telemetry


class _StubChatModel(BaseChatModel):
    """A minimal ``BaseChatModel`` subclass for testing."""

    def __init__(self, model="test-model", model_type="qwen_dashscope"):
        cfg = {"model": model, "model_type": model_type}
        super().__init__(cfg)
        self.use_raw_api = False

    def _chat_no_stream(self, messages, **kwargs):
        raise NotImplementedError

    def _chat_stream(self, messages, **kwargs):
        raise NotImplementedError

    def _chat_with_functions(self, messages, functions, **kwargs):
        raise NotImplementedError


class _StubAgent(Agent):
    """A minimal ``Agent`` subclass for testing."""

    @classmethod
    def create(cls, name="TestAgent", llm=None):
        obj = cls.__new__(cls)
        obj.name = name
        obj.description = "A test agent"
        obj.system_message = "You are a helpful assistant."
        obj.llm = llm
        obj.function_map = {}
        obj.extra_generate_cfg = {}
        return obj

    def _run(self, messages, **kwargs):
        raise NotImplementedError


class TestLLMChatSpan:
    """Verify ``BaseChatModel.chat()`` produces a correct chat (inference) span."""

    def test_non_stream_chat_creates_span(self, span_exporter, instrument):
        model = _StubChatModel(model="qwen-max", model_type="qwen_dashscope")
        fake_response = [Message(role="assistant", content="Hello there!")]

        with patch.object(
            _StubChatModel, "_chat_no_stream", return_value=fake_response
        ):
            result = model.chat(
                messages=[Message(role="user", content="Hi")], stream=False
            )

        assert result is not None

        spans = span_exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(chat_spans) >= 1

        span = chat_spans[0]
        assert span.name == "chat qwen-max"
        assert span.kind == SpanKind.CLIENT
        attrs = dict(span.attributes or {})
        assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "qwen-max"
        assert attrs.get(GenAIAttributes.GEN_AI_PROVIDER_NAME) == "dashscope"
        assert attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
        assert_no_removed_telemetry(spans)

    def test_non_stream_chat_records_token_usage(
        self, span_exporter, instrument
    ):
        model = _StubChatModel(
            model="deepseek-v3", model_type="qwen_dashscope"
        )
        fake_response = [
            Message(
                role="assistant",
                content="4",
                extra={
                    "model_service_info": SimpleNamespace(
                        usage={
                            "input_tokens": 21,
                            "output_tokens": 1,
                            "total_tokens": 22,
                            "prompt_tokens_details": {"cached_tokens": 4},
                        }
                    )
                },
            )
        ]

        with patch.object(
            _StubChatModel, "_chat_no_stream", return_value=fake_response
        ):
            model.chat(
                messages=[Message(role="user", content="What is 2+2?")],
                stream=False,
            )

        spans = span_exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(chat_spans) >= 1
        attrs = dict(chat_spans[0].attributes or {})
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 21
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 1
        assert (
            attrs.get(GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS)
            == 4
        )

    def test_stream_chat_creates_span(self, span_exporter, instrument):
        model = _StubChatModel(model="qwen-turbo", model_type="qwen_dashscope")
        chunk1 = [Message(role="assistant", content="Hello")]
        chunk2 = [Message(role="assistant", content="Hello world")]

        def fake_stream(messages, **kwargs):
            yield chunk1
            yield chunk2

        with patch.object(
            _StubChatModel, "_chat_stream", side_effect=fake_stream
        ):
            response_iter = model.chat(
                messages=[Message(role="user", content="Hi")], stream=True
            )
            responses = list(response_iter)

        assert len(responses) == 2

        spans = span_exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(chat_spans) >= 1
        span = chat_spans[0]
        assert span.name == "chat qwen-turbo"
        assert span.kind == SpanKind.CLIENT

    def test_stream_chat_records_token_usage(self, span_exporter, instrument):
        model = _StubChatModel(
            model="deepseek-v3", model_type="qwen_dashscope"
        )
        chunk1 = [
            Message(
                role="assistant",
                content="The",
                extra={
                    "model_service_info": {
                        "usage": {"input_tokens": 18, "output_tokens": 1}
                    }
                },
            )
        ]
        chunk2 = [
            Message(
                role="assistant",
                content="The answer is 4.",
                extra={
                    "model_service_info": {
                        "usage": {"input_tokens": 18, "output_tokens": 5}
                    }
                },
            )
        ]

        def fake_stream(messages, **kwargs):
            yield chunk1
            yield chunk2

        with patch.object(
            _StubChatModel, "_chat_stream", side_effect=fake_stream
        ):
            list(
                model.chat(
                    messages=[Message(role="user", content="What is 2+2?")],
                    stream=True,
                )
            )

        spans = span_exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(chat_spans) >= 1
        attrs = dict(chat_spans[0].attributes or {})
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 18
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 5

    def test_stream_chat_keeps_most_complete_usage(
        self, span_exporter, instrument
    ):
        model = _StubChatModel(
            model="deepseek-v3", model_type="qwen_dashscope"
        )
        chunk1 = [
            Message(
                role="assistant",
                content="The",
                extra={
                    "model_service_info": {
                        "usage": {"input_tokens": 18, "output_tokens": 1}
                    }
                },
            )
        ]
        chunk2 = [
            Message(
                role="assistant",
                content="The answer",
                extra={
                    "model_service_info": {
                        "usage": {"input_tokens": 18, "output_tokens": 5}
                    }
                },
            )
        ]
        chunk3 = [Message(role="assistant", content="The answer is 4.")]

        def fake_stream(messages, **kwargs):
            yield chunk1
            yield chunk2
            yield chunk3

        with patch.object(
            _StubChatModel, "_chat_stream", side_effect=fake_stream
        ):
            list(
                model.chat(
                    messages=[Message(role="user", content="What is 2+2?")],
                    stream=True,
                )
            )

        spans = span_exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(chat_spans) >= 1
        attrs = dict(chat_spans[0].attributes or {})
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 18
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 5

    def test_chat_with_function_call_response(self, span_exporter, instrument):
        model = _StubChatModel(model="qwen-max", model_type="qwen_dashscope")
        fake_response = [
            Message(
                role="assistant",
                content="",
                function_call=FunctionCall(
                    name="get_weather", arguments='{"city": "Beijing"}'
                ),
            )
        ]

        with patch.object(
            _StubChatModel, "_chat_no_stream", return_value=fake_response
        ):
            model.chat(
                messages=[
                    Message(role="user", content="What is the weather?")
                ],
                stream=False,
            )

        spans = span_exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(chat_spans) >= 1
        attrs = dict(chat_spans[0].attributes or {})
        assert attrs.get(GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS) == (
            "tool_calls",
        )

    def test_chat_error_creates_error_span(self, span_exporter, instrument):
        model = _StubChatModel(model="qwen-max", model_type="qwen_dashscope")

        with patch.object(
            _StubChatModel,
            "_chat_no_stream",
            side_effect=RuntimeError("API timeout"),
        ):
            with pytest.raises(RuntimeError, match="API timeout"):
                model.chat(
                    messages=[Message(role="user", content="Hi")],
                    stream=False,
                )

        spans = span_exporter.get_finished_spans()
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(chat_spans) >= 1
        span = chat_spans[0]
        assert span.status.status_code == StatusCode.ERROR
        attrs = dict(span.attributes or {})
        assert attrs.get("error.type") == "RuntimeError"


class TestAgentRunSpan:
    """Verify ``Agent.run()`` produces ``invoke_agent`` spans."""

    def test_agent_run_creates_invoke_agent_span(
        self, span_exporter, instrument
    ):
        llm = MagicMock()
        llm.model = "qwen-max"
        llm.model_type = "qwen_dashscope"
        agent = _StubAgent.create(name="WeatherBot", llm=llm)
        response_msgs = [Message(role="assistant", content="It is sunny.")]

        def fake_run(messages, **kwargs):
            yield response_msgs

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            results = list(
                agent.run([Message(role="user", content="Weather?")])
            )

        assert len(results) >= 1

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) >= 1
        span = agent_spans[0]
        assert span.name == "invoke_agent WeatherBot"
        assert span.kind == SpanKind.INTERNAL
        attrs = dict(span.attributes or {})
        assert (
            attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "invoke_agent"
        )
        assert_no_removed_telemetry(spans)

    def test_agent_run_nonstream_creates_invoke_agent_span(
        self, span_exporter, instrument
    ):
        llm = MagicMock()
        llm.model = "qwen-max"
        llm.model_type = "qwen_dashscope"
        agent = _StubAgent.create(name="ChatBot", llm=llm)
        response_msgs = [Message(role="assistant", content="Hello!")]

        def fake_run(messages, **kwargs):
            yield response_msgs

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            result = agent.run_nonstream([Message(role="user", content="Hi")])

        assert result is not None

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        # run_nonstream is not wrapped; only run() creates the span.
        assert len(agent_spans) == 1
        assert "ChatBot" in agent_spans[0].name

    def test_agent_run_error_creates_error_span(
        self, span_exporter, instrument
    ):
        llm = MagicMock()
        llm.model = "qwen-max"
        llm.model_type = "qwen_dashscope"
        agent = _StubAgent.create(name="FailBot", llm=llm)

        def fake_run(messages, **kwargs):
            if False:
                yield
            raise ValueError("Agent processing failed")

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            with pytest.raises(ValueError, match="Agent processing failed"):
                list(agent.run([Message(role="user", content="Go")]))

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) >= 1
        span = agent_spans[0]
        assert span.status.status_code == StatusCode.ERROR
        assert dict(span.attributes or {}).get("error.type") == "ValueError"

    def test_agent_run_multiple_yields(self, span_exporter, instrument):
        llm = MagicMock()
        llm.model = "qwen-max"
        llm.model_type = "qwen_dashscope"
        agent = _StubAgent.create(name="MultiYieldBot", llm=llm)

        def fake_run(messages, **kwargs):
            yield [Message(role="assistant", content="Thinking...")]
            yield [Message(role="assistant", content="Done!")]

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            results = list(agent.run([Message(role="user", content="Go")]))

        assert len(results) == 2

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) == 1

    def test_agent_run_records_only_final_output_message(
        self, span_exporter, instrument_with_content
    ):
        llm = MagicMock()
        llm.model = "qwen-plus"
        llm.model_type = "qwen_dashscope"
        agent = _StubAgent.create(name="OnePilotBot", llm=llm)

        response_msgs = [
            Message(
                role="assistant",
                content="",
                function_call=FunctionCall(
                    name="get_operational_snapshot",
                    arguments='{"incident_id": "INC-1"}',
                ),
            ),
            Message(
                role="function",
                name="get_operational_snapshot",
                content='{"p95_latency_ms": 1840}',
            ),
            Message(
                role="assistant",
                content="",
                function_call=FunctionCall(
                    name="score_bundle_plan",
                    arguments='{"plan_name": "gray-rollout"}',
                ),
            ),
            Message(
                role="function",
                name="score_bundle_plan",
                content='{"score": 80}',
            ),
            Message(role="assistant", content="Final verdict: continue."),
        ]

        def fake_run(messages, **kwargs):
            yield response_msgs

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            list(agent.run([Message(role="user", content="diagnose")]))

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) == 1

        attrs = dict(agent_spans[0].attributes or {})
        output = json.loads(attrs["gen_ai.output.messages"])
        assert len(output) == 1
        assert output[0]["role"] == "assistant"
        assert output[0]["finish_reason"] == "stop"
        assert output[0]["parts"] == [
            {"content": "Final verdict: continue.", "type": "text"}
        ]

    def test_agent_run_without_final_answer_skips_tool_output_messages(
        self, span_exporter, instrument_with_content
    ):
        llm = MagicMock()
        llm.model = "qwen-plus"
        llm.model_type = "qwen_dashscope"
        agent = _StubAgent.create(name="ToolOnlyBot", llm=llm)

        response_msgs = [
            Message(
                role="assistant",
                content="",
                function_call=FunctionCall(
                    name="get_operational_snapshot",
                    arguments='{"incident_id": "INC-1"}',
                ),
            ),
            Message(
                role="function",
                name="get_operational_snapshot",
                content='{"p95_latency_ms": 1840}',
            ),
        ]

        def fake_run(messages, **kwargs):
            yield response_msgs

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            list(agent.run([Message(role="user", content="diagnose")]))

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        assert len(agent_spans) == 1
        attrs = dict(agent_spans[0].attributes or {})
        assert "gen_ai.output.messages" not in attrs

    def test_nested_agent_run_creates_child_invoke_agent_span(
        self, span_exporter, instrument
    ):
        parent_llm = MagicMock()
        parent_llm.model = "qwen-turbo"
        parent_llm.model_type = "qwen_dashscope"
        child_llm = MagicMock()
        child_llm.model = "qwen-plus"
        child_llm.model_type = "qwen_dashscope"

        parent_agent = _StubAgent.create(name="ParentBot", llm=parent_llm)
        child_agent = _StubAgent.create(name="ChildBot", llm=child_llm)

        def fake_run(self, messages, **kwargs):
            if self is parent_agent:
                yield from child_agent.run(
                    [Message(role="user", content="child task")]
                )
                yield [Message(role="assistant", content="parent final")]
            elif self is child_agent:
                yield [Message(role="assistant", content="child final")]
            else:
                yield [Message(role="assistant", content="unexpected")]

        with patch.object(
            _StubAgent, "_run", autospec=True, side_effect=fake_run
        ):
            results = list(
                parent_agent.run([Message(role="user", content="parent task")])
            )

        assert len(results) == 2

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        span_by_name = {s.name: s for s in agent_spans}
        assert set(span_by_name) == {
            "invoke_agent ParentBot",
            "invoke_agent ChildBot",
        }

        parent_span = span_by_name["invoke_agent ParentBot"]
        child_span = span_by_name["invoke_agent ChildBot"]
        assert child_span.parent is not None
        assert child_span.parent.span_id == parent_span.context.span_id


class TestToolCallSpan:
    """Verify ``Agent._call_tool()`` produces an ``execute_tool`` span."""

    def _make_agent_with_tool(self, tool_name="get_weather"):
        agent = _StubAgent.create(name="ToolAgent")
        mock_tool = MagicMock()
        mock_tool.description = "Get weather information"
        mock_tool.call = MagicMock(return_value="Sunny, 25 degrees")
        agent.function_map = {tool_name: mock_tool}
        return agent, mock_tool

    def test_call_tool_creates_execute_tool_span(
        self, span_exporter, instrument
    ):
        agent, _ = self._make_agent_with_tool("get_weather")
        result = agent._call_tool("get_weather", '{"city": "Beijing"}')
        assert result == "Sunny, 25 degrees"

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) >= 1
        span = tool_spans[0]
        assert span.name == "execute_tool get_weather"
        attrs = dict(span.attributes or {})
        assert (
            attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "execute_tool"
        )
        assert_no_removed_telemetry(spans)

    def test_call_tool_with_dict_args(self, span_exporter, instrument):
        agent, mock_tool = self._make_agent_with_tool("search")
        mock_tool.call = MagicMock(return_value="Found 3 results")
        result = agent._call_tool("search", {"query": "OpenTelemetry"})
        assert result == "Found 3 results"

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) >= 1
        assert tool_spans[0].name == "execute_tool search"

    def test_call_tool_error_creates_span(self, span_exporter, instrument):
        agent, mock_tool = self._make_agent_with_tool("broken_tool")
        mock_tool.call = MagicMock(side_effect=RuntimeError("Tool crashed"))

        # qwen-agent catches the exception and returns an error string.
        result = agent._call_tool("broken_tool", "{}")
        assert isinstance(result, str)

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) >= 1
        assert tool_spans[0].name == "execute_tool broken_tool"

    def test_call_tool_returns_content_items(self, span_exporter, instrument):
        agent, mock_tool = self._make_agent_with_tool("image_gen")
        mock_tool.call = MagicMock(
            return_value=[ContentItem(text="Generated image description")]
        )
        result = agent._call_tool("image_gen", '{"prompt": "a cat"}')
        assert isinstance(result, list)

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) >= 1
        assert tool_spans[0].name == "execute_tool image_gen"

    def test_call_unknown_tool_no_crash(self, span_exporter, instrument):
        agent = _StubAgent.create(name="ToolAgent")
        agent.function_map = {}
        result = agent._call_tool("nonexistent", "{}")
        assert isinstance(result, str)

        spans = span_exporter.get_finished_spans()
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(tool_spans) >= 1


class TestSpanHierarchy:
    """Verify spans nest correctly when operations are composed."""

    def test_agent_run_with_llm_call_produces_nested_spans(
        self, span_exporter, instrument
    ):
        model = _StubChatModel(model="qwen-max", model_type="qwen_dashscope")
        agent = _StubAgent.create(name="NestBot", llm=model)

        llm_response = [
            Message(
                role="assistant",
                content="The answer is 42.",
                extra={
                    "model_service_info": {
                        "usage": {"input_tokens": 21, "output_tokens": 7}
                    }
                },
            )
        ]

        def fake_run(messages, **kwargs):
            with patch.object(
                _StubChatModel, "_chat_no_stream", return_value=llm_response
            ):
                agent.llm.chat(messages=messages, stream=False)
            yield [Message(role="assistant", content="The answer is 42.")]

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            list(agent.run([Message(role="user", content="What is 6*7?")]))

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        chat_spans = [s for s in spans if s.name.startswith("chat")]
        assert len(agent_spans) >= 1
        assert len(chat_spans) >= 1

        agent_span = agent_spans[0]
        chat_span = chat_spans[0]
        assert chat_span.context.trace_id == agent_span.context.trace_id
        assert chat_span.parent is not None
        assert chat_span.parent.span_id == agent_span.context.span_id

        # Nested LLM token usage rolls up onto the agent span.
        agent_attrs = dict(agent_span.attributes or {})
        assert agent_attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 21
        assert agent_attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 7

    def test_agent_run_with_tool_call_produces_nested_spans(
        self, span_exporter, instrument
    ):
        agent = _StubAgent.create(name="ToolNestBot")
        mock_tool = MagicMock()
        mock_tool.description = "Calculator tool"
        mock_tool.call = MagicMock(return_value="42")
        agent.function_map = {"calculator": mock_tool}

        def fake_run(messages, **kwargs):
            agent._call_tool("calculator", '{"expr": "6*7"}')
            yield [Message(role="assistant", content="The result is 42.")]

        with patch.object(_StubAgent, "_run", side_effect=fake_run):
            list(agent.run([Message(role="user", content="Calculate 6*7")]))

        spans = span_exporter.get_finished_spans()
        agent_spans = [s for s in spans if "invoke_agent" in s.name]
        tool_spans = [s for s in spans if "execute_tool" in s.name]
        assert len(agent_spans) >= 1
        assert len(tool_spans) >= 1

        agent_span = agent_spans[0]
        tool_span = tool_spans[0]
        assert tool_span.context.trace_id == agent_span.context.trace_id
        assert tool_span.parent is not None
        assert tool_span.parent.span_id == agent_span.context.span_id
        assert_no_removed_telemetry(spans)
