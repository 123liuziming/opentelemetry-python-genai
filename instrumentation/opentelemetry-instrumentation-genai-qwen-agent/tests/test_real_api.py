# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for Qwen-Agent instrumentation using VCR cassettes.

These tests replay recorded qwen-agent DashScope interactions so they run
without hitting the real API.
"""

from __future__ import annotations

import json

import pytest
from qwen_agent.agents import Assistant
from qwen_agent.llm import get_chat_model
from qwen_agent.tools.base import BaseTool, register_tool

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from ._test_helpers import assert_no_removed_telemetry


@pytest.mark.vcr()
def test_qwen_agent_basic_run(span_exporter, instrument):
    bot = Assistant(
        llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
        name="TestAssistant",
    )
    messages = [{"role": "user", "content": "Hello, what is 1+1?"}]
    list(bot.run(messages))

    spans = span_exporter.get_finished_spans()
    assert len(spans) >= 2

    agent_spans = [s for s in spans if "invoke_agent" in s.name]
    assert len(agent_spans) >= 1
    assert (
        agent_spans[0].attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )

    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1
    chat_span = chat_spans[0]
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "chat"
    )
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == "dashscope"
    )
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "qwen-max"
    )
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL)
        is not None
    )
    assert (
        chat_span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )
        is not None
    )
    assert_no_removed_telemetry(spans)


@pytest.mark.vcr()
def test_qwen_agent_stream_llm_with_ttft(span_exporter, instrument):
    llm = get_chat_model({"model": "qwen-max", "model_type": "qwen_dashscope"})
    messages = [{"role": "user", "content": "Say hello in one word."}]
    list(llm.chat(messages=messages, stream=True))

    spans = span_exporter.get_finished_spans()
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1
    chat_span = chat_spans[0]
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "chat"
    )
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == "dashscope"
    )
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "qwen-max"
    )
    assert (
        chat_span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )
        is not None
    )


@pytest.mark.vcr()
def test_non_stream_chat(span_exporter, instrument):
    llm = get_chat_model(
        {
            "model": "qwen-max",
            "model_type": "qwen_dashscope",
            "generate_cfg": {"use_raw_api": False},
        }
    )
    messages = [
        {
            "role": "user",
            "content": "What is 2+2? Answer with just the number.",
        }
    ]
    list(llm.chat(messages=messages, stream=False))

    spans = span_exporter.get_finished_spans()
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1
    chat_span = chat_spans[0]
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "chat"
    )
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == "dashscope"
    )
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "qwen-max"
    )
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL)
        is not None
    )
    assert (
        chat_span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )
        is not None
    )


@pytest.mark.vcr()
def test_agent_run_nonstream(span_exporter, instrument):
    bot = Assistant(
        llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
        name="NonStreamAssistant",
    )
    messages = [{"role": "user", "content": "Say 'OK' and nothing else."}]
    bot.run_nonstream(messages)

    spans = span_exporter.get_finished_spans()
    agent_spans = [
        s for s in spans if s.name == "invoke_agent NonStreamAssistant"
    ]
    assert len(agent_spans) >= 1
    assert (
        agent_spans[0].attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1


@pytest.mark.vcr()
def test_multi_turn_conversation(span_exporter, instrument):
    bot = Assistant(
        llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
        name="MultiTurnAssistant",
    )
    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What is my name?"},
    ]
    list(bot.run(messages))

    spans = span_exporter.get_finished_spans()
    agent_spans = [
        s for s in spans if s.name == "invoke_agent MultiTurnAssistant"
    ]
    assert len(agent_spans) >= 1

    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1
    chat_span = chat_spans[0]
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "qwen-max"
    )
    assert (
        chat_span.attributes.get(
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        )
        is not None
    )


@pytest.mark.vcr()
def test_react_multi_round(span_exporter, instrument):
    """A tool-enabled agent should nest chat/execute_tool spans under the
    single ``invoke_agent`` span — no ``react step`` spans are emitted."""

    @register_tool("calculator_react_test")
    class CalculatorTool(BaseTool):
        description = "Evaluate a simple arithmetic expression and return the numeric result."
        parameters = [
            {
                "name": "expression",
                "type": "string",
                "description": "The arithmetic expression to evaluate, e.g. '3 * 7'.",
                "required": True,
            }
        ]

        def call(self, params, **kwargs):
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except Exception:
                    params = {"expression": params}
            expr = params.get("expression", "0")
            try:
                return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
            except Exception as e:
                return f"Error: {e}"

    bot = Assistant(
        llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
        name="ReactAgent",
        function_list=["calculator_react_test"],
    )
    messages = [{"role": "user", "content": "What is 6 multiplied by 7?"}]
    list(bot.run(messages))

    spans = span_exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    agent_spans = [s for s in spans if s.name == "invoke_agent ReactAgent"]
    assert len(agent_spans) >= 1, span_names

    # No react step spans in the migrated instrumentation.
    assert [s for s in spans if s.name == "react step"] == []

    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1, span_names
    assert_no_removed_telemetry(spans)


@pytest.mark.vcr()
def test_qwen_agent_with_tool_call(span_exporter, instrument):
    @register_tool("get_current_weather_test")
    class GetCurrentWeatherTool(BaseTool):
        description = "Get the current weather for a given city."
        parameters = [
            {
                "name": "city",
                "type": "string",
                "description": "The city name to get weather for.",
                "required": True,
            }
        ]

        def call(self, params, **kwargs):
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except Exception:
                    params = {"city": params}
            city = (
                params.get("city", "unknown")
                if isinstance(params, dict)
                else "unknown"
            )
            return f"The weather in {city} is sunny and 22 degrees Celsius."

    bot = Assistant(
        llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
        name="WeatherAgent",
        function_list=["get_current_weather_test"],
    )
    messages = [
        {
            "role": "user",
            "content": "What is the weather in Beijing right now?",
        }
    ]
    list(bot.run(messages))

    spans = span_exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    agent_spans = [s for s in spans if "invoke_agent" in s.name]
    assert len(agent_spans) >= 1, span_names

    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1, span_names
    chat_span = chat_spans[0]
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "chat"
    )
    assert (
        chat_span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == "dashscope"
    )

    tool_spans = [s for s in spans if "execute_tool" in s.name]
    for tool_span in tool_spans:
        assert (
            tool_span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
            == "execute_tool"
        )
