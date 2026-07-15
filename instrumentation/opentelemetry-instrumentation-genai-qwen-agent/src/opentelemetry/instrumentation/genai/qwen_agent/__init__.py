# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
Qwen-Agent instrumentation supporting ``qwen-agent >= 0.0.20``.

Usage
-----
.. code:: python

    from qwen_agent.agents import Assistant

    from opentelemetry.instrumentation.genai.qwen_agent import (
        QwenAgentInstrumentor,
    )

    QwenAgentInstrumentor().instrument()

    bot = Assistant(
        llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
        name="my-assistant",
        system_message="You are a helpful assistant.",
    )

    messages = [{"role": "user", "content": "Hello!"}]
    for responses in bot.run(messages):
        pass

    QwenAgentInstrumentor().uninstrument()

API
---
"""

from __future__ import annotations

import logging
from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.qwen_agent.package import _instruments
from opentelemetry.instrumentation.genai.qwen_agent.patch import (
    wrap_agent_call_tool,
    wrap_agent_run,
    wrap_chat_model_chat,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler

logger = logging.getLogger(__name__)

_AGENT_MODULE = "qwen_agent.agent"
_LLM_MODULE = "qwen_agent.llm.base"

__all__ = ["QwenAgentInstrumentor"]


class QwenAgentInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for the Qwen-Agent framework.

    Instruments the following components:

    - ``Agent.run()``: agent execution spans (``invoke_agent``). ``run_nonstream``
      is not wrapped separately — it calls ``run()`` internally, so the
      ``invoke_agent`` span is created once by the ``run()`` wrapper.
    - ``BaseChatModel.chat()``: LLM call spans (``chat``).
    - ``Agent._call_tool()``: tool execution spans (``execute_tool``).
    """

    def __init__(self) -> None:
        super().__init__()
        self._handler: TelemetryHandler | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        self._handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )

        try:
            wrap_function_wrapper(
                module=_AGENT_MODULE,
                name="Agent.run",
                wrapper=lambda wrapped, instance, args, kwargs: wrap_agent_run(
                    wrapped, instance, args, kwargs, handler=self._handler
                ),
            )
        except Exception as e:
            logger.warning("Failed to instrument Agent.run: %s", e)

        try:
            wrap_function_wrapper(
                module=_LLM_MODULE,
                name="BaseChatModel.chat",
                wrapper=lambda wrapped, instance, args, kwargs: (
                    wrap_chat_model_chat(
                        wrapped, instance, args, kwargs, handler=self._handler
                    )
                ),
            )
        except Exception as e:
            logger.warning("Failed to instrument BaseChatModel.chat: %s", e)

        try:
            wrap_function_wrapper(
                module=_AGENT_MODULE,
                name="Agent._call_tool",
                wrapper=lambda wrapped, instance, args, kwargs: (
                    wrap_agent_call_tool(
                        wrapped, instance, args, kwargs, handler=self._handler
                    )
                ),
            )
        except Exception as e:
            logger.warning("Failed to instrument Agent._call_tool: %s", e)

    def _uninstrument(self, **kwargs: Any) -> None:
        del kwargs
        try:
            import qwen_agent.agent  # noqa: PLC0415

            unwrap(qwen_agent.agent.Agent, "run")
        except Exception as e:
            logger.warning("Failed to uninstrument Agent.run: %s", e)

        try:
            import qwen_agent.llm.base  # noqa: PLC0415

            unwrap(qwen_agent.llm.base.BaseChatModel, "chat")
        except Exception as e:
            logger.warning("Failed to uninstrument BaseChatModel.chat: %s", e)

        try:
            import qwen_agent.agent  # noqa: PLC0415

            unwrap(qwen_agent.agent.Agent, "_call_tool")
        except Exception as e:
            logger.warning("Failed to uninstrument Agent._call_tool: %s", e)

        self._handler = None
