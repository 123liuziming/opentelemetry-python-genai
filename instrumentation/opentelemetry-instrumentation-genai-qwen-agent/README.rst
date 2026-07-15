OpenTelemetry Qwen-Agent Instrumentation
=========================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-qwen-agent.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-qwen-agent/

This library allows tracing agent, LLM, and tool calls made by the
`Qwen-Agent <https://github.com/QwenLM/Qwen-Agent>`_ framework. It supports
``qwen-agent >= 0.0.20`` and emits telemetry through the shared
``opentelemetry-util-genai`` utilities following the GenAI semantic conventions.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-qwen-agent

Usage
-----

.. code-block:: python

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


Content capture
---------------

This instrumentation follows the shared ``opentelemetry-util-genai`` content
capture controls. Set ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to
select whether prompt/response content is attached to spans and/or events; see
the ``opentelemetry-util-genai`` documentation for the supported capture modes.

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry Python Examples <https://github.com/open-telemetry/opentelemetry-python/tree/main/docs/examples>`_
