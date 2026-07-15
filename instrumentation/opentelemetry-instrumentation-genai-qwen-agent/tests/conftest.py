# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration for the Qwen-Agent instrumentation package."""

from __future__ import annotations

import os

import pytest

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]

# Set DASHSCOPE_API_KEY before any dashscope/qwen-agent imports. The dashscope
# SDK reads environment variables at module import time and caches them.
if "DASHSCOPE_API_KEY" not in os.environ:
    os.environ["DASHSCOPE_API_KEY"] = "test_dashscope_api_key"


def _patch_vcr_response() -> None:
    """Add the ``version_string`` attribute VCR's HTTP stub is missing.

    Newer urllib3 requires ``version_string`` on HTTP responses, but VCR.py's
    ``VCRHTTPResponse`` stub does not set it, causing an ``AttributeError`` when
    the dashscope SDK streams SSE responses during replay.
    """
    try:
        from vcr.stubs import VCRHTTPResponse  # noqa: PLC0415
    except ImportError:
        return
    if not hasattr(VCRHTTPResponse, "version_string"):
        VCRHTTPResponse.version_string = "HTTP/1.1"


_patch_vcr_response()

from opentelemetry.instrumentation.genai.qwen_agent import (  # noqa: E402
    QwenAgentInstrumentor,
)
from opentelemetry.test_util_genai.instrumentor import (  # noqa: E402
    instrument as _instrument,
)


@pytest.fixture(scope="module")
def vcr_config():
    from opentelemetry.test_util_genai.vcr import (  # noqa: PLC0415
        scrub_response_headers,
    )

    return {
        "filter_headers": [
            ("authorization", "Bearer test_dashscope_api_key"),
            ("x-dashscope-api-key", "test_dashscope_api_key"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers(
            ["x-dashscope-request-id", "set-cookie"]
        ),
    }


@pytest.fixture
def instrument(tracer_provider, logger_provider, meter_provider):
    with _instrument(
        QwenAgentInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_no_content(tracer_provider, logger_provider, meter_provider):
    with _instrument(
        QwenAgentInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_with_content(tracer_provider, logger_provider, meter_provider):
    with _instrument(
        QwenAgentInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_with_content_and_events(
    tracer_provider, logger_provider, meter_provider
):
    with _instrument(
        QwenAgentInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_AND_EVENT",
        emit_event=True,
    ) as instrumentor:
        yield instrumentor
