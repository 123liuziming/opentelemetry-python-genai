# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test utility functions."""

from __future__ import annotations

from typing import List

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def find_spans_by_name_prefix(
    spans: List[ReadableSpan], prefix: str
) -> List[ReadableSpan]:
    """Find spans whose name starts with ``prefix``."""
    return [span for span in spans if span.name.startswith(prefix)]


def find_spans_by_operation(
    spans: List[ReadableSpan], operation_name: str
) -> List[ReadableSpan]:
    """Find spans by the ``gen_ai.operation.name`` attribute."""
    return [
        span
        for span in spans
        if span.attributes
        and span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == operation_name
    ]


def assert_no_removed_telemetry(spans: List[ReadableSpan]) -> None:
    """Assert that removed span kinds/attributes never appear.

    The migrated instrumentation drops the ``gen_ai.span.kind`` attribute
    (keeping only the standard ``gen_ai.operation.name``) and no longer emits
    ``react step`` spans.
    """
    for span in spans:
        assert "gen_ai.span.kind" not in (span.attributes or {}), (
            f"gen_ai.span.kind must not be set (span={span.name})"
        )
        assert span.name != "react step", (
            "react step spans must not be emitted"
        )
        assert (span.attributes or {}).get(
            "gen_ai.operation.name"
        ) != "react", "react operation must not be emitted"
