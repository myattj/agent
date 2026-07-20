"""Regression coverage for production GenAI trace payload redaction."""

from __future__ import annotations

import json
from pathlib import Path

from opentelemetry.instrumentation.botocore.extensions.bedrock_utils import (
    genai_capture_message_content,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from strands.telemetry.tracer import REDACTED_VALUE, Tracer


_MANIFEST = Path(__file__).resolve().parents[3] / "agentcore" / "agentcore.json"
_POLICY_ENV = "OTEL_SEMCONV_STABILITY_OPT_IN"
_REDACT_ALL_GENAI_CONTENT = "gen_ai_unredacted_attributes="
_BOTOCORE_CAPTURE_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"


def _runtime_env() -> dict[str, str]:
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    env_vars = manifest["runtimes"][0].get("envVars", [])
    return {item["name"]: item["value"] for item in env_vars}


def test_runtime_redacts_genai_payloads_but_keeps_usage(monkeypatch) -> None:
    """System, user, and output text must not reach the OTel exporter."""
    policy = _runtime_env().get(_POLICY_ENV)
    assert policy == _REDACT_ALL_GENAI_CONTENT
    monkeypatch.setenv(_POLICY_ENV, policy)

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = Tracer()
    # Keep this test isolated from any process-global provider configured by
    # pytest plugins or the AWS OTel distro.
    tracer.tracer = provider.get_tracer("test.telemetry-privacy")

    sentinels = (
        "SENSITIVE_SYSTEM_SENTINEL",
        "SENSITIVE_USER_SENTINEL",
        "SENSITIVE_OUTPUT_SENTINEL",
    )
    span = tracer.start_model_invoke_span(
        [{"role": "user", "content": [{"text": sentinels[1]}]}],
        model_id="test-model",
        system_prompt=sentinels[0],
    )
    tracer.end_model_invoke_span(
        span,
        {"role": "assistant", "content": [{"text": sentinels[2]}]},
        {"inputTokens": 7, "outputTokens": 3, "totalTokens": 10},
        {"latencyMs": 12},
        "end_turn",
    )

    finished = exporter.get_finished_spans()
    assert len(finished) == 1
    exported_events = [
        value
        for event in finished[0].events
        for value in event.attributes.values()
    ]
    serialized_events = repr(exported_events)

    assert all(sentinel not in serialized_events for sentinel in sentinels)
    assert exported_events.count(REDACTED_VALUE) == 3
    assert finished[0].attributes["gen_ai.usage.input_tokens"] == 7
    assert finished[0].attributes["gen_ai.usage.output_tokens"] == 3
    assert finished[0].attributes["gen_ai.server.request.duration"] == 12


def test_runtime_disables_botocore_bedrock_message_capture(monkeypatch) -> None:
    """AWS Distro botocore spans must not emit Bedrock request/response bodies."""
    capture_policy = _runtime_env().get(_BOTOCORE_CAPTURE_ENV)
    assert capture_policy == "false"
    monkeypatch.setenv(_BOTOCORE_CAPTURE_ENV, capture_policy)

    assert genai_capture_message_content() is False
