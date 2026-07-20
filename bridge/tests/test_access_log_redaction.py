"""Security regression tests for Uvicorn access-log query redaction."""

from __future__ import annotations

import io
import logging

from uvicorn.logging import AccessFormatter

from bridge.access_log import (
    QueryStringRedactionFilter,
    install_uvicorn_access_log_redaction,
)


def _render_access_log(target: str) -> str:
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(
        AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s')
    )
    logger = logging.Logger("test.uvicorn.access")
    logger.addFilter(QueryStringRedactionFilter())
    logger.addHandler(handler)
    logger.info(
        '%s - "%s %s HTTP/%s" %d',
        "127.0.0.1:12345",
        "GET",
        target,
        "1.1",
        302,
    )
    return output.getvalue()


def test_access_log_strips_the_entire_query_string() -> None:
    rendered = _render_access_log(
        "/slack/oauth/callback?code=oauth-secret&state=signed-state"
    )

    assert "GET /slack/oauth/callback HTTP/1.1" in rendered
    assert "302" in rendered
    assert "?" not in rendered
    assert "oauth-secret" not in rendered
    assert "signed-state" not in rendered


def test_access_log_preserves_a_path_without_a_query_string() -> None:
    rendered = _render_access_log("/healthz")

    assert "GET /healthz HTTP/1.1" in rendered
    assert "302" in rendered


def test_installation_is_idempotent() -> None:
    logger = logging.getLogger("uvicorn.access")

    install_uvicorn_access_log_redaction()
    install_uvicorn_access_log_redaction()

    matching_filters = [
        item for item in logger.filters if isinstance(item, QueryStringRedactionFilter)
    ]
    assert len(matching_filters) == 1
