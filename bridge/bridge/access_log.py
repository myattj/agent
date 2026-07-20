"""Access-log hardening for public HTTP endpoints.

Uvicorn's default access logger includes the raw query string in its request
target.  OAuth callbacks carry short-lived credentials in that query string,
so production logs must remove it before any handler formats the record.
"""

from __future__ import annotations

import logging


class QueryStringRedactionFilter(logging.Filter):
    """Remove query strings from Uvicorn's five-argument access records."""

    _redacts_uvicorn_query_strings = True

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if not isinstance(args, tuple) or len(args) != 5:
            return True

        client_addr, method, target, http_version, status_code = args
        if isinstance(target, str):
            path, separator, _query = target.partition("?")
            if separator:
                record.args = (
                    client_addr,
                    method,
                    path,
                    http_version,
                    status_code,
                )
        return True


def install_uvicorn_access_log_redaction() -> None:
    """Install the filter once on Uvicorn's access logger.

    Uvicorn configures logging before it imports the ASGI application, so the
    filter remains attached for every request handled by each worker process.
    """

    logger = logging.getLogger("uvicorn.access")
    if any(
        getattr(item, "_redacts_uvicorn_query_strings", False)
        for item in logger.filters
    ):
        return
    logger.addFilter(QueryStringRedactionFilter())
