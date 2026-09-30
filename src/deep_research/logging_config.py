"""Logging with run correlation.

Every log line from inside a run carries ``run=<id>`` thanks to a
contextvar-backed filter, which makes interleaved worker logs readable.
"""

from __future__ import annotations

import contextvars
import logging
import sys

_CONFIGURED = False

run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)


class RunContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = run_id_var.get() or "-"
        return True


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RunContextFilter())
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s [run=%(run_id)s] %(name)s :: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for noisy in ("httpx", "httpcore", "uvicorn.access", "langfuse"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_run_id(run_id: str | None):
    """Bind a run id to the current context; returns the reset token."""
    return run_id_var.set(run_id)
