"""Postgres connection pool and migration runner.

Deliberately small and dependency-light: ``psycopg`` only, no ORM. A LIFO queue
of autocommit connections is plenty for a worker pool, and it keeps the SQL in
plain sight, which matters for the queue-claim query.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .errors import StorageError
from .logging_config import get_logger

logger = get_logger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"


class Database:
    def __init__(self, dsn: str, size: int = 8, connect_timeout: int = 10) -> None:
        self.dsn = dsn
        self._size = max(1, size)
        self._connect_timeout = connect_timeout
        self._idle: queue.LifoQueue = queue.LifoQueue(maxsize=self._size)
        self._created = 0
        self._lock = threading.Lock()

    # -- pool ------------------------------------------------------------
    def _new_connection(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover
            raise StorageError("psycopg is required: pip install 'psycopg[binary]'") from exc

        return psycopg.connect(
            self.dsn,
            autocommit=True,
            connect_timeout=self._connect_timeout,
            row_factory=dict_row,
        )

    @staticmethod
    def _is_dead(conn: Any) -> bool:
        return bool(getattr(conn, "closed", False))

    def _checkout(self):
        while True:
            try:
                conn = self._idle.get_nowait()
            except queue.Empty:
                conn = None

            if conn is not None:
                if self._is_dead(conn):
                    with self._lock:
                        self._created -= 1
                    continue
                return conn

            with self._lock:
                if self._created < self._size:
                    self._created += 1
                    try:
                        return self._new_connection()
                    except Exception:
                        self._created -= 1
                        raise

            # Pool saturated: wait for a connection to come back.
            try:
                conn = self._idle.get(timeout=30)
            except queue.Empty as exc:
                raise StorageError("Timed out waiting for a database connection") from exc

            if not self._is_dead(conn):
                return conn
            with self._lock:
                self._created -= 1

    def _checkin(self, conn: Any, healthy: bool) -> None:
        if healthy and not self._is_dead(conn):
            try:
                self._idle.put_nowait(conn)
                return
            except queue.Full:
                pass
        try:
            conn.close()
        except Exception:  # pragma: no cover
            pass
        with self._lock:
            self._created -= 1

    @contextmanager
    def connection(self) -> Iterator[Any]:
        conn = self._checkout()
        try:
            yield conn
        except Exception:
            self._checkin(conn, healthy=False)
            raise
        else:
            self._checkin(conn, healthy=True)

    # -- convenience -----------------------------------------------------
    def execute(self, sql: str, params: tuple | None = None) -> list[dict]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                if cur.description is None:
                    return []
                return list(cur.fetchall())

    def execute_one(self, sql: str, params: tuple | None = None) -> dict | None:
        rows = self.execute(sql, params)
        return rows[0] if rows else None

    def ping(self) -> bool:
        try:
            self.execute("SELECT 1")
            return True
        except Exception as exc:
            logger.warning("Database ping failed: %s", exc)
            return False

    # -- migrations ------------------------------------------------------
    def migrate(self, directory: Path | None = None) -> list[str]:
        """Apply any unapplied ``*.sql`` files, in filename order."""
        directory = directory or MIGRATIONS_DIR
        if not directory.exists():
            raise StorageError(f"Migrations directory not found: {directory}")

        applied: list[str] = []
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        name TEXT PRIMARY KEY,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cur.execute("SELECT name FROM schema_migrations")
                done = {row["name"] for row in cur.fetchall()}

            for path in sorted(directory.glob("*.sql")):
                if path.name in done:
                    continue
                with conn.cursor() as cur:
                    cur.execute(path.read_text(encoding="utf-8"))
                    cur.execute(
                        "INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,)
                    )
                applied.append(path.name)
                logger.info("Applied migration %s", path.name)

        return applied

    def close(self) -> None:
        while True:
            try:
                conn = self._idle.get_nowait()
            except queue.Empty:
                break
            try:
                conn.close()
            except Exception:  # pragma: no cover
                pass
        with self._lock:
            self._created = 0
