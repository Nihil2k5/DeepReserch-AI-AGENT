"""Worker: claim queued runs and execute them.

Scaling is horizontal: start more workers, and the ``FOR UPDATE SKIP LOCKED``
claim in the repository guarantees each run goes to exactly one of them. A Redis
lock around execution protects against a stale worker still holding a run it was
already superseded on.

A crashed worker is handled by the stale-run sweep: its run flips back to
``queued`` and is picked up by a healthy worker, which resumes from the last
checkpoint rather than starting over.
"""

from __future__ import annotations

import argparse
import signal
import socket
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

from .agent import ResearchAgent
from .config import get_settings
from .errors import ConflictError, ResearchError
from .logging_config import configure_logging, get_logger, set_run_id
from .models import Run

logger = get_logger(__name__)


class Worker:
    def __init__(
        self,
        agent: ResearchAgent,
        worker_id: str | None = None,
        concurrency: int = 2,
        poll_interval: float = 2.0,
    ) -> None:
        self.agent = agent
        self.worker_id = worker_id or f"{socket.gethostname()}-{id(self) % 10000:04d}"
        self.concurrency = max(1, concurrency)
        self.poll_interval = poll_interval
        self._inflight: dict[Future, str] = {}
        self._lock = threading.Lock()
        self._sweep_counter = 0

    # -- one run ---------------------------------------------------------
    def process(self, run: Run) -> None:
        set_run_id(run.id)
        lock_name = f"lock:run:{run.id}"
        ttl = self.agent.settings.max_wall_clock_s + 120

        try:
            with self._coordinator_lock(lock_name, ttl):
                logger.info("Executing run (attempt %d)", run.attempt)
                self.agent.execute(run.id)
        except ConflictError:
            logger.warning("Run is already being executed elsewhere; skipping")
        except ResearchError as exc:
            logger.warning("Run failed: %s", exc)
        except Exception:
            logger.exception("Unexpected failure executing run")
        finally:
            set_run_id(None)

    def _coordinator_lock(self, name: str, ttl: int):
        if self.agent.coordinator is None:  # pragma: no cover - always set in practice
            from contextlib import nullcontext

            return nullcontext()
        return self.agent.coordinator.lock(name, ttl_s=ttl)

    # -- loop ------------------------------------------------------------
    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        stop = stop_event or threading.Event()
        self._install_signal_handlers(stop)

        logger.info(
            "Worker %s online (concurrency=%d, poll=%.1fs)",
            self.worker_id,
            self.concurrency,
            self.poll_interval,
        )

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            while not stop.is_set():
                self._reap(pool)
                self._sweep_stale(stop)

                with self._lock:
                    busy = len(self._inflight)
                if busy >= self.concurrency:
                    time.sleep(0.2)
                    continue

                run = self.agent.repo.claim_next_run(self.worker_id)
                if run is None:
                    stop.wait(self.poll_interval)
                    continue

                future = pool.submit(self.process, run)
                with self._lock:
                    self._inflight[future] = run.id

        self._drain(pool)
        logger.info("Worker %s shut down cleanly", self.worker_id)

    def run_once(self) -> bool:
        """Claim and execute a single run. Returns False when the queue is empty."""
        run = self.agent.repo.claim_next_run(self.worker_id)
        if run is None:
            return False
        self.process(run)
        return True

    # -- internals -------------------------------------------------------
    def _reap(self, pool: ThreadPoolExecutor) -> None:
        with self._lock:
            finished = [future for future in self._inflight if future.done()]
            for future in finished:
                run_id = self._inflight.pop(future)
                error = future.exception()
                if error is not None:  # already logged in process()
                    logger.debug("Run %s finished with error: %s", run_id, error)

    def _drain(self, pool: ThreadPoolExecutor) -> None:
        with self._lock:
            futures = list(self._inflight)
        for future in futures:
            try:
                future.result(timeout=30)
            except Exception:  # pragma: no cover
                pass
        pool.shutdown(wait=True)

    def _sweep_stale(self, stop: threading.Event) -> None:
        """Periodically requeue runs abandoned by crashed workers."""
        self._sweep_counter += 1
        if self._sweep_counter % 30 != 0:
            return
        try:
            requeued = self.agent.repo.reset_stale_runs(older_than_minutes=30)
            if requeued:
                logger.warning("Requeued %d stale run(s)", requeued)
        except Exception as exc:  # pragma: no cover
            logger.warning("Stale-run sweep failed: %s", exc)

    def _install_signal_handlers(self, stop: threading.Event) -> None:
        def handler(signum, _frame):  # pragma: no cover - signal path
            logger.info("Received signal %s; finishing in-flight runs", signum)
            stop.set()

        for name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is not None:
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):  # pragma: no cover - non-main thread
                    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-worker", description="Run the research worker.")
    parser.add_argument("--once", action="store_true", help="Claim and run a single job, then exit")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--worker-id", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    settings.validate_runtime()

    agent = ResearchAgent.from_settings(settings)
    worker = Worker(
        agent,
        worker_id=args.worker_id or settings.worker_id,
        concurrency=args.concurrency or settings.worker_concurrency,
        poll_interval=settings.worker_poll_interval_s,
    )

    try:
        if args.once:
            did_work = worker.run_once()
            print("processed one run" if did_work else "queue empty")
            return 0 if did_work else 1
        worker.run_forever()
    finally:
        agent.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
