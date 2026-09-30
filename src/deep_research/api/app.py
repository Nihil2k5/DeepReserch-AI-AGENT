"""FastAPI application factory and error handling."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..config import get_settings
from ..errors import ResearchError, error_payload
from ..logging_config import configure_logging, get_logger
from .deps import get_agent, reset_agent
from .routes import router

logger = get_logger(__name__)

#: Bundled single-page dashboard (same-origin, so no CORS setup needed).
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting %s (%s)", settings.app_name, settings.environment)
    # Fail fast if the infrastructure isn't reachable: better to refuse traffic
    # than to accept runs that can never execute.
    agent = get_agent()
    health = agent.health()
    if not health["database"]:
        logger.error("Database is unreachable at startup; runs will fail until it recovers")
    yield
    reset_agent()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Durable multi-agent web research: planning, parallel search, evidence "
            "verification, cited synthesis, plus span tracing and run budgets."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)

    if STATIC_DIR.is_dir():
        app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def index() -> RedirectResponse:
            return RedirectResponse(url="/ui/")
    else:  # pragma: no cover - only when static assets are missing
        logger.warning("Static UI directory not found at %s; /ui disabled", STATIC_DIR)

    @app.exception_handler(ResearchError)
    async def research_error_handler(request: Request, exc: ResearchError) -> JSONResponse:
        logger.warning("%s: %s", exc.__class__.__name__, exc)
        return JSONResponse(
            status_code=getattr(exc, "status_code", 500),
            content=error_payload(exc),
        )

    return app


app = create_app()
