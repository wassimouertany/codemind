"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from codemind.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm heavy models once, tear down on shutdown.

    On a 4GB card this is what keeps the reranker off the critical path:
    it is loaded here, on CPU, and never reloaded per request.
    """
    settings = get_settings()
    app.state.settings = settings
    # TODO(week1): app.state.embedder / reranker / qdrant client
    yield
    # TODO: close clients


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="CodeMind AI",
        version="0.1.0",
        docs_url="/docs" if settings.codemind_env != "prod" else None,
        lifespan=lifespan,
    )

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "env": settings.codemind_env}

    return app


app = create_app()
