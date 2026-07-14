"""FastAPI entry point (System Design §2.12, §6).

The API process. Routers map REST resources to service calls; a route body is
validate -> call -> serialize, no business logic. Owns the WebSocket event relay
and serves the SPA. Bootstrap scope: app factory, health + info routes, and the
container wired into app state. Resource routers land in Phase 1.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from vds.container import build_container
from vds.logging import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.container = build_container()
    log.info("api.startup")
    yield
    app.state.container.gpu.unload_all()
    log.info("api.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AutoDataForge",
        version="0.1.0",
        lifespan=_lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/info")
    def info() -> dict[str, object]:
        container = app.state.container
        return {
            "environment": container.settings.environment,
            "models": container.models.describe(),
            "llm_provider": container.settings.llm.provider,
            "vram_budget_mb": container.settings.gpu.vram_budget_mb,
        }

    # Phase 1: app.include_router(projects.router), ingest, review, export, ws...
    return app


app = create_app()
