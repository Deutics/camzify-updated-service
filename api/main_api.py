from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.routes.service_routes import router as service_router
from utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern lifespan handler for app startup and shutdown.
    Handles logging and graceful service cleanup.
    """
    logger.info(" FastAPI application startup — service API available at /service/*")
    try:
        yield
    except Exception as e:
        logger.exception(f"Error during app runtime: {e}")
        raise
    finally:
        logger.info("FastAPI application shutdown — stopping active service if any.")
        try:
            from api.service_controller import service_controller
            service_controller.stop_service()
            logger.info("Service stopped successfully on shutdown.")
        except Exception as e:
            logger.exception(f"Error while stopping service on shutdown: {e}")


def create_app() -> FastAPI:
    """Factory function to create and configure the FastAPI app."""
    app = FastAPI(
        title="Camzify Video Analytics API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Allow Swagger/UI from anywhere
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    #  Redirect root to service page
    @app.get("/", include_in_schema=False)
    async def root_redirect():
        """Redirect root URL to the service status endpoint."""
        return RedirectResponse(url="/docs", status_code=307)

    #  Include your service router (prefix/tags already inside router file)
    app.include_router(service_router)

    return app


#  Uvicorn entry point
app = create_app()
