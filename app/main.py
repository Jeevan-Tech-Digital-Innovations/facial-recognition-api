import logging
import sys
import os
from pathlib import Path

# Setup path for direct script execution
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
os.chdir(_project_root)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.exceptions import AppException, app_exception_handler, generic_exception_handler

# Initialize structured logging before anything else
setup_logging()
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events."""
    # Startup
    Path(settings.FACE_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    logger.info("Starting Facial Recognition API...")

    # Ensure database schema exists (idempotent)
    from app.core.database import init_db
    try:
        await init_db()
        logger.info("Database schema verified/created")
    except Exception as e:
        logger.error("Database initialization failed: %s", e)
        raise

    yield

    # Shutdown
    logger.info("Shutting down Facial Recognition API...")


# Create FastAPI app
app = FastAPI(
    title="Facial Recognition API",
    description="Employee registration and facial recognition system for canteen entry",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
cors_origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register exception handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Import and register routers
from app.routes import router as api_router
app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Facial Recognition API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "facial-recognition-api",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=True,
    )
