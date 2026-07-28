from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from typing import AsyncGenerator

from app.core.config import get_settings

settings = get_settings()

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


async def init_db() -> None:
    """Create pgvector extension, tables, and indexes if they don't exist.

    Idempotent - safe to run on every startup. Ensures a fresh database
    (e.g. a new staging/production instance) is usable without running
    setup_db.py manually.
    """
    from sqlalchemy import text

    # Import models so they register on Base.metadata
    from app.models import Employee, FaceEmbedding, EntryLog  # noqa: F401

    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text(
            """
            CREATE INDEX IF NOT EXISTS idx_face_embeddings_vector
            ON face_embeddings
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_entry_logs_entry_time "
            "ON entry_logs (entry_time DESC)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_entry_logs_employee_time "
            "ON entry_logs (employee_id, entry_time DESC)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_entry_logs_payroll_synced "
            "ON entry_logs (payroll_synced) WHERE payroll_synced = false"
        ))


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
