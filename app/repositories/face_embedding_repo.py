import logging
import math
from typing import Optional, List, Tuple

from sqlalchemy import select, func, delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models.face_embedding import FaceEmbedding
from app.models.employee import Employee

logger = logging.getLogger(__name__)


def _validate_embedding(embedding: List[float]) -> None:
    """
    Validate that an embedding vector contains only finite float values.

    Raises:
        ValueError: If embedding contains non-finite or non-numeric values
    """
    if not embedding:
        raise ValueError("Embedding vector must not be empty")

    for i, val in enumerate(embedding):
        if not isinstance(val, (int, float)):
            raise ValueError(f"Embedding value at index {i} is not numeric: {type(val)}")
        if not math.isfinite(val):
            raise ValueError(f"Embedding value at index {i} is not finite: {val}")


class FaceEmbeddingRepository:
    """Repository for face embedding data access with pgvector operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, embedding: FaceEmbedding) -> FaceEmbedding:
        """Create a new face embedding."""
        self.db.add(embedding)
        await self.db.flush()
        await self.db.refresh(embedding)
        return embedding

    async def get_by_id(self, embedding_id: int) -> Optional[FaceEmbedding]:
        """Get face embedding by ID."""
        result = await self.db.execute(
            select(FaceEmbedding)
            .options(joinedload(FaceEmbedding.employee))
            .where(FaceEmbedding.id == embedding_id)
        )
        return result.scalar_one_or_none()

    async def get_by_employee_id(self, employee_id: int) -> List[FaceEmbedding]:
        """Get all face embeddings for an employee."""
        result = await self.db.execute(
            select(FaceEmbedding)
            .where(FaceEmbedding.employee_id == employee_id)
            .order_by(FaceEmbedding.is_primary.desc(), FaceEmbedding.created_at.desc())
        )
        return list(result.scalars().all())

    async def count_by_employee_id(self, employee_id: int) -> int:
        """Count face embeddings for an employee."""
        result = await self.db.execute(
            select(func.count(FaceEmbedding.id)).where(
                FaceEmbedding.employee_id == employee_id
            )
        )
        return result.scalar() or 0

    async def find_similar(
        self,
        embedding: List[float],
        threshold: float = 0.55,
        limit: int = 1,
    ) -> List[Tuple[FaceEmbedding, float]]:
        """
        Find similar face embeddings using pgvector cosine distance.

        Uses pgvector's SQLAlchemy ORM integration for safe parameterized
        vector queries (no string interpolation).

        Args:
            embedding: 512-dimensional face embedding vector
            threshold: Maximum cosine distance for a match (lower = more similar)
            limit: Maximum number of results

        Returns:
            List of (FaceEmbedding, distance) tuples, sorted by distance ascending
        """
        # Validate embedding values are all finite numbers
        _validate_embedding(embedding)

        # Use pgvector's SQLAlchemy cosine_distance method which properly
        # parameterizes the vector value -- no f-string SQL interpolation
        distance_expr = FaceEmbedding.embedding.cosine_distance(embedding)

        # Step 1: Find matching IDs with distances using parameterized query
        query = (
            select(FaceEmbedding.id, distance_expr.label("distance"))
            .join(Employee, FaceEmbedding.employee_id == Employee.id)
            .where(Employee.is_active == True)
            .where(distance_expr < threshold)
            .order_by(distance_expr.asc())
            .limit(limit)
        )

        logger.info("Searching with threshold=%.3f, limit=%d", threshold, limit)

        result = await self.db.execute(query)
        rows = result.all()

        # Step 2: Fetch full objects with employee relationship
        embeddings_with_distance = []
        for row in rows:
            emb = await self.get_by_id(row.id)
            if emb:
                embeddings_with_distance.append((emb, float(row.distance)))

        return embeddings_with_distance

    async def find_best_match(
        self,
        embedding: List[float],
        threshold: float = 0.55,
    ) -> Optional[Tuple[FaceEmbedding, float]]:
        """
        Find the best matching face embedding.

        Args:
            embedding: 512-dimensional face embedding vector
            threshold: Maximum cosine distance for a match

        Returns:
            Tuple of (FaceEmbedding, distance) or None if no match found
        """
        results = await self.find_similar(embedding, threshold, limit=1)
        return results[0] if results else None

    async def delete(self, embedding_id: int) -> bool:
        """Delete a face embedding."""
        result = await self.db.execute(
            delete(FaceEmbedding).where(FaceEmbedding.id == embedding_id)
        )
        await self.db.flush()
        return result.rowcount > 0

    async def delete_by_employee_id(self, employee_id: int) -> int:
        """Delete all face embeddings for an employee."""
        result = await self.db.execute(
            delete(FaceEmbedding).where(FaceEmbedding.employee_id == employee_id)
        )
        await self.db.flush()
        return result.rowcount

    async def set_primary(self, embedding_id: int, employee_id: int) -> bool:
        """
        Set a face embedding as primary for an employee.
        Unsets any existing primary embedding.
        """
        # Unset existing primary (using ORM update instead of raw text)
        await self.db.execute(
            update(FaceEmbedding)
            .where(FaceEmbedding.employee_id == employee_id, FaceEmbedding.is_primary == True)
            .values(is_primary=False)
        )

        # Set new primary
        await self.db.execute(
            update(FaceEmbedding)
            .where(FaceEmbedding.id == embedding_id)
            .values(is_primary=True)
        )
        await self.db.flush()
        return True
