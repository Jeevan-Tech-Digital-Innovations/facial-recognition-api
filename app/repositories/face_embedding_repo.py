from typing import Optional, List, Tuple
from sqlalchemy import select, func, delete, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models.face_embedding import FaceEmbedding
from app.models.employee import Employee


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
        
        Args:
            embedding: 512-dimensional face embedding vector
            threshold: Maximum cosine distance for a match (lower = more similar)
            limit: Maximum number of results
            
        Returns:
            List of (FaceEmbedding, distance) tuples, sorted by distance ascending
        """
        # Use pgvector's cosine distance operator (<=>)
        # Cosine distance = 1 - cosine_similarity, so lower is more similar
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        
        # Use string formatting for the vector literal since asyncpg doesn't handle it well with params
        # First, let's get all matches without threshold to debug
        query = text(f"""
            SELECT fe.id, fe.employee_id, fe.image_path, fe.is_primary, fe.created_at,
                   fe.embedding <=> '{embedding_str}'::vector AS distance
            FROM face_embeddings fe
            JOIN employees e ON fe.employee_id = e.id
            WHERE e.is_active = true
              AND fe.embedding <=> '{embedding_str}'::vector < :threshold
            ORDER BY distance ASC
            LIMIT :limit
        """)
        
        # Log for debugging
        import logging
        logging.info(f"Searching with threshold: {threshold}")

        result = await self.db.execute(
            query,
            {"threshold": threshold, "limit": limit}
        )
        rows = result.fetchall()

        embeddings_with_distance = []
        for row in rows:
            # Fetch the full FaceEmbedding object with employee relationship
            emb = await self.get_by_id(row.id)
            if emb:
                embeddings_with_distance.append((emb, row.distance))

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
        # Unset existing primary
        await self.db.execute(
            text("""
                UPDATE face_embeddings 
                SET is_primary = false 
                WHERE employee_id = :employee_id AND is_primary = true
            """),
            {"employee_id": employee_id}
        )

        # Set new primary
        await self.db.execute(
            text("""
                UPDATE face_embeddings 
                SET is_primary = true 
                WHERE id = :embedding_id
            """),
            {"embedding_id": embedding_id}
        )
        await self.db.flush()
        return True
