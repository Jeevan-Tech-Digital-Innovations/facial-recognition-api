import logging
from typing import Optional

from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.config import get_settings
from app.core.exceptions import (
    FaceNotDetectedException,
    FaceNotRecognizedException,
    MultipleFacesException,
)
from app.services.face_service import get_face_service
from app.services.entry_service import EntryService
from app.repositories.face_embedding_repo import FaceEmbeddingRepository
from app.models.face_embedding import FaceEmbedding
from app.models.employee import Employee
from app.schemas.recognize import RecognizeResponse, RecognizeFailedResponse
from app.utils.image_utils import validate_image

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


@router.post("/debug")
async def debug_recognize(
    face_image: UploadFile = File(..., description="Face image to test"),
    db: AsyncSession = Depends(get_db),
):
    """Debug endpoint to check face distances without threshold.

    WARNING: This endpoint should be removed or protected in production.
    """
    image_bytes = await validate_image(face_image)
    face_service = get_face_service()

    embedding = await face_service.detect_and_embed(image_bytes)

    # Use pgvector's ORM cosine_distance (safe, parameterized)
    distance_expr = FaceEmbedding.embedding.cosine_distance(embedding)

    query = (
        select(
            FaceEmbedding.id,
            Employee.employee_id,
            Employee.name,
            distance_expr.label("distance"),
        )
        .join(Employee, FaceEmbedding.employee_id == Employee.id)
        .where(Employee.is_active == True)
        .order_by(distance_expr.asc())
        .limit(5)
    )

    result = await db.execute(query)
    rows = result.all()

    return {
        "current_threshold": settings.FACE_MATCH_THRESHOLD,
        "matches": [
            {
                "id": row.id,
                "employee_id": row.employee_id,
                "name": row.name,
                "distance": float(row.distance),
                "would_match": float(row.distance) < settings.FACE_MATCH_THRESHOLD,
            }
            for row in rows
        ],
    }


@router.post(
    "",
    response_model=RecognizeResponse,
    responses={
        200: {"model": RecognizeResponse, "description": "Face recognized successfully"},
        400: {"description": "No face detected or multiple faces"},
        404: {"model": RecognizeFailedResponse, "description": "Face not recognized"},
    },
)
async def recognize_face(
    face_image: UploadFile = File(..., description="Face image to recognize"),
    device_id: Optional[str] = Form(None, description="Optional kiosk/device ID"),
    log_entry: bool = Form(True, description="Whether to log the entry"),
    db: AsyncSession = Depends(get_db),
):
    """
    Recognize an employee from their face image.

    This endpoint:
    1. Detects the face in the uploaded image
    2. Generates a face embedding
    3. Searches for the closest match in the database
    4. If found, optionally logs the entry and returns employee info
    5. If not found, raises 404 with suggestion to use manual entry

    The image should contain exactly one face for best results.

    Raises FaceNotDetectedException (400) or MultipleFacesException (400)
    through the global exception handler for consistent error responses.
    """
    # Validate image
    image_bytes = await validate_image(face_image)

    # Use singleton face service (consistent model instance)
    face_service = get_face_service()
    embedding_repo = FaceEmbeddingRepository(db)

    # Detect face and generate embedding
    # FaceNotDetectedException / MultipleFacesException propagate
    # to the global exception handler for a consistent error response
    embedding = await face_service.detect_and_embed(image_bytes)

    # Search for matching face
    match_result = await embedding_repo.find_best_match(
        embedding=embedding,
        threshold=settings.FACE_MATCH_THRESHOLD,
    )

    if not match_result:
        raise FaceNotRecognizedException(
            message="Face not recognized. Please use manual entry.",
            details={"suggestion": "Use POST /api/entry/manual with your employee ID"},
        )

    face_embedding, distance = match_result
    employee = face_embedding.employee
    confidence = face_service.distance_to_confidence(distance)

    logger.info(
        "Face recognized: employee=%s confidence=%.4f",
        employee.employee_id, confidence,
    )

    # Log entry if requested
    entry_logged = False
    if log_entry:
        entry_service = EntryService(db)
        await entry_service.log_face_entry(
            employee_id=employee.id,
            confidence=confidence,
            device_id=device_id,
        )
        entry_logged = True

    return RecognizeResponse(
        success=True,
        employee_id=employee.employee_id,
        name=employee.name,
        department=employee.department,
        confidence=round(confidence, 4),
        entry_logged=entry_logged,
        message="Face recognized successfully",
    )
