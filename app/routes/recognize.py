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
from app.services.payroll_service import get_payroll_service
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
    payroll_synced = False
    payroll_punch_id = None

    if log_entry:
        entry_service = EntryService(db)
        entry = await entry_service.log_face_entry(
            employee_id=employee.id,
            confidence=confidence,
            device_id=device_id,
        )
        entry_logged = True

        # Server-to-server: trigger payroll punch
        payroll_service = get_payroll_service()
        if payroll_service.is_configured and employee.payroll_emp_no:
            try:
                direction = await payroll_service.determine_punch_direction(
                    employee.payroll_emp_no, device_id or ""
                )
                punch_result = await payroll_service.record_punch(
                    emp_no=employee.payroll_emp_no,
                    punch_direction=direction,
                    device_username=device_id or "",
                    face_match_confidence=confidence,
                )
                if punch_result:
                    payroll_synced = True
                    payroll_punch_id = (
                        punch_result.get("data", {}).get("punchId")
                        if isinstance(punch_result.get("data"), dict)
                        else None
                    )
                    # Update entry log with sync status
                    entry.payroll_synced = True
                    entry.payroll_punch_id = payroll_punch_id
                    await db.flush()
                else:
                    entry.payroll_synced = False
                    entry.sync_error = "Payroll API returned error"
                    await db.flush()
            except Exception as e:
                logger.error("Payroll sync failed: %s", str(e))
                entry.payroll_synced = False
                entry.sync_error = str(e)[:500]
                await db.flush()

    return RecognizeResponse(
        success=True,
        employee_id=employee.employee_id,
        name=employee.name,
        department=employee.department,
        confidence=round(confidence, 4),
        entry_logged=entry_logged,
        message="Face recognized successfully",
    )