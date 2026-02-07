from fastapi import APIRouter, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import Optional

from app.core.database import get_db
from app.core.config import get_settings
from app.services.face_service import FaceService
from app.services.entry_service import EntryService
from app.repositories.face_embedding_repo import FaceEmbeddingRepository
from app.schemas.recognize import RecognizeResponse, RecognizeFailedResponse
from app.utils.image_utils import validate_image
from app.core.exceptions import FaceNotDetectedException, MultipleFacesException

settings = get_settings()
router = APIRouter()


@router.post("/debug")
async def debug_recognize(
    face_image: UploadFile = File(..., description="Face image to test"),
    db: AsyncSession = Depends(get_db),
):
    """Debug endpoint to check face distances without threshold."""
    image_bytes = await validate_image(face_image)
    face_service = FaceService()
    
    try:
        embedding = await face_service.detect_and_embed(image_bytes)
    except Exception as e:
        return {"error": str(e)}
    
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
    
    # Get all faces with distances (no threshold)
    query = text(f"""
        SELECT fe.id, e.employee_id, e.name,
               fe.embedding <=> '{embedding_str}'::vector AS distance
        FROM face_embeddings fe
        JOIN employees e ON fe.employee_id = e.id
        WHERE e.is_active = true
        ORDER BY distance ASC
        LIMIT 5
    """)
    
    result = await db.execute(query)
    rows = result.fetchall()
    
    return {
        "current_threshold": settings.FACE_MATCH_THRESHOLD,
        "matches": [
            {
                "id": row.id,
                "employee_id": row.employee_id,
                "name": row.name,
                "distance": float(row.distance),
                "would_match": float(row.distance) < settings.FACE_MATCH_THRESHOLD
            }
            for row in rows
        ]
    }


@router.post(
    "",
    responses={
        200: {"model": RecognizeResponse, "description": "Face recognized successfully"},
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
    5. If not found, returns 404 with suggestion to use manual entry
    
    The image should contain exactly one face for best results.
    """
    # Validate image
    image_bytes = await validate_image(face_image)

    # Initialize services
    face_service = FaceService()
    embedding_repo = FaceEmbeddingRepository(db)

    try:
        # Detect face and generate embedding
        embedding = await face_service.detect_and_embed(image_bytes)
    except FaceNotDetectedException:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "No face detected in the image. Please ensure your face is clearly visible.",
                "suggestion": "Try again with better lighting or use manual entry.",
            },
        )
    except MultipleFacesException:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Multiple faces detected. Please provide an image with only one face.",
                "suggestion": "Ensure only one person is in the frame.",
            },
        )

    # Search for matching face
    match_result = await embedding_repo.find_best_match(
        embedding=embedding,
        threshold=settings.FACE_MATCH_THRESHOLD,
    )

    if not match_result:
        return JSONResponse(
            status_code=404,
            content=RecognizeFailedResponse(
                success=False,
                message="Face not recognized. Please use manual entry.",
                suggestion="Use POST /api/entry/manual with your employee ID",
            ).model_dump(),
        )

    face_embedding, distance = match_result
    employee = face_embedding.employee
    confidence = face_service.distance_to_confidence(distance)

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
