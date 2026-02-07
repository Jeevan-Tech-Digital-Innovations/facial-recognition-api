import io
import logging
import re
import uuid
from pathlib import Path
from typing import Tuple, Optional

import aiofiles
import aiofiles.os
import numpy as np
from PIL import Image
from fastapi import UploadFile

from app.core.config import get_settings
from app.core.exceptions import InvalidImageException, BadRequestException

logger = logging.getLogger(__name__)

settings = get_settings()

# Supported image formats
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Image constraints
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
MIN_IMAGE_DIMENSION = 100  # pixels
MAX_IMAGE_DIMENSION = 4096  # pixels
TARGET_SIZE = (640, 640)  # Resize target for face detection

# Pattern for safe directory names (alphanumeric, hyphens, underscores)
_SAFE_DIRNAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _sanitize_path_component(value: str) -> str:
    """
    Sanitize a string for safe use as a filesystem directory/file name.

    Strips path separators and ensures the value cannot escape the
    intended directory via traversal (e.g. '../../etc').

    Args:
        value: Raw string to sanitize

    Returns:
        Sanitized string safe for use in a path

    Raises:
        BadRequestException: If value is empty or contains only unsafe chars
    """
    # Strip whitespace and path separators
    cleaned = value.strip().replace("/", "").replace("\\", "").replace("..", "")

    if not cleaned or not _SAFE_DIRNAME_RE.match(cleaned):
        raise BadRequestException(
            message="Invalid identifier for file storage",
            details={"value": value, "allowed_pattern": "alphanumeric, hyphens, underscores"},
        )

    return cleaned


async def validate_image(file: UploadFile) -> bytes:
    """
    Validate uploaded image file.

    Args:
        file: Uploaded file from FastAPI

    Returns:
        Image bytes if valid

    Raises:
        InvalidImageException: If image is invalid
    """
    # Check content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise InvalidImageException(
            message=f"Invalid image type. Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
            details={"content_type": file.content_type},
        )

    # Check file extension
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise InvalidImageException(
                message=f"Invalid file extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
                details={"extension": ext},
            )

    # Read file content
    content = await file.read()
    await file.seek(0)  # Reset for potential re-read

    # Check file size
    if len(content) > MAX_IMAGE_SIZE:
        raise InvalidImageException(
            message=f"Image too large. Maximum size: {MAX_IMAGE_SIZE // (1024 * 1024)} MB",
            details={"size": len(content)},
        )

    if len(content) == 0:
        raise InvalidImageException(message="Empty image file")

    # Validate image can be opened
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()  # Verify it's a valid image
    except Exception as e:
        raise InvalidImageException(
            message="Corrupted or invalid image file",
            details={"error": str(e)},
        )

    # Re-open after verify (verify closes the file)
    img = Image.open(io.BytesIO(content))

    # Check dimensions
    width, height = img.size
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise InvalidImageException(
            message=f"Image too small. Minimum dimension: {MIN_IMAGE_DIMENSION}px",
            details={"width": width, "height": height},
        )

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise InvalidImageException(
            message=f"Image too large. Maximum dimension: {MAX_IMAGE_DIMENSION}px",
            details={"width": width, "height": height},
        )

    return content


def bytes_to_numpy(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to numpy array (RGB format).

    Applies EXIF orientation transforms so that phone-camera images
    (which store rotation in EXIF metadata rather than rotating the
    actual pixels) are correctly oriented before face detection.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Numpy array in RGB format with correct orientation
    """
    from PIL import ImageOps

    img = Image.open(io.BytesIO(image_bytes))

    # Apply EXIF rotation/flip metadata (e.g. iPhone portrait selfies)
    # Without this, a portrait photo arrives as landscape pixel data with
    # an EXIF tag saying "rotate 90°", causing face detection to fail or
    # produce a wrong-orientation embedding.
    img = ImageOps.exif_transpose(img)

    # Convert to RGB if necessary
    if img.mode != "RGB":
        img = img.convert("RGB")

    return np.array(img)


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.

    Args:
        image: Numpy array image
        target_size: Target (width, height)

    Returns:
        Resized numpy array
    """
    img = Image.fromarray(image)
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    return np.array(img)


async def save_face_image(
    image_bytes: bytes,
    employee_id: str,
    filename_prefix: Optional[str] = None,
) -> str:
    """
    Save face image to disk asynchronously.

    Sanitizes the employee_id to prevent path traversal attacks and
    verifies the final path stays within the configured images directory.

    Args:
        image_bytes: Raw image bytes
        employee_id: Employee ID for folder organization
        filename_prefix: Optional prefix for filename

    Returns:
        Relative path to saved image

    Raises:
        BadRequestException: If employee_id contains path traversal characters
    """
    # Sanitize employee_id to prevent path traversal
    safe_id = _sanitize_path_component(employee_id)

    # Build path and resolve to catch any remaining traversal
    base_dir = Path(settings.FACE_IMAGES_DIR).resolve()
    employee_dir = (base_dir / safe_id).resolve()

    # Verify the resolved path is still within the base directory
    if not str(employee_dir).startswith(str(base_dir)):
        logger.error(
            "Path traversal attempt detected: employee_id='%s' resolved to '%s'",
            employee_id, employee_dir,
        )
        raise BadRequestException(message="Invalid employee ID for file storage")

    # Create directory asynchronously
    await aiofiles.os.makedirs(employee_dir, exist_ok=True)

    # Sanitize prefix as well
    safe_prefix = ""
    if filename_prefix:
        safe_prefix = re.sub(r"[^a-zA-Z0-9_\-]", "", filename_prefix) + "_"

    # Generate unique filename
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{safe_prefix}{unique_id}.jpg"
    filepath = employee_dir / filename

    # Convert to JPEG bytes
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Save to bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=95)
    jpeg_bytes = buffer.getvalue()

    # Write asynchronously
    async with aiofiles.open(filepath, "wb") as f:
        await f.write(jpeg_bytes)

    logger.info("Saved face image: %s", filepath)

    # Return relative path
    return str(filepath)


async def delete_face_image(image_path: str) -> bool:
    """
    Delete a face image from disk asynchronously.

    Verifies the path is within the configured images directory before deleting.

    Args:
        image_path: Path to the image

    Returns:
        True if deleted, False if not found
    """
    try:
        base_dir = Path(settings.FACE_IMAGES_DIR).resolve()
        path = Path(image_path).resolve()

        # Verify the path is within the base directory
        if not str(path).startswith(str(base_dir)):
            logger.warning("Attempted to delete file outside image dir: %s", image_path)
            return False

        if await aiofiles.os.path.exists(path):
            await aiofiles.os.remove(path)
            logger.info("Deleted face image: %s", path)
            return True
        return False
    except Exception:
        logger.exception("Error deleting face image: %s", image_path)
        return False
