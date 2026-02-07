import io
import uuid
from pathlib import Path
from typing import Tuple, Optional

import aiofiles
import aiofiles.os
import numpy as np
from PIL import Image
from fastapi import UploadFile

from app.core.config import get_settings
from app.core.exceptions import InvalidImageException

settings = get_settings()

# Supported image formats
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Image constraints
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
MIN_IMAGE_DIMENSION = 100  # pixels
MAX_IMAGE_DIMENSION = 4096  # pixels
TARGET_SIZE = (640, 640)  # Resize target for face detection


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
            details={"content_type": file.content_type}
        )

    # Check file extension
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise InvalidImageException(
                message=f"Invalid file extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
                details={"extension": ext}
            )

    # Read file content
    content = await file.read()
    await file.seek(0)  # Reset for potential re-read

    # Check file size
    if len(content) > MAX_IMAGE_SIZE:
        raise InvalidImageException(
            message=f"Image too large. Maximum size: {MAX_IMAGE_SIZE // (1024 * 1024)} MB",
            details={"size": len(content)}
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
            details={"error": str(e)}
        )

    # Re-open after verify (verify closes the file)
    img = Image.open(io.BytesIO(content))

    # Check dimensions
    width, height = img.size
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise InvalidImageException(
            message=f"Image too small. Minimum dimension: {MIN_IMAGE_DIMENSION}px",
            details={"width": width, "height": height}
        )

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise InvalidImageException(
            message=f"Image too large. Maximum dimension: {MAX_IMAGE_DIMENSION}px",
            details={"width": width, "height": height}
        )

    return content


def bytes_to_numpy(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to numpy array (RGB format).
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Numpy array in RGB format
    """
    img = Image.open(io.BytesIO(image_bytes))

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
    filename_prefix: Optional[str] = None
) -> str:
    """
    Save face image to disk asynchronously.
    
    Args:
        image_bytes: Raw image bytes
        employee_id: Employee ID for folder organization
        filename_prefix: Optional prefix for filename
        
    Returns:
        Relative path to saved image
    """
    # Create directory structure
    base_dir = Path(settings.FACE_IMAGES_DIR)
    employee_dir = base_dir / employee_id
    
    # Create directory asynchronously
    await aiofiles.os.makedirs(employee_dir, exist_ok=True)

    # Generate unique filename
    unique_id = uuid.uuid4().hex[:8]
    prefix = f"{filename_prefix}_" if filename_prefix else ""
    filename = f"{prefix}{unique_id}.jpg"
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

    # Return relative path
    return str(filepath)


async def delete_face_image(image_path: str) -> bool:
    """
    Delete a face image from disk asynchronously.
    
    Args:
        image_path: Path to the image
        
    Returns:
        True if deleted, False if not found
    """
    try:
        path = Path(image_path)
        if await aiofiles.os.path.exists(path):
            await aiofiles.os.remove(path)
            return True
        return False
    except Exception:
        return False
