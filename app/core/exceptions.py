from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Any, Optional


class AppException(Exception):
    """Base exception for application errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Any] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class NotFoundException(AppException):
    """Resource not found."""

    def __init__(self, message: str = "Resource not found", details: Optional[Any] = None):
        super().__init__(message=message, status_code=404, details=details)


class BadRequestException(AppException):
    """Bad request - invalid input."""

    def __init__(self, message: str = "Bad request", details: Optional[Any] = None):
        super().__init__(message=message, status_code=400, details=details)


class ConflictException(AppException):
    """Conflict - resource already exists."""

    def __init__(self, message: str = "Resource already exists", details: Optional[Any] = None):
        super().__init__(message=message, status_code=409, details=details)


class FaceNotDetectedException(AppException):
    """No face detected in image."""

    def __init__(self, message: str = "No face detected in the image", details: Optional[Any] = None):
        super().__init__(message=message, status_code=400, details=details)


class FaceNotRecognizedException(AppException):
    """Face not recognized - no match found."""

    def __init__(self, message: str = "Face not recognized", details: Optional[Any] = None):
        super().__init__(message=message, status_code=404, details=details)


class MultipleFacesException(AppException):
    """Multiple faces detected in image."""

    def __init__(self, message: str = "Multiple faces detected, please provide image with single face", details: Optional[Any] = None):
        super().__init__(message=message, status_code=400, details=details)


class InvalidImageException(AppException):
    """Invalid image format or corrupted image."""

    def __init__(self, message: str = "Invalid image format", details: Optional[Any] = None):
        super().__init__(message=message, status_code=400, details=details)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Global exception handler for AppException."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.message,
            "details": exc.details,
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "details": str(exc) if str(exc) else None,
        },
    )
