from typing import Optional
from pydantic import BaseModel, Field


class RecognizeResponse(BaseModel):
    """Response schema for face recognition."""

    success: bool = True
    employee_id: str = Field(..., description="Matched employee ID")
    name: str = Field(..., description="Employee name")
    department: Optional[str] = Field(None, description="Employee department")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Match confidence score (0-1, higher is better)",
    )
    entry_logged: bool = Field(
        default=True, description="Whether entry was logged"
    )
    message: str = Field(default="Face recognized successfully")


class RecognizeFailedResponse(BaseModel):
    """Response schema for failed face recognition."""

    success: bool = False
    message: str = Field(
        default="Face not recognized. Please use manual entry.",
        description="Error message",
    )
    suggestion: str = Field(
        default="Use POST /api/entry/manual with your employee ID",
        description="Suggested action",
    )
