from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr


class EmployeeBase(BaseModel):
    """Base employee schema with common fields."""

    name: str = Field(..., min_length=1, max_length=255, description="Employee full name")
    department: Optional[str] = Field(None, max_length=100, description="Department name")
    email: Optional[EmailStr] = Field(None, description="Employee email address")
    phone: Optional[str] = Field(None, max_length=20, description="Phone number")


class EmployeeCreate(EmployeeBase):
    """Schema for creating a new employee."""

    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique employee ID (e.g., EMP-001)",
        examples=["EMP-001", "JT-2024-001"],
    )


class EmployeeUpdate(BaseModel):
    """Schema for updating an employee. All fields optional."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    department: Optional[str] = Field(None, max_length=100)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, max_length=20)
    is_active: Optional[bool] = None


class FaceEmbeddingInfo(BaseModel):
    """Face embedding info for employee response."""

    id: int
    image_path: str
    is_primary: bool
    created_at: datetime

    class Config:
        from_attributes = True


class EmployeeResponse(BaseModel):
    """Schema for employee response."""

    id: int
    employee_id: str
    name: str
    department: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime
    face_count: int = Field(default=0, description="Number of registered face images")

    class Config:
        from_attributes = True


class EmployeeDetailResponse(EmployeeResponse):
    """Detailed employee response with face embeddings info."""

    face_embeddings: List[FaceEmbeddingInfo] = []

    class Config:
        from_attributes = True


class EmployeeListResponse(BaseModel):
    """Response for list of employees."""

    items: List[EmployeeResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
