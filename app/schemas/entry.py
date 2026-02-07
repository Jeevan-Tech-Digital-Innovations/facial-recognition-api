from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, Field


class ManualEntryRequest(BaseModel):
    """Request schema for manual entry."""

    employee_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Employee ID for manual entry",
        examples=["EMP-001"],
    )
    device_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Optional kiosk/device identifier",
    )


class EntryLogResponse(BaseModel):
    """Response schema for a single entry log."""

    id: int
    employee_id: str = Field(..., description="Employee ID string")
    employee_name: str = Field(..., description="Employee name")
    entry_method: str = Field(..., description="Entry method: 'face' or 'manual'")
    match_confidence: Optional[float] = Field(
        None, description="Match confidence (null for manual entries)"
    )
    device_id: Optional[str] = Field(None, description="Device/kiosk ID")
    entry_time: datetime

    class Config:
        from_attributes = True


class ManualEntryResponse(BaseModel):
    """Response schema for successful manual entry."""

    success: bool = True
    message: str = "Manual entry logged successfully"
    employee_id: str
    employee_name: str
    entry_time: datetime


class EntryLogsResponse(BaseModel):
    """Response schema for entry logs list."""

    items: List[EntryLogResponse]
    total: int
    date: date
    face_entries: int = Field(..., description="Count of face-based entries")
    manual_entries: int = Field(..., description="Count of manual entries")


class EmployeeEntryLogsResponse(BaseModel):
    """Response schema for a specific employee's entry logs."""

    employee_id: str
    employee_name: str
    items: List[EntryLogResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    days_range: int = Field(..., description="Number of days included in the query")
    face_entries: int = Field(..., description="Count of face-based entries in current page")
    manual_entries: int = Field(..., description="Count of manual entries in current page")


class EntryStatsSummary(BaseModel):
    """Summary statistics for entries."""

    total_entries: int
    face_entries: int
    manual_entries: int
    face_percentage: float = Field(..., description="Percentage of face-based entries")
    unique_employees: int
