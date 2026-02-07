from datetime import date, datetime
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.enums import EntryMethod
from app.services.entry_service import EntryService
from app.schemas.entry import (
    ManualEntryRequest,
    ManualEntryResponse,
    EntryLogResponse,
    EntryLogsResponse,
    EmployeeEntryLogsResponse,
    EntryStatsSummary,
)
from app.schemas.common import APIResponse

router = APIRouter()


def _entry_to_response(entry) -> EntryLogResponse:
    """Map an EntryLog ORM object to its response schema."""
    return EntryLogResponse(
        id=entry.id,
        employee_id=entry.employee.employee_id,
        employee_name=entry.employee.name,
        entry_method=entry.entry_method,
        match_confidence=entry.match_confidence,
        device_id=entry.device_id,
        entry_time=entry.entry_time,
    )


@router.post("/manual", response_model=ManualEntryResponse)
async def manual_entry(
    request: ManualEntryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Log a manual entry using employee ID.

    This is the fallback endpoint when face recognition fails.
    The employee enters their ID manually to log their entry.
    """
    service = EntryService(db)
    entry_log, employee = await service.log_manual_entry(
        employee_id_str=request.employee_id,
        device_id=request.device_id,
    )

    return ManualEntryResponse(
        success=True,
        message="Manual entry logged successfully",
        employee_id=employee.employee_id,
        employee_name=employee.name,
        entry_time=entry_log.entry_time,
    )


@router.get("/logs", response_model=EntryLogsResponse)
async def get_today_entries(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=500, description="Items per page"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get today's entry logs with statistics.

    Returns a list of all entries for today along with
    counts of face-based vs manual entries.
    """
    service = EntryService(db)
    entries, total, stats = await service.get_today_entries(page, page_size)

    return EntryLogsResponse(
        items=[_entry_to_response(entry) for entry in entries],
        total=total,
        date=date.today(),
        face_entries=stats["face_entries"],
        manual_entries=stats["manual_entries"],
    )


@router.get("/logs/date/{target_date}", response_model=EntryLogsResponse)
async def get_entries_by_date(
    target_date: date,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=500, description="Items per page"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get entry logs for a specific date.

    Returns a list of all entries for the specified date along with
    counts of face-based vs manual entries.
    """
    service = EntryService(db)
    entries, total, stats = await service.get_entries_by_date(target_date, page, page_size)

    return EntryLogsResponse(
        items=[_entry_to_response(entry) for entry in entries],
        total=total,
        date=target_date,
        face_entries=stats["face_entries"],
        manual_entries=stats["manual_entries"],
    )


@router.get(
    "/logs/employee/{employee_id}",
    response_model=EmployeeEntryLogsResponse,
)
async def get_employee_entries(
    employee_id: str,
    days: int = Query(30, ge=1, le=365, description="Days to look back"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=500, description="Items per page"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get entry history for a specific employee.

    Returns the employee's entry logs for the specified number of days.
    """
    service = EntryService(db)
    entries, total, employee = await service.get_employee_entries(
        employee_id, days, page, page_size
    )

    total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0

    # Calculate stats for this employee (using enum instead of magic string)
    face_entries = sum(1 for e in entries if e.entry_method == EntryMethod.FACE)
    manual_entries = len(entries) - face_entries

    return EmployeeEntryLogsResponse(
        employee_id=employee.employee_id,
        employee_name=employee.name,
        items=[_entry_to_response(entry) for entry in entries],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        days_range=days,
        face_entries=face_entries,
        manual_entries=manual_entries,
    )


@router.get("/stats", response_model=EntryStatsSummary)
async def get_today_stats(
    db: AsyncSession = Depends(get_db),
):
    """
    Get entry statistics summary for today.

    Returns counts and percentages of face vs manual entries.
    """
    service = EntryService(db)
    _, _, stats = await service.get_today_entries(page=1, page_size=1)

    return EntryStatsSummary(
        total_entries=stats["total_entries"],
        face_entries=stats["face_entries"],
        manual_entries=stats["manual_entries"],
        face_percentage=round(stats["face_percentage"], 2),
        unique_employees=stats["unique_employees"],
    )


@router.get("/stats/date/{target_date}", response_model=EntryStatsSummary)
async def get_stats_by_date(
    target_date: date,
    db: AsyncSession = Depends(get_db),
):
    """
    Get entry statistics summary for a specific date.
    """
    service = EntryService(db)
    _, _, stats = await service.get_entries_by_date(target_date, page=1, page_size=1)

    return EntryStatsSummary(
        total_entries=stats["total_entries"],
        face_entries=stats["face_entries"],
        manual_entries=stats["manual_entries"],
        face_percentage=round(stats["face_percentage"], 2),
        unique_employees=stats["unique_employees"],
    )
