from datetime import date, datetime, time, timedelta
from typing import Optional, List, Tuple

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.config import get_settings
from app.core.enums import EntryMethod
from app.models.entry_log import EntryLog
from app.models.employee import Employee

settings = get_settings()


def _day_range(target_date: date) -> tuple[datetime, datetime]:
    """
    Return timezone-aware start/end datetimes for a given date
    in the application's configured timezone (APP_TIMEZONE).

    When PostgreSQL compares a ``timestamptz`` column against these
    timezone-aware boundaries, it correctly converts both sides to
    the same reference point.  So if APP_TIMEZONE=Asia/Kolkata,
    querying for "2026-02-08" builds the range
    ``2026-02-08 00:00:00+05:30`` to ``2026-02-08 23:59:59+05:30``,
    which matches entries stored at ``2026-02-07T18:30Z`` and later.
    """
    tz = settings.tz
    start_of_day = datetime.combine(target_date, time.min, tzinfo=tz)
    end_of_day = datetime.combine(target_date, time.max, tzinfo=tz)
    return start_of_day, end_of_day


def _today() -> date:
    """Return today's date in the application's configured timezone."""
    return datetime.now(settings.tz).date()


class EntryLogRepository:
    """Repository for entry log data access."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, entry_log: EntryLog) -> EntryLog:
        """Create a new entry log."""
        self.db.add(entry_log)
        await self.db.flush()
        await self.db.refresh(entry_log)
        return entry_log

    async def get_by_id(self, entry_id: int) -> Optional[EntryLog]:
        """Get entry log by ID."""
        result = await self.db.execute(
            select(EntryLog)
            .options(joinedload(EntryLog.employee))
            .where(EntryLog.id == entry_id)
        )
        return result.scalar_one_or_none()

    async def get_by_date(
        self,
        target_date: date,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[EntryLog], int]:
        """
        Get entry logs for a specific date.

        Returns:
            Tuple of (entry logs list, total count)
        """
        start_of_day, end_of_day = _day_range(target_date)

        # Base filter
        date_filter = and_(
            EntryLog.entry_time >= start_of_day,
            EntryLog.entry_time <= end_of_day,
        )

        # Get total count
        count_result = await self.db.execute(
            select(func.count(EntryLog.id)).where(date_filter)
        )
        total = count_result.scalar() or 0

        # Get paginated results
        result = await self.db.execute(
            select(EntryLog)
            .options(joinedload(EntryLog.employee))
            .where(date_filter)
            .order_by(EntryLog.entry_time.desc())
            .offset(offset)
            .limit(limit)
        )
        entries = list(result.scalars().all())

        return entries, total

    async def get_by_employee(
        self,
        employee_id: int,
        days: int = 30,
        offset: int = 0,
        limit: int = 100,
    ) -> Tuple[List[EntryLog], int]:
        """
        Get entry logs for a specific employee.

        Args:
            employee_id: Internal employee ID
            days: Number of days to look back
            offset: Pagination offset
            limit: Pagination limit

        Returns:
            Tuple of (entry logs list, total count)
        """
        start_date = datetime.now(settings.tz) - timedelta(days=days)

        # Base filter
        filters = and_(
            EntryLog.employee_id == employee_id,
            EntryLog.entry_time >= start_date,
        )

        # Get total count
        count_result = await self.db.execute(
            select(func.count(EntryLog.id)).where(filters)
        )
        total = count_result.scalar() or 0

        # Get paginated results
        result = await self.db.execute(
            select(EntryLog)
            .options(joinedload(EntryLog.employee))
            .where(filters)
            .order_by(EntryLog.entry_time.desc())
            .offset(offset)
            .limit(limit)
        )
        entries = list(result.scalars().all())

        return entries, total

    async def get_stats_by_date(self, target_date: date) -> dict:
        """
        Get entry statistics for a specific date.

        Returns:
            Dictionary with total, face_entries, manual_entries, unique_employees
        """
        start_of_day, end_of_day = _day_range(target_date)

        date_filter = and_(
            EntryLog.entry_time >= start_of_day,
            EntryLog.entry_time <= end_of_day,
        )

        # Total entries
        total_result = await self.db.execute(
            select(func.count(EntryLog.id)).where(date_filter)
        )
        total = total_result.scalar() or 0

        # Face entries
        face_result = await self.db.execute(
            select(func.count(EntryLog.id)).where(
                and_(date_filter, EntryLog.entry_method == EntryMethod.FACE)
            )
        )
        face_entries = face_result.scalar() or 0

        # Manual entries
        manual_entries = total - face_entries

        # Unique employees
        unique_result = await self.db.execute(
            select(func.count(func.distinct(EntryLog.employee_id))).where(date_filter)
        )
        unique_employees = unique_result.scalar() or 0

        return {
            "total_entries": total,
            "face_entries": face_entries,
            "manual_entries": manual_entries,
            "unique_employees": unique_employees,
            "face_percentage": (face_entries / total * 100) if total > 0 else 0.0,
        }

    async def get_today_entries(
        self, offset: int = 0, limit: int = 100
    ) -> Tuple[List[EntryLog], int]:
        """Get today's entry logs using the configured timezone."""
        return await self.get_by_date(_today(), offset, limit)

    async def employee_entered_today(self, employee_id: int) -> bool:
        """Check if employee has already entered today."""
        start_of_day, end_of_day = _day_range(_today())

        result = await self.db.execute(
            select(func.count(EntryLog.id)).where(
                and_(
                    EntryLog.employee_id == employee_id,
                    EntryLog.entry_time >= start_of_day,
                    EntryLog.entry_time <= end_of_day,
                )
            )
        )
        return (result.scalar() or 0) > 0
