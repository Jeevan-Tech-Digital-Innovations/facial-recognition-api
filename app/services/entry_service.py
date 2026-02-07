from datetime import date, datetime
from typing import Optional, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.entry_log import EntryLog
from app.models.employee import Employee
from app.repositories.entry_log_repo import EntryLogRepository
from app.repositories.employee_repo import EmployeeRepository
from app.core.exceptions import NotFoundException


class EntryService:
    """Service for entry logging business logic."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.entry_repo = EntryLogRepository(db)
        self.employee_repo = EmployeeRepository(db)

    async def log_face_entry(
        self,
        employee_id: int,
        confidence: float,
        device_id: Optional[str] = None,
    ) -> EntryLog:
        """
        Log a face-based entry.
        
        Args:
            employee_id: Internal employee ID (from DB)
            confidence: Match confidence score
            device_id: Optional device/kiosk identifier
            
        Returns:
            Created entry log
        """
        entry_log = EntryLog(
            employee_id=employee_id,
            entry_method="face",
            match_confidence=confidence,
            device_id=device_id,
        )
        return await self.entry_repo.create(entry_log)

    async def log_manual_entry(
        self,
        employee_id_str: str,
        device_id: Optional[str] = None,
    ) -> Tuple[EntryLog, Employee]:
        """
        Log a manual entry using employee ID string.
        
        Args:
            employee_id_str: Employee ID string (e.g., EMP-001)
            device_id: Optional device/kiosk identifier
            
        Returns:
            Tuple of (entry log, employee)
            
        Raises:
            NotFoundException: If employee not found or inactive
        """
        # Find active employee
        employee = await self.employee_repo.get_active_by_employee_id(employee_id_str)
        if not employee:
            raise NotFoundException(
                message=f"Employee with ID '{employee_id_str}' not found or inactive",
                details={"employee_id": employee_id_str},
            )

        # Create entry log
        entry_log = EntryLog(
            employee_id=employee.id,
            entry_method="manual",
            match_confidence=None,  # No confidence for manual entry
            device_id=device_id,
        )
        entry_log = await self.entry_repo.create(entry_log)

        return entry_log, employee

    async def get_today_entries(
        self,
        page: int = 1,
        page_size: int = 100,
    ) -> Tuple[List[EntryLog], int, dict]:
        """
        Get today's entry logs with statistics.
        
        Returns:
            Tuple of (entry logs, total count, stats dict)
        """
        offset = (page - 1) * page_size
        entries, total = await self.entry_repo.get_today_entries(offset, page_size)
        stats = await self.entry_repo.get_stats_by_date(date.today())

        return entries, total, stats

    async def get_entries_by_date(
        self,
        target_date: date,
        page: int = 1,
        page_size: int = 100,
    ) -> Tuple[List[EntryLog], int, dict]:
        """
        Get entry logs for a specific date with statistics.
        
        Returns:
            Tuple of (entry logs, total count, stats dict)
        """
        offset = (page - 1) * page_size
        entries, total = await self.entry_repo.get_by_date(target_date, offset, page_size)
        stats = await self.entry_repo.get_stats_by_date(target_date)

        return entries, total, stats

    async def get_employee_entries(
        self,
        employee_id_str: str,
        days: int = 30,
        page: int = 1,
        page_size: int = 100,
    ) -> Tuple[List[EntryLog], int, Employee]:
        """
        Get entry logs for a specific employee.
        
        Args:
            employee_id_str: Employee ID string
            days: Number of days to look back
            page: Page number
            page_size: Items per page
            
        Returns:
            Tuple of (entry logs, total count, employee)
            
        Raises:
            NotFoundException: If employee not found
        """
        # Find employee
        employee = await self.employee_repo.get_by_employee_id(employee_id_str)
        if not employee:
            raise NotFoundException(
                message=f"Employee with ID '{employee_id_str}' not found"
            )

        offset = (page - 1) * page_size
        entries, total = await self.entry_repo.get_by_employee(
            employee.id, days, offset, page_size
        )

        return entries, total, employee

    async def has_entered_today(self, employee_id_str: str) -> bool:
        """
        Check if an employee has already entered today.
        
        Args:
            employee_id_str: Employee ID string
            
        Returns:
            True if employee has entered today
        """
        employee = await self.employee_repo.get_by_employee_id(employee_id_str)
        if not employee:
            return False

        return await self.entry_repo.employee_entered_today(employee.id)
