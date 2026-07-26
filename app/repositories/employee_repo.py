from typing import Optional, List, Tuple
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.employee import Employee
from app.models.face_embedding import FaceEmbedding


class EmployeeRepository:
    """Repository for employee data access."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, employee: Employee) -> Employee:
        """Create a new employee."""
        self.db.add(employee)
        await self.db.flush()
        await self.db.refresh(employee)
        return employee

    async def get_by_id(self, employee_id: int) -> Optional[Employee]:
        """Get employee by internal ID."""
        result = await self.db.execute(
            select(Employee)
            .options(selectinload(Employee.face_embeddings))
            .where(Employee.id == employee_id)
        )
        return result.scalar_one_or_none()

    async def get_by_employee_id(self, employee_id: str) -> Optional[Employee]:
        """Get employee by employee_id string (e.g., EMP-001)."""
        result = await self.db.execute(
            select(Employee)
            .options(selectinload(Employee.face_embeddings))
            .where(Employee.employee_id == employee_id)
        )
        return result.scalar_one_or_none()

    async def get_active_by_employee_id(self, employee_id: str) -> Optional[Employee]:
        """Get active employee by employee_id string."""
        result = await self.db.execute(
            select(Employee)
            .options(selectinload(Employee.face_embeddings))
            .where(Employee.employee_id == employee_id, Employee.is_active == True)
        )
        return result.scalar_one_or_none()

    async def get_all(
        self,
        offset: int = 0,
        limit: int = 20,
        active_only: bool = True,
    ) -> Tuple[List[Employee], int]:
        """
        Get all employees with pagination.
        
        Returns:
            Tuple of (employees list, total count)
        """
        # Base query
        query = select(Employee)
        count_query = select(func.count(Employee.id))

        if active_only:
            query = query.where(Employee.is_active == True)
            count_query = count_query.where(Employee.is_active == True)

        # Get total count
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        # Get paginated results
        query = query.order_by(Employee.created_at.desc()).offset(offset).limit(limit)
        result = await self.db.execute(query)
        employees = list(result.scalars().all())

        # Get face counts for each employee
        for emp in employees:
            face_count_result = await self.db.execute(
                select(func.count(FaceEmbedding.id)).where(
                    FaceEmbedding.employee_id == emp.id
                )
            )
            emp.face_count = face_count_result.scalar() or 0

        return employees, total

    async def update(self, employee_id: int, **kwargs) -> Optional[Employee]:
        """Update employee by internal ID."""
        await self.db.execute(
            update(Employee).where(Employee.id == employee_id).values(**kwargs)
        )
        await self.db.flush()
        return await self.get_by_id(employee_id)

    async def soft_delete(self, employee_id: int) -> bool:
        """Soft delete employee (set is_active=False)."""
        result = await self.db.execute(
            update(Employee)
            .where(Employee.id == employee_id)
            .values(is_active=False)
        )
        await self.db.flush()
        return result.rowcount > 0

    async def get_by_payroll_emp_no(self, payroll_emp_no: str) -> Optional[Employee]:
        """Get employee by payroll employee number."""
        result = await self.db.execute(
            select(Employee)
            .options(selectinload(Employee.face_embeddings))
            .where(Employee.payroll_emp_no == payroll_emp_no)
        )
        return result.scalar_one_or_none()

    async def upsert_by_payroll_emp_no(
        self,
        payroll_emp_no: str,
        name: str,
        department: Optional[str] = None,
        email: Optional[str] = None,
        phone: Optional[str] = None,
    ) -> Employee:
        """Upsert employee by payroll_emp_no. Creates or updates."""
        existing = await self.get_by_payroll_emp_no(payroll_emp_no)
        if existing:
            existing.name = name
            if department is not None:
                existing.department = department
            if email is not None:
                existing.email = email
            if phone is not None:
                existing.phone = phone
            await self.db.flush()
            await self.db.refresh(existing)
            return existing
        else:
            new_emp = Employee(
                employee_id=f"SYNC-{payroll_emp_no}",
                name=name,
                department=department,
                email=email,
                phone=phone,
                payroll_emp_no=payroll_emp_no,
            )
            self.db.add(new_emp)
            await self.db.flush()
            await self.db.refresh(new_emp)
            return new_emp

    async def get_all_with_payroll_enrollment(self) -> List[dict]:
        """Get all payroll-synced employees with enrollment status."""
        result = await self.db.execute(
            select(Employee)
            .options(selectinload(Employee.face_embeddings))
            .where(Employee.payroll_emp_no.isnot(None))
            .order_by(Employee.name)
        )
        employees = list(result.scalars().all())
        return [
            {
                "payroll_emp_no": emp.payroll_emp_no,
                "name": emp.name,
                "department": emp.department,
                "email": emp.email,
                "enrolled": len(emp.face_embeddings) > 0 if emp.face_embeddings else False,
                "face_count": len(emp.face_embeddings) if emp.face_embeddings else 0,
                "is_active": emp.is_active,
            }
            for emp in employees
        ]

    async def exists_by_employee_id(self, employee_id: str) -> bool:
        """Check if employee exists by employee_id string."""
        result = await self.db.execute(
            select(func.count(Employee.id)).where(Employee.employee_id == employee_id)
        )
        return (result.scalar() or 0) > 0
