import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.payroll_service import get_payroll_service
from app.repositories.employee_repo import EmployeeRepository

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/employees")
async def sync_employees_from_payroll(
    db: AsyncSession = Depends(get_db),
):
    """
    Sync active employees from Payroll API into the Face Recognition DB.

    - Fetches all active employees from Payroll
    - Upserts them by payroll_emp_no (creates new or updates existing)
    - Returns sync summary
    """
    payroll_service = get_payroll_service()
    if not payroll_service.is_configured:
        return {
            "success": False,
            "message": "Payroll integration is not configured",
            "data": None,
        }

    employees = await payroll_service.get_active_employees()
    if employees is None:
        return {
            "success": False,
            "message": "Failed to fetch employees from Payroll API",
            "data": None,
        }

    emp_repo = EmployeeRepository(db)
    created_count = 0
    updated_count = 0
    error_count = 0

    for emp in employees:
        try:
            emp_no = emp.get("empNo") or emp.get("employeeNo") or emp.get("employeeId")
            name = emp.get("employeeName") or emp.get("name") or f"Unknown-{emp_no}"
            department = emp.get("departmentName") or emp.get("department")
            email = emp.get("email")
            phone = emp.get("phone") or emp.get("mobile")

            if not emp_no:
                logger.warning("Skipping employee without empNo: %s", emp)
                error_count += 1
                continue

            existing = await emp_repo.get_by_payroll_emp_no(emp_no)
            await emp_repo.upsert_by_payroll_emp_no(
                payroll_emp_no=str(emp_no),
                name=name,
                department=department,
                email=email,
                phone=phone,
            )
            if existing:
                updated_count += 1
            else:
                created_count += 1

        except Exception as e:
            logger.error("Error syncing employee: %s - %s", emp, str(e))
            error_count += 1

    await db.commit()

    result = {
        "success": True,
        "message": f"Sync complete: {created_count} created, {updated_count} updated, {error_count} errors",
        "data": {
            "total_fetched": len(employees),
            "created": created_count,
            "updated": updated_count,
            "errors": error_count,
        },
    }
    logger.info("Employee sync: %s", result["message"])
    return result


@router.get("/enrollment-status")
async def get_enrollment_status(
    db: AsyncSession = Depends(get_db),
):
    """
    Get enrollment status of all payroll-synced employees.

    Returns which payroll employees have face data enrolled and which don't.
    """
    emp_repo = EmployeeRepository(db)
    employees = await emp_repo.get_all_with_payroll_enrollment()

    enrolled = [e for e in employees if e["enrolled"]]
    not_enrolled = [e for e in employees if not e["enrolled"]]

    return {
        "success": True,
        "message": f"{len(enrolled)} enrolled, {len(not_enrolled)} pending enrollment",
        "data": {
            "total": len(employees),
            "enrolled_count": len(enrolled),
            "pending_count": len(not_enrolled),
            "enrolled": enrolled,
            "not_enrolled": not_enrolled,
        },
    }
