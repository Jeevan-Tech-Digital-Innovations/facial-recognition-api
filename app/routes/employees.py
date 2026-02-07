from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.employee_service import EmployeeService
from app.schemas.employee import (
    EmployeeCreate,
    EmployeeUpdate,
    EmployeeResponse,
    EmployeeDetailResponse,
    EmployeeListResponse,
    FaceEmbeddingInfo,
)
from app.schemas.common import APIResponse

router = APIRouter()


@router.post("/register", response_model=APIResponse[EmployeeResponse], status_code=201)
async def register_employee(
    employee_id: str = Form(..., description="Unique employee ID (e.g., EMP-001)"),
    name: str = Form(..., description="Employee full name"),
    department: Optional[str] = Form(None, description="Department name"),
    email: Optional[str] = Form(None, description="Email address"),
    phone: Optional[str] = Form(None, description="Phone number"),
    face_image: UploadFile = File(..., description="Face image for recognition"),
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new employee with face image.
    
    The face image will be processed for facial recognition.
    Only one face should be present in the image.
    """
    service = EmployeeService(db)

    employee_data = EmployeeCreate(
        employee_id=employee_id,
        name=name,
        department=department,
        email=email,
        phone=phone,
    )

    employee = await service.register_employee(employee_data, face_image)

    return APIResponse(
        success=True,
        message="Employee registered successfully",
        data=EmployeeResponse(
            id=employee.id,
            employee_id=employee.employee_id,
            name=employee.name,
            department=employee.department,
            email=employee.email,
            phone=employee.phone,
            is_active=employee.is_active,
            created_at=employee.created_at,
            updated_at=employee.updated_at,
            face_count=len(employee.face_embeddings) if employee.face_embeddings else 0,
        ),
    )


@router.post(
    "/{employee_id}/add-face",
    response_model=APIResponse[EmployeeDetailResponse],
)
async def add_face_image(
    employee_id: str,
    face_image: UploadFile = File(..., description="Additional face image"),
    db: AsyncSession = Depends(get_db),
):
    """
    Add additional face image for an employee.
    
    Multiple face images (up to 3) improve recognition accuracy
    with different angles and lighting conditions.
    """
    service = EmployeeService(db)
    employee = await service.add_face_image(employee_id, face_image)

    return APIResponse(
        success=True,
        message="Face image added successfully",
        data=EmployeeDetailResponse(
            id=employee.id,
            employee_id=employee.employee_id,
            name=employee.name,
            department=employee.department,
            email=employee.email,
            phone=employee.phone,
            is_active=employee.is_active,
            created_at=employee.created_at,
            updated_at=employee.updated_at,
            face_count=len(employee.face_embeddings) if employee.face_embeddings else 0,
            face_embeddings=[
                FaceEmbeddingInfo(
                    id=fe.id,
                    image_path=fe.image_path,
                    is_primary=fe.is_primary,
                    created_at=fe.created_at,
                )
                for fe in (employee.face_embeddings or [])
            ],
        ),
    )


@router.get("", response_model=EmployeeListResponse)
async def list_employees(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    active_only: bool = Query(True, description="Only show active employees"),
    db: AsyncSession = Depends(get_db),
):
    """List all employees with pagination."""
    service = EmployeeService(db)
    employees, total = await service.list_employees(page, page_size, active_only)

    total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0

    return EmployeeListResponse(
        items=[
            EmployeeResponse(
                id=emp.id,
                employee_id=emp.employee_id,
                name=emp.name,
                department=emp.department,
                email=emp.email,
                phone=emp.phone,
                is_active=emp.is_active,
                created_at=emp.created_at,
                updated_at=emp.updated_at,
                face_count=getattr(emp, "face_count", 0),
            )
            for emp in employees
        ],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{employee_id}", response_model=APIResponse[EmployeeDetailResponse])
async def get_employee(
    employee_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get employee details by employee ID."""
    service = EmployeeService(db)
    employee = await service.get_employee(employee_id)

    return APIResponse(
        success=True,
        message="Employee retrieved successfully",
        data=EmployeeDetailResponse(
            id=employee.id,
            employee_id=employee.employee_id,
            name=employee.name,
            department=employee.department,
            email=employee.email,
            phone=employee.phone,
            is_active=employee.is_active,
            created_at=employee.created_at,
            updated_at=employee.updated_at,
            face_count=len(employee.face_embeddings) if employee.face_embeddings else 0,
            face_embeddings=[
                FaceEmbeddingInfo(
                    id=fe.id,
                    image_path=fe.image_path,
                    is_primary=fe.is_primary,
                    created_at=fe.created_at,
                )
                for fe in (employee.face_embeddings or [])
            ],
        ),
    )


@router.put("/{employee_id}", response_model=APIResponse[EmployeeResponse])
async def update_employee(
    employee_id: str,
    update_data: EmployeeUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update employee information."""
    service = EmployeeService(db)
    employee = await service.update_employee(employee_id, update_data)

    return APIResponse(
        success=True,
        message="Employee updated successfully",
        data=EmployeeResponse(
            id=employee.id,
            employee_id=employee.employee_id,
            name=employee.name,
            department=employee.department,
            email=employee.email,
            phone=employee.phone,
            is_active=employee.is_active,
            created_at=employee.created_at,
            updated_at=employee.updated_at,
            face_count=len(employee.face_embeddings) if employee.face_embeddings else 0,
        ),
    )


@router.delete("/{employee_id}", response_model=APIResponse)
async def delete_employee(
    employee_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Soft delete an employee.
    
    The employee will be marked as inactive but their data
    and entry logs will be preserved.
    """
    service = EmployeeService(db)
    await service.delete_employee(employee_id)

    return APIResponse(
        success=True,
        message=f"Employee '{employee_id}' has been deactivated",
        data=None,
    )


@router.delete(
    "/{employee_id}/faces/{face_id}",
    response_model=APIResponse[EmployeeDetailResponse],
)
async def delete_face_image(
    employee_id: str,
    face_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a specific face image for an employee.
    
    At least one face image must remain for the employee.
    """
    service = EmployeeService(db)
    employee = await service.delete_face_image(employee_id, face_id)

    return APIResponse(
        success=True,
        message="Face image deleted successfully",
        data=EmployeeDetailResponse(
            id=employee.id,
            employee_id=employee.employee_id,
            name=employee.name,
            department=employee.department,
            email=employee.email,
            phone=employee.phone,
            is_active=employee.is_active,
            created_at=employee.created_at,
            updated_at=employee.updated_at,
            face_count=len(employee.face_embeddings) if employee.face_embeddings else 0,
            face_embeddings=[
                FaceEmbeddingInfo(
                    id=fe.id,
                    image_path=fe.image_path,
                    is_primary=fe.is_primary,
                    created_at=fe.created_at,
                )
                for fe in (employee.face_embeddings or [])
            ],
        ),
    )
