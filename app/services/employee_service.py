from typing import Optional, List, Tuple
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.employee import Employee
from app.models.face_embedding import FaceEmbedding
from app.repositories.employee_repo import EmployeeRepository
from app.repositories.face_embedding_repo import FaceEmbeddingRepository
from app.schemas.employee import EmployeeCreate, EmployeeUpdate, EmployeeResponse
from app.core.exceptions import (
    NotFoundException,
    ConflictException,
    BadRequestException,
)
from app.utils.image_utils import validate_image, save_face_image, delete_face_image


class EmployeeService:
    """Service for employee business logic."""

    MAX_FACE_IMAGES = 3  # Maximum face images per employee

    def __init__(self, db: AsyncSession):
        self.db = db
        self.employee_repo = EmployeeRepository(db)
        self.face_embedding_repo = FaceEmbeddingRepository(db)
        self._face_service = None  # Lazy loaded to avoid circular import

    @property
    def face_service(self):
        """Lazy load face service to avoid circular import."""
        if self._face_service is None:
            from app.services.face_service import FaceService
            self._face_service = FaceService()
        return self._face_service

    async def register_employee(
        self,
        employee_data: EmployeeCreate,
        face_image: UploadFile,
    ) -> Employee:
        """
        Register a new employee with face image.
        
        Args:
            employee_data: Employee information
            face_image: Face image file
            
        Returns:
            Created employee
            
        Raises:
            ConflictException: If employee_id already exists
            BadRequestException: If face detection fails
        """
        # Check if employee_id already exists
        if await self.employee_repo.exists_by_employee_id(employee_data.employee_id):
            raise ConflictException(
                message=f"Employee with ID '{employee_data.employee_id}' already exists"
            )

        # Validate and process face image
        image_bytes = await validate_image(face_image)

        # Detect face and generate embedding
        embedding = await self.face_service.detect_and_embed(image_bytes)

        # Create employee
        employee = Employee(
            employee_id=employee_data.employee_id,
            name=employee_data.name,
            department=employee_data.department,
            email=employee_data.email,
            phone=employee_data.phone,
        )
        employee = await self.employee_repo.create(employee)

        # Save face image to disk (async)
        image_path = await save_face_image(
            image_bytes, employee_data.employee_id, filename_prefix="primary"
        )

        # Create face embedding record
        face_embedding = FaceEmbedding(
            employee_id=employee.id,
            embedding=embedding,
            image_path=image_path,
            is_primary=True,
        )
        await self.face_embedding_repo.create(face_embedding)

        # Refresh to get relationships
        return await self.employee_repo.get_by_id(employee.id)

    async def add_face_image(
        self,
        employee_id: str,
        face_image: UploadFile,
    ) -> Employee:
        """
        Add additional face image for an employee.
        
        Args:
            employee_id: Employee ID string
            face_image: Face image file
            
        Returns:
            Updated employee
            
        Raises:
            NotFoundException: If employee not found
            BadRequestException: If max face images reached or face detection fails
        """
        # Get employee
        employee = await self.employee_repo.get_active_by_employee_id(employee_id)
        if not employee:
            raise NotFoundException(f"Employee with ID '{employee_id}' not found")

        # Check face image count
        face_count = await self.face_embedding_repo.count_by_employee_id(employee.id)
        if face_count >= self.MAX_FACE_IMAGES:
            raise BadRequestException(
                message=f"Maximum {self.MAX_FACE_IMAGES} face images allowed per employee"
            )

        # Validate and process face image
        image_bytes = await validate_image(face_image)

        # Detect face and generate embedding
        embedding = await self.face_service.detect_and_embed(image_bytes)

        # Save face image to disk (async)
        image_path = await save_face_image(
            image_bytes, employee_id, filename_prefix=f"face_{face_count + 1}"
        )

        # Create face embedding record
        face_embedding = FaceEmbedding(
            employee_id=employee.id,
            embedding=embedding,
            image_path=image_path,
            is_primary=False,
        )
        await self.face_embedding_repo.create(face_embedding)

        # Refresh and return
        return await self.employee_repo.get_by_id(employee.id)

    async def get_employee(self, employee_id: str) -> Employee:
        """
        Get employee by employee_id string.
        
        Raises:
            NotFoundException: If employee not found
        """
        employee = await self.employee_repo.get_by_employee_id(employee_id)
        if not employee:
            raise NotFoundException(f"Employee with ID '{employee_id}' not found")
        return employee

    async def get_active_employee(self, employee_id: str) -> Employee:
        """
        Get active employee by employee_id string.
        
        Raises:
            NotFoundException: If employee not found or inactive
        """
        employee = await self.employee_repo.get_active_by_employee_id(employee_id)
        if not employee:
            raise NotFoundException(
                f"Active employee with ID '{employee_id}' not found"
            )
        return employee

    async def list_employees(
        self,
        page: int = 1,
        page_size: int = 20,
        active_only: bool = True,
    ) -> Tuple[List[Employee], int]:
        """
        List employees with pagination.
        
        Returns:
            Tuple of (employees list, total count)
        """
        offset = (page - 1) * page_size
        return await self.employee_repo.get_all(
            offset=offset, limit=page_size, active_only=active_only
        )

    async def update_employee(
        self,
        employee_id: str,
        update_data: EmployeeUpdate,
    ) -> Employee:
        """
        Update employee information.
        
        Raises:
            NotFoundException: If employee not found
        """
        employee = await self.employee_repo.get_by_employee_id(employee_id)
        if not employee:
            raise NotFoundException(f"Employee with ID '{employee_id}' not found")

        # Filter out None values
        update_dict = {k: v for k, v in update_data.model_dump().items() if v is not None}

        if update_dict:
            employee = await self.employee_repo.update(employee.id, **update_dict)

        return employee

    async def delete_employee(self, employee_id: str) -> bool:
        """
        Soft delete an employee.
        
        Raises:
            NotFoundException: If employee not found
        """
        employee = await self.employee_repo.get_by_employee_id(employee_id)
        if not employee:
            raise NotFoundException(f"Employee with ID '{employee_id}' not found")

        return await self.employee_repo.soft_delete(employee.id)

    async def delete_face_image(self, employee_id: str, face_id: int) -> Employee:
        """
        Delete a specific face image for an employee.
        
        Raises:
            NotFoundException: If employee or face image not found
            BadRequestException: If trying to delete the only face image
        """
        employee = await self.employee_repo.get_by_employee_id(employee_id)
        if not employee:
            raise NotFoundException(f"Employee with ID '{employee_id}' not found")

        # Get face embedding
        face_embedding = await self.face_embedding_repo.get_by_id(face_id)
        if not face_embedding or face_embedding.employee_id != employee.id:
            raise NotFoundException(f"Face image with ID {face_id} not found")

        # Check if it's the only face image
        face_count = await self.face_embedding_repo.count_by_employee_id(employee.id)
        if face_count <= 1:
            raise BadRequestException(
                message="Cannot delete the only face image. Add another image first."
            )

        # Delete image file (async)
        await delete_face_image(face_embedding.image_path)

        # Delete from database
        await self.face_embedding_repo.delete(face_id)

        # If deleted was primary, set another as primary
        if face_embedding.is_primary:
            embeddings = await self.face_embedding_repo.get_by_employee_id(employee.id)
            if embeddings:
                await self.face_embedding_repo.set_primary(
                    embeddings[0].id, employee.id
                )

        return await self.employee_repo.get_by_id(employee.id)
