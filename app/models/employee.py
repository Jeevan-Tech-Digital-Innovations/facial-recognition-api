from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from sqlalchemy import String, Boolean, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.face_embedding import FaceEmbedding
    from app.models.entry_log import EntryLog


class Employee(Base):
    """Employee model - stores employee information."""

    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    employee_id: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    department: Mapped[str] = mapped_column(String(100), nullable=True)
    email: Mapped[str] = mapped_column(String(255), nullable=True)
    phone: Mapped[str] = mapped_column(String(20), nullable=True)
    payroll_emp_no: Mapped[Optional[str]] = mapped_column(
        String(50), unique=True, nullable=True, index=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    face_embeddings: Mapped[List["FaceEmbedding"]] = relationship(
        "FaceEmbedding", back_populates="employee", cascade="all, delete-orphan"
    )
    entry_logs: Mapped[List["EntryLog"]] = relationship(
        "EntryLog", back_populates="employee"
    )

    def __repr__(self) -> str:
        return f"<Employee(id={self.id}, employee_id={self.employee_id}, name={self.name})>"
