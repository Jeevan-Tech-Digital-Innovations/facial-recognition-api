from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import String, Float, DateTime, ForeignKey, Integer, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.employee import Employee


class EntryLog(Base):
    """Entry log model - tracks canteen entries (face or manual)."""

    __tablename__ = "entry_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    employee_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("employees.id", ondelete="CASCADE"), nullable=False, index=True
    )
    entry_method: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # 'face' or 'manual'
    match_confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )  # null if manual
    device_id: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True
    )  # optional kiosk identifier
    entry_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # Relationships
    employee: Mapped["Employee"] = relationship("Employee", back_populates="entry_logs")

    def __repr__(self) -> str:
        return f"<EntryLog(id={self.id}, employee_id={self.employee_id}, method={self.entry_method})>"
