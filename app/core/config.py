from zoneinfo import ZoneInfo
from pydantic_settings import BaseSettings
from functools import lru_cache
from urllib.parse import quote_plus


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database - separate fields to handle special chars in password
    DB_HOST: str
    DB_PORT: int = 5432
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str

    # Face recognition
    FACE_MATCH_THRESHOLD: float = 0.55
    FACE_MODEL: str = "ArcFace"
    FACE_DETECTOR: str = "mtcnn"

    # Application
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    CORS_ORIGINS: str = "*"

    # Timezone for date-based queries (entry logs, stats).
    # Must be a valid IANA timezone name (e.g. "Asia/Kolkata", "US/Eastern").
    # This ensures "today" and date-range queries match the facility's
    # local wall-clock time, not UTC.
    APP_TIMEZONE: str = "UTC"

    # Storage
    FACE_IMAGES_DIR: str = "face_images"

    # Payroll API Integration
    PAYROLL_API_BASE_URL: str = ""
    PAYROLL_API_KEY: str = ""
    PAYROLL_EMPLOYEE_SYNC_ENABLED: bool = False

    @property
    def tz(self) -> ZoneInfo:
        """Get the configured timezone as a ZoneInfo object."""
        return ZoneInfo(self.APP_TIMEZONE)

    @property
    def DATABASE_URL(self) -> str:
        """Construct async database URL with properly encoded password."""
        encoded_password = quote_plus(self.DB_PASSWORD)
        return f"postgresql+asyncpg://{self.DB_USER}:{encoded_password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def DATABASE_URL_SYNC(self) -> str:
        """Construct sync database URL for setup scripts."""
        encoded_password = quote_plus(self.DB_PASSWORD)
        return f"postgresql://{self.DB_USER}:{encoded_password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
