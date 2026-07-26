import logging
from typing import Optional, List, Dict, Any

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class PayrollService:
    """Client service for server-to-server communication with the Payroll API."""

    def __init__(self):
        self.base_url = settings.PAYROLL_API_BASE_URL
        self.api_key = settings.PAYROLL_API_KEY
        self.sync_enabled = settings.PAYROLL_EMPLOYEE_SYNC_ENABLED
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client singleton."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    @property
    def is_configured(self) -> bool:
        """Check if payroll integration is configured."""
        return bool(self.base_url and self.api_key and self.sync_enabled)

    async def record_punch(
        self,
        emp_no: str,
        punch_direction: str,
        device_username: str,
        face_match_confidence: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Record a face-recognition punch in the Payroll system.

        Args:
            emp_no: Payroll employee number
            punch_direction: "IN" or "OUT"
            device_username: Device username of the kiosk
            face_match_confidence: Face match confidence (0.0-1.0)

        Returns:
            Punch response dict from Payroll API, or None if failed
        """
        if not self.is_configured:
            logger.debug("Payroll integration not configured, skipping punch sync")
            return None

        payload = {
            "empNo": emp_no,
            "punchDirection": punch_direction,
            "deviceUsername": device_username,
            "faceMatchConfidence": round(face_match_confidence, 4),
        }

        try:
            client = await self._get_client()
            response = await client.post("/attendance/punch/face", json=payload)
            response.raise_for_status()
            data = response.json()
            logger.info(
                "Payroll punch recorded: empNo=%s direction=%s confidence=%.4f",
                emp_no, punch_direction, face_match_confidence,
            )
            return data
        except httpx.HTTPStatusError as e:
            logger.error(
                "Payroll punch failed: status=%s body=%s",
                e.response.status_code, e.response.text[:500],
            )
            return None
        except httpx.RequestError as e:
            logger.error("Payroll punch request error: %s", str(e))
            return None

    async def get_active_employees(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch active employees from the Payroll system for sync.

        Returns:
            List of employee dicts, or None if failed
        """
        if not self.is_configured:
            logger.debug("Payroll integration not configured, skipping employee sync")
            return None

        try:
            client = await self._get_client()
            response = await client.get(
                "/employees",
                params={"page": 0, "size": 1000, "status": "ACTIVE"},
            )
            response.raise_for_status()
            data = response.json()
            # Payroll returns PaginatedResponse with 'content' list
            employees = data.get("data", {})
            if isinstance(employees, dict):
                return employees.get("content", [])
            return employees if isinstance(employees, list) else []
        except httpx.HTTPStatusError as e:
            logger.error(
                "Employee sync failed: status=%s", e.response.status_code
            )
            return None
        except httpx.RequestError as e:
            logger.error("Employee sync request error: %s", str(e))
            return None

    async def determine_punch_direction(
        self, emp_no: str, device_username: str
    ) -> str:
        """
        Determine punch direction based on employee's last punch today.

        Logic:
        - No punch today or last was OUT -> IN
        - Last was IN -> OUT

        Args:
            emp_no: Payroll employee number
            device_username: Device username (for context)

        Returns:
            "IN" or "OUT"
        """
        if not self.is_configured:
            return "IN"  # Default to IN when not configured

        try:
            client = await self._get_client()
            from datetime import date
            today = date.today().isoformat()
            response = await client.get(
                f"/attendance/summary/{emp_no}",
                params={"date": today},
            )
            if response.status_code == 200:
                data = response.json().get("data", {})
                # If employee has a first_in_time but no last_out_time, they're still in
                if data.get("firstInTime") and not data.get("lastOutTime"):
                    return "OUT"
                elif data.get("lastOutTime"):
                    return "IN"
                else:
                    return "IN"
            return "IN"  # Default: no record today -> punch IN
        except Exception as e:
            logger.warning(
                "Could not determine punch direction for %s: %s. Defaulting to IN.",
                emp_no, str(e),
            )
            return "IN"


# Singleton instance
_payroll_service: Optional[PayrollService] = None


def get_payroll_service() -> PayrollService:
    """Get singleton payroll service instance."""
    global _payroll_service
    if _payroll_service is None:
        _payroll_service = PayrollService()
    return _payroll_service
