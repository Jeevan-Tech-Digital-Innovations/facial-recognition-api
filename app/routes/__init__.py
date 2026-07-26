from fastapi import APIRouter

from app.routes.employees import router as employees_router
from app.routes.recognize import router as recognize_router
from app.routes.entry import router as entry_router
from app.routes.sync import router as sync_router

router = APIRouter()

# Register all routers
router.include_router(employees_router, prefix="/employees", tags=["Employees"])
router.include_router(recognize_router, prefix="/recognize", tags=["Recognition"])
router.include_router(entry_router, prefix="/entry", tags=["Entry"])
router.include_router(sync_router, prefix="/sync", tags=["Sync"])
