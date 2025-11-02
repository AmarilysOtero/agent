"""API routers"""

from .health import router as health_router
from .machine import router as machine_router
from .graph import router as graph_router
from .upload import router as upload_router

__all__ = ["health_router", "machine_router", "graph_router", "upload_router"]


