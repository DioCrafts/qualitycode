"""
Executive reports API endpoints.
"""

from .dora import router as dora_router
from .executive import router as executive_router
from .kpis import router as kpis_router

__all__ = ["dora_router", "executive_router", "kpis_router"]
