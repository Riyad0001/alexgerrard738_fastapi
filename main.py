"""Production ASGI entry point: ``uvicorn main:app``."""
from key_matching.production_api import app

__all__ = ["app"]
