from __future__ import annotations

import os

import uvicorn

from backend.config import settings


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("APP_HOST", settings.app_host),
        port=int(os.getenv("APP_PORT", str(settings.app_port))),
        reload=False,
    )
