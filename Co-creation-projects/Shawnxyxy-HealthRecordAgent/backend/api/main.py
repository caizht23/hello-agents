from fastapi import FastAPI
from api.routes.health import router as health_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="HealthRecordAgent API",
    version="1.0.0"
)

app.include_router(health_router, prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段允许全部
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)