from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from io import BytesIO
import pdfplumber

from uuid import uuid4
import asyncio

from service.health_analysis import HealthAnalysisService

router = APIRouter()

class HealthRequest(BaseModel):
    report_text: str

@router.post("/health/analysis")
async def analysis_health(request: HealthRequest):
    task_id = str(uuid4())

    service = HealthAnalysisService(task_id=task_id)
    # 异步启动分析，不阻塞接口返回
    asyncio.create_task(service.run(request.report_text))

    return {"task_id": task_id}

@router.post("/health/analysis/pdf")
async def analysis_health_pdf(file: UploadFile = File(...)):

    # 读取 PDF 二进制
    contents = await file.read()

    text = ""

    # 使用 pdfplumber 提取文本
    with pdfplumber.open(BytesIO(contents)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    if not text.strip():
        return {"error": "无法从PDF中提取文本"}

    task_id = str(uuid4())
    service = HealthAnalysisService(task_id=task_id)

    asyncio.create_task(service.run(text))

    return {"task_id": task_id}

@router.get("/health/task_status/{task_id}")
async def task_status(task_id: str):
    from agents.base import get_task_status

    status = get_task_status(task_id)
    if not status:
        return {"error": "task not found"}

    return status