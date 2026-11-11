"""API FastAPI của Trợ lý phân tích dữ liệu CSV.

Chạy:  uvicorn app.main:app --reload
Tài liệu API tự sinh: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import uuid
from collections import OrderedDict
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from app.analyzer import CSVError, load_csv, sample_rows, summarize
from app.config import get_settings
from app.schemas import AnalyzeResponse, AskRequest, AskResponse, ExplainResponse, HealthResponse
from app.services.llm_service import LLMError, LLMService

app = FastAPI(
    title="CSV AI Assistant",
    version="1.0.0",
    description="Tải file CSV lên để nhận thống kê (Pandas) và nhận xét, hỏi đáp bằng AI (Gemini).",
)

# Lưu tạm bộ dữ liệu trong bộ nhớ theo dataset_id (đủ cho demo, không cần database).
# Giới hạn số lượng để không tốn RAM: vượt quá thì xóa bộ cũ nhất.
MAX_DATASETS = 20
DATASETS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()


def get_llm() -> LLMService:
    """Dependency: tạo LLMService từ cấu hình.

    Trong test, ta thay hàm này bằng LLM giả qua app.dependency_overrides.
    """
    settings = get_settings()
    return LLMService(api_key=settings.gemini_api_key, model=settings.gemini_model)


@app.get("/health", response_model=HealthResponse, tags=["Hệ thống"])
def health() -> HealthResponse:
    """Kiểm tra server có đang chạy không."""
    return HealthResponse(status="ok")


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Phân tích"])
def analyze(file: UploadFile = File(...), llm: LLMService = Depends(get_llm)) -> AnalyzeResponse:
    """Nhận file CSV → thống kê bằng Pandas → AI nhận xét."""
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file .csv")

    content = file.file.read()
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File quá lớn (tối đa {get_settings().max_upload_mb} MB)")

    try:
        df = load_csv(content)
    except CSVError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary = summarize(df)
    samples = sample_rows(df)

    dataset_id = uuid.uuid4().hex
    DATASETS[dataset_id] = {"filename": filename, "summary": summary, "samples": samples}
    while len(DATASETS) > MAX_DATASETS:
        DATASETS.popitem(last=False)

    # Nếu AI lỗi, vẫn trả về thống kê, chỉ kèm thông báo lỗi.
    explanation, llm_error = None, None
    try:
        explanation = llm.explain(summary, samples)
    except LLMError as exc:
        llm_error = str(exc)

    return AnalyzeResponse(
        dataset_id=dataset_id,
        filename=filename,
        summary=summary,
        samples=samples,
        explanation=explanation,
        llm_error=llm_error,
    )


@app.post("/explain/{dataset_id}", response_model=ExplainResponse, tags=["Phân tích"])
def explain(dataset_id: str, llm: LLMService = Depends(get_llm)) -> ExplainResponse:
    """Yêu cầu AI nhận xét lại một bộ dữ liệu đã tải lên (dùng khi lần trước AI bị lỗi)."""
    dataset = DATASETS.get(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy bộ dữ liệu. Hãy tải file lên lại.")
    try:
        explanation = llm.explain(dataset["summary"], dataset["samples"])
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ExplainResponse(explanation=explanation)


@app.post("/ask", response_model=AskResponse, tags=["Phân tích"])
def ask(request: AskRequest, llm: LLMService = Depends(get_llm)) -> AskResponse:
    """Hỏi đáp về một bộ dữ liệu đã tải lên trước đó."""
    dataset = DATASETS.get(request.dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy bộ dữ liệu. Hãy tải file lên lại.")

    try:
        answer = llm.ask(dataset["summary"], dataset["samples"], request.question)
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AskResponse(answer=answer)
