"""Pydantic models: mô tả dữ liệu vào/ra của API.

FastAPI dùng các model này để tự kiểm tra dữ liệu (sai kiểu → lỗi 422)
và tự sinh tài liệu API tại /docs.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NumericStats(BaseModel):
    mean: float | None
    std: float | None
    min: float | None
    q1: float | None
    median: float | None
    q3: float | None
    max: float | None


class ColumnSummary(BaseModel):
    name: str
    dtype: str
    missing: int
    missing_pct: float
    unique: int
    stats: NumericStats | None = None  # có giá trị nếu là cột số
    top_values: dict[str, int] | None = None  # có giá trị nếu là cột chữ


class DatasetSummary(BaseModel):
    n_rows: int
    n_cols: int
    total_missing: int
    duplicate_rows: int
    columns: list[ColumnSummary]


class HealthResponse(BaseModel):
    status: str


class AnalyzeResponse(BaseModel):
    dataset_id: str
    filename: str
    summary: DatasetSummary
    samples: list[dict[str, Any]]
    explanation: str | None = Field(None, description="Nhận xét của AI")
    llm_error: str | None = Field(None, description="Lý do nếu AI không trả lời được")


class AskRequest(BaseModel):
    dataset_id: str
    question: str = Field(..., min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str


class ExplainResponse(BaseModel):
    explanation: str
