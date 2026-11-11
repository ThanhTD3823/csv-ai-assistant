"""Đọc cấu hình từ file .env (tham khảo cách tổ chức của ai-avatar-demo/config).

Giữ API key trong .env, không viết thẳng vào code, và không commit .env lên GitHub.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()  # nạp các biến trong file .env vào biến môi trường


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    max_upload_mb: int


@lru_cache
def get_settings() -> Settings:
    """Đọc cấu hình một lần rồi lưu lại (lru_cache) cho các lần gọi sau."""
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip(),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "10")),
    )
