"""Kiểm tra API key và đo tốc độ các mô hình Gemini Flash.

Chạy:  python check_gemini.py
Sau đó chọn mô hình "OK" nhanh nhất, ghi vào dòng GEMINI_MODEL trong file .env.
"""
import time

from app.config import get_settings
from app.services.llm_service import LLMError, LLMService, rank_flash_models

PROMPT = "Trả lời đúng một câu bằng tiếng Việt: 2 + 2 bằng mấy?"

settings = get_settings()
service = LLMService(api_key=settings.gemini_api_key, model=settings.gemini_model)

try:
    client = service._get_client()
    all_models = [
        m.name.removeprefix("models/")
        for m in client.models.list()
        if "generateContent" in (getattr(m, "supported_actions", None) or [])
    ]
except LLMError as exc:
    raise SystemExit(f"LỖI: {exc}")
except Exception as exc:
    raise SystemExit(f"LỖI: không lấy được danh sách mô hình, kiểm tra API key. Chi tiết: {exc}")

flash = rank_flash_models(all_models)
print(f"Key hợp lệ. Có {len(flash)} mô hình Flash khả dụng.")
print(f"Mô hình đang cấu hình trong .env: {settings.gemini_model}\n")

for model in flash[:8]:
    started = time.perf_counter()
    try:
        response = service._call(PROMPT, model)
        status = "OK  " + (response.text or "").strip().replace("\n", " ")[:50]
    except Exception as exc:
        status = "LỖI " + str(exc).split(".")[0][:70]
    print(f"{model:35s} {time.perf_counter() - started:5.1f}s  {status}")

print("\nChọn mô hình có chữ OK và thời gian nhỏ nhất, ghi vào .env:  GEMINI_MODEL=<tên mô hình>")
