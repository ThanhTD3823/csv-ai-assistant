@echo off
chcp 65001 >nul
call .venv\Scripts\activate.bat
echo API dang chay tai http://127.0.0.1:8000  (tai lieu: http://127.0.0.1:8000/docs)
echo Giu cua so nay mo. Nhan Ctrl+C de dung.
python -m uvicorn app.main:app --reload
