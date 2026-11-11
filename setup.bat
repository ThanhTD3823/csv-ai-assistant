@echo off
chcp 65001 >nul
echo === Tao moi truong ao .venv ===
python -m venv .venv || goto :error
call .venv\Scripts\activate.bat
echo === Cai thu vien ===
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :error
if not exist .env (
    copy .env.example .env >nul
    echo === Da tao file .env - hay mo file .env va dien GEMINI_API_KEY ===
)
echo === Chay test ===
python -m pytest -q
echo.
echo HOAN TAT. Tiep theo: dien API key vao .env, chay run_api.bat roi run_ui.bat
pause
exit /b 0
:error
echo.
echo CO LOI. Hay chup man hinh gui cho Claude.
pause
exit /b 1
