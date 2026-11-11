@echo off
chcp 65001 >nul
call .venv\Scripts\activate.bat
echo Giao dien se mo trong trinh duyet. Giu cua so nay mo. Nhan Ctrl+C de dung.
python -m streamlit run ui/streamlit_app.py
