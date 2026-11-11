# 📊 Trợ lý phân tích dữ liệu CSV bằng AI

Ứng dụng cho phép tải lên file CSV, tự động **thống kê bằng Pandas/NumPy**, sau đó dùng mô hình ngôn ngữ lớn **Gemini** để **nhận xét** và **trả lời câu hỏi** về dữ liệu.

**Công nghệ:** Python · Pandas · NumPy · FastAPI · Pydantic · Gemini API (google-genai) · Streamlit · pytest · Git

## Kiến trúc

```
┌──────────────┐  HTTP   ┌──────────────────┐        ┌───────────────────────┐
│  Streamlit   │ ──────▶ │  FastAPI          │ ─────▶ │ analyzer.py (Pandas)  │
│  (giao diện) │ ◀────── │  /health          │        │ → bản tóm tắt thống kê│
└──────────────┘  JSON   │  /analyze  /ask   │        └───────────┬───────────┘
                         └──────────────────┘                    │ tóm tắt + 5 dòng mẫu
                                                     ┌───────────▼───────────┐
                                                     │ llm_service.py        │
                                                     │ → Gemini nhận xét/trả │
                                                     │   lời                 │
                                                     └───────────────────────┘
```

**Nguyên tắc thiết kế:** chỉ gửi *bản tóm tắt + 5 dòng mẫu* cho AI, không gửi toàn bộ file. Cách này tiết kiệm token, xử lý được file lớn và hạn chế lộ dữ liệu. Mọi con số đều do Pandas tính và được kiểm chứng bằng test, AI chỉ diễn giải.

## Cấu trúc thư mục

| File | Vai trò |
|---|---|
| `app/analyzer.py` | Đọc CSV, tính thống kê (Pandas/NumPy) |
| `app/services/llm_service.py` | Gọi Gemini, xây dựng prompt |
| `app/config.py` | Đọc cấu hình từ `.env` |
| `app/schemas.py` | Pydantic models, định nghĩa dữ liệu vào/ra |
| `app/main.py` | API FastAPI: `/health`, `/analyze`, `/ask` |
| `ui/streamlit_app.py` | Giao diện người dùng |
| `tests/` | 16 test tự động (pytest) |
| `data/sample_sales.csv` | Dữ liệu mẫu: 303 dòng bán hàng, có giá trị thiếu và dòng trùng |

---

## Cài đặt và chạy trên Windows

### Bước 0. Chuẩn bị (làm một lần)
1. Cài **Python 3.11 trở lên** từ https://www.python.org/downloads/. Khi cài, **tick ô "Add python.exe to PATH"**.
2. Cài **VS Code** (https://code.visualstudio.com) và **Git** (https://git-scm.com/download/win).
3. Lấy API key Gemini miễn phí: https://aistudio.google.com/app/apikey

### Bước 1. Cài đặt tự động
Giải nén thư mục, **nháy đúp `setup.bat`**. File này sẽ:
- tạo môi trường ảo `.venv`
- cài các thư viện cần thiết
- tạo file `.env`
- chạy test (kết quả mong đợi: `16 passed`)

### Bước 2. Điền API key
Mở file `.env` bằng VS Code, thay `dien-api-key-cua-ban-vao-day` bằng key của bạn, rồi lưu lại.
> ⚠️ Không gửi key cho ai và không commit file `.env` lên GitHub (đã có `.gitignore` chặn).

### Bước 3. Chạy ứng dụng (cần 2 cửa sổ)
1. Nháy đúp **`run_api.bat`** và giữ cửa sổ này mở. Có thể kiểm tra tại http://127.0.0.1:8000/docs
2. Nháy đúp **`run_ui.bat`**. Trình duyệt sẽ mở giao diện tại http://localhost:8501
3. Tải `data/sample_sales.csv` lên để thử.

### Chạy bằng lệnh (nếu không dùng file .bat)
Mở **Command Prompt** trong thư mục dự án:
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m pytest -v
python -m uvicorn app.main:app --reload
```
Mở thêm một cửa sổ Command Prompt khác:
```bat
.venv\Scripts\activate
python -m streamlit run ui/streamlit_app.py
```

### Lỗi thường gặp
| Lỗi | Cách sửa |
|---|---|
| `'python' is not recognized` | Cài lại Python và tick "Add to PATH", hoặc dùng `py` thay cho `python` |
| PowerShell báo `running scripts is disabled` | Dùng **Command Prompt** thay cho PowerShell, hoặc chạy `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `Chưa cấu hình GEMINI_API_KEY` | Chưa điền key vào `.env`, hoặc chưa khởi động lại `run_api.bat` sau khi điền |
| `404 ... model not found` | Đổi `GEMINI_MODEL` trong `.env` sang tên mô hình Flash hiện có trong AI Studio |
| `429 ... quota` | Hết hạn mức miễn phí, đợi một lúc rồi thử lại |
| Giao diện báo "Không kết nối được API" | Chưa chạy `run_api.bat` |

---

## Đưa lên GitHub theo từng phần

Tạo repo trống `csv-ai-assistant` trên GitHub (không tick README), sau đó chạy lần lượt trong Command Prompt:

```bat
git init
git branch -M main
git add .gitignore requirements.txt .env.example README.md setup.bat run_api.bat run_ui.bat
git commit -m "Khởi tạo dự án: cấu trúc, thư viện, cấu hình"
git add app/__init__.py app/analyzer.py data/ tests/__init__.py tests/test_analyzer.py
git commit -m "Thêm module phân tích CSV bằng Pandas/NumPy và test"
git add app/config.py app/services/
git commit -m "Thêm dịch vụ gọi Gemini và cấu hình"
git add app/schemas.py app/main.py tests/test_api.py
git commit -m "Thêm API FastAPI và test API với LLM giả lập"
git add ui/
git commit -m "Thêm giao diện Streamlit"
git remote add origin https://github.com/<ten-tai-khoan>/csv-ai-assistant.git
git push -u origin main
```
Kiểm tra bằng `git status`: file `.env` **không được** xuất hiện trong danh sách.

---

## Thứ tự đọc code để hiểu dự án
1. `app/analyzer.py`: thống kê dữ liệu. Đọc kèm `tests/test_analyzer.py` để thấy ví dụ đầu vào và đầu ra.
2. `app/services/llm_service.py`: cách xây prompt và gọi Gemini.
3. `app/schemas.py` rồi `app/main.py`: API nhận và trả dữ liệu thế nào.
4. `tests/test_api.py`: cách giả lập (mock) AI khi test.
5. `ui/streamlit_app.py`: giao diện gọi API.

## Hướng phát triển
- Hỗ trợ file Excel và phát hiện giá trị ngoại lai (outlier).
- Sinh biểu đồ theo câu hỏi bằng ngôn ngữ tự nhiên.
- Lưu dữ liệu bằng cơ sở dữ liệu thay vì bộ nhớ tạm.
- RAG / agent tự chọn giữa truy vấn SQL và tìm kiếm tài liệu.
- Đóng gói Docker và triển khai lên cloud.

## Tài liệu tham khảo
- FastAPI: https://fastapi.tiangolo.com
- Pandas: https://pandas.pydata.org/docs/
- Gemini API: https://ai.google.dev/gemini-api/docs
- Streamlit: https://docs.streamlit.io
- Tham khảo kiến trúc: `ai-avatar-demo`, `video-rag-gemini` trong https://github.com/patchy631/ai-engineering-hub
