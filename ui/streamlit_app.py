"""Giao diện Streamlit (tham khảo cách dùng session_state và chat của video-rag-gemini).

Giao diện KHÔNG tự xử lý dữ liệu hay gọi AI. Nó chỉ gọi API FastAPI:
  Streamlit (giao diện)  →  FastAPI (xử lý)  →  Pandas + Gemini

Chạy:  streamlit run ui/streamlit_app.py   (nhớ chạy API trước)
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 180  # giây; API tự giới hạn thời gian chờ Gemini nên sẽ trả lời trước mốc này

st.set_page_config(page_title="Trợ lý phân tích CSV", page_icon="📊", layout="wide")


# ---------- Trạng thái phiên làm việc ----------
def reset_state():
    st.session_state.result = None  # kết quả /analyze
    st.session_state.file_key = None  # nhận biết người dùng đổi file
    st.session_state.messages = []  # lịch sử hỏi đáp


if "result" not in st.session_state:
    reset_state()


def api_error_message(res: requests.Response) -> str:
    try:
        return str(res.json().get("detail", res.text))
    except ValueError:
        return res.text


# ---------- Thanh bên ----------
with st.sidebar:
    st.header("⚙️ Hệ thống")
    try:
        ok = requests.get(f"{API_URL}/health", timeout=3).status_code == 200
    except requests.RequestException:
        ok = False
    if ok:
        st.success("API đang chạy")
    else:
        st.error(f"Không kết nối được API tại {API_URL}. Hãy chạy run_api.bat trước.")
    st.caption(f"API: {API_URL}")
    if st.button("🔄 Làm mới", width="stretch"):
        reset_state()
        st.rerun()

st.title("📊 Trợ lý phân tích dữ liệu CSV bằng AI")
st.caption("Thống kê bằng Pandas/NumPy · Nhận xét và hỏi đáp bằng Gemini")

# ---------- Tải file ----------
uploaded = st.file_uploader("Chọn file CSV", type=["csv"])

if uploaded is not None:
    file_key = f"{uploaded.name}-{uploaded.size}"
    if st.session_state.file_key != file_key:  # file mới → gọi API phân tích
        with st.spinner("Đang phân tích dữ liệu và chờ AI nhận xét..."):
            try:
                res = requests.post(
                    f"{API_URL}/analyze",
                    files={"file": (uploaded.name, uploaded.getvalue(), "text/csv")},
                    timeout=TIMEOUT,
                )
            except requests.RequestException as exc:
                st.error(f"Không gọi được API: {exc}")
                st.stop()
        if res.status_code != 200:
            st.error(f"Lỗi: {api_error_message(res)}")
            st.stop()
        st.session_state.result = res.json()
        st.session_state.file_key = file_key
        st.session_state.messages = []

result = st.session_state.result
if not result:
    st.info("Hãy tải lên một file CSV để bắt đầu (có thể dùng data/sample_sales.csv).")
    st.stop()

summary = result["summary"]

# ---------- Chỉ số tổng quan ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Số dòng", f"{summary['n_rows']:,}")
c2.metric("Số cột", summary["n_cols"])
c3.metric("Giá trị thiếu", f"{summary['total_missing']:,}")
c4.metric("Dòng trùng lặp", summary["duplicate_rows"])

# ---------- Nhận xét của AI ----------
st.subheader("🤖 Nhận xét của AI")
if result.get("explanation"):
    st.markdown(result["explanation"])
else:
    st.warning(f"AI chưa nhận xét được: {result.get('llm_error')}")
    if st.button("🔁 Thử lại nhận xét của AI"):
        with st.spinner("Đang yêu cầu AI nhận xét lại..."):
            try:
                res = requests.post(f"{API_URL}/explain/{result['dataset_id']}", timeout=TIMEOUT)
            except requests.RequestException as exc:
                st.error(f"Không gọi được API: {exc}")
                st.stop()
        if res.status_code == 200:
            result["explanation"] = res.json()["explanation"]
            result["llm_error"] = None
            st.rerun()
        elif res.status_code == 404:
            st.error("API đã khởi động lại nên mất dữ liệu tạm. Hãy bấm 🔄 Làm mới rồi tải file lên lại.")
        else:
            result["llm_error"] = api_error_message(res)
            st.rerun()

# ---------- Chi tiết ----------
tab_cols, tab_chart, tab_sample = st.tabs(["📋 Thông tin các cột", "📈 Biểu đồ", "🔎 Dữ liệu mẫu"])

columns = summary["columns"]
with tab_cols:
    table = pd.DataFrame(
        [
            {
                "Cột": c["name"],
                "Kiểu": c["dtype"],
                "Thiếu": c["missing"],
                "% thiếu": c["missing_pct"],
                "Số giá trị khác nhau": c["unique"],
                "Trung bình": (c["stats"] or {}).get("mean"),
                "Nhỏ nhất": (c["stats"] or {}).get("min"),
                "Lớn nhất": (c["stats"] or {}).get("max"),
            }
            for c in columns
        ]
    )
    st.dataframe(table, width="stretch", hide_index=True)

with tab_chart:
    left, right = st.columns(2)
    def horizontal_bar(data: dict, label: str, value: str):
        """Biểu đồ cột NGANG: tên nằm ngang bên trái nên dễ đọc, xếp từ lớn đến nhỏ."""
        df_chart = pd.DataFrame({label: list(data.keys()), value: list(data.values())})
        st.bar_chart(
            df_chart,
            x=label,
            y=value,
            horizontal=True,
            sort=f"-{value}",  # giá trị lớn nhất ở trên cùng
            height=max(220, 38 * len(df_chart)),  # nhiều cột thì biểu đồ cao hơn
        )

    with left:
        st.markdown("**Số giá trị thiếu theo cột**")
        horizontal_bar({c["name"]: c["missing"] for c in columns}, "Cột", "Số giá trị thiếu")
    with right:
        name = st.selectbox("Xem chi tiết cột", [c["name"] for c in columns])
        col = next(c for c in columns if c["name"] == name)
        non_missing = summary["n_rows"] - col["missing"]
        if col["stats"]:
            st.markdown("**Thống kê (cột số)**")
            st.dataframe(pd.DataFrame([col["stats"]]), hide_index=True, width="stretch")
        elif non_missing > 1 and col["unique"] == non_missing:
            # Mỗi dòng một giá trị riêng (vd mã giao dịch) → biểu đồ tần suất không có ý nghĩa.
            st.info(
                f"Cột **{name}** có {col['unique']:,} giá trị khác nhau, mỗi dòng một giá trị "
                "(giống mã định danh), nên không vẽ biểu đồ tần suất."
            )
        elif col["top_values"]:
            st.markdown("**Top giá trị xuất hiện nhiều nhất**")
            horizontal_bar(col["top_values"], "Giá trị", "Số lần")
        else:
            st.info("Cột này không có dữ liệu.")

with tab_sample:
    st.dataframe(pd.DataFrame(result["samples"]), width="stretch", hide_index=True)

# ---------- Hỏi đáp ----------
st.subheader("💬 Hỏi đáp về dữ liệu")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ví dụ: Cột nào thiếu dữ liệu nhiều nhất?")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("AI đang trả lời..."):
            try:
                res = requests.post(
                    f"{API_URL}/ask",
                    json={"dataset_id": result["dataset_id"], "question": question},
                    timeout=TIMEOUT,
                )
                answer = res.json()["answer"] if res.status_code == 200 else f"⚠️ {api_error_message(res)}"
            except requests.RequestException as exc:
                answer = f"⚠️ Không gọi được API: {exc}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
