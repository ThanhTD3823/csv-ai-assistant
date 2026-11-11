"""Test cho API FastAPI.

Phần LLM được thay bằng FakeLLM (mock) qua app.dependency_overrides:
- test chạy nhanh, không cần internet, không tốn hạn mức API;
- kết quả luôn ổn định, không phụ thuộc câu trả lời ngẫu nhiên của AI.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_llm
from app.services.llm_service import LLMError

CSV_OK = b"ten,tuoi\nAn,20\nBinh,\nChi,30\n"


class FakeLLM:
    def explain(self, summary, samples):
        return f"Bo du lieu co {summary['n_rows']} dong."

    def ask(self, summary, samples, question):
        return f"Tra loi: {question}"


class BrokenLLM:
    """LLM giả lập trường hợp lỗi (mất mạng, sai key...)."""

    def explain(self, summary, samples):
        raise LLMError("mat mang")

    def ask(self, summary, samples, question):
        raise LLMError("mat mang")


@pytest.fixture
def client():
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def broken_client():
    app.dependency_overrides[get_llm] = lambda: BrokenLLM()
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(c, content=CSV_OK, name="data.csv"):
    return c.post("/analyze", files={"file": (name, content, "text/csv")})


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_analyze_csv_hop_le(client):
    res = upload(client)
    assert res.status_code == 200
    body = res.json()
    assert body["summary"]["n_rows"] == 3
    assert body["summary"]["total_missing"] == 1
    assert body["explanation"] == "Bo du lieu co 3 dong."
    assert body["llm_error"] is None
    assert body["dataset_id"]


def test_analyze_tu_choi_file_khong_phai_csv(client):
    res = upload(client, name="anh.png")
    assert res.status_code == 400


def test_analyze_tu_choi_csv_rong(client):
    res = upload(client, content=b"")
    assert res.status_code == 400


def test_ask_dataset_khong_ton_tai(client):
    res = client.post("/ask", json={"dataset_id": "khong-co", "question": "Co bao nhieu dong?"})
    assert res.status_code == 404


def test_ask_hop_le(client):
    dataset_id = upload(client).json()["dataset_id"]
    res = client.post("/ask", json={"dataset_id": dataset_id, "question": "Co bao nhieu dong?"})
    assert res.status_code == 200
    assert res.json()["answer"] == "Tra loi: Co bao nhieu dong?"


def test_ask_cau_hoi_qua_ngan_bi_tu_choi(client):
    dataset_id = upload(client).json()["dataset_id"]
    res = client.post("/ask", json={"dataset_id": dataset_id, "question": "?"})
    assert res.status_code == 422  # Pydantic tự kiểm tra min_length=3


def test_analyze_van_tra_thong_ke_khi_ai_loi(broken_client):
    res = upload(broken_client)
    assert res.status_code == 200
    body = res.json()
    assert body["summary"]["n_rows"] == 3
    assert body["explanation"] is None
    assert "mat mang" in body["llm_error"]


def test_ask_tra_503_khi_ai_loi(broken_client):
    dataset_id = upload(broken_client).json()["dataset_id"]
    res = broken_client.post("/ask", json={"dataset_id": dataset_id, "question": "Co bao nhieu dong?"})
    assert res.status_code == 503


def test_explain_thu_lai_thanh_cong(client):
    dataset_id = upload(client).json()["dataset_id"]
    res = client.post(f"/explain/{dataset_id}")
    assert res.status_code == 200
    assert res.json()["explanation"] == "Bo du lieu co 3 dong."


def test_explain_dataset_khong_ton_tai(client):
    assert client.post("/explain/khong-co").status_code == 404


def test_explain_tra_503_khi_ai_loi(broken_client):
    dataset_id = upload(broken_client).json()["dataset_id"]
    assert broken_client.post(f"/explain/{dataset_id}").status_code == 503
