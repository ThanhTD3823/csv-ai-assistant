"""Test cho module analyzer: kiểm tra các con số do Pandas/NumPy tính là đúng."""
import pytest

from app.analyzer import CSVError, load_csv, sample_rows, summarize

CSV_OK = b"""ten,tuoi,thanh_pho
An,20,Ha Noi
Binh,,Ha Noi
Chi,30,Da Nang
Chi,30,Da Nang
"""


def test_load_csv_hop_le():
    df = load_csv(CSV_OK)
    assert df.shape == (4, 3)


def test_load_csv_rong_bao_loi():
    with pytest.raises(CSVError):
        load_csv(b"   ")


def test_load_csv_tieng_viet_utf8():
    df = load_csv("ten,que\nAn,Hà Nội\n".encode("utf-8"))
    assert df.loc[0, "que"] == "Hà Nội"


def test_dem_dung_gia_tri_thieu():
    summary = summarize(load_csv(CSV_OK))
    tuoi = next(c for c in summary["columns"] if c["name"] == "tuoi")
    assert tuoi["missing"] == 1
    assert tuoi["missing_pct"] == 25.0
    assert summary["total_missing"] == 1


def test_tinh_dung_trung_binh():
    summary = summarize(load_csv(CSV_OK))
    tuoi = next(c for c in summary["columns"] if c["name"] == "tuoi")
    # (20 + 30 + 30) / 3 = 26.6667 (bỏ qua giá trị thiếu)
    assert tuoi["stats"]["mean"] == pytest.approx(26.6667, abs=1e-3)
    assert tuoi["stats"]["max"] == 30


def test_dem_dong_trung_lap_va_top_values():
    summary = summarize(load_csv(CSV_OK))
    assert summary["duplicate_rows"] == 1
    tp = next(c for c in summary["columns"] if c["name"] == "thanh_pho")
    assert tp["top_values"] == {"Ha Noi": 2, "Da Nang": 2}
    assert tp["stats"] is None


def test_sample_rows_doi_nan_thanh_none():
    rows = sample_rows(load_csv(CSV_OK), n=2)
    assert len(rows) == 2
    assert rows[1]["tuoi"] is None
