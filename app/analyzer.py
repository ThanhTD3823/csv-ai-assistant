"""Module phân tích dữ liệu CSV bằng Pandas và NumPy.

Ba hàm chính:
- load_csv(content)  : đọc nội dung file (bytes) thành DataFrame, báo lỗi rõ ràng nếu file hỏng.
- summarize(df)      : tạo bản tóm tắt (dict) gồm số dòng, số cột, giá trị thiếu, thống kê từng cột.
- sample_rows(df, n) : lấy n dòng đầu làm mẫu để gửi cho LLM.

Lưu ý thiết kế: mọi con số đều do Pandas/NumPy tính. LLM chỉ đọc bản tóm tắt này
và diễn giải lại, nên các con số luôn chính xác và kiểm chứng được bằng test.
"""
from __future__ import annotations

import io
import json
import math
from typing import Any

import numpy as np
import pandas as pd


class CSVError(ValueError):
    """Lỗi khi file CSV không hợp lệ (rỗng, sai định dạng...)."""


def load_csv(content: bytes) -> pd.DataFrame:
    """Đọc nội dung file CSV (dạng bytes) thành DataFrame.

    Thử mã hóa UTF-8 trước (phổ biến, hỗ trợ tiếng Việt); nếu lỗi thì thử latin-1.
    """
    if not content or not content.strip():
        raise CSVError("File rỗng, không có dữ liệu.")

    df = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=encoding)
            break
        except UnicodeDecodeError:
            continue  # thử mã hóa tiếp theo
        except pd.errors.EmptyDataError as exc:
            raise CSVError("File không có dữ liệu.") from exc
        except pd.errors.ParserError as exc:
            raise CSVError(f"Không đọc được file CSV: {exc}") from exc

    if df is None:
        raise CSVError("Không nhận diện được mã hóa ký tự của file.")
    if df.shape[1] == 0 or df.empty:
        raise CSVError("File CSV không có dòng dữ liệu nào.")
    return df


def _to_python(value: Any, ndigits: int = 4) -> Any:
    """Chuyển kiểu NumPy sang kiểu Python thường để trả về được dạng JSON.

    NaN và vô cực được đổi thành None (JSON không hỗ trợ NaN).
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, ndigits)
    return value


def _numeric_stats(series: pd.Series) -> dict[str, Any] | None:
    """Thống kê cho cột số, tính bằng NumPy."""
    values = series.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return None
    return {
        "mean": _to_python(np.mean(values)),
        "std": _to_python(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": _to_python(np.min(values)),
        "q1": _to_python(np.percentile(values, 25)),
        "median": _to_python(np.median(values)),
        "q3": _to_python(np.percentile(values, 75)),
        "max": _to_python(np.max(values)),
    }


def _top_values(series: pd.Series, k: int = 5) -> dict[str, int]:
    """Top k giá trị xuất hiện nhiều nhất cho cột chữ (dùng value_counts)."""
    counts = series.dropna().astype(str).value_counts().head(k)
    return {str(key): int(val) for key, val in counts.items()}


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    """Tạo bản tóm tắt toàn bộ bảng dữ liệu."""
    n_rows, n_cols = df.shape
    columns = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        info: dict[str, Any] = {
            "name": str(col),
            "dtype": str(series.dtype),
            "missing": missing,
            "missing_pct": round(100 * missing / n_rows, 2) if n_rows else 0.0,
            "unique": int(series.nunique(dropna=True)),
            "stats": None,
            "top_values": None,
        }
        # Cột kiểu bool cũng là "numeric" trong Pandas, nên loại ra và coi như cột phân loại.
        is_number = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        if is_number:
            info["stats"] = _numeric_stats(series)
        else:
            info["top_values"] = _top_values(series)
        columns.append(info)

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "total_missing": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
    }


def sample_rows(df: pd.DataFrame, n: int = 5) -> list[dict[str, Any]]:
    """Lấy n dòng đầu. Đi qua to_json để NaN thành null và ngày tháng thành chuỗi."""
    return json.loads(df.head(n).to_json(orient="records", force_ascii=False, date_format="iso"))
