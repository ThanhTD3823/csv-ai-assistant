"""Test cơ chế tự chuyển mô hình khi tên mô hình không còn tồn tại (lỗi 404 NOT_FOUND)."""
from types import SimpleNamespace

import pytest

import app.services.llm_service as llm_module
from app.services.llm_service import LLMError, LLMService, pick_flash_model


def test_pick_flash_model_uu_tien_ban_on_dinh_moi_nhat():
    names = [
        "gemini-2.5-flash",
        "gemini-3.8-flash",
        "gemini-3.8-flash-lite",
        "gemini-3.9-flash-preview",
        "gemini-3.8-flash-tts",
        "gemini-3.8-pro",
    ]
    assert pick_flash_model(names) == "gemini-3.8-flash"


def test_pick_flash_model_khong_co_flash():
    assert pick_flash_model(["gemini-3.8-pro"]) is None


class FakeModels:
    """Giả lập client.models: model cũ báo 404, model mới trả lời được."""

    def generate_content(self, model, contents, config):
        if model == "gemini-flash-latest":
            raise RuntimeError("404 NOT_FOUND. models/gemini-flash-latest is not found")
        return SimpleNamespace(text=f"ok tu {model}")

    def list(self):
        return [
            SimpleNamespace(name="models/gemini-3.8-flash", supported_actions=["generateContent"]),
            SimpleNamespace(name="models/text-embedding", supported_actions=["embedContent"]),
        ]


@pytest.fixture(autouse=True)
def reset_resolved_model():
    llm_module._resolved_model = None
    llm_module._available_models = None
    llm_module._thinking_supported = True
    yield
    llm_module._resolved_model = None
    llm_module._available_models = None
    llm_module._thinking_supported = True


def make_service(model):
    service = LLMService(api_key="fake", model=model)
    service._client = SimpleNamespace(models=FakeModels())
    return service


def test_tu_chuyen_sang_model_kha_dung_khi_bi_404():
    service = make_service("gemini-flash-latest")
    assert service.generate("xin chao") == "ok tu gemini-3.8-flash"
    # Lần tạo service sau dùng luôn model đã tìm được
    assert LLMService(api_key="fake", model="gemini-flash-latest").model == "gemini-3.8-flash"


def test_het_han_muc_o_moi_mo_hinh_thi_bao_loi_de_hieu():
    service = make_service("gemini-3.8-flash")
    service._client.models.generate_content = lambda **kw: (_ for _ in ()).throw(RuntimeError("429 quota"))
    with pytest.raises(LLMError, match="hết hạn mức"):
        service.generate("xin chao")


def test_thieu_api_key_bao_loi_ro_rang():
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        LLMService(api_key="", model="gemini-3.8-flash").generate("xin chao")


def test_timeout_bao_loi_than_thien():
    class Timeout(Exception):
        pass

    service = make_service("gemini-3.8-flash")
    service._client.models.generate_content = lambda **kw: (_ for _ in ()).throw(Timeout("read timed out"))
    with pytest.raises(LLMError, match="không phản hồi"):
        service.generate("xin chao")


def test_mo_hinh_khong_ho_tro_thinking_thi_goi_lai_khong_kem_tuy_chon():
    calls = []

    def fake_generate(model, contents, config):
        calls.append(config.thinking_config)
        if config.thinking_config is not None:
            raise RuntimeError("400 INVALID_ARGUMENT: thinking_level is not supported")
        return SimpleNamespace(text="ok")

    service = make_service("gemini-3.8-flash")
    service._client.models.generate_content = fake_generate
    assert service.generate("xin chao") == "ok"
    assert calls[0] is not None and calls[1] is None
    assert llm_module._thinking_supported is False


def test_client_that_co_timeout_va_khong_thu_lai():
    client = LLMService(api_key="fake-key", model="gemini-3.8-flash")._get_client()
    opts = client._api_client._http_options
    assert opts.timeout == llm_module.LLM_TIMEOUT_SECONDS * 1000
    assert opts.retry_options.attempts == 1


class OverloadedModels:
    """Giả lập lỗi 503 bạn gặp: gemini-3.8-flash quá tải, mô hình khác vẫn trả lời được."""

    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        if model == "gemini-3.8-flash":
            raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
        return SimpleNamespace(text=f"ok tu {model}")

    def list(self):
        return [
            SimpleNamespace(name=f"models/{n}", supported_actions=["generateContent"])
            for n in ("gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-2.5-flash")
        ]


def test_qua_tai_503_thi_chuyen_sang_mo_hinh_khac():
    service = LLMService(api_key="fake", model="gemini-3.8-flash")
    service._client = SimpleNamespace(models=OverloadedModels())
    assert service.generate("xin chao") == "ok tu gemini-2.5-flash"
    assert service._client.models.calls == ["gemini-3.8-flash", "gemini-2.5-flash"]
    # 503 chỉ là tạm thời → lần sau vẫn ưu tiên mô hình đã cấu hình
    assert LLMService(api_key="fake", model="gemini-3.8-flash").model == "gemini-3.8-flash"


def test_tat_ca_mo_hinh_qua_tai_thi_bao_loi_de_hieu_va_dung_sau_3_lan():
    models = OverloadedModels()
    models.generate_content_orig = models.generate_content

    def always_503(model, contents, config):
        models.calls.append(model)
        raise RuntimeError("503 UNAVAILABLE")

    models.generate_content = always_503
    service = LLMService(api_key="fake", model="gemini-3.8-flash")
    service._client = SimpleNamespace(models=models)
    with pytest.raises(LLMError, match="quá tải"):
        service.generate("xin chao")
    assert models.calls == ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-3.8-flash-lite"]
