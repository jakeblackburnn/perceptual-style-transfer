import io

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from PIL import Image

from style_transfer import inference
from style_transfer.models import StyleTransferModel
from style_transfer_web.app import create_app

INSTALLED = ["candy", "kanagawa"]


def png_bytes(size=(32, 32)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 80, 40)).save(buffer, format="PNG")
    return buffer.getvalue()


def post_stylize(client, model, data=None):
    return client.post(
        "/api/stylize",
        files={"file": ("content.png", data if data is not None else png_bytes(), "image/png")},
        data={"model": model},
    )


@pytest.fixture
def loads(monkeypatch):
    """Install fake list_models/load_named_model; return the list of loaded names."""
    loaded_names = []

    def fake_list_models(models_dir):
        return [
            {"name": name, "model_size": "small", "weights": f"{name}.pth",
             "loss": "default", "path": models_dir / name}
            for name in INSTALLED
        ]

    def fake_load_named_model(name, models_dir, device="cpu"):
        if name not in INSTALLED:
            raise FileNotFoundError(name)
        loaded_names.append(name)
        return StyleTransferModel(size_config="small").eval()

    monkeypatch.setattr(inference, "list_models", fake_list_models, raising=False)
    monkeypatch.setattr(inference, "load_named_model", fake_load_named_model, raising=False)
    return loaded_names


@pytest.fixture
def client(tmp_path, loads):
    return TestClient(create_app(tmp_path, "cpu"))


def test_index_serves_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "/static/script.js" in response.text
    assert "/static/styles.css" in response.text


def test_script_is_served_and_has_no_remote_urls(client):
    response = client.get("/static/script.js")
    assert response.status_code == 200
    assert "run.app" not in response.text
    assert "http://localhost" not in response.text
    assert "/api/stylize" in response.text


def test_docs_pages_are_disabled(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404


def test_models_lists_installed_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    assert response.json() == [
        {"name": "candy", "model_size": "small", "loss": "default"},
        {"name": "kanagawa", "model_size": "small", "loss": "default"},
    ]


def test_stylize_returns_png_of_input_size(client):
    response = post_stylize(client, "candy", png_bytes((48, 32)))
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    result = Image.open(io.BytesIO(response.content))
    assert result.format == "PNG"
    assert result.size == (48, 32)


def test_unknown_model_is_404(client):
    response = post_stylize(client, "nope")
    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


def test_garbage_upload_is_400(client, loads):
    response = post_stylize(client, "candy", b"this is not an image")
    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str)
    assert loads == []


def test_value_error_from_stylize_is_400(client, monkeypatch):
    def too_small(content_img, model, device="cpu", max_side=None):
        raise ValueError("image is too small")

    monkeypatch.setattr(inference, "stylize_image", too_small)
    response = post_stylize(client, "candy")
    assert response.status_code == 400
    assert response.json()["detail"] == "image is too small"


def test_oversized_upload_is_413(client, monkeypatch):
    monkeypatch.setattr("style_transfer_web.app.MAX_UPLOAD_BYTES", 100)
    response = post_stylize(client, "candy", b"x" * 101)
    assert response.status_code == 413
    assert isinstance(response.json()["detail"], str)


def test_model_is_cached_by_name(client, loads):
    assert post_stylize(client, "candy").status_code == 200
    assert post_stylize(client, "candy").status_code == 200
    assert loads == ["candy"]

    assert post_stylize(client, "kanagawa").status_code == 200
    assert loads == ["candy", "kanagawa"]
