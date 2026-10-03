"""The style-transfer command, run in-process against a temp models dir."""

import json
import sys

import pytest
import torch
from PIL import Image

from style_transfer.cli import main
from style_transfer.models import StyleTransferModel


@pytest.fixture
def models_dir(tmp_path):
    folder = tmp_path / "models" / "tiny"
    folder.mkdir(parents=True)
    torch.save(StyleTransferModel("small").state_dict(), folder / "tiny.pth")
    (folder / "model.json").write_text(json.dumps(
        {"name": "tiny", "model_size": "small", "weights": "tiny.pth", "loss": "johnson-sum"}))
    return tmp_path / "models"


def _photo(path, size=(37, 22)):
    Image.new("RGB", size, (200, 30, 90)).save(path)
    return path


def _stylize(models_dir, *args):
    return main(["--models-dir", str(models_dir), "stylize", *map(str, args), "--model", "tiny", "--device", "cpu"])


def test_list_shows_installed_models_and_experiments(models_dir, capsys):
    assert main(["--models-dir", str(models_dir), "list"]) == 0
    out = capsys.readouterr().out
    assert "tiny" in out and "johnson-sum" in out
    assert "mini_kanagawa" in out  # a registered training experiment


def test_list_with_no_models_says_how_to_get_one(tmp_path, capsys):
    assert main(["--models-dir", str(tmp_path), "list"]) == 0
    out = capsys.readouterr().out
    assert "none" in out
    assert "train --experiment" in out and "write_model_json.py" in out


def test_stylize_file(models_dir, tmp_path, capsys):
    photo = _photo(tmp_path / "photo.jpg")
    out = tmp_path / "out" / "styled.jpg"

    assert _stylize(models_dir, photo, "-o", out) == 0

    with Image.open(out) as result:
        assert result.size == (37, 22)
        assert result.format == "JPEG"
    assert capsys.readouterr().out.count("\n") == 1


def test_stylize_directory(models_dir, tmp_path, capsys):
    photos = tmp_path / "photos"
    photos.mkdir()
    _photo(photos / "b.jpg")
    _photo(photos / "a.png")
    _photo(photos / "c.JPEG")
    (photos / "notes.txt").write_text("not an image")
    out_dir = tmp_path / "styled"

    assert _stylize(models_dir, photos, "--output-dir", out_dir, "--limit", 2) == 0

    assert sorted(p.name for p in out_dir.iterdir()) == ["a.png", "b.png"]
    assert capsys.readouterr().out.count("\n") == 2


def test_stylize_refuses_to_overwrite(models_dir, tmp_path, capsys):
    photo = _photo(tmp_path / "photo.jpg")
    out = tmp_path / "out.png"
    out.write_bytes(b"keep me")

    assert _stylize(models_dir, photo, "-o", out) == 2
    assert out.read_bytes() == b"keep me"
    assert "--overwrite" in capsys.readouterr().err

    assert _stylize(models_dir, photo, "-o", out, "--overwrite") == 0
    with Image.open(out) as result:
        assert result.size == (37, 22)


def test_stylize_detects_output_name_collision(models_dir, tmp_path, capsys):
    photos = tmp_path / "photos"
    photos.mkdir()
    _photo(photos / "a.jpg")
    _photo(photos / "a.png")
    out_dir = tmp_path / "styled"

    assert _stylize(models_dir, photos, "--output-dir", out_dir) == 2

    err = capsys.readouterr().err
    assert "a.jpg" in err and "a.png" in err
    assert not out_dir.exists()


def test_stylize_file_requires_output_file(models_dir, tmp_path):
    photo = _photo(tmp_path / "photo.jpg")
    assert _stylize(models_dir, photo, "--output-dir", tmp_path / "styled") == 2


def test_stylize_unreadable_image_is_a_processing_failure(models_dir, tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_text("not an image")
    assert _stylize(models_dir, broken, "-o", tmp_path / "out.png") == 1
    assert not (tmp_path / "out.png").exists()


def test_stylize_missing_model(models_dir, tmp_path, capsys):
    photo = _photo(tmp_path / "photo.jpg")
    code = main(["--models-dir", str(models_dir), "stylize", str(photo), "--model", "absent",
                 "-o", str(tmp_path / "out.png")])
    assert code == 2
    assert "tiny" in capsys.readouterr().err  # lists what is available


def test_train_calls_train_model_with_models_dir(models_dir, tmp_path, monkeypatch, capsys):
    import style_transfer.train
    from style_transfer.config import Models

    calls = []

    def fake_train_model(model_name, model_config, device, models_dir="artifacts/models"):
        calls.append((model_name, model_config, str(device), models_dir))
        return models_dir / model_name / f"{model_name}.pth"

    monkeypatch.setattr(style_transfer.train, "train_model", fake_train_model)

    assert main(["--models-dir", str(models_dir), "train", "--experiment", "mini_kanagawa", "--device", "cpu"]) == 0

    assert calls == [("mini_kanagawa", Models["mini_kanagawa"], "cpu", models_dir)]
    out = capsys.readouterr().out
    assert str(models_dir / "mini_kanagawa" / "mini_kanagawa.pth") in out
    assert "stylize" in out and "--model mini_kanagawa" in out


def test_train_unknown_experiment(models_dir, capsys):
    assert main(["--models-dir", str(models_dir), "train", "--experiment", "absent"]) == 2
    assert "mini_kanagawa" in capsys.readouterr().err


def test_serve_without_web_package_hints_at_ui_extra(models_dir, monkeypatch, capsys):
    # None makes the import fail; .app too, since test_web may have imported it already
    monkeypatch.setitem(sys.modules, "style_transfer_web", None)
    monkeypatch.setitem(sys.modules, "style_transfer_web.app", None)
    assert main(["--models-dir", str(models_dir), "serve"]) == 2
    assert "pip install -e '.[ui]'" in capsys.readouterr().err


def test_serve_starts_the_web_app_on_the_models_dir(models_dir, monkeypatch):
    app = pytest.importorskip("style_transfer_web.app")
    calls = []
    monkeypatch.setattr(app, "run", lambda models_dir, port, device: calls.append((models_dir, port, device)))
    assert main(["--models-dir", str(models_dir), "serve", "--port", "8123"]) == 0
    assert calls == [(models_dir, 8123, None)]
