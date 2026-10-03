"""Local web UI: one FastAPI process serving the static page and the API.

Bound to 127.0.0.1 only. One request per stylization: the browser posts the
image and the model name, the response body is the stylized PNG. Nothing is
written to disk. Errors are JSON {"detail": "..."} (FastAPI's HTTPException).

The inference functions are looked up on the `inference` module at request
time (not imported by name) so tests can replace them.
"""

import io
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from style_transfer import inference

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def create_app(models_dir, device) -> FastAPI:
    # openapi/docs pages are off: they load their assets from a CDN.
    app = FastAPI(title="Style Transfer (local)", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # One loaded model at a time. The lock covers both the cache and the
    # forward pass, so two requests never run inference at once.
    lock = threading.Lock()
    loaded = {"name": None, "model": None}

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/models")
    def models():
        return [
            {"name": m["name"], "model_size": m["model_size"], "loss": m["loss"]}
            for m in inference.list_models(models_dir)
        ]

    # Plain `def`: FastAPI runs it in a threadpool, off the event loop.
    @app.post("/api/stylize")
    def stylize(file: UploadFile = File(...), model: str = Form(...)):
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image is larger than the 20 MB limit.")

        try:
            content_img = Image.open(io.BytesIO(data))
            content_img.load()
        except Exception:  # PIL raises several unrelated types for bad image data
            raise HTTPException(400, "Could not read that file as an image.")

        with lock:
            if loaded["name"] != model:
                try:
                    loaded["model"] = inference.load_named_model(model, models_dir, device)
                except FileNotFoundError:
                    raise HTTPException(404, f"Model '{model}' is not installed.")
                loaded["name"] = model
            try:
                output_img = inference.stylize_image(content_img, loaded["model"], device)
            except ValueError as error:
                raise HTTPException(400, str(error))

        buffer = io.BytesIO()
        output_img.save(buffer, format="PNG")
        return Response(buffer.getvalue(), media_type="image/png")

    return app


def run(models_dir, port=8000, device=None):
    import uvicorn

    if device is None:
        device = inference.select_device()
    print(f"http://127.0.0.1:{port}")
    uvicorn.run(create_app(models_dir, device), host="127.0.0.1", port=port)
