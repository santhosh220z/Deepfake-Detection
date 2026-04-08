from pathlib import Path
import logging
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from model import load_model_bundle
from predict import predict_image
from utils import load_image_from_bytes, validate_image_upload, validate_video_upload
from video_predict import load_video_model_bundle, predict_video

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
LOGGER = logging.getLogger(__name__)

IMAGE_MODEL_READY = False
IMAGE_MODEL_ERROR: str | None = None

VIDEO_MODEL_READY = False
VIDEO_MODEL_ERROR: str | None = None

VIDEO_ANALYSIS_FRAMES = max(1, int(os.getenv("VIDEO_ANALYSIS_FRAMES", "24")))

app = FastAPI(
    title="Deepfake Image Detection API",
    description=(
        "Classifies uploaded images and videos as real or deepfake using a "
        "dedicated image model and video model."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def ensure_image_model_ready() -> None:
    global IMAGE_MODEL_READY, IMAGE_MODEL_ERROR

    try:
        load_model_bundle()
        IMAGE_MODEL_READY = True
        IMAGE_MODEL_ERROR = None
    except Exception as exc:  # noqa: BLE001
        IMAGE_MODEL_READY = False
        IMAGE_MODEL_ERROR = str(exc)
        LOGGER.exception("Image model initialization failed")
        raise


def ensure_video_model_ready() -> None:
    global VIDEO_MODEL_READY, VIDEO_MODEL_ERROR

    try:
        load_video_model_bundle()
        VIDEO_MODEL_READY = True
        VIDEO_MODEL_ERROR = None
    except Exception as exc:  # noqa: BLE001
        VIDEO_MODEL_READY = False
        VIDEO_MODEL_ERROR = str(exc)
        LOGGER.exception("Video model initialization failed")
        raise


@app.on_event("startup")
async def warmup_model() -> None:
    try:
        ensure_image_model_ready()
    except Exception:
        # Keep API available so users can still access health and frontend.
        return


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str | None]:
    return {
        "status": "ok",
        "image_model_status": "ready" if IMAGE_MODEL_READY else "not_ready",
        "image_model_error": IMAGE_MODEL_ERROR,
        "video_model_status": "ready" if VIDEO_MODEL_READY else "not_ready",
        "video_model_error": VIDEO_MODEL_ERROR,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not IMAGE_MODEL_READY:
        try:
            ensure_image_model_ready()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail=f"Image model is not ready: {IMAGE_MODEL_ERROR or str(exc)}",
            ) from exc

    try:
        validate_image_upload(file.filename, file.content_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image_bytes = await file.read()

    try:
        image = load_image_from_bytes(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        return predict_image(image)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Model inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(exc)}",
        ) from exc


@app.post("/predict-video")
async def predict_video_endpoint(file: UploadFile = File(...)) -> dict:
    try:
        validate_video_upload(file.filename, file.content_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty.")

    if not VIDEO_MODEL_READY:
        try:
            ensure_video_model_ready()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail=f"Video model is not ready: {VIDEO_MODEL_ERROR or str(exc)}",
            ) from exc

    suffix = Path(file.filename or "upload.mp4").suffix.lower() or ".mp4"
    temp_file_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(video_bytes)
            temp_file_path = Path(temp_file.name)

        return predict_video(
            video_path=temp_file_path,
            num_frames=VIDEO_ANALYSIS_FRAMES,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Video inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"Video inference failed: {str(exc)}",
        ) from exc
    finally:
        if temp_file_path and temp_file_path.exists():
            temp_file_path.unlink(missing_ok=True)
