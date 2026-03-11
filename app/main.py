"""
FastAPI Backend for Deepfake Detection
Handles image/video uploads, runs preprocessing + model inference, returns predictions.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.model import DeepfakeDetector, load_model, create_model
from src.face_detector import FaceDetector
from src.preprocessing import preprocess_uploaded_file

# ─── App Setup ──────────────────────────────────────────────
app = FastAPI(
    title="Deepfake Detection API",
    description="AI-powered deepfake and AI-generated image detection using EfficientNet-B4",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ───────────────────────────────────────────
model = None
face_detector = None
device = None
config = None


@app.on_event("startup")
async def startup():
    """Load model and face detector on startup."""
    global model, face_detector, device, config

    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'model': {'input_size': 380, 'dropout': 0.3},
            'face_detection': {'confidence_threshold': 0.9, 'margin': 20, 'min_face_size': 40},
            'video': {'frame_interval': 10, 'max_frames': 50}
        }

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")

    # Load model
    model_path = Path(__file__).parent.parent / "models" / "best_model.pth"
    if model_path.exists():
        print(f"📦 Loading trained model from {model_path}...")
        model = load_model(str(model_path), device=device)
    else:
        print(f"⚠️  No trained model found at {model_path}")
        print(f"    Loading pretrained EfficientNet-B4 for demo (untrained classifier)...")
        model = create_model(pretrained=True, dropout=config['model']['dropout'], device=device)
        model.eval()

    # Initialize face detector
    print(f"👤 Initializing face detector (MTCNN)...")
    fd_config = config.get('face_detection', {})
    face_detector = FaceDetector(
        device=device,
        confidence_threshold=fd_config.get('confidence_threshold', 0.9),
        margin=fd_config.get('margin', 20),
        min_face_size=fd_config.get('min_face_size', 40)
    )
    print(f"✅ System ready!")


# ─── Static Files ──────────────────────────────────────────
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ─── Routes ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = static_dir / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Frontend not found. Place index.html in app/static/</h1>")
    return HTMLResponse(index_path.read_text(encoding='utf-8'))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "face_detector": face_detector is not None
    }


@app.post("/api/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict whether an uploaded image is real or fake.

    Returns:
        JSON with prediction, confidence, and details
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    allowed_image_types = {'image/jpeg', 'image/png', 'image/bmp', 'image/webp', 'image/tiff'}
    if file.content_type not in allowed_image_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Accepted: JPEG, PNG, BMP, WEBP, TIFF"
        )

    try:
        file_bytes = await file.read()
        input_size = config['model']['input_size']

        # Preprocess
        tensors = preprocess_uploaded_file(
            file_bytes, file.filename,
            face_detector=face_detector,
            input_size=input_size
        )

        # Predict
        model.eval()
        with torch.no_grad():
            tensor = tensors[0].to(device)
            output = model(tensor)
            probability = output.item()

        prediction = "Fake" if probability > 0.5 else "Real"
        confidence = probability if probability > 0.5 else (1.0 - probability)

        return {
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "raw_probability": round(probability, 6),
            "filename": file.filename,
            "type": "image"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/api/predict/video")
async def predict_video(file: UploadFile = File(...)):
    """
    Predict whether an uploaded video contains deepfake content.
    Analyzes multiple frames and returns aggregate results.

    Returns:
        JSON with overall prediction, per-frame results, and aggregate confidence
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    allowed_video_types = {'video/mp4', 'video/avi', 'video/x-msvideo', 'video/quicktime',
                           'video/x-matroska', 'video/webm', 'application/octet-stream'}
    if file.content_type not in allowed_video_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Accepted: MP4, AVI, MOV, MKV, WEBM"
        )

    try:
        file_bytes = await file.read()
        input_size = config['model']['input_size']
        video_config = config.get('video', {})

        # Preprocess (extracts frames + face detection)
        tensors = preprocess_uploaded_file(
            file_bytes, file.filename,
            face_detector=face_detector,
            input_size=input_size,
            frame_interval=video_config.get('frame_interval', 10),
            max_frames=video_config.get('max_frames', 50)
        )

        if not tensors:
            raise HTTPException(status_code=400, detail="Could not extract frames from video")

        # Predict each frame
        model.eval()
        frame_results = []
        all_probs = []

        with torch.no_grad():
            for i, tensor in enumerate(tensors):
                tensor = tensor.to(device)
                output = model(tensor)
                prob = output.item()
                all_probs.append(prob)
                frame_results.append({
                    "frame": i + 1,
                    "probability": round(prob, 6),
                    "prediction": "Fake" if prob > 0.5 else "Real"
                })

        # Aggregate prediction
        avg_prob = sum(all_probs) / len(all_probs)
        fake_count = sum(1 for p in all_probs if p > 0.5)
        total_frames = len(all_probs)

        overall_prediction = "Fake" if avg_prob > 0.5 else "Real"
        overall_confidence = avg_prob if avg_prob > 0.5 else (1.0 - avg_prob)

        return {
            "prediction": overall_prediction,
            "confidence": round(overall_confidence * 100, 2),
            "raw_probability": round(avg_prob, 6),
            "frames_analyzed": total_frames,
            "fake_frames": fake_count,
            "real_frames": total_frames - fake_count,
            "frame_results": frame_results,
            "filename": file.filename,
            "type": "video"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ─── Run ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    host = "0.0.0.0"
    port = 8000
    if config:
        host = config.get('app', {}).get('host', host)
        port = config.get('app', {}).get('port', port)
    uvicorn.run(app, host=host, port=port)
