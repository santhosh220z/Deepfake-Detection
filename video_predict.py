from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import cv2
from huggingface_hub import hf_hub_download
import numpy as np
import torch
import yaml

from video_model.GenConViT.model.genconvit_ed import GenConViTED

BASE_DIR = Path(__file__).resolve().parent
VIDEO_MODEL_ROOT = BASE_DIR / "video_model" / "GenConViT"
VIDEO_MODEL_CONFIG_PATH = VIDEO_MODEL_ROOT / "model" / "config.yaml"
VIDEO_MODEL_WEIGHT_DIR = VIDEO_MODEL_ROOT / "weight"

VIDEO_MODEL_HF_REPO = os.getenv("VIDEO_MODEL_HF_REPO", "Deressa/GenConViT")
VIDEO_MODEL_WEIGHT_STEM = os.getenv("VIDEO_MODEL_WEIGHT_STEM", "genconvit_ed_inference")
VIDEO_MODEL_AUTO_DOWNLOAD = os.getenv("VIDEO_MODEL_AUTO_DOWNLOAD", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}

VIDEO_FINAL_FAKE_LABEL_THRESHOLD = float(
    os.getenv("VIDEO_FINAL_FAKE_LABEL_THRESHOLD", "0.70")
)

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

HAAR_CASCADE_PATH = (
    Path(cv2.__file__).resolve().parent / "data" / "haarcascade_frontalface_default.xml"
)
FACE_CASCADE = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))


def _load_video_model_config() -> dict[str, Any]:
    if not VIDEO_MODEL_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "Video model config not found. Expected: "
            f"{VIDEO_MODEL_CONFIG_PATH}"
        )

    with VIDEO_MODEL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}

    return state_dict


def _resolve_video_weight_path() -> Path:
    custom_path = os.getenv("VIDEO_MODEL_WEIGHT_PATH")
    if custom_path:
        resolved = Path(custom_path).expanduser().resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(
            f"VIDEO_MODEL_WEIGHT_PATH points to missing file: {resolved}"
        )

    default_path = VIDEO_MODEL_WEIGHT_DIR / f"{VIDEO_MODEL_WEIGHT_STEM}.pth"
    if default_path.exists():
        return default_path

    if not VIDEO_MODEL_AUTO_DOWNLOAD:
        raise FileNotFoundError(
            "GenConViT ED weight is missing. Place "
            f"{VIDEO_MODEL_WEIGHT_STEM}.pth in {VIDEO_MODEL_WEIGHT_DIR} "
            "or set VIDEO_MODEL_WEIGHT_PATH."
        )

    VIDEO_MODEL_WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=VIDEO_MODEL_HF_REPO,
        filename=f"{VIDEO_MODEL_WEIGHT_STEM}.pth",
        local_dir=str(VIDEO_MODEL_WEIGHT_DIR),
    )
    return Path(downloaded)


@lru_cache(maxsize=1)
def load_video_model_bundle() -> tuple[GenConViTED, torch.device]:
    config = _load_video_model_config()
    weight_path = _resolve_video_weight_path()

    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = _normalize_state_dict(state_dict)

    model = GenConViTED(config=config, pretrained=False)
    model.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, device


def _sample_video_frames(video_path: Path, num_frames: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Unable to open the uploaded video file.")

    frames: list[np.ndarray] = []
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        if total_frames > 0:
            sample_count = max(1, min(num_frames, total_frames))
            sample_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

            for frame_index in sample_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                success, frame = capture.read()
                if success and frame is not None:
                    frames.append(frame)
        else:
            while len(frames) < max(1, num_frames):
                success, frame = capture.read()
                if not success or frame is None:
                    break
                frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise ValueError("No frames could be extracted from the uploaded video.")

    return frames


def _center_crop(frame_rgb: np.ndarray) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return frame_rgb[top : top + side, left : left + side]


def _extract_face_or_center(frame_rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    if FACE_CASCADE.empty():
        return _center_crop(frame_rgb), False

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48),
    )

    if len(faces) == 0:
        return _center_crop(frame_rgb), False

    x, y, width, height = max(faces, key=lambda item: int(item[2]) * int(item[3]))
    margin_x = int(width * 0.2)
    margin_y = int(height * 0.2)

    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(frame_rgb.shape[1], x + width + margin_x)
    y1 = min(frame_rgb.shape[0], y + height + margin_y)

    if x1 <= x0 or y1 <= y0:
        return _center_crop(frame_rgb), False

    return frame_rgb[y0:y1, x0:x1], True


def _preprocess_frames(
    frames_bgr: list[np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    processed: list[np.ndarray] = []
    detected_faces = 0

    for frame_bgr in frames_bgr:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        region, has_face = _extract_face_or_center(frame_rgb)
        if has_face:
            detected_faces += 1

        resized = cv2.resize(region, (224, 224), interpolation=cv2.INTER_AREA)
        processed.append(resized)

    array = np.stack(processed, axis=0)
    tensor = torch.from_numpy(array).float().permute(0, 3, 1, 2) / 255.0
    tensor = tensor.to(device)

    mean = MEAN.to(device)
    std = STD.to(device)
    tensor = (tensor - mean) / std

    return tensor, detected_faces


def predict_video(video_path: str | Path, num_frames: int = 24) -> dict[str, Any]:
    video_path = Path(video_path)
    if not video_path.exists():
        raise ValueError("Uploaded video file was not found on disk.")

    model, device = load_video_model_bundle()

    frames = _sample_video_frames(video_path, num_frames=max(1, num_frames))
    frame_tensor, detected_faces = _preprocess_frames(frames, device)

    with torch.no_grad():
        logits = model(frame_tensor)
        frame_probabilities = torch.softmax(logits, dim=1)
        mean_probabilities = frame_probabilities.mean(dim=0).cpu().numpy()

    # GenConViT ED output index mapping: 0=fake, 1=real.
    fake_probability = float(mean_probabilities[0])
    real_probability = float(mean_probabilities[1])

    is_fake = fake_probability >= VIDEO_FINAL_FAKE_LABEL_THRESHOLD
    label = "fake" if is_fake else "real"
    confidence = fake_probability if is_fake else real_probability

    sampled_frames = len(frames)
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": [real_probability, fake_probability],
        "class_names": ["real", "fake"],
        "sampled_frames": sampled_frames,
        "face_detected_frames": detected_faces,
        "face_detected_ratio": detected_faces / max(1, sampled_frames),
        "fake_probability": fake_probability,
        "fake_label_threshold": VIDEO_FINAL_FAKE_LABEL_THRESHOLD,
    }
