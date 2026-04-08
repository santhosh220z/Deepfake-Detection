from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png"}

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mpeg", ".mpg", ".webm", ".m4v"}
VIDEO_CONTENT_TYPES = {
    "video/mp4",
    "video/x-msvideo",
    "video/quicktime",
    "video/mpeg",
    "video/webm",
    "application/octet-stream",
}


def validate_image_upload(filename: str | None, content_type: str | None) -> None:
    if not filename:
        raise ValueError("Please upload an image file.")

    extension = Path(filename).suffix.lower()
    if extension not in IMAGE_EXTENSIONS:
        raise ValueError("Unsupported format. Allowed formats: jpg, jpeg, png.")

    if content_type and content_type.lower() not in IMAGE_CONTENT_TYPES:
        raise ValueError("Invalid content type. Please upload a JPG or PNG image.")


def validate_video_upload(filename: str | None, content_type: str | None) -> None:
    if not filename:
        raise ValueError("Please upload a video file.")

    extension = Path(filename).suffix.lower()
    if extension not in VIDEO_EXTENSIONS:
        raise ValueError(
            "Unsupported video format. Allowed formats: mp4, avi, mov, mpeg, mpg, webm, m4v."
        )

    if content_type and content_type.lower() not in VIDEO_CONTENT_TYPES:
        raise ValueError(
            "Invalid content type. Please upload a supported video file."
        )


# Backward-compatible alias used by existing image endpoint code.
validate_upload = validate_image_upload


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("Uploaded file is empty.")

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if decoded is None:
        raise ValueError("Could not decode image. Please upload a valid JPG or PNG file.")

    # OpenCV decodes as BGR, so convert to RGB before creating PIL image.
    rgb_image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)
