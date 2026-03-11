"""
Preprocessing Module
Handles video frame extraction, image resizing, normalization, and the full preprocessing pipeline.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torchvision import transforms


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transform(input_size=380):
    """
    Get the standard inference transform pipeline.
    
    Args:
        input_size: Target image size
        
    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def extract_frames(video_path, frame_interval=10, max_frames=50):
    """
    Extract frames from a video file.

    Args:
        video_path: Path to video file
        frame_interval: Extract every Nth frame
        max_frames: Maximum number of frames to extract

    Returns:
        List of PIL Images (RGB)
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    frame_count = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)

        frame_count += 1

    cap.release()
    return frames


def preprocess_image(image, input_size=380):
    """
    Preprocess a single image for model inference.

    Args:
        image: PIL Image (RGB)
        input_size: Target size

    Returns:
        torch.Tensor of shape (1, 3, input_size, input_size)
    """
    transform = get_inference_transform(input_size)
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension


def process_input(input_path, face_detector=None, input_size=380, frame_interval=10, max_frames=50):
    """
    Full preprocessing pipeline for an image or video input.

    Args:
        input_path: Path to image or video file
        face_detector: FaceDetector instance (optional)
        input_size: Target image size
        frame_interval: Frame extraction interval for videos
        max_frames: Max frames for videos

    Returns:
        List of preprocessed torch.Tensor (each shape: (1, 3, input_size, input_size))
    """
    input_path = Path(input_path)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    ext = input_path.suffix.lower()

    if ext in video_extensions:
        # Video: extract frames
        frames = extract_frames(str(input_path), frame_interval, max_frames)
        images = frames
    elif ext in image_extensions:
        # Single image
        image = Image.open(str(input_path)).convert('RGB')
        images = [image]
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Face detection
    if face_detector is not None:
        images = [face_detector.extract_largest_face(img) for img in images]

    # Preprocess each image
    tensors = [preprocess_image(img, input_size) for img in images]

    return tensors


def preprocess_uploaded_file(file_bytes, filename, face_detector=None, input_size=380,
                              frame_interval=10, max_frames=50):
    """
    Preprocess an uploaded file (from web interface) given raw bytes.

    Args:
        file_bytes: Raw file bytes
        filename: Original filename (used to determine type)
        face_detector: FaceDetector instance (optional)
        input_size: Target image size
        frame_interval: Frame extraction interval for videos
        max_frames: Max frames for videos

    Returns:
        List of preprocessed torch.Tensor
    """
    import tempfile
    import os

    ext = Path(filename).suffix.lower()
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    if ext in image_extensions:
        from io import BytesIO
        image = Image.open(BytesIO(file_bytes)).convert('RGB')

        if face_detector is not None:
            image = face_detector.extract_largest_face(image)

        return [preprocess_image(image, input_size)]

    elif ext in video_extensions:
        # Save to temp file for OpenCV
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            frames = extract_frames(tmp_path, frame_interval, max_frames)

            if face_detector is not None:
                frames = [face_detector.extract_largest_face(f) for f in frames]

            return [preprocess_image(f, input_size) for f in frames]
        finally:
            os.unlink(tmp_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
