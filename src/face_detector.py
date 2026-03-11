"""
Face Detection Module using MTCNN
Detects and crops faces from images for deepfake analysis.
"""

import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch


class FaceDetector:
    """MTCNN-based face detector for preprocessing deepfake detection inputs."""

    def __init__(self, device=None, confidence_threshold=0.9, margin=20, min_face_size=40):
        """
        Initialize the face detector.

        Args:
            device: torch device (auto-detected if None)
            confidence_threshold: Minimum detection confidence
            margin: Pixel margin around detected face
            min_face_size: Minimum face size in pixels
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.margin = margin

        self.detector = MTCNN(
            image_size=380,
            margin=margin,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=device,
            keep_all=True
        )

    def detect_faces(self, image):
        """
        Detect faces in an image.

        Args:
            image: PIL Image or numpy array (RGB)

        Returns:
            list of dicts with 'box' (x1,y1,x2,y2), 'confidence', and 'face' (cropped PIL Image)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        boxes, confidences = self.detector.detect(image)

        results = []
        if boxes is not None:
            for box, conf in zip(boxes, confidences):
                if conf is not None and conf >= self.confidence_threshold:
                    # Crop face with margin
                    x1, y1, x2, y2 = [int(b) for b in box]
                    w, h = image.size
                    x1 = max(0, x1 - self.margin)
                    y1 = max(0, y1 - self.margin)
                    x2 = min(w, x2 + self.margin)
                    y2 = min(h, y2 + self.margin)

                    face = image.crop((x1, y1, x2, y2))
                    results.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': float(conf),
                        'face': face
                    })

        return results

    def extract_largest_face(self, image):
        """
        Extract the largest face from an image.
        Falls back to the full image if no face is detected.

        Args:
            image: PIL Image or numpy array (RGB)

        Returns:
            PIL Image of the largest face (or full image as fallback)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        faces = self.detect_faces(image)

        if not faces:
            # Fallback: return the full image
            return image

        # Return the largest face by area
        largest = max(faces, key=lambda f: (f['box'][2] - f['box'][0]) * (f['box'][3] - f['box'][1]))
        return largest['face']

    def detect_batch(self, images):
        """
        Detect faces in a batch of images.

        Args:
            images: List of PIL Images or numpy arrays

        Returns:
            List of face images (largest face per image)
        """
        return [self.extract_largest_face(img) for img in images]
