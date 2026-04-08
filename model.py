from functools import lru_cache
import logging

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_NAME = "dima806/deepfake_vs_real_image_detection"
LOGGER = logging.getLogger(__name__)


def _normalize_label(label: str) -> str:
    lowered = str(label).strip().lower()
    if "real" in lowered:
        return "real"
    if "fake" in lowered:
        return "fake"
    return lowered


def _load_processor():
    try:
        return AutoImageProcessor.from_pretrained(MODEL_NAME)
    except ImportError as exc:
        if "Torchvision" not in str(exc):
            raise
        LOGGER.warning(
            "Torchvision is unavailable; retrying with use_fast=False for image processing."
        )
        return AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=False)


@lru_cache(maxsize=1)
def load_model_bundle():
    processor = _load_processor()
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    raw_mapping = getattr(model.config, "id2label", {}) or {}
    if raw_mapping:
        id2label = {
            int(index): _normalize_label(name)
            for index, name in raw_mapping.items()
        }
    else:
        id2label = {0: "real", 1: "fake"}

    return processor, model, device, id2label
