from typing import Any

import torch
from PIL import Image

from model import load_model_bundle


def predict_image(image: Image.Image) -> dict[str, Any]:
    processor, model, device, id2label = load_model_bundle()

    inputs = processor(images=image, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities_tensor = (
            torch.nn.functional.softmax(outputs.logits, dim=1)
            .squeeze(0)
            .cpu()
        )

    probabilities = [float(value) for value in probabilities_tensor.tolist()]
    class_names = [
        id2label.get(index, f"class_{index}")
        for index in range(len(probabilities))
    ]
    predicted_index = int(probabilities_tensor.argmax().item())

    return {
        "label": class_names[predicted_index],
        "confidence": probabilities[predicted_index],
        "probabilities": probabilities,
        "class_names": class_names,
    }
