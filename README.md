# Deepfake Detection System (Image + Video)

A web application that supports:

- Image deepfake detection
- Video deepfake detection using GenConViT from the `video_model/GenConViT` folder

Backend: FastAPI + PyTorch + Transformers
Frontend: HTML/CSS/JavaScript

## Features

- Unified UI with Image and Video modes
- Drag-and-drop or click-to-upload workflow
- Image prediction with confidence and class probabilities
- Video prediction with confidence, class probabilities, and sampled frame count
- Input validation and API error handling
- Responsive preview panel for images and videos

## How Video Detection Works

`/predict-video` loads the GenConViT ED model from `video_model/GenConViT`, samples frames from the uploaded video, applies face-focused preprocessing, and predicts a final real/fake score from the video model outputs.

## Video Model Setup

The project expects this repository structure:

```text
video_model/
`-- GenConViT/
```

If missing, clone it:

```bash
git clone https://github.com/erprogs/GenConViT.git video_model/GenConViT
```

For model weights, either:

1. Place `genconvit_ed_inference.pth` in `video_model/GenConViT/weight`, or
2. Let the app auto-download from Hugging Face on first video prediction (enabled by default).

## Project Structure

```text
.
|-- app.py
|-- model.py
|-- predict.py
|-- video_predict.py
|-- utils.py
|-- requirements.txt
|-- render.yaml
`-- static/
    |-- index.html
    |-- style.css
    `-- script.js
```

## Setup

1. Create and activate virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

4. Open:

- http://127.0.0.1:8000

## API

### `POST /predict`

Image prediction endpoint.

- Content-Type: `multipart/form-data`
- Field name: `file`
- Allowed formats: `jpg`, `jpeg`, `png`

Example response:

```json
{
  "label": "real",
  "confidence": 0.91,
  "probabilities": [0.91, 0.09],
  "class_names": ["real", "fake"]
}
```

### `POST /predict-video`

Video prediction endpoint (GenConViT ED inference from `video_model/GenConViT`).

- Content-Type: `multipart/form-data`
- Field name: `file`
- Allowed formats: `mp4`, `avi`, `mov`, `mpeg`, `mpg`, `webm`, `m4v`

Example response:

```json
{
  "label": "fake",
  "confidence": 0.87,
  "probabilities": [0.13, 0.87],
  "class_names": ["real", "fake"],
  "sampled_frames": 16
}
```

## Notes

- The image model performs best on face-centric visuals.
- Video predictions rely on GenConViT ED weights being available.
- `timm==0.6.5` is required for compatibility with provided GenConViT inference weights.
- You can tune frame count in `app.py` / `video_predict.py` for speed vs stability.
