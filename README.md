# 🛡️ DeepGuard AI — Deepfake Detection System

> AI-powered deepfake and AI-generated image detection using **EfficientNet-B4** with MTCNN face detection, built with **PyTorch** and **FastAPI**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📋 Overview

DeepGuard AI is a complete deepfake detection system that:

- **Detects deepfake faces** and AI-generated images using EfficientNet-B4
- **Processes images and videos** with automatic face detection (MTCNN)
- **Provides confidence scores** with visual indicators
- **Includes training & evaluation** pipelines with full metrics
- **Deploys as a web app** with a premium glassmorphism UI

### System Pipeline

```
Input (Image/Video)
    → Frame Extraction (if video)
    → Face Detection (MTCNN)
    → Preprocessing (resize 380×380, normalize)
    → EfficientNet-B4 Classification
    → Probability Score
    → Prediction: Real / Fake
```

---

## 📁 Project Structure

```
Deepfake-Detection/
├── config/
│   └── config.yaml              # Hyperparameters & settings
├── data/                         # Dataset directory
│   ├── train/real/ & fake/
│   ├── val/real/ & fake/
│   └── test/real/ & fake/
├── src/
│   ├── face_detector.py         # MTCNN face detection
│   ├── preprocessing.py         # Video frames, transforms
│   ├── dataset.py               # DataLoader + augmentation
│   ├── model.py                 # EfficientNet-B4 classifier
│   ├── train.py                 # Training pipeline
│   └── evaluate.py              # Evaluation & plots
├── app/
│   ├── main.py                  # FastAPI backend
│   └── static/                  # Frontend (HTML/CSS/JS)
├── models/                      # Saved checkpoints
├── outputs/                     # Evaluation plots
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your images in the data directory:

```
data/
├── train/
│   ├── real/    ← real images here
│   └── fake/    ← deepfake/AI-generated images here
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

**Supported datasets:**
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [DFDC](https://ai.facebook.com/datasets/dfdc/)
- Any GAN/Diffusion-generated image datasets

### 3. Train the Model

```bash
python -m src.train --config config/config.yaml
```

Training features:
- Adam optimizer (lr=0.0001)
- Early stopping (patience=5)
- LR scheduling (ReduceLROnPlateau)
- Best model checkpointing
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

### 4. Evaluate

```bash
python -m src.evaluate --model models/best_model.pth --data data --output outputs
```

Generates:
- Confusion Matrix
- ROC Curve
- Training vs Validation Loss/Accuracy graphs
- Classification Report

### 5. Run Web App

```bash
cd "d:\presonal projects\Deepfake-Detection"
python -m app.main
```

Open **http://localhost:8000** in your browser.

> **Note:** If no trained model is found, the app loads a pretrained EfficientNet-B4 for demo purposes (untrained classifier head — predictions will be random until you train on your data).

---

## ⚙️ Configuration

All settings are in `config/config.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `model.input_size` | 380 | Image input size (380×380) |
| `training.epochs` | 30 | Max training epochs |
| `training.batch_size` | 16 | Training batch size |
| `training.learning_rate` | 0.0001 | Adam learning rate |
| `training.early_stopping_patience` | 5 | Early stopping patience |
| `augmentation.*` | Various | Data augmentation settings |
| `video.frame_interval` | 10 | Extract every Nth frame |
| `video.max_frames` | 50 | Max frames per video |

---

## 🧠 Model Architecture

- **Backbone:** EfficientNet-B4 (pretrained on ImageNet)
- **Classification Head:** `AdaptiveAvgPool2d → Dropout(0.3) → Linear(1792, 1) → Sigmoid`
- **Task:** Binary Classification (Real vs Fake)
- **Input:** 380×380 RGB images (ImageNet normalized)

### Data Augmentation

| Transform | Details |
|-----------|---------|
| Horizontal Flip | 50% probability |
| Random Rotation | ±15° |
| Brightness | ±20% |
| Gaussian Noise | σ=0.02 |
| Gaussian Blur | kernel=3 |
| JPEG Compression | quality 70-100 |

---

## 🖥️ Hardware Requirements

| Component | Minimum |
|-----------|---------|
| GPU | NVIDIA with ≥8GB VRAM (for training) |
| RAM | 16GB |
| Storage | 100GB (with datasets) |

> Inference can run on CPU (slower).

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/health` | GET | Health check |
| `/api/predict/image` | POST | Image prediction |
| `/api/predict/video` | POST | Video prediction |

### Example Response (Image)

```json
{
    "prediction": "Fake",
    "confidence": 94.23,
    "raw_probability": 0.942312,
    "filename": "test_image.jpg",
    "type": "image"
}
```

---

## 📄 License

MIT License — feel free to use for research and educational purposes.
