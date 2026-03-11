"""
Dataset Module
PyTorch Dataset and DataLoader for deepfake detection training.
Includes data augmentation with domain-specific transforms.
"""

import os
import random
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from io import BytesIO

from src.preprocessing import IMAGENET_MEAN, IMAGENET_STD


class JPEGCompression:
    """Custom transform: apply random JPEG compression artifacts."""

    def __init__(self, quality_range=(70, 100)):
        self.quality_range = quality_range

    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')


class GaussianNoise:
    """Custom transform: add Gaussian noise to a tensor."""

    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


def get_train_transforms(input_size=380, augmentation_config=None):
    """
    Get training transforms with data augmentation.

    Args:
        input_size: Target image size
        augmentation_config: Dict of augmentation settings

    Returns:
        torchvision.transforms.Compose
    """
    aug = augmentation_config or {}

    transform_list = []

    # Pre-tensor transforms (on PIL Image)
    transform_list.append(transforms.Resize((input_size, input_size)))

    # JPEG compression (applied before other transforms)
    if aug.get('jpeg_compression_quality'):
        transform_list.append(JPEGCompression(quality_range=tuple(aug['jpeg_compression_quality'])))

    if aug.get('horizontal_flip', True):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    rotation_deg = aug.get('rotation_degrees', 15)
    if rotation_deg > 0:
        transform_list.append(transforms.RandomRotation(rotation_deg))

    brightness = aug.get('brightness_range', [0.8, 1.2])
    if brightness:
        brightness_factor = brightness[1] - 1.0
        transform_list.append(transforms.ColorJitter(brightness=brightness_factor))

    if aug.get('blur_kernel', 0) > 0:
        transform_list.append(transforms.GaussianBlur(kernel_size=aug['blur_kernel']))

    # Convert to tensor
    transform_list.append(transforms.ToTensor())

    # Gaussian noise (applied after ToTensor)
    noise_std = aug.get('gaussian_noise_std', 0.0)
    if noise_std > 0:
        transform_list.append(GaussianNoise(std=noise_std))

    # Normalize
    transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return transforms.Compose(transform_list)


def get_val_transforms(input_size=380):
    """
    Get validation/test transforms (no augmentation).

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


class DeepfakeDataset(Dataset):
    """
    Dataset for loading real/fake images for deepfake detection.
    
    Expected directory structure:
        root_dir/
            real/
                img1.jpg
                img2.jpg
                ...
            fake/
                img1.jpg
                img2.jpg
                ...
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(self, root_dir, transform=None, face_detector=None):
        """
        Args:
            root_dir: Root directory containing 'real' and 'fake' subdirectories
            transform: torchvision transforms to apply
            face_detector: Optional FaceDetector for face cropping
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.face_detector = face_detector
        self.samples = []  # List of (path, label) tuples

        # Load real images (label = 0)
        real_dir = self.root_dir / 'real'
        if real_dir.exists():
            for img_path in sorted(real_dir.iterdir()):
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((str(img_path), 0))

        # Load fake images (label = 1)
        fake_dir = self.root_dir / 'fake'
        if fake_dir.exists():
            for img_path in sorted(fake_dir.iterdir()):
                if img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.samples.append((str(img_path), 1))

        if len(self.samples) == 0:
            print(f"Warning: No images found in {root_dir}. "
                  f"Expected 'real/' and 'fake/' subdirectories with images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image on error
            image = Image.new('RGB', (380, 380), (0, 0, 0))

        # Optional face detection
        if self.face_detector is not None:
            image = self.face_detector.extract_largest_face(image)

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


def create_dataloaders(data_dir, input_size=380, batch_size=16, num_workers=4,
                       augmentation_config=None, face_detector=None):
    """
    Create train, validation, and test DataLoaders.

    Args:
        data_dir: Root data directory containing train/val/test subdirectories
        input_size: Image size
        batch_size: Batch size
        num_workers: DataLoader workers
        augmentation_config: Augmentation settings dict
        face_detector: Optional FaceDetector

    Returns:
        dict with 'train', 'val', 'test' DataLoaders (keys present only if data exists)
    """
    data_dir = Path(data_dir)
    dataloaders = {}

    splits = {
        'train': get_train_transforms(input_size, augmentation_config),
        'val': get_val_transforms(input_size),
        'test': get_val_transforms(input_size)
    }

    for split, transform in splits.items():
        split_dir = data_dir / split
        if split_dir.exists() and any(split_dir.iterdir()):
            dataset = DeepfakeDataset(
                root_dir=split_dir,
                transform=transform,
                face_detector=face_detector
            )
            if len(dataset) > 0:
                dataloaders[split] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=(split == 'train'),
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=(split == 'train')
                )
                print(f"  {split}: {len(dataset)} images")

    return dataloaders
