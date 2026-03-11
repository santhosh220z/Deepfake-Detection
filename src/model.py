"""
Model Module
EfficientNet-B4 based deepfake detection classifier.
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path


class DeepfakeDetector(nn.Module):
    """
    EfficientNet-B4 based binary classifier for deepfake detection.
    
    Architecture:
        - EfficientNet-B4 backbone (pretrained on ImageNet)
        - Custom classification head: AdaptiveAvgPool → Dropout → Linear → Sigmoid
    """

    def __init__(self, pretrained=True, dropout=0.3):
        """
        Args:
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate before final classification layer
        """
        super().__init__()

        # Load EfficientNet-B4 backbone
        self.backbone = timm.create_model(
            'efficientnet_b4',
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
            global_pool=''  # Remove global pooling (we add our own)
        )

        # Get feature dimensions
        # EfficientNet-B4 outputs 1792 channels
        self.num_features = 1792

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 380, 380)

        Returns:
            Tensor of shape (batch, 1) with probability scores (0=real, 1=fake)
        """
        features = self.backbone(x)
        output = self.classifier(features)
        return output

    def get_param_count(self):
        """Get total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def create_model(pretrained=True, dropout=0.3, device=None):
    """
    Create and return a DeepfakeDetector model.

    Args:
        pretrained: Use pretrained weights
        dropout: Dropout rate
        device: Target device

    Returns:
        DeepfakeDetector model on the specified device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DeepfakeDetector(pretrained=pretrained, dropout=dropout)
    model = model.to(device)

    total, trainable = model.get_param_count()
    print(f"Model: EfficientNet-B4 Deepfake Detector")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Device: {device}")

    return model


def save_model(model, path, epoch=None, optimizer=None, val_loss=None, val_acc=None):
    """
    Save model checkpoint.

    Args:
        model: DeepfakeDetector model
        path: Save path
        epoch: Current epoch number
        optimizer: Optimizer state
        val_loss: Validation loss
        val_acc: Validation accuracy
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_features': model.num_features,
        }
    }

    if epoch is not None:
        checkpoint['epoch'] = epoch
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if val_loss is not None:
        checkpoint['val_loss'] = val_loss
    if val_acc is not None:
        checkpoint['val_acc'] = val_acc

    torch.save(checkpoint, str(path))
    print(f"Model saved to {path}")


def load_model(path, device=None, pretrained=False):
    """
    Load model from checkpoint.

    Args:
        path: Checkpoint path
        device: Target device
        pretrained: Whether to load pretrained backbone (usually False when loading checkpoint)

    Returns:
        DeepfakeDetector model with loaded weights
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DeepfakeDetector(pretrained=pretrained)
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    if 'val_acc' in checkpoint:
        print(f"  Val Acc: {checkpoint['val_acc']:.4f}")

    return model
