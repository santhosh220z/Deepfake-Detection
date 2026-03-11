"""
Training Script
Complete training pipeline for the EfficientNet-B4 deepfake detector.
"""

import os
import sys
import time
import yaml
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import create_model, save_model
from src.dataset import create_dataloaders


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\nEarly stopping triggered after {self.patience} epochs without improvement.")
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (outputs.detach() > 0.5).float()
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = outputs.cpu().numpy().flatten()
            preds = (outputs > 0.5).float().cpu().numpy().flatten()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    return epoch_loss, metrics


def compute_metrics(labels, preds, probs=None):
    """Compute all evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
    }

    if probs is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(labels, probs)
        except ValueError:
            metrics['roc_auc'] = 0.0

    return metrics


def train(config_path='config/config.yaml'):
    """
    Main training function.

    Args:
        config_path: Path to configuration YAML file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Device setup
    device_str = config['training'].get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"\n{'='*60}")
    print(f"  Deepfake Detection - Training Pipeline")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # Create data loaders
    print(f"\nLoading datasets...")
    dataloaders = create_dataloaders(
        data_dir=config['data'].get('train_dir', 'data').replace('/train', '').replace('\\train', ''),
        input_size=config['model']['input_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        augmentation_config=config.get('augmentation', {}),
    )

    if 'train' not in dataloaders:
        print("\nERROR: No training data found!")
        print("Please place images in:")
        print(f"  data/train/real/  (real images)")
        print(f"  data/train/fake/  (fake/deepfake images)")
        return

    # Create model
    print(f"\nCreating model...")
    model = create_model(
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout'],
        device=device
    )

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-5)
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training'].get('early_stopping_patience', 5)
    )

    # Training loop
    epochs = config['training']['epochs']
    best_val_loss = float('inf')
    checkpoint_dir = Path(config['model']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [],
        'val_f1': [], 'val_roc_auc': []
    }

    print(f"\nStarting training for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)

        # Validate
        val_loss, val_metrics = 0.0, {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'roc_auc': 0}
        if 'val' in dataloaders:
            val_loss, val_metrics = validate(model, dataloaders['val'], criterion, device)

        elapsed = time.time() - start_time

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])

        # Print epoch summary
        print(f"Epoch [{epoch}/{epochs}] ({elapsed:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        if 'val' in dataloaders:
            print(f"  Val   - Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | "
                  f"P: {val_metrics['precision']:.4f} | R: {val_metrics['recall']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['roc_auc']:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(
                model, checkpoint_dir / 'best_model.pth',
                epoch=epoch, optimizer=optimizer,
                val_loss=val_loss, val_acc=val_metrics['accuracy']
            )

        # Learning rate scheduling
        if 'val' in dataloaders:
            scheduler.step(val_loss)

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.should_stop:
            break

        print()

    # Save final model
    save_model(model, checkpoint_dir / 'final_model.pth', epoch=epoch, optimizer=optimizer)

    # Save training history
    import json
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()

    train(args.config)
