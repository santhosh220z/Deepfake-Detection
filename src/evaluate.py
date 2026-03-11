"""
Evaluation Module
Generate evaluation metrics, confusion matrix, ROC curve, and training history plots.
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import load_model
from src.dataset import create_dataloaders


def evaluate_model(model, dataloader, device):
    """
    Run model on test data and collect predictions.

    Returns:
        labels, predictions, probabilities (numpy arrays)
    """
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            probs = outputs.cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().flatten())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(float)

    return all_labels, all_preds, all_probs


def plot_confusion_matrix(labels, preds, output_path):
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                ax=ax, annot_kws={'size': 16})
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Confusion matrix saved to {output_path}")


def plot_roc_curve(labels, probs, output_path):
    """Generate and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#2196F3', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curve', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ROC curve saved to {output_path}")


def plot_training_history(history_path, output_dir):
    """Generate training/validation accuracy and loss plots from saved history."""
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('Training vs Validation Loss', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = output_dir / 'training_validation_loss.png'
    fig.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Loss plot saved to {loss_path}")

    # Accuracy plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
    ax.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Training vs Validation Accuracy', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    acc_path = output_dir / 'training_validation_accuracy.png'
    fig.savefig(acc_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Accuracy plot saved to {acc_path}")

    # Metrics plot (Precision, Recall, F1, AUC)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['val_precision'], linewidth=2, label='Precision', marker='^', markersize=4)
    ax.plot(epochs, history['val_recall'], linewidth=2, label='Recall', marker='v', markersize=4)
    ax.plot(epochs, history['val_f1'], linewidth=2, label='F1 Score', marker='D', markersize=4)
    ax.plot(epochs, history['val_roc_auc'], linewidth=2, label='ROC-AUC', marker='*', markersize=6)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Validation Metrics Over Epochs', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.0, 1.05])
    plt.tight_layout()
    metrics_path = output_dir / 'validation_metrics.png'
    fig.savefig(metrics_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Metrics plot saved to {metrics_path}")


def run_evaluation(model_path='models/best_model.pth', data_dir='data',
                   output_dir='outputs', config_path='config/config.yaml'):
    """
    Run full evaluation pipeline.

    Args:
        model_path: Path to model checkpoint
        data_dir: Root data directory
        output_dir: Directory to save evaluation outputs
        config_path: Path to config file
    """
    import yaml

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Deepfake Detection - Evaluation")
    print(f"{'='*60}")
    print(f"  Device: {device}")

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path, device=device)

    # Load test data
    print(f"\nLoading test data...")
    dataloaders = create_dataloaders(
        data_dir=data_dir,
        input_size=config['model']['input_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )

    if 'test' not in dataloaders:
        print("WARNING: No test data found. Using validation data instead.")
        if 'val' not in dataloaders:
            print("ERROR: No test or validation data found!")
            return
        test_loader = dataloaders['val']
    else:
        test_loader = dataloaders['test']

    # Evaluate
    print(f"\nRunning evaluation...")
    labels, preds, probs = evaluate_model(model, test_loader, device)

    # Print metrics
    print(f"\n{'='*40}")
    print(f"  Test Results")
    print(f"{'='*40}")
    print(f"  Accuracy:  {accuracy_score(labels, preds):.4f}")
    print(f"  Precision: {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(labels, preds, zero_division=0):.4f}")
    print(f"  F1 Score:  {f1_score(labels, preds, zero_division=0):.4f}")
    try:
        print(f"  ROC-AUC:   {roc_auc_score(labels, probs):.4f}")
    except ValueError:
        print(f"  ROC-AUC:   N/A")
    print(f"{'='*40}")

    print(f"\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Real', 'Fake']))

    # Generate plots
    print(f"\nGenerating evaluation plots...")
    plot_confusion_matrix(labels, preds, output_dir / 'confusion_matrix.png')
    plot_roc_curve(labels, probs, output_dir / 'roc_curve.png')

    # Plot training history if available
    history_path = Path('models/training_history.json')
    if history_path.exists():
        print(f"\nGenerating training history plots...")
        plot_training_history(history_path, output_dir)

    print(f"\nAll evaluation outputs saved to {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Deepfake Detection Model')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='Model checkpoint path')
    parser.add_argument('--data', type=str, default='data', help='Data directory')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file path')
    args = parser.parse_args()

    run_evaluation(args.model, args.data, args.output, args.config)
