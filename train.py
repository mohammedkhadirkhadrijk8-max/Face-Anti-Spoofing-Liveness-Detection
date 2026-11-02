"""
train.py  –  Train the Face Anti-Spoofing Model
================================================
• 15 epochs (configurable).
• Logs per-epoch training loss, training accuracy, and test accuracy.
• Evaluates with ISO/IEC 30107-3 metrics via metrics.py.
• Saves the best model weights to  best_antispoof_model.pth.
• Saves a combined training-curves + ROC plot to  results.png.
"""

import os
import sys
import ssl
import time
import numpy as np
import matplotlib

# ── SSL workaround for macOS (missing Install Certificates.command) ──
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
matplotlib.use("Agg")           # headless backend – no display needed
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Local modules
from dataset import build_dataloaders
from model   import FaceAntiSpoofNet
from metrics import full_evaluation, print_metrics


# ──────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────
NUM_EPOCHS   = 15
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "best_antispoof_model.pth")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results.png")


# ──────────────────────────────────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        rgb   = batch["rgb"].to(device)
        ycrcb = batch["ycrcb"].to(device)
        lbp   = batch["lbp"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(rgb, ycrcb, lbp)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


# ──────────────────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    """
    Returns
    -------
    all_labels : list[int]
    all_scores : list[float]   – softmax probability for class 1 (bonafide)
    accuracy   : float
    """
    model.eval()
    all_labels, all_scores = [], []
    correct, total = 0, 0

    for batch in loader:
        rgb   = batch["rgb"].to(device)
        ycrcb = batch["ycrcb"].to(device)
        lbp   = batch["lbp"].to(device)
        labels = batch["label"].to(device)

        logits = model(rgb, ycrcb, lbp)
        probs  = torch.softmax(logits, dim=1)[:, 1]  # P(bonafide)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_scores.extend(probs.cpu().numpy().tolist())

    accuracy = correct / total
    return all_labels, all_scores, accuracy


# ──────────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────────

def save_results_plot(train_losses, train_accs, test_accs, roc_fpr, roc_tpr,
                      eer, save_path):
    """Save a 2×2 grid: loss curve, accuracy curves, ROC curve, EER bar."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Face Anti-Spoofing – Training Results", fontsize=16,
                 fontweight="bold")
    epochs = range(1, len(train_losses) + 1)

    # ── Loss ──
    ax = axes[0, 0]
    ax.plot(epochs, train_losses, "o-", color="#e74c3c", linewidth=2,
            markersize=5, label="Train Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    # ── Accuracy ──
    ax = axes[0, 1]
    ax.plot(epochs, train_accs, "o-", color="#2ecc71", linewidth=2,
            markersize=5, label="Train Acc")
    ax.plot(epochs, test_accs, "s-", color="#3498db", linewidth=2,
            markersize=5, label="Test Acc")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curves"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # ── ROC Curve ──
    ax = axes[1, 0]
    ax.plot(roc_fpr, roc_tpr, color="#9b59b6", linewidth=2.5,
            label=f"ROC (EER = {eer*100:.2f}%)")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)

    # ── Metric summary text ──
    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        f"Equal Error Rate (EER):   {eer * 100:.2f}%\n"
        f"Final Train Accuracy:     {train_accs[-1] * 100:.2f}%\n"
        f"Final Test Accuracy:      {test_accs[-1] * 100:.2f}%\n"
        f"Final Train Loss:         {train_losses[-1]:.4f}\n"
        f"Epochs:                   {len(train_losses)}\n"
        f"Device:                   {DEVICE}"
    )
    ax.text(0.1, 0.5, summary, transform=ax.transAxes, fontsize=13,
            verticalalignment="center", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#ecf0f1",
                      edgecolor="#bdc3c7"))
    ax.set_title("Summary")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n[train] Results plot saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  Face Anti-Spoofing Training Pipeline")
    print(f"{'='*60}")
    print(f"  Device       : {DEVICE}")
    print(f"  Epochs       : {NUM_EPOCHS}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Learning rate: {LR}")
    print(f"{'='*60}\n")

    # ── Data ──
    train_loader, test_loader = build_dataloaders(batch_size=BATCH_SIZE)

    # ── Model ──
    model = FaceAntiSpoofNet(backbone="mobilenet_v3_small", pretrained=True).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Parameters: {total_params:,}  (trainable: {trainable:,})\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # ── Training ──
    train_losses, train_accs, test_accs = [], [], []
    best_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, DEVICE)
        _, _, test_acc = evaluate(model, test_loader, DEVICE)
        scheduler.step()

        train_losses.append(loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
              f"Loss: {loss:.4f}  "
              f"Train Acc: {train_acc*100:.2f}%  "
              f"Test Acc: {test_acc*100:.2f}%  "
              f"LR: {lr_now:.6f}  "
              f"({elapsed:.1f}s)")

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ↳ Best model saved → {SAVE_PATH}")

    # ── Final evaluation on test set with ISO metrics ──
    print("\n" + "─" * 60)
    print("  Running final ISO/IEC 30107-3 evaluation on test set …")
    print("─" * 60)

    # Load best weights for final eval
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))
    y_true, y_scores, final_acc = evaluate(model, test_loader, DEVICE)

    results = full_evaluation(np.array(y_true), np.array(y_scores))
    print_metrics(results)

    # ── Save plots ──
    save_results_plot(train_losses, train_accs, test_accs,
                      results["fpr"], results["tpr"], results["eer"],
                      RESULTS_PATH)

    print(f"\n✅  Training complete.  Best test accuracy: {best_acc*100:.2f}%")
    print(f"    Model weights : {SAVE_PATH}")
    print(f"    Results plot   : {RESULTS_PATH}\n")


if __name__ == "__main__":
    main()
