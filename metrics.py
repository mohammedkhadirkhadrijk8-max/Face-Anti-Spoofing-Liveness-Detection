"""
metrics.py  –  ISO/IEC 30107-3 Anti-Spoofing Metrics
=====================================================
Provides:
    • EER  (Equal Error Rate) and its threshold via ROC curve
    • APCER  (Attack Presentation Classification Error Rate)
    • BPCER  (Bona Fide Presentation Classification Error Rate)
    • HTER   (Half Total Error Rate)
    • Confusion matrix (TP, TN, FP, FN)

All functions expect:
    y_true   – ground-truth labels (0 = spoof/attack, 1 = real/bonafide)
    y_scores – model confidence scores for the *real/bonafide* class (higher → more real)
"""

import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix


# ---------------------------------------------------------------------------
#  EER – Equal Error Rate
# ---------------------------------------------------------------------------

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray):
    """
    Compute the Equal Error Rate and the corresponding threshold.

    Returns
    -------
    eer       : float   (in [0, 1])
    threshold : float   (score value where FAR ≈ FRR)
    fpr       : ndarray (false positive rates along ROC)
    tpr       : ndarray (true positive rates along ROC)
    thresholds: ndarray (thresholds along ROC)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr  # false negative rate

    # Find the point where |FPR − FNR| is minimised
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    return eer, threshold, fpr, tpr, thresholds


# ---------------------------------------------------------------------------
#  APCER / BPCER / HTER  (ISO/IEC 30107-3)
# ---------------------------------------------------------------------------

def compute_apcer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Attack Presentation Classification Error Rate.

    APCER = (# attack samples classified as bonafide) / (# total attack samples)

    Parameters
    ----------
    y_true : array-like  –  ground-truth labels (0 = attack, 1 = bonafide)
    y_pred : array-like  –  predicted labels after thresholding
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    attack_mask = (y_true == 0)
    if attack_mask.sum() == 0:
        return 0.0
    # Attacks misclassified as bonafide (false accepts)
    false_accepts = ((y_pred == 1) & attack_mask).sum()
    return float(false_accepts) / float(attack_mask.sum())


def compute_bpcer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Bona Fide Presentation Classification Error Rate.

    BPCER = (# bonafide samples classified as attack) / (# total bonafide samples)

    Parameters
    ----------
    y_true : array-like  –  ground-truth labels (0 = attack, 1 = bonafide)
    y_pred : array-like  –  predicted labels after thresholding
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    bonafide_mask = (y_true == 1)
    if bonafide_mask.sum() == 0:
        return 0.0
    # Bonafides misclassified as attack (false rejects)
    false_rejects = ((y_pred == 0) & bonafide_mask).sum()
    return float(false_rejects) / float(bonafide_mask.sum())


def compute_hter(apcer: float, bpcer: float) -> float:
    """Half Total Error Rate  =  (APCER + BPCER) / 2."""
    return (apcer + bpcer) / 2.0


# ---------------------------------------------------------------------------
#  Confusion matrix helper
# ---------------------------------------------------------------------------

def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Returns (tn, fp, fn, tp) from sklearn's confusion matrix.

    Labels convention: 0 = spoof, 1 = real.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp


# ---------------------------------------------------------------------------
#  All-in-one evaluation
# ---------------------------------------------------------------------------

def full_evaluation(y_true: np.ndarray, y_scores: np.ndarray):
    """
    Run the complete ISO metric suite.

    Parameters
    ----------
    y_true   : (N,) ground-truth labels (0 = attack, 1 = bonafide)
    y_scores : (N,) model scores for the bonafide class

    Returns
    -------
    results : dict with keys eer, eer_threshold, apcer, bpcer, hter, tn, fp, fn, tp,
              fpr, tpr, thresholds
    """
    y_true   = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    eer, eer_thresh, fpr, tpr, thresholds = compute_eer(y_true, y_scores)

    # Binary predictions at EER threshold
    y_pred = (y_scores >= eer_thresh).astype(int)

    apcer = compute_apcer(y_true, y_pred)
    bpcer = compute_bpcer(y_true, y_pred)
    hter  = compute_hter(apcer, bpcer)

    tn, fp, fn, tp = compute_confusion(y_true, y_pred)

    return {
        "eer":           eer,
        "eer_threshold": eer_thresh,
        "apcer":         apcer,
        "bpcer":         bpcer,
        "hter":          hter,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "fpr": fpr, "tpr": tpr, "thresholds": thresholds,
    }


# ---------------------------------------------------------------------------
#  Pretty printer
# ---------------------------------------------------------------------------

def print_metrics(results: dict) -> None:
    """Print a formatted metrics report to stdout."""
    print("\n" + "=" * 60)
    print("     ISO/IEC 30107-3  Anti-Spoofing Evaluation Report")
    print("=" * 60)
    print(f"  EER            : {results['eer'] * 100:6.2f} %")
    print(f"  EER Threshold  : {results['eer_threshold']:.4f}")
    print(f"  APCER          : {results['apcer'] * 100:6.2f} %")
    print(f"  BPCER          : {results['bpcer'] * 100:6.2f} %")
    print(f"  HTER           : {results['hter'] * 100:6.2f} %")
    print("-" * 60)
    print(f"  Confusion Matrix (threshold = {results['eer_threshold']:.4f})")
    print(f"    TP (Real→Real)    : {results['tp']}")
    print(f"    TN (Spoof→Spoof) : {results['tn']}")
    print(f"    FP (Spoof→Real)  : {results['fp']}")
    print(f"    FN (Real→Spoof)  : {results['fn']}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
#  Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.RandomState(0)
    # Synthetic scores: real samples get higher scores, spoof lower
    y_true   = np.array([1]*50 + [0]*50)
    y_scores = np.concatenate([rng.uniform(0.5, 1.0, 50),
                                rng.uniform(0.0, 0.5, 50)])
    results = full_evaluation(y_true, y_scores)
    print_metrics(results)
