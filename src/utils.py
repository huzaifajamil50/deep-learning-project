"""
utils.py
--------
Utility functions for evaluation metrics, result visualization,
and experiment management.

Provides:
  - AUC / pAUC computation
  - Mahalanobis distance anomaly scoring
  - Confusion matrix plotting
  - Training curve plotting
  - ROC curve plotting
  - Result table generation
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
from scipy.spatial.distance import mahalanobis


# ------------------------------------------------------------------
#  Metrics
# ------------------------------------------------------------------

def compute_auc(labels, scores):
    """
    Compute the area under the ROC curve.

    Parameters
    ----------
    labels : array-like of {0, 1}
    scores : array-like of float

    Returns
    -------
    auc : float
    """
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, scores)


def compute_pauc(labels, scores, max_fpr=0.1):
    """
    Compute the partial AUC (pAUC) up to a maximum false-positive rate.

    The DCASE 2023 Task 2 evaluation uses max_fpr = 0.1.

    Parameters
    ----------
    labels : array-like of {0, 1}
    scores : array-like of float
    max_fpr : float
        Upper limit of the FPR integration range.

    Returns
    -------
    pauc : float
        Normalized so that chance-level = 0.5.
    """
    if len(np.unique(labels)) < 2:
        return 0.5
    pauc = roc_auc_score(labels, scores, max_fpr=max_fpr)
    return pauc


def compute_all_metrics(labels, scores, threshold=None):
    """
    Compute AUC, pAUC, precision, recall, and F1-score.

    If threshold is None, the optimal threshold (Youden's index) is used.

    Returns
    -------
    metrics : dict
    """
    auc = compute_auc(labels, scores)
    pauc = compute_pauc(labels, scores)

    # determine threshold
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        idx = np.argmax(tpr - fpr)
        threshold = thresholds[idx]

    preds = (np.array(scores) >= threshold).astype(int)

    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    return {
        "AUC": round(auc * 100, 2),
        "pAUC": round(pauc * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall": round(rec * 100, 2),
        "F1": round(f1 * 100, 2),
        "Threshold": round(threshold, 6)
    }


# ------------------------------------------------------------------
#  Mahalanobis Distance Scoring
# ------------------------------------------------------------------

def fit_mahalanobis(latent_vectors):
    """
    Fit a Gaussian to the training latent vectors and return
    the mean and inverse covariance matrix for Mahalanobis scoring.

    Parameters
    ----------
    latent_vectors : np.ndarray, shape (N, latent_dim)

    Returns
    -------
    mean : np.ndarray, shape (latent_dim,)
    inv_cov : np.ndarray, shape (latent_dim, latent_dim)
    """
    mean = np.mean(latent_vectors, axis=0)
    cov = np.cov(latent_vectors.T)
    # regularize to ensure invertibility
    cov += np.eye(cov.shape[0]) * 1e-6
    inv_cov = np.linalg.inv(cov)
    return mean, inv_cov


def compute_mahalanobis_scores(latent_vectors, mean, inv_cov):
    """
    Compute the Mahalanobis distance for each sample.

    Parameters
    ----------
    latent_vectors : np.ndarray, shape (N, latent_dim)
    mean : np.ndarray
    inv_cov : np.ndarray

    Returns
    -------
    scores : np.ndarray, shape (N,)
    """
    scores = []
    for vec in latent_vectors:
        dist = mahalanobis(vec, mean, inv_cov)
        scores.append(dist)
    return np.array(scores)


# ------------------------------------------------------------------
#  Visualization
# ------------------------------------------------------------------

def plot_training_curves(train_losses, val_losses=None,
                         save_path=None, title="Training Loss Curve"):
    """
    Plot training (and optionally validation) loss over epochs.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, "b-", linewidth=1.5, label="Train Loss")
    if val_losses is not None:
        ax.plot(epochs, val_losses, "r--", linewidth=1.5, label="Val Loss")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_roc_curve(labels, scores, save_path=None,
                   title="ROC Curve"):
    """Plot the ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(labels, scores)
    auc_val = compute_auc(labels, scores)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, "b-", linewidth=2,
            label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_confusion_matrix(labels, preds, save_path=None,
                           title="Confusion Matrix"):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_score_distribution(normal_scores, anomaly_scores,
                             save_path=None,
                             title="Anomaly Score Distribution"):
    """
    Plot histograms of anomaly scores for normal and anomalous samples.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(normal_scores, bins=50, alpha=0.6, color="green",
            label="Normal", density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.6, color="red",
            label="Anomaly", density=True)

    ax.set_xlabel("Anomaly Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ------------------------------------------------------------------
#  Result Tables
# ------------------------------------------------------------------

def format_results_table(results_dict):
    """
    Format a dictionary of per-machine results into a printable table.

    Parameters
    ----------
    results_dict : dict
        {machine_type: {metric_name: value, ...}, ...}

    Returns
    -------
    table_str : str
    """
    header = f"{'Machine Type':<15} {'AUC(src)':>10} {'AUC(tgt)':>10} "
    header += f"{'pAUC':>8} {'F1(src)':>8} {'F1(tgt)':>8}"
    lines = [header, "-" * len(header)]

    for mtype, metrics in results_dict.items():
        line = f"{mtype:<15} "
        line += f"{metrics.get('AUC_source', 0):>10.2f} "
        line += f"{metrics.get('AUC_target', 0):>10.2f} "
        line += f"{metrics.get('pAUC', 0):>8.2f} "
        line += f"{metrics.get('F1_source', 0):>8.2f} "
        line += f"{metrics.get('F1_target', 0):>8.2f}"
        lines.append(line)

    return "\n".join(lines)


def save_results_csv(results_dict, save_path):
    """Save results dictionary to a CSV file."""
    import csv
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Machine Type", "AUC (source)", "AUC (target)",
            "pAUC", "Precision (source)", "Precision (target)",
            "Recall (source)", "Recall (target)",
            "F1 (source)", "F1 (target)"
        ])
        for mtype, m in results_dict.items():
            writer.writerow([
                mtype,
                m.get("AUC_source", ""),
                m.get("AUC_target", ""),
                m.get("pAUC", ""),
                m.get("Precision_source", ""),
                m.get("Precision_target", ""),
                m.get("Recall_source", ""),
                m.get("Recall_target", ""),
                m.get("F1_source", ""),
                m.get("F1_target", ""),
            ])
