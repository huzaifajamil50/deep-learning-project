"""
test.py
-------
Evaluation script for the DCASE 2023 Task 2 ASD models.

For each machine type:
  1. Load the trained model checkpoint.
  2. Extract features from test audio files.
  3. Compute anomaly scores (MSE or Mahalanobis).
  4. Aggregate frame-level scores to file-level scores.
  5. Separate source-domain and target-domain test files.
  6. Compute AUC, pAUC, precision, recall, F1 for each domain.
  7. Save CSV results and generate plots.

Usage:
    python -m src.test --config configs/config.yaml --model baseline
    python -m src.test --config configs/config.yaml --model improved
"""

import os
import argparse
import yaml
import numpy as np
import torch

from .models.baseline_ae import BaselineAutoEncoder
from .models.improved_ae import ImprovedAutoEncoder
from .feature_extraction import extract_features_from_file
from .dataset import (
    get_machine_type_files, separate_by_domain, get_labels
)
from .utils import (
    compute_all_metrics, compute_mahalanobis_scores,
    plot_roc_curve, plot_confusion_matrix,
    plot_score_distribution, save_results_csv,
    format_results_table
)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(config, model_type, machine_type, device):
    """Load a trained model from checkpoint."""
    if model_type == "baseline":
        cfg = config["baseline_ae"]
        model = BaselineAutoEncoder(
            input_dim=cfg["input_dim"],
            hidden_dims=cfg["hidden_dims"],
            latent_dim=cfg["latent_dim"],
            use_batch_norm=cfg["use_batch_norm"],
            dropout=cfg["dropout"]
        )
    else:
        cfg = config["improved_ae"]
        model = ImprovedAutoEncoder(
            input_dim=cfg["input_dim"],
            hidden_dims=cfg["hidden_dims"],
            latent_dim=cfg["latent_dim"],
            use_batch_norm=cfg["use_batch_norm"],
            dropout=cfg["dropout"],
            use_skip_connections=cfg["use_skip_connections"],
            use_attention=cfg["use_attention"],
            attention_reduction=cfg["attention_reduction"]
        )

    ckpt_path = os.path.join(
        config["paths"]["model_dir"],
        f"{model_type}_{machine_type}_best.pth"
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device)
    )
    model = model.to(device)
    model.eval()
    return model


def compute_file_scores(model, file_list, config, device,
                         score_mode="mse", mahal_mean=None,
                         mahal_inv_cov=None):
    """
    Compute an anomaly score for each file.

    For each file:
      1. Extract frame-level features.
      2. Compute frame-level anomaly scores.
      3. Average to get a single file-level score.

    Parameters
    ----------
    model : nn.Module
    file_list : list of str
    config : dict
    device : torch.device
    score_mode : "mse" or "mahalanobis"
    mahal_mean, mahal_inv_cov : np.ndarray or None
        Required when score_mode == "mahalanobis".

    Returns
    -------
    file_scores : np.ndarray, shape (len(file_list),)
    """
    audio_cfg = config["audio"]
    file_scores = []

    for fpath in file_list:
        vectors = extract_features_from_file(
            fpath,
            sr=audio_cfg["sr"],
            n_fft=audio_cfg["n_fft"],
            hop_length=audio_cfg["hop_length"],
            n_mels=audio_cfg["n_mels"],
            n_frames=audio_cfg["frames"],
            duration=audio_cfg["duration"]
        )
        x = torch.tensor(vectors, dtype=torch.float32).to(device)

        with torch.no_grad():
            if score_mode == "mse":
                scores = model.compute_anomaly_score(x)
                file_score = scores.mean().item()
            elif score_mode == "mahalanobis":
                z = model.get_latent(x).cpu().numpy()
                dists = compute_mahalanobis_scores(
                    z, mahal_mean, mahal_inv_cov
                )
                file_score = np.mean(dists)
            else:
                raise ValueError(f"Unknown score_mode: {score_mode}")

        file_scores.append(file_score)

    return np.array(file_scores)


def evaluate_machine_type(model, config, machine_type, model_type,
                           device, score_mode="mse"):
    """
    Full evaluation pipeline for one machine type.

    Returns
    -------
    results : dict with source and target metrics
    """
    data_root = config["dataset"]["dcase2023_root"]

    # get test files
    test_files = get_machine_type_files(data_root, machine_type, split="test")
    if not test_files:
        print(f"  [WARN] No test files for {machine_type}")
        return {}

    source_files, target_files = separate_by_domain(test_files)

    # load Mahalanobis params if needed
    mahal_mean, mahal_inv_cov = None, None
    if score_mode == "mahalanobis":
        save_dir = config["paths"]["model_dir"]
        mahal_mean = np.load(
            os.path.join(save_dir,
                         f"{model_type}_{machine_type}_mahal_mean.npy")
        )
        mahal_inv_cov = np.load(
            os.path.join(save_dir,
                         f"{model_type}_{machine_type}_mahal_invcov.npy")
        )

    results = {}
    fig_dir = config["paths"]["figure_dir"]
    os.makedirs(fig_dir, exist_ok=True)

    # evaluate source domain
    if source_files:
        source_labels = get_labels(source_files)
        source_scores = compute_file_scores(
            model, source_files, config, device,
            score_mode, mahal_mean, mahal_inv_cov
        )
        src_metrics = compute_all_metrics(source_labels, source_scores)
        results["AUC_source"] = src_metrics["AUC"]
        results["pAUC_source"] = src_metrics["pAUC"]
        results["Precision_source"] = src_metrics["Precision"]
        results["Recall_source"] = src_metrics["Recall"]
        results["F1_source"] = src_metrics["F1"]

        # ROC curve
        plot_roc_curve(
            source_labels, source_scores,
            save_path=os.path.join(
                fig_dir,
                f"roc_{model_type}_{machine_type}_source.png"
            ),
            title=f"ROC — {model_type} / {machine_type} (source)"
        )

    # evaluate target domain
    if target_files:
        target_labels = get_labels(target_files)
        target_scores = compute_file_scores(
            model, target_files, config, device,
            score_mode, mahal_mean, mahal_inv_cov
        )
        tgt_metrics = compute_all_metrics(target_labels, target_scores)
        results["AUC_target"] = tgt_metrics["AUC"]
        results["pAUC_target"] = tgt_metrics["pAUC"]
        results["Precision_target"] = tgt_metrics["Precision"]
        results["Recall_target"] = tgt_metrics["Recall"]
        results["F1_target"] = tgt_metrics["F1"]

        # ROC curve
        plot_roc_curve(
            target_labels, target_scores,
            save_path=os.path.join(
                fig_dir,
                f"roc_{model_type}_{machine_type}_target.png"
            ),
            title=f"ROC — {model_type} / {machine_type} (target)"
        )

    # combined pAUC (harmonic mean of source and target)
    if "pAUC_source" in results and "pAUC_target" in results:
        ps = results["pAUC_source"]
        pt = results["pAUC_target"]
        if ps + pt > 0:
            results["pAUC"] = round(2 * ps * pt / (ps + pt), 2)
        else:
            results["pAUC"] = 0.0

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ASD models on DCASE 2023 Task 2"
    )
    parser.add_argument("--config", type=str,
                        default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "improved"])
    parser.add_argument("--score", type=str, default="mse",
                        choices=["mse", "mahalanobis"])
    parser.add_argument("--machine", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    machine_types = config["dataset"]["dcase2023_machine_types"]
    if args.machine:
        machine_types = [args.machine]

    all_results = {}

    for mtype in machine_types:
        print(f"\nEvaluating {args.model} on {mtype} ...")
        try:
            model = load_model(config, args.model, mtype, device)
        except FileNotFoundError:
            print(f"  [WARN] No checkpoint for {mtype}, skipping.")
            continue

        results = evaluate_machine_type(
            model, config, mtype, args.model,
            device, score_mode=args.score
        )
        all_results[mtype] = results

    # print summary table
    print("\n" + "=" * 70)
    print(f"  Results Summary — {args.model} AE ({args.score} scoring)")
    print("=" * 70)
    print(format_results_table(all_results))

    # save CSV
    result_dir = config["paths"]["result_dir"]
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(
        result_dir, f"results_{args.model}_{args.score}.csv"
    )
    save_results_csv(all_results, csv_path)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
