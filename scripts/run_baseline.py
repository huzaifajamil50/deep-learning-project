"""
run_baseline.py
---------------
Convenience script to run the complete baseline experiment pipeline:
  1. Feature extraction
  2. Training baseline AE for all machine types
  3. Evaluation with MSE scoring
  4. Evaluation with Mahalanobis scoring
  5. Result summary

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --machine fan
"""

import os
import sys
import argparse

# add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import load_config, set_seed, build_model, train_model
from src.train import fit_mahalanobis_params
from src.test import evaluate_machine_type, load_model
from src.feature_extraction import extract_features_from_directory
from src.utils import format_results_table, save_results_csv

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--machine", default=None,
                        help="Specific machine type (default: all)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    machine_types = config["dataset"]["dcase2023_machine_types"]
    if args.machine:
        machine_types = [args.machine]

    audio_cfg = config["audio"]
    data_root = config["dataset"]["dcase2023_root"]

    print("=" * 60)
    print("  DCASE 2023 Task 2 — Baseline Autoencoder Experiment")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Machine types: {machine_types}")

    # --- Phase 1: Train ---
    for mtype in machine_types:
        train_dir = os.path.join(data_root, mtype, "train")
        if not os.path.isdir(train_dir):
            print(f"\n  [SKIP] {mtype}: train dir not found at {train_dir}")
            continue

        print(f"\n--- Training baseline AE on {mtype} ---")
        feats, _, _ = extract_features_from_directory(
            train_dir, sr=audio_cfg["sr"], n_fft=audio_cfg["n_fft"],
            hop_length=audio_cfg["hop_length"], n_mels=audio_cfg["n_mels"],
            n_frames=audio_cfg["frames"], duration=audio_cfg["duration"]
        )
        model = build_model(config, model_type="baseline")
        model, losses = train_model(
            config, model, feats, model_type="baseline",
            machine_type=mtype, device=device
        )
        mean, inv_cov = fit_mahalanobis_params(model, feats, device)
        save_dir = config["paths"]["model_dir"]
        np.save(os.path.join(save_dir, f"baseline_{mtype}_mahal_mean.npy"), mean)
        np.save(os.path.join(save_dir, f"baseline_{mtype}_mahal_invcov.npy"), inv_cov)

    # --- Phase 2: Evaluate ---
    all_results_mse = {}
    all_results_mahal = {}

    for mtype in machine_types:
        try:
            model = load_model(config, "baseline", mtype, device)
        except Exception as e:
            print(f"  [SKIP] {mtype}: {e}")
            continue

        # MSE scoring
        res_mse = evaluate_machine_type(
            model, config, mtype, "baseline", device, score_mode="mse"
        )
        all_results_mse[mtype] = res_mse

        # Mahalanobis scoring
        res_mahal = evaluate_machine_type(
            model, config, mtype, "baseline", device, score_mode="mahalanobis"
        )
        all_results_mahal[mtype] = res_mahal

    # --- Phase 3: Summary ---
    result_dir = config["paths"]["result_dir"]
    os.makedirs(result_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Results — Baseline AE (MSE Scoring)")
    print("=" * 60)
    print(format_results_table(all_results_mse))
    save_results_csv(all_results_mse,
                     os.path.join(result_dir, "results_baseline_mse.csv"))

    print("\n" + "=" * 60)
    print("  Results — Baseline AE (Mahalanobis Scoring)")
    print("=" * 60)
    print(format_results_table(all_results_mahal))
    save_results_csv(all_results_mahal,
                     os.path.join(result_dir, "results_baseline_mahalanobis.csv"))


if __name__ == "__main__":
    main()
