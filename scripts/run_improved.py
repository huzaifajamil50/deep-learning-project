"""
run_improved.py
---------------
Convenience script to run the improved model experiment:
  1. Train improved AE (with attention + skip connections + combined loss)
  2. Evaluate on DCASE 2023 dataset
  3. Evaluate on MIMII dataset (additional dataset — Assignment 3 requirement)
  4. Compare results with baseline

Usage:
    python scripts/run_improved.py
    python scripts/run_improved.py --machine fan
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import load_config, set_seed, build_model, train_model
from src.train import fit_mahalanobis_params
from src.test import evaluate_machine_type, load_model
from src.feature_extraction import extract_features_from_directory
from src.utils import format_results_table, save_results_csv

import numpy as np
import torch


def run_on_dataset(config, machine_types, data_root, model_type,
                   audio_cfg, device, dataset_name="dcase2023"):
    """
    Train and evaluate a model on the given dataset.

    Returns
    -------
    results_mse : dict, results_mahal : dict
    """
    # --- Train ---
    for mtype in machine_types:
        train_dir = os.path.join(data_root, mtype, "train")
        if not os.path.isdir(train_dir):
            print(f"  [SKIP] {mtype}: {train_dir} not found")
            continue

        print(f"\n--- Training {model_type} on {mtype} ({dataset_name}) ---")
        feats, _, _ = extract_features_from_directory(
            train_dir, sr=audio_cfg["sr"], n_fft=audio_cfg["n_fft"],
            hop_length=audio_cfg["hop_length"], n_mels=audio_cfg["n_mels"],
            n_frames=audio_cfg["frames"], duration=audio_cfg["duration"]
        )
        if len(feats) == 0:
            continue

        model = build_model(config, model_type=model_type)
        model, losses = train_model(
            config, model, feats, model_type=model_type,
            machine_type=f"{dataset_name}_{mtype}", device=device
        )
        mean, inv_cov = fit_mahalanobis_params(model, feats, device)
        save_dir = config["paths"]["model_dir"]
        np.save(os.path.join(
            save_dir, f"{model_type}_{dataset_name}_{mtype}_mahal_mean.npy"
        ), mean)
        np.save(os.path.join(
            save_dir, f"{model_type}_{dataset_name}_{mtype}_mahal_invcov.npy"
        ), inv_cov)

    # --- Evaluate ---
    results_mse = {}
    for mtype in machine_types:
        try:
            model = load_model(config, model_type,
                              f"{dataset_name}_{mtype}", device)
        except Exception:
            continue
        res = evaluate_machine_type(
            model, config, mtype, model_type, device, score_mode="mse"
        )
        results_mse[mtype] = res

    return results_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--machine", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_cfg = config["audio"]

    print("=" * 60)
    print("  Assignment 3 — Improved AE Experiments")
    print("=" * 60)
    print(f"  Device: {device}")

    # --- Experiment 1: Improved AE on DCASE 2023 ---
    dcase_types = config["dataset"]["dcase2023_machine_types"]
    if args.machine:
        dcase_types = [args.machine]

    dcase_root = config["dataset"]["dcase2023_root"]
    dcase_results = run_on_dataset(
        config, dcase_types, dcase_root, "improved",
        audio_cfg, device, "dcase2023"
    )

    print("\n" + "=" * 60)
    print("  Results — Improved AE on DCASE 2023")
    print("=" * 60)
    print(format_results_table(dcase_results))

    result_dir = config["paths"]["result_dir"]
    os.makedirs(result_dir, exist_ok=True)
    save_results_csv(dcase_results,
                     os.path.join(result_dir, "results_improved_dcase2023.csv"))

    # --- Experiment 2: Improved AE on MIMII (Additional Dataset) ---
    mimii_types = config["dataset"]["mimii_machine_types"]
    mimii_root = config["dataset"]["mimii_root"]

    if os.path.isdir(mimii_root):
        mimii_results = run_on_dataset(
            config, mimii_types, mimii_root, "improved",
            audio_cfg, device, "mimii"
        )
        print("\n" + "=" * 60)
        print("  Results — Improved AE on MIMII (Additional Dataset)")
        print("=" * 60)
        print(format_results_table(mimii_results))
        save_results_csv(
            mimii_results,
            os.path.join(result_dir, "results_improved_mimii.csv")
        )
    else:
        print(f"\n  [INFO] MIMII dataset not found at {mimii_root}")
        print("         Run: python scripts/download_data.py --dataset mimii")

    print("\n=== All improved model experiments complete ===")


if __name__ == "__main__":
    main()
