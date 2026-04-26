"""
train.py
--------
Training loop for both baseline and improved autoencoders.

Supports:
  - MSE, spectral convergence, and combined loss functions
  - Cosine annealing learning rate scheduling
  - Early stopping based on validation loss
  - Checkpointing best model weights
  - Per-epoch training loss logging
  - Mahalanobis covariance fitting after training completes

Usage (from project root):
    python -m src.train --config configs/config.yaml --model baseline
    python -m src.train --config configs/config.yaml --model improved
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# project imports
from .models.baseline_ae import BaselineAutoEncoder
from .models.improved_ae import ImprovedAutoEncoder
from .losses import get_loss_function
from .dataset import build_dataloader
from .feature_extraction import extract_features_from_directory
from .utils import fit_mahalanobis, plot_training_curves


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def build_model(config, model_type="baseline"):
    """
    Instantiate the autoencoder model.

    Parameters
    ----------
    config : dict
    model_type : str, "baseline" or "improved"

    Returns
    -------
    model : nn.Module
    """
    if model_type == "baseline":
        cfg = config["baseline_ae"]
        model = BaselineAutoEncoder(
            input_dim=cfg["input_dim"],
            hidden_dims=cfg["hidden_dims"],
            latent_dim=cfg["latent_dim"],
            use_batch_norm=cfg["use_batch_norm"],
            dropout=cfg["dropout"]
        )
    elif model_type == "improved":
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
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """
    Train for a single epoch.

    Returns
    -------
    avg_loss : float
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        x_hat, _ = model(batch)
        loss = loss_fn(batch, x_hat)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_model(config, model, train_features, model_type="baseline",
                machine_type="unknown", device=None):
    """
    Full training procedure for one machine type.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    model : nn.Module
    train_features : np.ndarray, shape (N, input_dim)
    model_type : str
    machine_type : str
    device : torch.device or None

    Returns
    -------
    model : nn.Module (trained)
    train_losses : list of float
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    tcfg = config["training"]
    lcfg = config["loss"]

    # setup loss, optimizer, scheduler
    if model_type == "improved":
        loss_fn = get_loss_function("combined",
                                    spectral_weight=lcfg["spectral_weight"])
    else:
        loss_fn = get_loss_function("mse")

    optimizer = Adam(model.parameters(),
                     lr=tcfg["learning_rate"],
                     weight_decay=tcfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])

    dataloader = build_dataloader(train_features,
                                   batch_size=tcfg["batch_size"],
                                   shuffle=True)

    # training loop
    train_losses = []
    best_loss = float("inf")
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training {model_type} AE on [{machine_type}]")
    print(f"  Device:     {device}")
    print(f"  Samples:    {len(train_features)}")
    print(f"  Epochs:     {tcfg['epochs']}")
    print(f"  Batch size: {tcfg['batch_size']}")
    print(f"  LR:         {tcfg['learning_rate']}")
    print(f"{'='*60}")

    for epoch in range(1, tcfg["epochs"] + 1):
        loss = train_one_epoch(model, dataloader, optimizer,
                                loss_fn, device)
        scheduler.step()
        train_losses.append(loss)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>4d}/{tcfg['epochs']}  "
                  f"Loss: {loss:.6f}  LR: {lr_now:.6f}")

        # early stopping check
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            # save best model checkpoint
            save_dir = config["paths"]["model_dir"]
            os.makedirs(save_dir, exist_ok=True)
            ckpt_path = os.path.join(
                save_dir,
                f"{model_type}_{machine_type}_best.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        if patience_counter >= tcfg["early_stopping_patience"]:
            print(f"  Early stopping at epoch {epoch}")
            break

    # load best weights
    ckpt_path = os.path.join(
        config["paths"]["model_dir"],
        f"{model_type}_{machine_type}_best.pth"
    )
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # save training curve
    fig_dir = config["paths"]["figure_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    plot_training_curves(
        train_losses,
        save_path=os.path.join(
            fig_dir,
            f"loss_{model_type}_{machine_type}.png"
        ),
        title=f"Training Loss — {model_type} AE on {machine_type}"
    )

    print(f"  Training complete.  Best loss: {best_loss:.6f}")
    return model, train_losses


def fit_mahalanobis_params(model, train_features, device):
    """
    After training, compute the Mahalanobis parameters (mean, inv_cov)
    from the training set latent representations.

    Returns
    -------
    mean, inv_cov : np.ndarray
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(train_features, dtype=torch.float32).to(device)
        z = model.get_latent(x).cpu().numpy()
    mean, inv_cov = fit_mahalanobis(z)
    return mean, inv_cov


# ------------------------------------------------------------------
#  Main entry point
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train ASD autoencoder models"
    )
    parser.add_argument("--config", type=str,
                        default="configs/config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "improved"],
                        help="Model type to train")
    parser.add_argument("--machine", type=str, default=None,
                        help="Specific machine type (default: train all)")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # determine machine types to process
    machine_types = config["dataset"]["dcase2023_machine_types"]
    if args.machine:
        machine_types = [args.machine]

    data_root = config["dataset"]["dcase2023_root"]
    audio_cfg = config["audio"]

    for mtype in machine_types:
        train_dir = os.path.join(data_root, mtype, "train")

        if not os.path.isdir(train_dir):
            print(f"  [WARN] Training directory not found: {train_dir}")
            continue

        # extract features
        print(f"\nExtracting features for {mtype} ...")
        train_feats, _, _ = extract_features_from_directory(
            train_dir,
            sr=audio_cfg["sr"],
            n_fft=audio_cfg["n_fft"],
            hop_length=audio_cfg["hop_length"],
            n_mels=audio_cfg["n_mels"],
            n_frames=audio_cfg["frames"],
            duration=audio_cfg["duration"]
        )

        if len(train_feats) == 0:
            print(f"  [WARN] No training features for {mtype}, skipping.")
            continue

        # build and train model
        model = build_model(config, model_type=args.model)
        model, losses = train_model(
            config, model, train_feats,
            model_type=args.model,
            machine_type=mtype,
            device=device
        )

        # fit Mahalanobis parameters
        mean, inv_cov = fit_mahalanobis_params(model, train_feats, device)
        save_dir = config["paths"]["model_dir"]
        np.save(os.path.join(save_dir,
                             f"{args.model}_{mtype}_mahal_mean.npy"), mean)
        np.save(os.path.join(save_dir,
                             f"{args.model}_{mtype}_mahal_invcov.npy"), inv_cov)

    print("\n=== Training finished for all machine types ===")


if __name__ == "__main__":
    main()
