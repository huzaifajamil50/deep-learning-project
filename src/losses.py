"""
losses.py
---------
Loss functions for training the anomalous sound detection models.

1. MSE Loss (baseline)
   - Standard mean squared error between input and reconstruction.

2. Spectral Convergence Loss (proposed)
   - Measures the Frobenius-norm ratio between the error and the
     original signal.  Commonly used in audio synthesis tasks,
     it penalises perceptually important spectral differences more
     than a plain L2 loss.

3. Combined Loss (proposed)
   - Weighted combination: L = alpha * L_MSE + (1 - alpha) * L_SC
"""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """Standard MSE reconstruction loss (baseline)."""

    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x, x_hat):
        return self.mse(x_hat, x)


class SpectralConvergenceLoss(nn.Module):
    """
    Spectral convergence loss.

    L_SC = || x - x_hat ||_F  /  || x ||_F

    where ||.||_F is the Frobenius norm computed over the batch.
    This loss is scale-invariant and emphasises relative error,
    which is useful when the magnitude of mel spectrograms varies
    across frequency bands.
    """

    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x, x_hat):
        error_norm = torch.norm(x - x_hat, p="fro")
        signal_norm = torch.norm(x, p="fro")
        # add small epsilon to avoid division by zero
        loss = error_norm / (signal_norm + 1e-8)
        return loss


class CombinedLoss(nn.Module):
    """
    Weighted combination of MSE and spectral convergence losses.

    L_total = alpha * L_MSE + (1 - alpha) * L_SC
    """

    def __init__(self, alpha=0.7):
        """
        Parameters
        ----------
        alpha : float
            Weight for MSE loss.  (1 - alpha) is applied to the
            spectral convergence loss.
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = MSELoss()
        self.sc_loss = SpectralConvergenceLoss()

    def forward(self, x, x_hat):
        l_mse = self.mse_loss(x, x_hat)
        l_sc = self.sc_loss(x, x_hat)
        return self.alpha * l_mse + (1 - self.alpha) * l_sc


def get_loss_function(loss_type="mse", spectral_weight=0.3):
    """
    Factory function to create the appropriate loss.

    Parameters
    ----------
    loss_type : str
        "mse", "spectral", or "combined".
    spectral_weight : float
        Weight of the spectral convergence term in combined loss.
        alpha = 1 - spectral_weight  (so MSE weight = 1 - spectral_weight).

    Returns
    -------
    loss_fn : nn.Module
    """
    if loss_type == "mse":
        return MSELoss()
    elif loss_type == "spectral":
        return SpectralConvergenceLoss()
    elif loss_type == "combined":
        alpha = 1.0 - spectral_weight
        return CombinedLoss(alpha=alpha)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
