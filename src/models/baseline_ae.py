"""
baseline_ae.py
--------------
Baseline Autoencoder architecture for DCASE 2023 Task 2.

This reproduces the official baseline from:
  Dohi et al., "Description and Discussion on DCASE 2023 Challenge Task 2:
  First-Shot Unsupervised Anomalous Sound Detection for Machine Condition
  Monitoring," arXiv:2305.07828, 2023.

Architecture:
  Encoder: input_dim -> 128 -> 128 -> 128 -> 128 -> latent_dim (8)
  Decoder: latent_dim (8) -> 128 -> 128 -> 128 -> 128 -> input_dim

Each hidden layer uses:  Linear -> BatchNorm -> ReLU
The output layer uses:   Linear (no activation)

The model is trained with MSE loss between input and reconstructed output.
Anomaly scores at test time can be computed via:
  (a) Reconstruction error (MSE) -- Simple Autoencoder mode
  (b) Mahalanobis distance in the bottleneck space -- Selective Mahalanobis mode
"""

import torch
import torch.nn as nn


class BaselineAutoEncoder(nn.Module):
    """
    Fully-connected autoencoder matching the DCASE 2023 Task 2 baseline.
    """

    def __init__(self, input_dim=640, hidden_dims=None,
                 latent_dim=8, use_batch_norm=True, dropout=0.0):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input feature vectors (n_mels * n_frames = 128*5 = 640).
        hidden_dims : list of int
            Sizes of hidden layers. Default is [128, 128, 128, 128].
        latent_dim : int
            Size of the bottleneck layer.
        use_batch_norm : bool
            Whether to apply BatchNorm after each hidden layer.
        dropout : float
            Dropout probability (0.0 means no dropout).
        """
        super(BaselineAutoEncoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 128]

        # ----- build encoder -----
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                encoder_layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        # bottleneck projection
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        if use_batch_norm:
            encoder_layers.append(nn.BatchNorm1d(latent_dim))
        encoder_layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*encoder_layers)

        # ----- build decoder -----
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                decoder_layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim

        # output projection (no activation)
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, input_dim).

        Returns
        -------
        x_hat : torch.Tensor
            Reconstructed input, shape (batch_size, input_dim).
        z : torch.Tensor
            Bottleneck representation, shape (batch_size, latent_dim).
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def get_latent(self, x):
        """Extract the bottleneck representation only."""
        return self.encoder(x)

    def compute_anomaly_score(self, x):
        """
        Compute per-sample MSE reconstruction error as anomaly score.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, input_dim)

        Returns
        -------
        scores : torch.Tensor, shape (batch,)
        """
        self.eval()
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            scores = torch.mean((x - x_hat) ** 2, dim=1)
        return scores
