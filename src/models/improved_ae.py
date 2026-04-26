"""
improved_ae.py
--------------
Proposed improved autoencoder for DCASE 2023 Task 2.

Improvements over the baseline:

1. Skip Connections (U-Net style)
   - Encoder hidden activations are forwarded to the corresponding
     decoder layers, helping the decoder recover fine spectral detail.

2. Squeeze-and-Excitation (SE) Attention at the Bottleneck
   - A channel-attention block adaptively re-weights the latent
     dimensions, allowing the model to focus on the most discriminative
     features for each machine type.

3. Optional dropout for regularization.

Motivation:
  The baseline AE loses fine-grained spectral information through the
  narrow bottleneck (dim=8). Skip connections alleviate this, while
  SE attention lets the model learn which latent features are most
  informative for detecting anomalies.

Note: total parameter count is kept comparable to the baseline so
that performance gains are due to architecture, not model capacity.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for 1-D feature vectors.

    Given an input of shape (batch, dim), this module computes a
    per-dimension attention weight using a small bottleneck MLP:
        squeeze  : global description = input itself (already 1-D)
        excite   : dim -> dim//r -> dim  (with ReLU + Sigmoid)
        scale    : input * attention_weights
    """

    def __init__(self, dim, reduction=4):
        super(SEBlock, self).__init__()
        mid = max(dim // reduction, 2)
        self.fc = nn.Sequential(
            nn.Linear(dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.fc(x)          # (batch, dim)
        return x * weights            # channel-wise reweighting


class EncoderBlock(nn.Module):
    """Single encoder layer: Linear -> BN -> ReLU (-> Dropout)."""

    def __init__(self, in_dim, out_dim, use_bn=True, dropout=0.0):
        super(EncoderBlock, self).__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Single decoder layer with optional skip connection input."""

    def __init__(self, in_dim, out_dim, use_bn=True, dropout=0.0):
        super(DecoderBlock, self).__init__()
        layers = [nn.Linear(in_dim, out_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ImprovedAutoEncoder(nn.Module):
    """
    Autoencoder with skip connections and SE-attention at the bottleneck.
    """

    def __init__(self, input_dim=640, hidden_dims=None,
                 latent_dim=8, use_batch_norm=True, dropout=0.1,
                 use_skip_connections=True, use_attention=True,
                 attention_reduction=4):
        """
        Parameters
        ----------
        input_dim : int
            Input feature dimensionality (640 by default).
        hidden_dims : list of int
            Encoder hidden layer sizes. Decoder mirrors this in reverse.
        latent_dim : int
            Bottleneck size.
        use_batch_norm : bool
            Apply batch normalization in each block.
        dropout : float
            Dropout rate.
        use_skip_connections : bool
            If True, add encoder hidden outputs to decoder inputs.
        use_attention : bool
            If True, apply SE-attention at the bottleneck.
        attention_reduction : int
            Reduction ratio for the SE block.
        """
        super(ImprovedAutoEncoder, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 128]

        self.use_skip = use_skip_connections
        self.use_attn = use_attention

        # ----- Encoder -----
        self.enc_blocks = nn.ModuleList()
        prev = input_dim
        for h in hidden_dims:
            self.enc_blocks.append(
                EncoderBlock(prev, h, use_bn=use_batch_norm, dropout=dropout)
            )
            prev = h

        # bottleneck projection
        self.bottleneck_enc = nn.Sequential(
            nn.Linear(prev, latent_dim),
            nn.BatchNorm1d(latent_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        # SE attention at bottleneck
        if self.use_attn:
            self.se_block = SEBlock(latent_dim,
                                    reduction=attention_reduction)

        # ----- Decoder -----
        self.dec_blocks = nn.ModuleList()
        rev_dims = list(reversed(hidden_dims))

        # first decoder layer takes latent_dim input
        first_in = latent_dim
        if self.use_skip:
            # skip from last encoder block adds hidden_dims[-1]
            first_in = latent_dim + hidden_dims[-1]
        self.dec_blocks.append(
            DecoderBlock(first_in, rev_dims[0],
                         use_bn=use_batch_norm, dropout=dropout)
        )

        # remaining decoder layers
        for i in range(1, len(rev_dims)):
            d_in = rev_dims[i - 1]
            if self.use_skip:
                # skip from corresponding encoder block
                d_in = rev_dims[i - 1] + hidden_dims[-(i + 1)]
            self.dec_blocks.append(
                DecoderBlock(d_in, rev_dims[i],
                             use_bn=use_batch_norm, dropout=dropout)
            )

        # output projection
        self.output_layer = nn.Linear(rev_dims[-1], input_dim)

        # store for skip connection indexing
        self.hidden_dims = hidden_dims

    def forward(self, x):
        """
        Forward pass with optional skip connections and attention.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, input_dim)

        Returns
        -------
        x_hat : torch.Tensor, shape (batch, input_dim)
            Reconstructed output.
        z : torch.Tensor, shape (batch, latent_dim)
            Bottleneck representation (after attention if enabled).
        """
        # encode - save intermediate activations for skip connections
        enc_outputs = []
        h = x
        for block in self.enc_blocks:
            h = block(h)
            enc_outputs.append(h)

        # bottleneck
        z = self.bottleneck_enc(h)

        # apply attention
        if self.use_attn:
            z = self.se_block(z)

        # decode with skip connections
        d = z
        for i, block in enumerate(self.dec_blocks):
            if self.use_skip:
                # grab the encoder output from the mirror position
                skip_idx = len(enc_outputs) - 1 - i
                skip = enc_outputs[skip_idx]
                d = torch.cat([d, skip], dim=1)
            d = block(d)

        x_hat = self.output_layer(d)
        return x_hat, z

    def get_latent(self, x):
        """Extract bottleneck representation."""
        h = x
        for block in self.enc_blocks:
            h = block(h)
        z = self.bottleneck_enc(h)
        if self.use_attn:
            z = self.se_block(z)
        return z

    def compute_anomaly_score(self, x):
        """MSE reconstruction error as anomaly score."""
        self.eval()
        with torch.no_grad():
            x_hat, _ = self.forward(x)
            scores = torch.mean((x - x_hat) ** 2, dim=1)
        return scores
