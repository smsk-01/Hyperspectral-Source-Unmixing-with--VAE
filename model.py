import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from torch.distributions import Dirichlet

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, nb_channels=156, n_sources=3, hidden_dims=None, min_alpha=1e-6):
        super().__init__()
        self.nb_channels = nb_channels
        self.n_sources = n_sources
        self.min_alpha = min_alpha
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # --- Encoder backbone ---
        modules = []
        in_channels = nb_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.1),
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        
        # --- Head predicting Dirichlet concentration parameters α > 0 ---
        self.alpha_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[-1], n_sources),
            nn.Softplus()  # ensures strictly positive α
        )

        # --- Decoder dictionary (endmembers) ---
        self.endmembers = nn.Parameter(
            torch.rand(n_sources, nb_channels) * 0.4 + 0.3
        )

    def encode(self, x):
        """Encodes the input spectra into a latent representation."""
        h = self.encoder(x)
        if torch.isnan(h).any():
            raise ValueError("NaNs detected in encoder output")
        return h

    def decode(self, abundances):
        """Reconstructs spectra from abundances and endmembers."""
        endmembers_clamped = torch.clamp(self.endmembers, 0, 1)
        spectra = torch.matmul(abundances, endmembers_clamped)
        return torch.clamp(spectra, 0, 1)

    def forward(self, x, tau: float = 1.0):
        """
        Forward pass:
        - Encodes input spectra into hidden features.
        - Predicts Dirichlet parameters α.
        - Samples abundances from Dir(α) (training) or uses the mean (inference).
        - Decodes abundances to reconstruct spectra.

        Args:
            x: input spectra, shape (B, nb_channels)
            tau: optional scaling factor for α to control variance
        """
        h = self.encode(x)

        # Ensure α ≥ min_alpha
        alpha = self.alpha_head(h) + 1.0  # shift slightly above 1
        alpha = torch.clamp(alpha, min=self.min_alpha)

        # Optionally scale α to control concentration (variance of Dirichlet)
        if tau is not None and tau > 0:
            alpha_effective = torch.clamp(alpha * tau, min=self.min_alpha)
        else:
            alpha_effective = alpha

        # --- Sample latent abundances on the simplex ---
        if self.training:
            # Reparameterized sampling using implicit gradients
            dist = torch.distributions.Dirichlet(alpha_effective)
            abundances = dist.rsample()  # shape (B, n_sources), sum = 1
        else:
            # Use mean of the Dirichlet distribution for inference
            abundances = alpha_effective / alpha_effective.sum(dim=1, keepdim=True)

        # --- Decode to reconstruct the spectra ---
        output = self.decode(abundances)

        return output, alpha, abundances