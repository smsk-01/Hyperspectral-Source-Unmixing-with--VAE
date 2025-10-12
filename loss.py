import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import math

def SAD(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Spectral Angle Distance (SAD) loss.
    Measures the angle (in radians) between two spectral vectors.
    Lower values mean more similar spectra.
    """

    # L2 normalization of both tensors
    y_true_norm = F.normalize(y_true, p=2, dim=-1)
    y_pred_norm = F.normalize(y_pred, p=2, dim=-1)

    # Compute cosine similarity between normalized vectors
    cos_theta = torch.sum(y_true_norm * y_pred_norm, dim=-1)

    # Clamp for numerical stability (avoid values slightly > 1 or < -1)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Convert cosine similarity to angle in radians
    sad = torch.acos(cos_theta)

    # Return mean angle over the batch
    return sad.mean()


def loss_function(x, x_recon, alpha,
                  lambda_recon=1.0, lambda_kl=0.0001
                  ):
    """
    Computes the total loss for the Dirichlet-VAE without an explicit alpha prior.

    Args:
        x: (batch, channels) original input spectra
        x_recon: (batch, channels) reconstructed spectra
        alpha: (batch, n_sources) Dirichlet parameters predicted by the encoder
        lambda_recon: weight for reconstruction loss
        lambda_kl: weight for KL divergence term
    """
    batch_size = x.size(0)

    # --- 1. Reconstruction loss (SAD + MSE) ---
    x_norm = F.normalize(x, p=2, dim=1)
    x_recon_norm = F.normalize(x_recon, p=2, dim=1)
    cos_sim = torch.sum(x_norm * x_recon_norm, dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
    sad = torch.acos(cos_sim) / math.pi  # normalize to [0,1]
    mse = torch.mean((x - x_recon) ** 2, dim=1)
    recon_loss = torch.mean(sad + 0.1 * mse)

    # --- 2. KL Divergence KL(Dir(alpha) || Dir(1)) ---
    alpha = torch.clamp(alpha, min=1e-6)
    alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
    n_sources = alpha.size(1)
    
    kl_div = (
        torch.lgamma(alpha_sum)
        - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        - torch.lgamma(torch.tensor(n_sources, dtype=torch.float32))
        + torch.sum((alpha - 1) * (torch.digamma(alpha) - torch.digamma(alpha_sum)), dim=1, keepdim=True)
    )
    kl_loss = torch.mean(kl_div)

    # --- Total loss ---
    total_loss = (
        lambda_recon * recon_loss +
        lambda_kl * kl_loss 
    )

    return total_loss, recon_loss, kl_loss
