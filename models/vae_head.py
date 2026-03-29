"""
models/vae_head.py
------------------
VAE head that sits on top of the frozen TRM's z_H state.

z_H shape: (batch, seq_len + puzzle_emb_len, hidden_size=512)
We use z_H[:, 0] as the summary vector (same as q_head in trm.py)
to produce mu and logvar, then sample z_sampled.

z_sampled is then broadcast across all sequence positions and added
to z_H to produce perturbed hidden states, which are decoded by
the existing lm_head to produce output logits.

This means:
  - We reuse the existing lm_head (no new output head needed)
  - Each sample produces a different logit distribution per token
  - K samples = K different output grids
  - Majority vote per token position across K samples = final prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEHead(nn.Module):
    """
    Variational head on top of TRM's z_H.

    Takes z_H[:, 0] (batch, hidden_size) -> mu, logvar
    Samples z_sampled and broadcasts it across sequence positions.
    The perturbed z_H is then decoded by the frozen lm_head.

    Args:
        hidden_size: must match TRM hidden_size (512)
    """

    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size

        # Two small linear heads to produce mu and logvar
        # from the summary token z_H[:, 0]
        self.mu_head     = nn.Linear(hidden_size, hidden_size)
        self.logvar_head = nn.Linear(hidden_size, hidden_size)

        # Small MLP to blend sampled noise back into z_H
        # Takes (z_H_position + z_sampled) -> refined hidden state
        """
        self.blend = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        """
        self.noise_scale = 1.0
        # Init mu_head close to identity, logvar_head close to zero
        # This makes initial samples close to deterministic TRM output
        nn.init.eye_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        nn.init.constant_(self.logvar_head.bias, -4.0)  # small initial variance

    def encode(self, z_H: torch.Tensor):
        """
        z_H: (batch, seq_len, hidden_size)
        Returns mu, logvar: (batch, hidden_size) from summary token
        """
        summary = z_H[:, 0]  # (batch, hidden_size) — same as q_head
        mu      = self.mu_head(summary)
        logvar  = self.logvar_head(summary).clamp(-6, 2)  # stability
        return mu, logvar

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Single reparameterised sample. Different each call due to randn."""
        std = torch.exp(0.5 * logvar) + 0.5
        eps = torch.randn_like(std)
        return mu + eps * std   # (batch, hidden_size)

    def perturb_z_H(self, z_H: torch.Tensor, z_sampled: torch.Tensor):
        """
        Add sampled noise directly to z_H instead of blending.
        This bypasses the learned suppression problem.
        """
        # Project z_sampled to same shape and add directly
        z_exp = z_sampled.unsqueeze(1).expand(-1, z_H.size(1), -1)
        return z_H + self.noise_scale * z_exp                      # (batch, seq_len, H)

    def forward(self, z_H: torch.Tensor):
        """
        Single sample forward pass.
        Returns perturbed z_H, mu, logvar.
        """
        mu, logvar = self.encode(z_H)
        z_sampled  = self.sample(mu, logvar)
        z_perturbed = self.perturb_z_H(z_H, z_sampled)
        return z_perturbed, mu, logvar

    def sample_n(self, z_H: torch.Tensor, K: int):
        """
        K independent samples from the same z_H.
        Returns:
            all_z: (K, batch, seq_len, hidden_size)
            mu:    (batch, hidden_size)
            logvar:(batch, hidden_size)
        """
        mu, logvar = self.encode(z_H)
        all_z = torch.stack(
            [self.perturb_z_H(z_H, self.sample(mu, logvar)) for _ in range(K)],
            dim=0
        )
        return all_z, mu, logvar


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.5):
    """
    KL divergence from N(mu, sigma) to N(0, 1).
    free_bits prevents posterior collapse by not penalising
    dimensions with KL below the threshold.
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (batch, hidden_size)
    kl = kl.clamp(min=free_bits)
    return kl.mean()
