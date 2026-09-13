"""Shared Koopman operator model for EDKCD baseline.

Replaces the per-edge operators K[i,j,:,:,:] with one shared NC×NC×lag operator,
following the design of NeuroKoopman's shared Koopman operator.
"""
import torch
import torch.nn as nn
from edkcd.models import VARP


class VARP_SharedK(VARP):
    """VARP with shared Koopman operator (NC×NC) instead of per-edge operators.

    Inheritance approach: keeps encoder/decoder from VARP, replaces the
    per-edge fusion kernel K with a single shared operator K_shared.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        C = self.n_channels
        # Remove the per-edge K, replace with shared operator
        del self.K
        NC = self.c_in * C
        self.K_shared = nn.Parameter(
            torch.zeros(NC, NC, self.lag).normal_(0, 0.01)
        )

    def forward(self, x, var_fusion_enabled):
        B, N, T = x.shape
        C = self.n_channels
        x_enc = self.encoder(x.reshape(B * N, 1, T))
        x_enc = self.identity_pool(x_enc)

        if var_fusion_enabled:
            x_pad = self.const_pad(x_enc)
            x_unfold = x_pad.unfold(2, self.lag, 1)
            x_reshaped = x_unfold.reshape(B, N, C, -1, self.lag)
            x_perm = x_reshaped.permute(0, 1, 3, 2, 4)   # (B, N, T', C, lag)

            # Shared operator: view K_shared as (N, C, N, C, lag)
            # Use same einsum pattern as original, no extra reshape needed
            K_view = self.K_shared.view(N, C, N, C, self.lag)
            # 'bitcl,icjdl->bjtd': keep x_perm layout, apply shared sub-blocks
            x_fused = torch.einsum('bitcl,icjdl->bjtd', x_perm, K_view)
            # (B, N, T', C) -> (B*N, C, T')
            x_fused = x_fused.permute(0, 1, 3, 2).reshape(B * N, C, -1)

            x_out = self._decode(x_fused)
        else:
            x_out = self._decode(x_enc)

        x_out = x_out.reshape(B, N, -1)
        return x_out

    def compute_latent_fusion(self, x_enc):
        """Latent fusion for computing linear loss (lin_weight)."""
        B_merged = x_enc.size(0) // self.c_in
        x_pad = self.const_pad(x_enc)
        x_unfold = x_pad.unfold(2, self.lag, 1)
        x_reshaped = x_unfold.reshape(B_merged, self.c_in, self.n_channels, -1, self.lag)
        x_perm = x_reshaped.permute(0, 1, 3, 2, 4)

        N = self.c_in
        C = self.n_channels
        K_view = self.K_shared.view(N, C, N, C, self.lag)
        x_fused = torch.einsum('bitcl,icjdl->bjtd', x_perm, K_view)
        x_fused = x_fused.permute(0, 1, 3, 2).reshape(B_merged * N, C, -1)
        return x_fused

    def get_regularized_params(self, stage=1):
        if stage == 0:
            if self.decoder_mode == 'kan':
                return list(self.kan_decoder.parameters())
            else:
                return list(self.mlp_decoder.parameters())
        else:
            decoder_params = super().get_regularized_params(stage=1)
            return [self.K_shared] + decoder_params

    def get_trainable_params(self):
        return [self.K_shared]


# ---------------------------------------------------------------------------
# Helper: extract per-edge scores from the shared Koopman operator
# ---------------------------------------------------------------------------
def compute_sharedk_edge_scores(K_shared, n_vars, n_channels):
    """Extract per-edge causal scores from the shared NC×NC operator.

    Vectorized: reshape K_shared to (N, C, N, C, lag) and norm over (C, C, lag).
    """
    N = n_vars
    C = n_channels
    # K_shared: (N*C, N*C, lag) -> (N, C, N, C, -1) -> (N, N)
    blocks = K_shared.view(N, C, N, C, -1)
    # Frobenius norm over (C, C, lag) via sum of squares (supports 3+ dims)
    scores = torch.sqrt((blocks ** 2).sum(dim=(1, 3, 4)))
    return scores


def compute_sharedk_group_lasso(K_shared, n_vars, n_channels, group_lambda):
    """Group lasso over sub-blocks of the shared operator (vectorized)."""
    N = n_vars
    C = n_channels
    blocks = K_shared.view(N, C, N, C, -1)
    norms = torch.sqrt((blocks ** 2).sum(dim=(1, 3, 4)))  # (N, N)
    return group_lambda * norms.mean()