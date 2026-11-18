"""DBCL equations (18)-(23), DualUCOD, paper page 7.

The paper does not uniquely specify every internal architectural detail.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import LayerNorm2D, PositionwiseFeedForward


class CSE(nn.Module):
    def __init__(self, d_in, d_hidden, proj_dim=128, n_heads=8, pool_size=2):
        super().__init__()
        # Eq.18 lists convolution, BN, ReLU, average pooling and flattening.
        # Retain the local convolution/normalization stem. Pooling size is
        # explicitly an unresolved implementation choice.
        self.conv = nn.Sequential(
            nn.Conv2d(d_in, d_hidden, 1), nn.BatchNorm2d(d_hidden), nn.ReLU(),
            nn.Conv2d(d_hidden, d_hidden, 1), LayerNorm2D(d_hidden), nn.ReLU(),
            nn.AvgPool2d(pool_size),
        )
        self.mha_orig = nn.MultiheadAttention(d_hidden, n_heads, batch_first=True)
        self.mha_aug = nn.MultiheadAttention(d_hidden, n_heads, batch_first=True)
        self.g_orig = PositionwiseFeedForward(d_hidden, d_hidden * 2)
        self.g_aug = PositionwiseFeedForward(d_hidden, d_hidden * 2)
        self.projector = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.ReLU(inplace=True),
            nn.Linear(d_hidden, proj_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x, attention, ffn):
        tokens = self.conv(x).flatten(2).transpose(1, 2)
        query = (tokens.softmax(dim=1) * tokens).sum(dim=1, keepdim=True)
        attended, _ = attention(query, tokens, tokens, need_weights=False)
        return F.normalize(self.projector(ffn(attended).squeeze(1)), dim=1)

    @staticmethod
    def contrastive_loss(u_orig, u_aug, tau=0.1):
        """Eq.23: augmented query i contrasted against B original-view keys."""
        if tau <= 0:
            raise ValueError("tau must be positive")
        if u_orig.ndim != 2 or u_orig.shape != u_aug.shape or u_orig.shape[0] == 0:
            raise ValueError("Both views must have the same nonempty B x D shape")
        logits = u_aug @ u_orig.T / tau
        return F.cross_entropy(logits, torch.arange(u_orig.shape[0], device=logits.device))

    def forward(self, x_orig, x_aug, tau=0.1):
        u_orig = self.encode(x_orig, self.mha_orig, self.g_orig)
        u_aug = self.encode(x_aug, self.mha_aug, self.g_aug)
        return self.contrastive_loss(u_orig, u_aug, tau)
