"""DualUCOD stage-II network, reviewed against the published method equations."""

import torch
from torch import nn
from torch.nn import functional as F

from hubconf import dinov2_vitl14
from lib.new_Modules import GPM, REM11
from lib.cse import CSE


class FeatureAdapter(nn.Module):
    """Retain the adapter structure present in the supplied local source."""

    def __init__(self, in_dim=1024, out_dim=1024, num_heads=8, dropout=0.1):
        super().__init__()
        if in_dim != out_dim:
            raise ValueError("The local residual adapter requires in_dim == out_dim")
        self.query_project = nn.Linear(in_dim, out_dim)
        self.key_project = nn.Linear(in_dim, out_dim)
        self.value_project = nn.Linear(in_dim, out_dim)
        self.multihead_attn = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(out_dim * 4, out_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n, c, h, w = x.shape
        tokens = x.flatten(2).permute(2, 0, 1)
        attended, _ = self.multihead_attn(
            self.query_project(tokens), self.key_project(tokens),
            self.value_project(tokens), need_weights=False,
        )
        tokens = self.norm(attended + tokens)
        tokens = self.norm(tokens + self.dropout(self.feed_forward(tokens)))
        return tokens.permute(1, 2, 0).reshape(n, c, h, w)


class Network(nn.Module):
    """Frozen shared backbone, two adapters/decoders, shared contrastive encoder.

    Inputs use ImageNet normalization, as in utils.data_val. phase='train'
    accepts a photometric second view and returns (preds, aug_preds, Lcon, None).
    phase='test' returns (preds, None, None, None). Without phase, the nine
    original-view prediction tensors are returned for the legacy test API.
    A prediction sequence contains prior, four foreground, four background logits.
    """

    def __init__(self, channels=64, pretrained=True, backbone=None, feature_dim=1024):
        super().__init__()
        if backbone is None and feature_dim != 1024:
            raise ValueError("The paper backbone DINOv2 ViT-L/14 has 1024 channels")
        self.backbone = dinov2_vitl14(pretrained=pretrained) if backbone is None else backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.GPM = GPM(channels=feature_dim)
        self.Aug_GPM = GPM(channels=feature_dim)
        self.REM11 = REM11(feature_dim, channels)
        self.Aug_REM11 = REM11(feature_dim, channels)
        self.x_adapter = FeatureAdapter(feature_dim, feature_dim)
        self.y_adapter = FeatureAdapter(feature_dim, feature_dim)
        self.CSE = CSE(feature_dim, feature_dim, proj_dim=feature_dim)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def _rgb(image):
        mean = image.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = image.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (image * std + mean).clamp(0, 1)

    def _view(self, image, adapter, gpm, decoder):
        with torch.no_grad():
            features = list(self.backbone.get_intermediate_layers(image, n=4, reshape=True))
        features[-1] = adapter(features[-1])
        prior = gpm(features[-1])
        full_prior = F.interpolate(prior, size=image.shape[-2:], mode='bilinear', align_corners=False)
        decoded = decoder(features, prior, self._rgb(image))
        return [full_prior, *decoded], features[-1]

    def forward(self, x, phase=None, y=None):
        if phase not in (None, 'train', 'test'):
            raise ValueError("phase must be 'train', 'test', or None")
        if phase == 'train' and y is None:
            raise ValueError("Training requires the photometric second view y")
        preds, feature = self._view(x, self.x_adapter, self.GPM, self.REM11)
        if phase == 'train':
            aug_preds, aug_feature = self._view(y, self.y_adapter, self.Aug_GPM, self.Aug_REM11)
            return preds, aug_preds, self.CSE(feature, aug_feature), None
        if phase == 'test':
            return preds, None, None, None
        return tuple(preds)
