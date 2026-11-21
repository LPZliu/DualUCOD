"""Dual-Eigenvector Spectral Pseudo-Labeling (DESPL), paper Eqs. (1)-(10)."""

from .despl import DESPLConfig, generate_pseudo_label_with_augmentation

__all__ = ["DESPLConfig", "generate_pseudo_label_with_augmentation"]
