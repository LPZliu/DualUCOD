"""Compatibility import for the historical misspelled module name.

Training and inference now construct the same full DualUCOD model. Legacy
single-branch checkpoints require their original implementation.
"""
from lib.new_dino_network import Network

__all__ = ["Network"]
