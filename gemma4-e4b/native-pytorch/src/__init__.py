"""Gemma 4 E4B-it on Trainium2 — native PyTorch + TP=2."""
from .tp_plan import build_e4b_tp_plan

__all__ = ["build_e4b_tp_plan"]
