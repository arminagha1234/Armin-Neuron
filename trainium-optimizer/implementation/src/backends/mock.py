"""
Mock backend — lets the orchestrator, guardrails, and reporting be developed
and tested with zero Trainium hardware.

It models the two things that actually shape the loop's behavior:
  1. Compile time dominates cost (5-20 min per candidate).
  2. Different config axes have different, interacting effects on throughput.

The throughput model is a toy, but it is monotone-ish and noisy in the right
places, so a search strategy that works against it is at least not obviously
broken before it ever touches hardware. It is NOT a performance predictor.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from backends.base import (
    Artifact,
    Backend,
    Measurements,
    Neff,
    OpSite,
    Profile,
)


class MockBackend:
    """A deterministic-with-seed stand-in for a real Neuron backend."""

    name = "mock"

    def __init__(self, seed: int = 0, hbm_gb: float = 96.0) -> None:
        self._rng = random.Random(seed)
        self._hbm_gb = hbm_gb

    def build_baseline(self, model_id: str) -> Artifact:
        return Artifact(
            model_id=model_id,
            backend=self.name,
            config={
                "tp_degree": 1,
                "weights_dtype": "bf16",
                "kv_cache_dtype": "bf16",
                "batching": "static",
                "attention_kernel": "default",
            },
        )

    def config_axes(self) -> dict[str, list[Any]]:
        return {
            "tp_degree": [1, 2, 4, 8, 16, 32],
            "weights_dtype": ["fp32", "bf16", "fp8"],
            "kv_cache_dtype": ["bf16", "fp8"],
            "batching": ["static", "dynamic", "continuous"],
            "attention_kernel": ["default", "paged", "flash", "kv_parallel"],
        }

    def apply_config(self, artifact: Artifact, config: dict[str, Any]) -> Artifact:
        merged = {**artifact.config, **config}
        return Artifact(model_id=artifact.model_id, backend=self.name, config=merged)

    def compile(self, artifact: Artifact) -> Neff:
        # Model compile cost as a function of config complexity, so the loop's
        # cost accounting is exercised. fp8 and flash cost more to compile.
        base = 300.0
        cfg = artifact.config
        if cfg.get("weights_dtype") == "fp8":
            base += 180
        if cfg.get("attention_kernel") in ("flash", "kv_parallel"):
            base += 150
        base += cfg.get("tp_degree", 1) * 8
        jitter = self._rng.uniform(-30, 30)
        return Neff(artifact=artifact, path=f"mock://{_hash(cfg)}",
                    compile_seconds=max(60.0, base + jitter))

    def measure(self, neff: Neff, shape: str, batch: int) -> Measurements:
        tps = self._throughput(neff.artifact.config, shape, batch)
        hbm = self._hbm(neff.artifact.config, shape, batch)
        return Measurements(
            metric=tps,
            metric_p50=tps,
            metric_p99=tps * 0.92,
            ttft_ms_p50=1000.0 / max(tps, 1) * batch,
            hbm_peak_gb=hbm,
            hbm_available_gb=self._hbm_gb,
            mfu_percent=self._mfu(neff.artifact.config, tps),
            shape=shape,
            batch=batch,
            warmup_iters=3,
            measured_iters=10,
        )

    def profile(self, neff: Neff, shape: str) -> Profile:
        cfg = neff.artifact.config
        # Bottleneck flips with config, so symptom-based retrieval has something
        # to bite on: default attention is collective/compute bound; once a
        # better attention kernel is in, it shifts toward dma.
        if cfg.get("attention_kernel") == "default":
            bottleneck, engines = "collective_bound", {"PE": 0.35, "DMA": 0.4, "CC": 0.7}
        elif cfg.get("attention_kernel") in ("flash", "kv_parallel"):
            bottleneck, engines = "dma_bound", {"PE": 0.55, "DMA": 0.75, "CC": 0.2}
        else:
            bottleneck, engines = "compute_bound", {"PE": 0.7, "DMA": 0.4, "CC": 0.3}
        return Profile(
            op_sites=[
                OpSite("attention_prefill", 0.47, cfg.get("attention_kernel", "default")),
                OpSite("mlp", 0.20, "default"),
                OpSite("rmsnorm", 0.08, "default"),
            ],
            bottleneck=bottleneck,
            engine_utilization=engines,
            raw_path=neff.path,
        )

    def kernel_swap_points(self, artifact: Artifact) -> list[OpSite]:
        return [
            OpSite("attention_prefill", 0.47, artifact.config.get("attention_kernel", "default")),
            OpSite("rmsnorm", 0.08, "default"),
        ]

    def toolchain_stamp(self) -> dict[str, str]:
        return {
            "backend": self.name,
            "neuron_sdk": "2.28.0",
            "neuronx_cc": "2.26.6360.0-mock",
            "instance_type": "mock",
        }

    # -- toy models ----------------------------------------------------------

    def _throughput(self, cfg: dict[str, Any], shape: str, batch: int) -> float:
        base = 600.0
        # TP helps up to a point, then over-sharding hurts (models the real
        # weight-spill anti-pattern the bank should learn).
        tp = cfg.get("tp_degree", 1)
        base *= {1: 1.0, 2: 1.8, 4: 3.2, 8: 5.0, 16: 3.5, 32: 2.0}.get(tp, 1.0)
        base *= {"fp32": 0.6, "bf16": 1.0, "fp8": 1.35}.get(cfg.get("weights_dtype"), 1.0)
        base *= {"bf16": 1.0, "fp8": 1.15}.get(cfg.get("kv_cache_dtype"), 1.0)
        base *= {"static": 1.0, "dynamic": 1.1, "continuous": 1.4}.get(cfg.get("batching"), 1.0)
        base *= {"default": 1.0, "paged": 1.3, "flash": 2.6, "kv_parallel": 2.9}.get(
            cfg.get("attention_kernel"), 1.0)
        base *= 1.0 + 0.02 * batch
        return round(base * self._rng.uniform(0.98, 1.02), 1)

    def _hbm(self, cfg: dict[str, Any], shape: str, batch: int) -> float:
        tp = cfg.get("tp_degree", 1)
        weight_gb = 62.0 / tp
        kv_mult = 0.5 if cfg.get("kv_cache_dtype") == "fp8" else 1.0
        ctx = {"chat 1k/512": 4, "rag 10k/512": 12, "generate 512/10k": 12,
               "stress 64k/64k": 60}.get(shape, 6)
        kv_gb = ctx * batch * kv_mult / tp
        return round(weight_gb + kv_gb, 1)

    def _mfu(self, cfg: dict[str, Any], tps: float) -> float:
        # Illustrative only.
        return round(min(60.0, tps / 900.0), 2)


def _hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha1(repr(sorted(cfg.items())).encode()).hexdigest()[:10]


# Structural check: MockBackend must satisfy the Backend protocol.
_: Backend = MockBackend()
