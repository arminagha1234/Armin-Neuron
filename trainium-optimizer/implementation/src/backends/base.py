"""
Backend interface — isolates the ~20% of the framework that is
backend-specific, so the XLA -> native-PyTorch migration is an added file
rather than a rewrite.

Everything above this line in the stack (bank, search loop, guardrails,
ledger, reporting, and the NKI kernels themselves) is backend-independent.

Concrete backends:
  vllm_neuron_xla.py   — V1 primary. Production-representative, proven at a large (multiple-x).
  nxdi_xla.py          — Stage 0 baseline producer (what autoport targets).
  native_pytorch.py    — added when the beta is viable and TP=8 works on Trn2.

See ../../../architecture.md#backend-adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Artifact:
    """A configured, not-yet-compiled model implementation."""

    model_id: str
    backend: str
    config: dict[str, Any] = field(default_factory=dict)
    workdir: str = ""            # path to the fork/checkout this artifact edits


@dataclass
class Neff:
    """A compiled artifact ready to run. `path` may be empty for eager backends."""

    artifact: Artifact
    path: str = ""
    compile_seconds: float = 0.0


@dataclass
class Measurements:
    """One (shape, batch) measurement. Metric units are track-specific."""

    metric: float                 # tok/s for track A; images/s, RTF, etc. elsewhere
    metric_p50: float = 0.0
    metric_p99: float = 0.0
    ttft_ms_p50: float = 0.0
    ttft_ms_p99: float = 0.0
    hbm_peak_gb: float = 0.0
    hbm_available_gb: float = 0.0
    mfu_percent: float = -1.0
    shape: str = ""
    batch: int = 1
    warmup_iters: int = 0
    measured_iters: int = 0

    @property
    def hbm_utilization(self) -> float:
        if self.hbm_available_gb <= 0:
            return 0.0
        return self.hbm_peak_gb / self.hbm_available_gb


@dataclass
class OpSite:
    """A point in the graph where a kernel can be substituted (Stages 2-4)."""

    op_name: str                  # e.g. "attention_prefill", "rmsnorm"
    cost_share: float = 0.0       # fraction of step time, from the profile
    current_kernel: str = ""      # what is there now
    shape_signature: dict[str, Any] = field(default_factory=dict)


@dataclass
class Profile:
    """Parsed profiler output. The 'what to fix' signal for Stages 2-5."""

    op_sites: list[OpSite] = field(default_factory=list)
    bottleneck: str = ""          # "compute_bound" | "dma_bound" | "collective_bound" | ...
    engine_utilization: dict[str, float] = field(default_factory=dict)  # PE, DMA, CC, ...
    raw_path: str = ""

    def hottest(self, n: int = 3) -> list[OpSite]:
        return sorted(self.op_sites, key=lambda s: s.cost_share, reverse=True)[:n]


@runtime_checkable
class Backend(Protocol):
    """What every serving backend must provide.

    Kept deliberately small. Anything a stage needs that is not here is, by
    definition, backend-independent and belongs in the core.
    """

    name: str

    def build_baseline(self, model_id: str) -> Artifact:
        """Stage 0. Produce a runnable, correct implementation from a model id."""
        ...

    def config_axes(self) -> dict[str, list[Any]]:
        """Stage 1. Backend-specific knob names and their legal values."""
        ...

    def apply_config(self, artifact: Artifact, config: dict[str, Any]) -> Artifact:
        ...

    def compile(self, artifact: Artifact) -> Neff:
        """No-op (returns immediately, compile_seconds=0) for eager backends."""
        ...

    def measure(self, neff: Neff, shape: str, batch: int) -> Measurements:
        ...

    def profile(self, neff: Neff, shape: str) -> Profile:
        ...

    def kernel_swap_points(self, artifact: Artifact) -> list[OpSite]:
        """Stages 2-4. Where a NKI kernel can be substituted."""
        ...

    def toolchain_stamp(self) -> dict[str, str]:
        """Full version capture for reproducibility. neuronx_cc especially."""
        ...
