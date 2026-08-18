"""
Native PyTorch (TorchNeuron / Beta 3) backend — STUB.

This is the real backend the user wants: native PyTorch on Trainium, NOT XLA.
It is intentionally a stub with the integration points marked, because the real
implementation must run on a trn2 with the Beta 3 DLC — it cannot be written or
tested from a laptop.

Everything a real backend must satisfy is in ../base.py (the Backend Protocol).
This file documents HOW each method maps onto the Beta 3 native-PyTorch stack,
so whoever finishes it (Claude Code, on-device) has the exact patterns.

CRITICAL — Beta 3 patterns (from internal Neuron Beta 3 setup docs, a hard rule):
  - device string is torch.device("neuron")   NOT "privateuseone:N"  (Beta 2)
  - torch.compile(model, backend="neuron", dynamic=False)  — dynamic unsupported
  - distributed rendezvous: torchrun --rdzv_backend c10d
  - driver MUST be the Beta 3 one from runtime_artifacts/*.deb (public DLAMI
    driver is incompatible)
  - NEVER use Beta 2 patterns (privateuseone, backend="neuron" PG init)

See ENVIRONMENT.md for the DLC pull + driver install.
"""

from __future__ import annotations

from typing import Any

from backends.base import (
    Artifact,
    Measurements,
    Neff,
    OpSite,
    Profile,
)

# Set True only inside the Beta 3 DLC with a working `neuron` device. The
# import guard keeps this module importable on a laptop for type-checking and
# for the CLI to give a clean error instead of an ImportError.
_ON_DEVICE = False


class NativePyTorchBackend:
    """Native PyTorch (TorchNeuron, Beta 3) backend.

    STATUS: stub. Each method raises NotImplementedError with the exact
    integration note. Finish these on-device, in order:
      1. build_baseline + compile + measure  (enough to run Stage 1)
      2. profile                              (enables Stages 2-5)
      3. kernel_swap_points                   (enables kernel work)
    """

    name = "native-pytorch-beta3"

    def __init__(self, instance_type: str = "trn2.48xlarge",
                 core_count: int = 64) -> None:
        self.instance_type = instance_type
        self.core_count = core_count
        if not _ON_DEVICE:
            # Not fatal at import; fatal at first use. Lets the CLI print a
            # helpful "you are not on a Beta 3 device" message.
            self._device_ready = False
        else:
            self._device_ready = True

    # -- Stage 0 -------------------------------------------------------------

    def build_baseline(self, model_id: str) -> Artifact:
        """Load the HF model onto the neuron device in eager mode.

        Reference (Beta 3):
            from transformers import AutoModelForCausalLM
            import torch
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, attn_implementation="eager"
            ).to(torch.device("neuron"))

        For TP>1, use torchrun + init_process_group. NOTE the open question:
        cross-chip TP (>=4) is documented failing on Trn1 with
        'Failed to execute the device barrier 1'. Whether it works on Trn2 is
        UNTESTED and gates the whole native-PyTorch-as-primary decision.
        Run the TP=8 smoke test (see ENVIRONMENT.md) before trusting this path
        for the 30B seed models.
        """
        raise NotImplementedError(
            "build_baseline: load HF model to torch.device('neuron') in eager "
            "mode. See docstring for the Beta 3 pattern."
        )

    # -- Stage 1 -------------------------------------------------------------

    def config_axes(self) -> dict[str, list[Any]]:
        """Config knobs meaningful for native PyTorch.

        Note this differs from vLLM-Neuron's axes: no continuous batching or
        paged attention here (native PyTorch has no serving scheduler). The
        axes are mostly compile + parallelism + dtype.
        """
        return {
            "tp_degree": [1, 2, 4, 8],       # 16/32 need cross-chip; verify first
            "weights_dtype": ["bf16", "fp32"],   # fp8 not in Beta 3 (planned pre-GA)
            "compile_mode": ["eager", "compile-default"],  # torch.compile backend=neuron
            "attn_implementation": ["eager", "sdpa"],      # sdpa = native FlashAttention
        }

    def apply_config(self, artifact: Artifact, config: dict[str, Any]) -> Artifact:
        return Artifact(model_id=artifact.model_id, backend=self.name,
                        config={**artifact.config, **config})

    def compile(self, artifact: Artifact) -> Neff:
        """Eager mode: no compile (return compile_seconds=0). compile-default
        mode: torch.compile(model, backend='neuron', dynamic=False) and time
        the first forward (which triggers the NEFF build).

        Beta 3 has a persistent NEFF cache on by default, so a repeat of the
        same config compiles ~instantly — the ledger should reflect that
        (cheap re-runs of seen configs).
        """
        raise NotImplementedError(
            "compile: eager -> 0s; compile-default -> torch.compile(backend="
            "'neuron', dynamic=False), time the first forward."
        )

    def measure(self, neff: Neff, shape: str, batch: int) -> Measurements:
        """Run the shape/batch, time it, read HBM peak.

        MUST measure HBM at FULL sequence occupancy (end of generation), not
        step 0 — see guardrails. For the stress shape (64k/64k) peak HBM is at
        token 65,536, and that is where OOM actually happens.

        Report p50 AND p99 over >= 10 measured iterations after >= 3 warmup.
        """
        raise NotImplementedError(
            "measure: run shape@batch, time p50/p99 over >=10 iters, read peak "
            "HBM at full KV occupancy."
        )

    # -- Stages 2-5 ----------------------------------------------------------

    def profile(self, neff: Neff, shape: str) -> Profile:
        """Capture a Neuron profile and classify the bottleneck.

        Delegate to the neuron-nki-profile-analysis-agent / neuron-profile
        tooling. Populate engine_utilization (PE, DMA, CC) and set `bottleneck`
        to one of compute_bound | dma_bound | collective_bound — that string is
        what the bank's symptom query keys on.
        """
        raise NotImplementedError(
            "profile: capture NEFF+NTFF, parse to op_sites + engine util + "
            "bottleneck classification."
        )

    def kernel_swap_points(self, artifact: Artifact) -> list[OpSite]:
        """Where a NKI kernel can be substituted in the native-PyTorch graph.

        For torch.compile(backend='neuron'), custom kernels are registered as
        custom ops. This enumerates the hot ops (attention, rmsnorm, moe, ...)
        that a harvested/borrowed/invented kernel could replace.
        """
        raise NotImplementedError(
            "kernel_swap_points: enumerate substitutable ops in the compiled "
            "graph."
        )

    # -- reproducibility -----------------------------------------------------

    def toolchain_stamp(self) -> dict[str, str]:
        """Full version capture. On-device, read the actual installed versions;
        the values below are the Beta 3 expected set from the steering file."""
        return {
            "backend": self.name,
            "stack": "native-pytorch-beta3",
            "torch": "2.11.0+cpu",
            "torch_neuronx": "2.11.3.0.1254+",
            "neuronx_cc": "2.0.253257.0a0+",
            "nki": "0.4.0b4+",
            "device_string": "neuron",
            "instance_type": self.instance_type,
            "_note": "read real versions with pip show on-device; these are "
                     "the Beta 3 expected values from internal Neuron Beta 3 setup docs",
        }
