"""
Tests for publish (recipe bundle) and that the native_pytorch stub is a
structurally valid Backend even though its methods raise.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backends.mock import MockBackend
from bank import Applicability, Confidence, KnowledgeBank, Lesson, LessonType, Tier
from guardrails import Guardrails
from ledger import Layer, Ledger
from orchestrator import ModelSpec, Orchestrator
from publish import publish, _slug


def _bank(root: Path) -> KnowledgeBank:
    bank = KnowledgeBank(root)
    bank.save(Lesson(
        lesson_id="p", type=LessonType.CONFIG_PRIOR,
        applicability=Applicability("dense_causal_lm", (20e9, 40e9),
                                    neuron_sdk_versions=["2.28.*"]),
        layer=Layer.CONFIG, migration_risk="medium", tier=Tier.VERIFIED,
        intervention={"spec": {"batching": "continuous", "attention_kernel": "flash"}},
        confidence=Confidence(3, 1, True), last_reverified_sdk="2.28.0",
    ))
    return bank


def test_publish_creates_deliverable(tmp_path: Path):
    run = tmp_path / "run"
    orch = Orchestrator(
        backend=MockBackend(seed=3), bank=_bank(tmp_path / "bank"),
        guards=Guardrails(), ledger=Ledger(run), sdk_version="2.28.0",
    )
    orch.ledger.init()
    orch.run_stage1_config(ModelSpec(
        model_id="google/gemma-4-31B", family="dense_causal_lm",
        param_count=31e9, parent="gemma",
    ))

    dest = publish(
        run_dir=run, out_root=tmp_path / "optimized_models",
        model_id="google/gemma-4-31B", backend="mock",
        toolchain=orch.backend.toolchain_stamp(),
        kernel_provenance=[{"op": "attention", "origin": "harvested",
                            "source": "nki-library@abc"}],
    )

    assert dest.name == "gemma-4-31b"
    assert (dest / "recipe.json").exists()
    assert (dest / "RECIPE.md").exists()
    assert (dest / "reproduce.sh").exists()
    assert (dest / "results.tsv").exists()          # evidence copied

    recipe = json.loads((dest / "recipe.json").read_text())
    assert recipe["model_id"] == "google/gemma-4-31B"
    assert recipe["speedup"] > 1.0
    assert recipe["best_metric"] > recipe["baseline_metric"]
    assert recipe["kernels"][0]["origin"] == "harvested"

    # reproduce.sh must be executable
    assert (dest / "reproduce.sh").stat().st_mode & 0o111


def test_slug():
    assert _slug("google/gemma-4-31B") == "gemma-4-31b"
    assert _slug("Qwen/Qwen3.8-27B") == "qwen3-8-27b"
    assert _slug("meta-models/Muse-Glimmer-30B") == "muse-glimmer-30b"


def test_publish_refuses_empty_run(tmp_path: Path):
    Ledger(tmp_path / "empty").init()
    with pytest.raises(SystemExit):
        publish(run_dir=tmp_path / "empty", out_root=tmp_path / "out",
                model_id="x/y", backend="mock", toolchain={})


def test_native_pytorch_is_structurally_a_backend():
    """The stub must satisfy the Backend protocol shape (methods present),
    even though they raise NotImplementedError when called."""
    from backends.base import Backend
    from backends.native_pytorch import NativePyTorchBackend

    be = NativePyTorchBackend()
    assert isinstance(be, Backend)          # runtime_checkable protocol
    assert be.name == "native-pytorch-beta3"
    # config_axes and toolchain_stamp are safe to call off-device
    assert "tp_degree" in be.config_axes()
    assert be.toolchain_stamp()["device_string"] == "neuron"
    # the on-device methods raise until implemented
    with pytest.raises(NotImplementedError):
        be.build_baseline("google/gemma-4-31B")
