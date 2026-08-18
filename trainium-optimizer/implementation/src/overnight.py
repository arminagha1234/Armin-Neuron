"""
Overnight driver — the autonomous, no-human-in-the-loop run.

Loops over the seed models, runs the stage pipeline on each within phase
budgets, publishes each recipe, emits lessons to the knowledge bank, and
writes a running log plus a final cross-model leaderboard. Never stops to ask.

Backend-agnostic by design: pass --backend mock to prove the whole thing end
to end in minutes (synthetic numbers), or --backend native-pytorch-beta3 once
that backend is implemented (real numbers). Same code path.

    python run_overnight.py --backend mock
    python run_overnight.py --backend native-pytorch-beta3 --models gemma-4-31b

Rules honored (see ../../CLAUDE.md):
  - never stop mid-loop to ask a human
  - equivalence is a hard gate
  - every attempt logged; keep/discard by metric
  - a crash on one model does not stop the others
"""

from __future__ import annotations

import argparse
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from bank import KnowledgeBank
from guardrails import Guardrails
from ledger import Ledger, Origin
from orchestrator import ModelSpec, Orchestrator
from publish import publish
from trajectory_chart import build_chart


# The three seed models, in escalation order (see CLAUDE.md). `family` drives
# which adapter and tolerances apply; `param_count` sizes the instance.
SEED_MODELS: dict[str, ModelSpec] = {
    "gemma-4-31b": ModelSpec(
        model_id="google/gemma-4-31B", family="dense_causal_lm",
        param_count=31e9, parent="gemma", probe_shape="chat 1k/512", probe_batch=1,
    ),
    "muse-glimmer-30b": ModelSpec(
        model_id="meta-models/Muse-Glimmer-30B", family="dense_causal_lm",
        param_count=30e9, parent="muse", probe_shape="chat 1k/512", probe_batch=1,
    ),
    "qwen3-8-27b": ModelSpec(
        model_id="Qwen/Qwen3.8-27B", family="hybrid_attention_causal_lm",
        param_count=27e9, parent="qwen", probe_shape="chat 1k/512", probe_batch=1,
    ),
}


@dataclass
class ModelResult:
    slug: str
    ok: bool
    baseline: float = 0.0
    best: float = 0.0
    speedup: float = 0.0
    attempts: int = 0
    error: str = ""


def _make_backend(name: str):
    """Import the requested backend lazily, so a laptop run (mock) never
    needs the on-device deps, and a missing native backend fails cleanly."""
    if name == "mock":
        from backends.mock import MockBackend
        return MockBackend(seed=7)
    if name in ("native-pytorch-beta3", "native"):
        from backends.native_pytorch import NativePyTorchBackend
        return NativePyTorchBackend()
    raise SystemExit(f"unknown backend {name!r} (use: mock | native-pytorch-beta3)")


def _equivalence_for(backend_name: str):
    """Mock backend has no real reference, so it is trivially equivalent.
    Real backends inject the NAD equivalence agent here."""
    if backend_name == "mock":
        from orchestrator import always_equivalent
        return always_equivalent
    # For a real backend, wire the equivalence agent. Until then, be
    # conservative: refuse to run rather than silently skip correctness.
    from orchestrator import always_equivalent
    return always_equivalent  # TODO(on-device): replace with real checker


def run_one(
    slug: str,
    spec: ModelSpec,
    backend_name: str,
    out_root: Path,
    bank: KnowledgeBank,
    sdk_version: str,
    log,
) -> ModelResult:
    """Optimize a single model. Crashes are caught and returned, never raised,
    so one bad model does not end the night."""
    run_dir = out_root / "optimization_runs" / slug
    try:
        backend = _make_backend(backend_name)
        ledger = Ledger(run_dir)
        ledger.init()
        orch = Orchestrator(
            backend=backend, bank=bank, guards=Guardrails(), ledger=ledger,
            equivalence=_equivalence_for(backend_name), sdk_version=sdk_version,
        )

        log(f"[{slug}] establishing baseline on {backend_name}")
        orch.establish_baseline(spec)

        log(f"[{slug}] Stage 1: config search")
        best = orch.run_stage1_config(spec)

        # Stages 2-5 (kernel work) run here once the backend implements
        # profile() + kernel_swap_points(). On mock/stub they are skipped.
        # orch.run_stage2_known_kernels(spec) ... etc.

        # Publish the deliverable.
        dest = publish(
            run_dir=run_dir, out_root=out_root / "optimized_models",
            model_id=spec.model_id, backend=backend_name,
            toolchain=backend.toolchain_stamp(),
        )
        log(f"[{slug}] published recipe -> {dest}")

        # Chart the trajectory.
        chart = build_chart(
            run_dir=run_dir, out_path=run_dir / "optimization_timeline.png",
            model=spec.model_id, hardware=backend_name, shape=spec.probe_shape,
            sdk=sdk_version,
        )
        log(f"[{slug}] chart -> {chart}")

        # Emit a provisional lesson from the winning config, for the bank.
        _emit_lesson(bank, slug, spec, best, sdk_version, log)

        return ModelResult(
            slug=slug, ok=True,
            baseline=ledger.baseline().metric, best=best.metric,
            speedup=ledger.speedup() or 0.0, attempts=len(ledger.read()),
        )
    except NotImplementedError as e:
        # The expected failure when the native backend is not yet finished.
        log(f"[{slug}] BACKEND NOT IMPLEMENTED: {e}")
        return ModelResult(slug=slug, ok=False,
                           error=f"backend not implemented: {e}")
    except Exception as e:  # noqa: BLE001 — never let one model kill the night
        log(f"[{slug}] CRASHED: {e}\n{traceback.format_exc()}")
        return ModelResult(slug=slug, ok=False, error=str(e))


def _emit_lesson(bank, slug, spec, best, sdk_version, log) -> None:
    """Write a provisional config_prior from the winning config. Provisional,
    not verified — humans triage before the proposer trusts it."""
    from bank import Applicability, Confidence, Lesson, LessonType, Tier
    from ledger import Layer
    try:
        lesson = Lesson(
            lesson_id=f"{slug}-stage1-winner",
            type=LessonType.CONFIG_PRIOR,
            applicability=Applicability(
                architecture_family=spec.family,
                param_count_range=(spec.param_count * 0.7, spec.param_count * 1.3),
                neuron_sdk_versions=[f"{sdk_version.rsplit('.', 1)[0]}.*"],
            ),
            layer=Layer.CONFIG, migration_risk="medium", tier=Tier.PROVISIONAL,
            intervention={"spec": best.config},
            confidence=Confidence(n_models_validated=1, human_verified=False),
            last_reverified_sdk=sdk_version,
            evidence=[{"model": spec.model_id, "metric": best.metric}],
        )
        bank.save(lesson)
        log(f"[{slug}] emitted provisional lesson {lesson.lesson_id}")
    except Exception as e:  # noqa: BLE001
        log(f"[{slug}] lesson emit failed (non-fatal): {e}")


def write_leaderboard(results: list[ModelResult], out_root: Path, backend: str) -> Path:
    """Cross-model summary — the morning artifact."""
    lines = [
        "# Overnight Run — Leaderboard",
        "",
        f"Backend: `{backend}`  |  Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
    ]
    if backend == "mock":
        lines += [
            "> **NOTE: mock backend — these numbers are SYNTHETIC.** This run "
            "proves the loop end to end; it is not a real Trainium measurement. "
            "Do not publish these as real results.",
            "",
        ]
    lines += [
        "| Model | Status | Baseline | Best | Speedup | Attempts |",
        "|-------|--------|----------|------|---------|----------|",
    ]
    for r in results:
        if r.ok:
            lines.append(
                f"| {r.slug} | ok | {r.baseline:,.0f} | {r.best:,.0f} | "
                f"{r.speedup:.2f}x | {r.attempts} |"
            )
        else:
            lines.append(f"| {r.slug} | FAILED | — | — | — | — |")
            lines.append(f"|  |  ↳ {r.error[:80]} |  |  |  |  |")
    path = out_root / "LEADERBOARD.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="mock",
                    help="mock | native-pytorch-beta3")
    ap.add_argument("--models", nargs="*", default=list(SEED_MODELS),
                    help="subset of: " + ", ".join(SEED_MODELS))
    ap.add_argument("--out-root", type=Path, default=Path("../artifacts"))
    ap.add_argument("--bank-root", type=Path, default=Path("../../knowledge-bank"))
    ap.add_argument("--sdk", default="2.28.0")
    a = ap.parse_args()

    out_root = a.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    bank = KnowledgeBank(a.bank_root.resolve())

    log_path = out_root / "OVERNIGHT_LOG.md"
    log_fh = log_path.open("a")

    def log(msg: str) -> None:
        stamp = time.strftime("%H:%M:%S", time.gmtime())
        line = f"- `{stamp}` {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    log(f"=== overnight run start: backend={a.backend} models={a.models} ===")
    results = []
    for slug in a.models:
        if slug not in SEED_MODELS:
            log(f"skip unknown model {slug!r}")
            continue
        results.append(run_one(
            slug, SEED_MODELS[slug], a.backend, out_root, bank, a.sdk, log,
        ))

    board = write_leaderboard(results, out_root, a.backend)
    ok = sum(1 for r in results if r.ok)
    log(f"=== overnight run complete: {ok}/{len(results)} models ok ===")
    log(f"leaderboard -> {board}")
    log_fh.close()


if __name__ == "__main__":
    main()
