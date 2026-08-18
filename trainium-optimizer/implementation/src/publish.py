"""
Publish — turn a completed optimization run into a deliverable recipe bundle.

An "optimized model" here is NOT new weights (we do not retrain). It is a
recipe: the winning config, the custom/harvested NKI kernels, the backend fork
diff, the measurements, the full toolchain stamp, and a reproduction script.
That bundle is what a human or a downstream service actually consumes.

Two output trees, deliberately separated:

  optimization_runs/<slug>/   the search TRACE — ledger, charts, per-candidate
                              profiles. The messy history. (process)

  optimized_models/<slug>/    the DELIVERABLE — best recipe, kernels, repro
                              script, measurements. The clean handoff. (product)

This module reads the former and writes the latter.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ledger import Ledger, Row


@dataclass
class Recipe:
    """The deliverable. Everything needed to reproduce the optimized model."""

    model_id: str
    backend: str
    baseline_metric: float
    best_metric: float
    speedup: float
    metric_label: str
    config: dict[str, Any]              # the winning config
    toolchain: dict[str, str]           # full SDK/compiler stamp — reproducibility
    kernels: list[dict[str, Any]] = field(default_factory=list)  # provenance per kernel
    measurements: dict[str, Any] = field(default_factory=dict)   # per-shape, per-batch
    generated_at: str = ""


def publish(
    run_dir: Path | str,
    out_root: Path | str,
    model_id: str,
    backend: str,
    toolchain: dict[str, str],
    metric_label: str = "tok/s",
    full_measurements: dict[str, Any] | None = None,
    kernel_provenance: list[dict[str, Any]] | None = None,
    fork_diff_path: Path | str | None = None,
) -> Path:
    """Write optimized_models/<slug>/ from a completed run.

    The slug is derived from the model id ("google/gemma-4-31B" -> "gemma-4-31b")
    so the folder name is filesystem-safe and stable.
    """
    import time

    led = Ledger(run_dir)
    base = led.baseline()
    best = led.incumbent()
    if base is None or best is None:
        raise SystemExit("run has no baseline or no incumbent — nothing to publish")

    slug = _slug(model_id)
    dest = Path(out_root) / slug
    dest.mkdir(parents=True, exist_ok=True)

    recipe = Recipe(
        model_id=model_id,
        backend=backend,
        baseline_metric=base.metric,
        best_metric=best.metric,
        speedup=round(best.metric / base.metric, 3) if base.metric else 0.0,
        metric_label=metric_label,
        config=_config_from_ledger(led),
        toolchain=toolchain,
        kernels=kernel_provenance or [],
        measurements=full_measurements or {},
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    # 1. recipe.json — the machine-readable deliverable
    (dest / "recipe.json").write_text(json.dumps(asdict(recipe), indent=2, default=str))

    # 2. reproduce.sh — the exact commands to rebuild this result
    (dest / "reproduce.sh").write_text(_reproduce_script(recipe))
    (dest / "reproduce.sh").chmod(0o755)

    # 3. copy the trajectory chart + ledger from the run (the evidence)
    for artifact in ("optimization_timeline.png", "trajectory.png", "results.tsv"):
        src = Path(run_dir) / artifact
        if src.exists():
            shutil.copy2(src, dest / artifact)

    # 4. the backend fork diff, if provided (the actual code changes)
    if fork_diff_path and Path(fork_diff_path).exists():
        shutil.copy2(fork_diff_path, dest / "backend.diff")

    # 5. a human-readable summary
    (dest / "RECIPE.md").write_text(_recipe_markdown(recipe))

    return dest


def _config_from_ledger(led: Ledger) -> dict[str, Any]:
    """Best-effort: the incumbent's config is not stored column-wise in the
    ledger (descriptions are prose), so real usage passes the config in. Here
    we expose what the ledger knows; the orchestrator should hand the actual
    incumbent config to publish() in production. Placeholder until wired."""
    best = led.incumbent()
    return {"_note": "pass incumbent.config to publish() in production",
            "best_metric": best.metric if best else None}


def _reproduce_script(r: Recipe) -> str:
    cfg = " ".join(f"--set {k}={v}" for k, v in r.config.items()
                   if not k.startswith("_"))
    return f"""#!/usr/bin/env bash
# Reproduce the optimized {r.model_id} recipe.
# Backend: {r.backend}
# Expected: {r.best_metric:.0f} {r.metric_label} ({r.speedup}x baseline)
# Toolchain at publish time: {json.dumps(r.toolchain)}
set -euo pipefail

# See ENVIRONMENT.md for backend setup (DLC pull, driver, venv).
python -m optimizer.apply \\
    --model {r.model_id} \\
    --backend {r.backend} \\
    {cfg}

# Then measure to confirm you land within tolerance of {r.best_metric:.0f} {r.metric_label}.
python -m optimizer.measure --model {r.model_id} --backend {r.backend} --all-shapes
"""


def _recipe_markdown(r: Recipe) -> str:
    kernels = "\n".join(
        f"- `{k.get('op', '?')}` — {k.get('origin', '?')}"
        + (f" (from {k.get('source')})" if k.get("source") else "")
        for k in r.kernels
    ) or "- (none — config-only recipe)"
    return f"""# Optimized Recipe: {r.model_id}

**{r.best_metric:,.0f} {r.metric_label}** — {r.speedup}x over baseline
({r.baseline_metric:,.0f} {r.metric_label}).

Backend: `{r.backend}`
Generated: {r.generated_at}

## Winning config

```json
{json.dumps(r.config, indent=2)}
```

## Kernels

{kernels}

## Toolchain (reproducibility)

```json
{json.dumps(r.toolchain, indent=2)}
```

## Reproduce

```bash
./reproduce.sh
```

See `results.tsv` for the full search trace and the trajectory chart for how
this recipe was reached.
"""


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace(".", "-").replace("_", "-")
