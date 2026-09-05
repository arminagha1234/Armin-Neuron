#!/usr/bin/env python3
"""Generate a text-only `vllm_neuron/model/qwen3` package for dense Qwen3-8B.

vllm_neuron 0.21 registers only 4 architectures (Eagle3Llama, GptOss, Llama,
Qwen3VL) -- no dense Qwen3. But `qwen3_vl` already contains a correct Qwen3
text decoder: QK-norm is wired through NF.qkv_proj in both prefill
(`qk_norm_pre_rope_*`) and decode (`rmsnorm_QK_pre_rope_W_*`). So rather than
take llama3 and add QK-norm to several attention call sites, we take qwen3_vl
and strip the vision half. Every edit is asserted; nothing is silently skipped.

Patches applied to the copy:
  1. HF_TEXT_PREFIX "model.language_model" -> "model"  (dense checkpoint layout)
  2. drop the vision/MRoPE Protocol bases from the top-level class
  3. do not construct the vision encoder
  4. do not load vision weights
  5. delete get_mrope_input_positions + build_vision_synthetic_inputs so the
     runtime_checkable Protocols no longer match -> runner sends 1D positions
  6. forward(): rotary_position_ids becomes optional, defaults to positions
  7. from_configs(): accept a FLAT dense HF config and synthesize rope_parameters
"""
import os
import sys
import ast
import shutil

FAILS = []


def edit(text, old, new, label, count=1):
    if old not in text:
        FAILS.append(f"{label}: PATTERN NOT FOUND")
        return text
    n = text.count(old)
    if n != count:
        FAILS.append(f"{label}: expected {count} occurrence(s), found {n}")
        return text
    print(f"  [ok] {label}")
    return text.replace(old, new, count)


def main():
    model_dir = None
    for base in sys.path:
        cand = os.path.join(base, "vllm_neuron", "model")
        if os.path.isdir(cand):
            model_dir = cand
            break
    assert model_dir, "vllm_neuron/model not found on sys.path"
    src = os.path.join(model_dir, "qwen3_vl")
    dst = os.path.join(model_dir, "qwen3")
    assert os.path.isdir(src), f"missing {src}"
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"[mk_qwen3] copied qwen3_vl -> {dst}")

    mp = os.path.join(dst, "model_bf16.py")
    s = open(mp).read()

    # 1. dense checkpoint prefix
    s = edit(s, 'HF_TEXT_PREFIX = "model.language_model"',
             'HF_TEXT_PREFIX = "model"  # dense Qwen3: no .language_model nesting',
             "prefix -> model")

    # 2. drop vision/MRoPE protocol bases
    s = edit(s,
             "class Qwen3VLForConditionalGeneration(nn.Module, SupportsVisionWarmup, SupportsMRoPE):",
             "class Qwen3ForCausalLM(nn.Module):",
             "top-level class decl")

    # 3. no vision encoder
    s = edit(s,
             "        self._vision_captures: tuple[torch.Tensor, ...] = ()\n"
             "        self.visual = Qwen3VLVisionModel(config.vision_config, dtype=torch.bfloat16)\n"
             "        vision_config = config.vision_config\n",
             "        # text-only port: no vision encoder\n"
             "        self._vision_captures: tuple[torch.Tensor, ...] = ()\n"
             "        self.visual = None\n",
             "skip vision encoder construction")

    # 4. no vision weight load
    s = edit(s,
             "        # Load vision encoder weights (uses vision TP group, not text TP).\n"
             "        self.visual.load_weights(checkpoint_path, device=\"cpu\", cpu_mode=True)",
             "        # text-only port: no vision weights to load",
             "skip vision weight load")

    # 6. rotary_position_ids optional
    s = edit(s,
             "        input_ids: torch.LongTensor,\n"
             "        positions: torch.Tensor,\n"
             "        rotary_position_ids: torch.Tensor,\n"
             "        attn_metadata: object | None = None,",
             "        input_ids: torch.LongTensor,\n"
             "        positions: torch.Tensor,\n"
             "        rotary_position_ids: torch.Tensor | None = None,\n"
             "        attn_metadata: object | None = None,",
             "forward: rotary_position_ids optional (both forwards)", count=2)
    s = edit(s,
             "        positions = positions.to(torch.int32)\n"
             "\n"
             "        first_layer_name = \"layers.0.self_attn\"",
             "        positions = positions.to(torch.int32)\n"
             "        if rotary_position_ids is None:\n"
             "            # Text-only: M-RoPE degenerates to standard RoPE because all\n"
             "            # three sections carry identical positions, so the interleave\n"
             "            # is a no-op. The rotary module expands 1D [T] internally.\n"
             "            rotary_position_ids = positions\n"
             "\n"
             "        first_layer_name = \"layers.0.self_attn\"",
             "forward: default rotary ids to positions")

    # 7. from_configs on a FLAT dense config
    s = edit(s,
             "        config = Qwen3VLConfig.from_configs(\n"
             "            hf_config,\n"
             "            text_neuron_config=text_neuron_config,\n"
             "            vision_neuron_config=vision_neuron_config,\n"
             "        )\n"
             "        return cls(config)",
             "        config = _text_only_config(hf_config, text_neuron_config)\n"
             "        return cls(config)",
             "from_configs -> flat dense config")
    s = edit(s,
             "    @classmethod\n"
             "    def from_configs(\n"
             "        cls,\n"
             "        hf_config: PretrainedConfig,\n"
             "        text_neuron_config: NeuronConfig,\n"
             "        vision_neuron_config: VisionNeuronConfig,\n"
             "    ):",
             "    @classmethod\n"
             "    def from_configs(\n"
             "        cls,\n"
             "        hf_config: PretrainedConfig,\n"
             "        neuron_config: NeuronConfig | None = None,\n"
             "        text_neuron_config: NeuronConfig | None = None,\n"
             "        vision_neuron_config: VisionNeuronConfig | None = None,\n"
             "        **kwargs,\n"
             "    ):\n"
             "        text_neuron_config = neuron_config or text_neuron_config",
             "from_configs signature")

    # helper appended at module scope
    s += '''

# ---------------------------------------------------------------------------
# Text-only config adapter for dense Qwen3 (no nested text_config/vision_config)
# ---------------------------------------------------------------------------
class _TextOnlyConfigShim:
    """Stands in for Qwen3VLConfig: only `.text_config` is ever read once the
    vision encoder is gone."""

    def __init__(self, text_config):
        self.text_config = text_config
        self.vision_config = None


def _mrope_sections(head_dim: int) -> list[int]:
    """M-RoPE sections must sum to head_dim//2 (the inv_freq length). For a
    text-only model the split is arbitrary -- all three sections receive the
    same positions -- but the sum must be exact."""
    n = head_dim // 2
    third = n // 3
    return [n - 2 * third, third, third]


def _text_only_config(hf_config, neuron_config):
    """Build a Qwen3VLTextConfig from a FLAT dense Qwen3 HF config.

    `_from_hf_sub_config` filters to dataclass fields, so a flat config works
    directly. The one missing piece is `rope_parameters`, which dense Qwen3
    spells as a top-level `rope_theta`.
    """
    if isinstance(hf_config, PretrainedConfig):
        d = hf_config.to_dict()
    elif isinstance(hf_config, dict):
        d = dict(hf_config)
    else:
        raise TypeError(f"unsupported hf_config type: {type(hf_config)}")
    # a nested text_config still wins if one is present
    if isinstance(d.get("text_config"), dict):
        inner = d["text_config"]
        d = {**d, **inner}
    head_dim = d.get("head_dim") or (
        d["hidden_size"] // d["num_attention_heads"]
    )
    d["head_dim"] = head_dim
    if "rope_parameters" not in d:
        d["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": float(d.get("rope_theta", 1000000.0)),
            "mrope_interleaved": True,
            "mrope_section": _mrope_sections(head_dim),
        }
    sec = d["rope_parameters"].get("mrope_section") or _mrope_sections(head_dim)
    assert sum(sec) == head_dim // 2, (
        f"mrope_section {sec} must sum to head_dim//2 ={head_dim // 2}"
    )
    d["rope_parameters"]["mrope_section"] = sec
    tc = Qwen3VLTextConfig.from_hf_config(d, neuron_config)
    print(
        f"[qwen3] text-only config: layers={tc.num_hidden_layers} "
        f"hidden={tc.hidden_size} heads={tc.num_attention_heads} "
        f"kv={tc.num_key_value_heads} head_dim={tc.head_dim} "
        f"vocab={tc.vocab_size} rope_theta={tc.rope_parameters['rope_theta']} "
        f"tie={tc.tie_word_embeddings}",
        flush=True,
    )
    return _TextOnlyConfigShim(tc)
'''

    # 5. remove the two Protocol-satisfying methods by renaming them
    tree = ast.parse(s)
    for name in ("get_mrope_input_positions", "build_vision_synthetic_inputs",
                 "embed_multimodal"):
        if f"def {name}(" in s:
            s = s.replace(f"    def {name}(", f"    def _disabled_{name}(", 1)
            print(f"  [ok] disabled Protocol method {name}")
        else:
            print(f"  [--] {name} absent (fine)")

    open(mp, "w").write(s)

    # factory
    fp = os.path.join(dst, "factory.py")
    open(fp, "w").write('''# SPDX-License-Identifier: Apache-2.0
"""Factory for dense text-only Qwen3 (Qwen3ForCausalLM)."""
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen3ForCausalLM(nn.Module):
    """Dense Qwen3 text decoder. Delegates to the qwen3_vl text backbone with
    the vision half removed (see model_bf16.py header)."""

    def __init__(
        self,
        hf_config: PretrainedConfig = None,
        neuron_config: NeuronConfig | None = None,
        *,
        text_neuron_config: NeuronConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._model = (
            self._select_implementation(hf_config, neuron_config or text_neuron_config)
            if hf_config is not None
            else None
        )

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
        *,
        text_neuron_config: NeuronConfig | None = None,
        **kwargs,
    ) -> nn.Module:
        return cls._select_implementation(
            hf_config, neuron_config or text_neuron_config
        )

    @classmethod
    def _select_implementation(cls, hf_config, neuron_config) -> nn.Module:
        from .model_bf16 import Qwen3ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)
''')
    print("  [ok] wrote factory.py")

    open(os.path.join(dst, "__init__.py"), "w").write(
        "# SPDX-License-Identifier: Apache-2.0\n"
        "from .factory import Qwen3ForCausalLM\n\n"
        '__all__ = ["Qwen3ForCausalLM"]\n'
    )
    print("  [ok] wrote __init__.py")

    # registry
    reg = os.path.join(model_dir, "registry.py")
    r = open(reg).read()
    if "from .qwen3 import Qwen3ForCausalLM" not in r:
        lines = r.splitlines()
        idx = max(i for i, l in enumerate(lines) if l.startswith("from ."))
        lines.insert(idx + 1, "from .qwen3 import Qwen3ForCausalLM")
        r = "\n".join(lines) + "\n"
    if '("Qwen3ForCausalLM"' not in r:
        anchor = "    models = ["
        assert anchor in r, "registry anchor missing"
        r = r.replace(
            anchor,
            anchor + '\n        ("Qwen3ForCausalLM", Qwen3ForCausalLM),\n',
            1,
        )
    bak = reg + ".bak_qwen3"
    if not os.path.exists(bak):
        shutil.copy2(reg, bak)
    open(reg, "w").write(r)
    print("  [ok] registered Qwen3ForCausalLM in registry.py")

    # syntax gate on everything we touched
    for f in (mp, fp, reg, os.path.join(dst, "__init__.py")):
        ast.parse(open(f).read())
    print("  [ok] all touched files parse")

    if FAILS:
        print("\n[mk_qwen3] FAILED PATCHES:")
        for f in FAILS:
            print("   -", f)
        sys.exit(1)
    print("[mk_qwen3] SUCCESS - all patches applied")


if __name__ == "__main__":
    main()
