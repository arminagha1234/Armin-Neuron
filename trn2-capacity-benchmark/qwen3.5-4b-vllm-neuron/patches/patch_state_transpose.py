#!/usr/bin/env python3
"""Qwen3.5 DeltaNet decode: store the recurrent state transposed.

WHY. The decode recurrence writes two matrix-vector products as broadcast
multiply + reduce over dim=-2:

    kv_mem  = (new_state * k_t.unsqueeze(-1)).sum(dim=-2)
    out_one = (new_state * q_t.unsqueeze(-1)).sum(dim=-2)

new_state is [B, num_v_heads, head_k_dim, head_v_dim] fp32 = 2.1 MB per
sequence per layer. Two costs follow:
  1. each broadcast materialises a FULL fp32 copy of the state;
  2. dim=-2 maps to the PARTITION axis on trn2, so the reduce is done through
     the PE array as a transpose.
The device profile of this graph attributes 86% of decode FLOPs (796 of 924
GFLOP) to transposes, with DMA 100.4% active, arithmetic intensity 9.6, and
12.0 GB of HBM read per token against ~2.2 GB of weights.

WHAT. Store the state as [B, num_v_heads, head_v_dim, head_k_dim] instead.
Then both reductions are over the LAST (free) axis, which the vector engine
does directly:

    kv_mem  = (st * k_t.unsqueeze(-2)).sum(dim=-1)
    st      = st + delta.unsqueeze(-1) * k_t.unsqueeze(-2)
    out_one = (st * q_t.unsqueeze(-2)).sum(dim=-1)

Algebraically identical: st[v,k] == state[k,v], so sum_k st[v,k]*k_t[k] ==
sum_k state[k,v]*k_t[k].

head_k_dim == head_v_dim == 128, and the state is kept FLAT in k_cache with an
explicit reshape on read, so the layout swap needs no cache resize. Prefill
writes the state transposed once per request; decode then reads and writes it
transposed, so no per-token transpose survives. Prefill pays 1 transpose per
request instead of decode paying 2 per layer per token (x50 tokens x24 layers).

Gated by QWEN35_STATE_T=1. Default off => byte-identical graph to baseline.
Idempotent; backs up once; py_compiles and auto-reverts on syntax error.
"""
import os, py_compile, shutil, sys

M = "/opt/conda/lib/python3.13/site-packages/vllm_neuron/model/qwen3_5/model_bf16.py"

FLAG = '''
# --- QWEN35_STATE_T: store DeltaNet recurrent state as [B,H,v_dim,k_dim] so the
# decode reductions run over the free axis instead of the partition axis. ---
_STATE_T = __import__("os").environ.get("QWEN35_STATE_T", "0") == "1"
'''

READ_OLD = """        return flat.reshape(
            batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim
        ).to(self.dtype).float()"""

READ_NEW = """        if _STATE_T:
            # transposed layout: [B, H, head_v_dim, head_k_dim]
            return flat.reshape(
                batch_size, self.num_v_heads, self.head_v_dim, self.head_k_dim
            ).to(self.dtype).float()
        return flat.reshape(
            batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim
        ).to(self.dtype).float()"""

PF_OLD = """        rec_state_size = self.num_v_heads * self.head_k_dim * self.head_v_dim
        self._write_recurrent_state_to_cache(final_state, batch_size, rec_state_size)"""

PF_NEW = """        rec_state_size = self.num_v_heads * self.head_k_dim * self.head_v_dim
        if _STATE_T:
            # one transpose per request so decode needs none per token
            final_state = final_state.transpose(-1, -2)
        self._write_recurrent_state_to_cache(final_state, batch_size, rec_state_size)"""

DEC_OLD = """        new_state = recurrent_state * g_t
        # kv_mem[b, h, v_dim] = sum_k new_state[b, h, k_dim, v_dim] * k_t[b, h, k_dim]
        kv_mem = (new_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        # outer product update: new_state += k_t \u2297 delta
        new_state = new_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # output = sum_k new_state * q_t  \u2192  [B, H, head_v_dim]
        out_one = (new_state * q_t.unsqueeze(-1)).sum(dim=-2)"""

DEC_NEW = """        if _STATE_T:
            # recurrent_state is [B, H, head_v_dim, head_k_dim]; every reduce
            # below is over the LAST axis => vector engine, no PE transpose.
            new_state = recurrent_state * g_t
            kv_mem = (new_state * k_t.unsqueeze(-2)).sum(dim=-1)
            delta = (v_t - kv_mem) * beta_t
            new_state = new_state + delta.unsqueeze(-1) * k_t.unsqueeze(-2)
            out_one = (new_state * q_t.unsqueeze(-2)).sum(dim=-1)
        else:
            new_state = recurrent_state * g_t
            # kv_mem[b,h,v] = sum_k new_state[b,h,k,v] * k_t[b,h,k]
            kv_mem = (new_state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            new_state = new_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            out_one = (new_state * q_t.unsqueeze(-1)).sum(dim=-2)"""

src = open(M).read()
if "_STATE_T" in src:
    print("STATE_T_ALREADY_APPLIED"); sys.exit(0)

# the module imports os as `_os`? detect and normalise the flag accordingly
if "import os as _os" not in src:
    if "\nimport os\n" in src:
        flag = FLAG.replace("_os.environ", "os.environ")
    else:
        flag = FLAG.replace("_os.environ", "__import__('os').environ")
else:
    flag = FLAG

missing = [n for n, o in (("read", READ_OLD), ("prefill", PF_OLD), ("decode", DEC_OLD)) if o not in src]
if missing:
    print("STATE_T_ANCHOR_NOT_FOUND:", missing); sys.exit(2)
for n, o in (("read", READ_OLD), ("prefill", PF_OLD), ("decode", DEC_OLD)):
    if src.count(o) != 1:
        print(f"STATE_T_ANCHOR_AMBIGUOUS {n}: {src.count(o)} matches"); sys.exit(2)

bak = M + ".pre_state_t"
if not os.path.exists(bak):
    shutil.copy2(M, bak)

# insert the flag after the last module-level import
lines = src.splitlines()
last_imp = max(i for i, l in enumerate(lines)
               if l.startswith("import ") or l.startswith("from "))
lines.insert(last_imp + 1, flag)
src = "\n".join(lines)
src = src.replace(READ_OLD, READ_NEW, 1)
src = src.replace(PF_OLD, PF_NEW, 1)
src = src.replace(DEC_OLD, DEC_NEW, 1)
open(M, "w").write(src)
try:
    py_compile.compile(M, doraise=True)
    print(f"STATE_T_OK backup={bak}")
except py_compile.PyCompileError as e:
    shutil.copy2(bak, M)
    print("STATE_T_REVERTED syntax error:", e); sys.exit(3)
