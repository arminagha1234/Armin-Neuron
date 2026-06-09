#!/usr/bin/env python3
"""
In-image surgical patch of vllm_neuron NeuronModelRunner to add a pooling/embed
output path. Run INSIDE the vllm_neuron container (after restoring .orig).

vLLM core's step_with_batch_queue does, for pooling models:
    if self.is_pooling_model or not model_executed:
        future = exec_future          # uses execute_model() return DIRECTLY
    else:
        future = sample_tokens(...)    # generate path

The stock neuron execute_model always stores state and returns None (generate
design), so for pooling future.result() is None -> "unexpected error".

Fix: three edits —
  1. get_supported_tasks(): advertise embed/encode when is_pooling_model.
  2. __init__: set is_pooling_model from runner/convert config (not hardcoded).
  3. execute_model(): when is_pooling_model, build a ModelRunnerOutput with
     pooler_output from model_output_tensor and RETURN it (instead of storing
     state + returning None).
"""
F = "/opt/conda/lib/python3.12/site-packages/vllm_neuron/vllm/worker/neuron_model_runner.py"
MARKER = "# BERT_ENCODER_POOLING_PATCH"

src = open(F).read()
if MARKER in src:
    print("ALREADY PATCHED — restore .orig first"); raise SystemExit(0)

# ---- Patch 1: get_supported_tasks -> add embed/encode ----------------------
pat1 = '''        return ("generate",)'''
rep1 = '''        # BERT_ENCODER_POOLING_PATCH: advertise embedding tasks for pooling models
        if getattr(self, "is_pooling_model", False):
            return ("encode", "embed")
        return ("generate",)'''
assert pat1 in src, "get_supported_tasks return not found"
src = src.replace(pat1, rep1, 1)

# ---- Patch 2: is_pooling_model from config ---------------------------------
pat2 = "        self.is_pooling_model = False"
rep2 = '''        # BERT_ENCODER_POOLING_PATCH: detect pooling/embed models from config
        try:
            _runner = getattr(model_config, "runner_type", None) or getattr(
                model_config, "runner", None
            )
            _conv = getattr(model_config, "convert_type", None) or getattr(
                model_config, "convert", None
            )
            self.is_pooling_model = (_runner == "pooling") or (_conv == "embed")
        except Exception:
            self.is_pooling_model = False'''
assert pat2 in src, "is_pooling_model init not found"
src = src.replace(pat2, rep2, 1)

# ---- Patch 3: execute_model returns pooling output -------------------------
anchor = '''        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            model_output_tensor,
            spec_decode_metadata,
            positions,
            logits_indices,
            aux_hidden_states,
            input_ids,
            attn_metadata,
        )

        self.kv_connector_output = kv_connector_output
        return None'''
inject = '''        # BERT_ENCODER_POOLING_PATCH: pooling/embed models consume execute_model's
        # return value directly (vLLM core never calls sample_tokens for them).
        # model_output_tensor is our encoder's embedding (already on CPU).
        if getattr(self, "is_pooling_model", False):
            import torch as _torch
            from vllm.v1.outputs import ModelRunnerOutput as _MRO
            _req_ids = list(self.input_batch.req_ids)
            _n = len(_req_ids)
            _t = model_output_tensor
            if hasattr(_t, "detach"):
                _t = _t.detach()
            # Neuron lazy runtime: move to CPU at the SAME dtype first, then
            # cast (a dtype change during device->host copy trips a dtype
            # assertion in the Neuron runtime).
            try:
                _t = _t.cpu()
            except Exception:
                pass
            _t = _t.to(_torch.float32)
            if _t.dim() == 1:
                _rows = [_t for _ in range(_n)]
            elif _t.shape[0] == _n:
                _rows = [_t[_i] for _i in range(_n)]
            else:
                _rows = [_t.reshape(-1)[: _t.shape[-1]] for _ in range(_n)]
            return _MRO(
                req_ids=_req_ids,
                req_id_to_index={_r: _i for _i, _r in enumerate(_req_ids)},
                sampled_token_ids=[[] for _ in range(_n)],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=list(_rows),
                kv_connector_output=kv_connector_output,
            )

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            model_output_tensor,
            spec_decode_metadata,
            positions,
            logits_indices,
            aux_hidden_states,
            input_ids,
            attn_metadata,
        )

        self.kv_connector_output = kv_connector_output
        return None'''
assert anchor in src, "execute_model state-store anchor not found"
src = src.replace(anchor, inject, 1)

open(F, "w").write(src)
print("PATCHED OK (execute_model pooling return)")
