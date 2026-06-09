#!/usr/bin/env python3
"""
Serve BERT embeddings through vLLM-Neuron.

Requires:
  1. our BERT encoder class registered (architecture 'BertModel'), AND
  2. the in-image runner patch applied (patch_runner_inimage.py) which adds
     the pooling/embed output path to NeuronModelRunner.

With both in place, llm.embed() returns real embeddings computed on Trainium.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL = os.environ.get("BERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_LEN = int(os.environ.get("MAX_LEN", "128"))


def register():
    from vllm import ModelRegistry
    from bert_neuron_impl import BertModel
    ModelRegistry.register_model("BertModel", BertModel)
    print("[bert-vllm] registered BertModel ->", BertModel)


def main():
    register()
    from vllm import LLM

    print(f"[bert-vllm] building LLM for {MODEL} (pooling, max_len={MAX_LEN})")
    t0 = time.time()
    llm = LLM(
        model=MODEL,
        runner="pooling",
        convert="embed",
        max_model_len=MAX_LEN,
        max_num_seqs=4,
        tensor_parallel_size=1,
        dtype="bfloat16",
    )
    print(f"[bert-vllm] engine ready in {time.time()-t0:.1f}s")

    prompts = [
        "Encoder benchmark sentence one",
        "second sentence for embedding",
        "the quick brown fox",
    ]
    t1 = time.time()
    out = llm.embed(prompts)
    dt = time.time() - t1
    for i, o in enumerate(out):
        e = o.outputs.embedding
        print(f"  prompt[{i}] dim={len(e)} first3={[round(float(v),4) for v in e[:3]]}")
    dims = {len(o.outputs.embedding) for o in out}
    assert len(dims) == 1, f"inconsistent dims {dims}"
    print(f"[bert-vllm] embedded {len(out)} prompts in {dt*1000:.1f} ms total")
    print(f"[bert-vllm] PASS — BERT embeddings served through vLLM-Neuron, dim={dims.pop()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[bert-vllm] FAIL: {type(e).__name__}: {e}")
        sys.exit(1)
