#!/usr/bin/env python3
"""
Standalone validation of the Triton Python backend code without needing
tritonserver. Imports the same model.py that Triton would load and exercises
its initialize() / execute() lifecycle against a mocked Triton API.

Run inside the native_beta3 container:
    python3 /tmp/standalone_test.py
"""
import os
import sys
import time
import types

import numpy as np

# === Mock pb_utils so we can run model.py outside Triton ===================

class _MockTensor:
    def __init__(self, name, np_arr):
        self.name = name
        self._a = np_arr
    def as_numpy(self):
        return self._a

class _MockResp:
    def __init__(self, output_tensors):
        self.outputs = output_tensors

class _MockLogger:
    @staticmethod
    def log_info(msg):
        print(msg, flush=True)

class _MockTensorFactory:
    def __init__(self, name, arr):
        self.name = name
        self.arr = arr

mock_pb = types.SimpleNamespace(
    Logger=_MockLogger,
    Tensor=_MockTensorFactory,
    InferenceResponse=_MockResp,
    get_input_tensor_by_name=lambda req, name: req[name],
)
sys.modules["triton_python_backend_utils"] = mock_pb

# === Load the actual model.py ==============================================

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PY = os.path.join(ROOT, "model_repo", "bert_embed", "1")
# When invoked inside the container at /tmp/standalone_test.py, model_repo is
# also at /tmp/model_repo
if not os.path.isdir(MODEL_PY):
    MODEL_PY = "/tmp/model_repo/bert_embed/1"
sys.path.insert(0, MODEL_PY)
from model import TritonPythonModel  # noqa


def main():
    print("=== initializing model (this includes torch.compile per bucket) ===")
    t0 = time.time()
    m = TritonPythonModel()
    m.initialize({})
    print(f"=== init done in {time.time() - t0:.1f}s ===\n")

    # Build a request matching Triton's STRING tensor shape: numpy bytes
    prompts = [
        "Encoder benchmark sentence via Triton",
        "second sentence for embedding",
        "the quick brown fox",
    ]
    arr = np.array([p.encode("utf-8") for p in prompts], dtype=object).reshape(-1)
    req = {"PROMPTS": _MockTensor("PROMPTS", arr)}

    print(f"=== execute() with N={len(prompts)} prompts ===")
    t1 = time.time()
    resps = m.execute([req])
    dt = time.time() - t1
    out = resps[0].outputs[0].arr
    print(f"output shape: {out.shape}, dtype={out.dtype}")
    print(f"first row first 5: {[round(float(v), 4) for v in out[0][:5]]}")
    print(f"execute() (1st call) took {dt*1000:.2f} ms")

    # Second call same shape — should hit warm cache
    t1b = time.time()
    _ = m.execute([req])
    dt_b = time.time() - t1b
    print(f"execute() (2nd call) took {dt_b*1000:.2f} ms\n")

    # Larger batch
    big = (prompts * 200)[:128]
    arr_big = np.array([p.encode("utf-8") for p in big], dtype=object).reshape(-1)
    req_big = {"PROMPTS": _MockTensor("PROMPTS", arr_big)}
    t2 = time.time()
    resps = m.execute([req_big])
    dt2 = time.time() - t2
    out2 = resps[0].outputs[0].arr
    print(f"=== execute() with N=128 prompts ===")
    print(f"output shape: {out2.shape}")
    print(f"execute() (1st) took {dt2*1000:.2f} ms = {len(big)/dt2:.0f} seq/s")
    # 2nd, 3rd
    t2b = time.time(); _ = m.execute([req_big]); dt2b = time.time() - t2b
    print(f"execute() (2nd) took {dt2b*1000:.2f} ms = {len(big)/dt2b:.0f} seq/s")
    t2c = time.time(); _ = m.execute([req_big]); dt2c = time.time() - t2c
    print(f"execute() (3rd) took {dt2c*1000:.2f} ms = {len(big)/dt2c:.0f} seq/s")

    # Verify N=1 row in big batch matches the single-call N=1 result
    print("\n=== sanity: same prompt across batch sizes should give same emb ===")
    cos = float(np.dot(out[0], out2[0]) / (np.linalg.norm(out[0]) * np.linalg.norm(out2[0]) + 1e-12))
    print(f"cosine(prompt[0] in N=3 vs prompt[0] in N=128) = {cos:.5f}")
    print("PASS" if cos > 0.999 else "FAIL — different embedding for same prompt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"FAIL: {type(e).__name__}: {e}")
        sys.exit(1)
