# SPDX-License-Identifier: Apache-2.0
"""
Triton Python backend wrapping native PyTorch BERT + torch.compile on Trainium.

Lifecycle:
  initialize()  - load HF model, copy weights into our BertEncoder, move to
                  Neuron device, torch.compile per-bucket, warm each bucket.
  execute()     - tokenize incoming prompts, route to nearest compiled bucket,
                  pad up to bucket size, forward, slice back to real N, return.

Buckets: BERT_BUCKETS env var, default "1,8,32,128,512".

Tested against:
  sentence-transformers/all-MiniLM-L6-v2  (384-dim)
  BAAI/bge-base-en-v1.5                   (768-dim)
"""
import json
import os
import sys

import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except ImportError:  # standalone-test path
    pb_utils = None

# Allow `model.py` to import sibling `native_bert_model.py` placed in the
# version directory next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TritonPythonModel:
    def initialize(self, args):
        import torch
        import torch_neuronx  # noqa: F401  registers neuron device
        from transformers import AutoModel, AutoTokenizer
        from native_bert_model import BertEncoder, load_from_hf

        model_name = os.environ.get(
            "BERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.max_len = int(os.environ.get("BERT_MAX_LEN", "128"))
        self.buckets = sorted(
            int(b) for b in os.environ.get("BERT_BUCKETS", "1,8,32,128,512").split(",")
        )
        self.dtype = torch.bfloat16
        self.device = torch.device("neuron")

        self._log(f"loading {model_name}, buckets={self.buckets}, max_len={self.max_len}")

        hf = AutoModel.from_pretrained(model_name, return_dict=False).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden = hf.config.hidden_size

        ours = BertEncoder(hf.config, dtype=self.dtype).eval()
        load_from_hf(hf, ours)
        ours = ours.to(self.dtype).to(self.device)
        self._eager = ours

        # Compile + warm each bucket.
        self.compiled = {}
        for B in self.buckets:
            self._log(f"torch.compile + warm batch={B}...")
            cm = torch.compile(self._eager, backend="neuron", dynamic=False)
            ids, pos, am = self._tokenize_pad(["x"] * B)
            with torch.no_grad():
                _ = cm(ids, pos, am)
                torch_neuronx.synchronize()
            self.compiled[B] = cm

        self._log("ready")

    def _log(self, msg):
        if pb_utils is not None:
            pb_utils.Logger.log_info(f"[bert_embed] {msg}")
        else:
            print(f"[bert_embed] {msg}", flush=True)

    def _tokenize_pad(self, prompts):
        import torch
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding="max_length",
            max_length=self.max_len, truncation=True,
        )
        ids = enc["input_ids"].to(self.device)
        am = enc["attention_mask"].to(self.device).to(self.dtype)
        pos = torch.arange(self.max_len, device=self.device).unsqueeze(0).expand_as(ids)
        return ids, pos, am

    def _route_to_bucket(self, n):
        for B in self.buckets:
            if B >= n:
                return B
        return self.buckets[-1]

    def execute(self, requests):
        import torch
        import torch_neuronx
        responses = []
        for req in requests:
            prompts_in = pb_utils.get_input_tensor_by_name(req, "PROMPTS").as_numpy()
            # Triton STRING tensors arrive as numpy bytes; decode each row.
            prompts = [
                p.decode("utf-8") if isinstance(p, (bytes, bytearray)) else str(p)
                for p in prompts_in.reshape(-1).tolist()
            ]
            n = len(prompts)
            B = self._route_to_bucket(n)

            # Pad up to bucket size with empty strings; their rows are sliced off.
            padded = prompts + [""] * (B - n)
            ids, pos, am = self._tokenize_pad(padded)

            with torch.no_grad():
                emb = self.compiled[B](ids, pos, am)
                torch_neuronx.synchronize()

            # Slice off padding, move to CPU fp32, return as numpy.
            out = emb[:n].detach().to(torch.float32).cpu().numpy()
            out_t = pb_utils.Tensor("EMBEDDING", out)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_t]))
        return responses

    def finalize(self):
        self._log("finalize")
