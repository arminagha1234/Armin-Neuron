"""Proof-of-path: AOT-trace the Mimi codec single-frame decode at a FIXED shape.

Demonstrates that fixed-shape compilation (torch_neuronx.trace) removes the per-call
recompile/dispatch overhead that the lazy-offload streaming path suffers. Compiles
once for (1, 32, 1) codes -> 1 frame (1920 samples), then times stable warm calls.
"""
import time, torch, statistics
import torch_neuronx
from transformers import CsmForConditionalGeneration

MODEL = "/scratch/csm/csm_1b"


class CodecFrameDecoder(torch.nn.Module):
    """Fixed-shape wrapper: (1, num_codebooks, 1) int codes -> (samples,) audio."""
    def __init__(self, codec):
        super().__init__()
        self.codec = codec

    def forward(self, codes):
        out = self.codec.decode(codes)
        a = out[0] if isinstance(out, tuple) else getattr(out, "audio_values", out)
        return a


def main():
    m = CsmForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).eval()
    nq = m.config.num_codebooks
    dec = CodecFrameDecoder(m.codec_model).eval()

    ex = torch.randint(0, 2047, (1, nq, 1), dtype=torch.long)
    print("[trace] compiling codec frame-decode at fixed shape (1, %d, 1)..." % nq)
    t0 = time.time()
    traced = torch_neuronx.trace(dec, ex)
    print(f"[trace] compiled in {time.time()-t0:.1f}s")

    # warm
    _ = traced(ex)
    ts = []
    for _ in range(20):
        codes = torch.randint(0, 2047, (1, nq, 1), dtype=torch.long)
        t0 = time.perf_counter()
        a = traced(codes)
        ts.append((time.perf_counter() - t0) * 1000)
    print(f"[trace] FIXED-SHAPE codec frame decode: median {statistics.median(ts):.2f} ms "
          f"(min {min(ts):.2f}, max {max(ts):.2f}) over 20 calls; out {tuple(a.shape)}")
    print("  vs lazy-offload codec in stream run: ~hundreds of ms w/ recompiles")


if __name__ == "__main__":
    main()
