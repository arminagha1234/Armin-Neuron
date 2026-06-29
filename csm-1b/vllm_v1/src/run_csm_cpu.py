"""CSM-1B CPU reference (oracle).

Generates speech for a fixed text deterministically (greedy) and saves the audio
waveform + stats as the correctness oracle for the Trainium port.
"""
import sys, time, torch, numpy as np
from transformers import AutoProcessor, CsmForConditionalGeneration

MODEL = "/scratch/csm/csm_1b"
TEXT = "[0]Hello from Trainium, this is a conversational speech model test."
OUT_WAV = "/scratch/csm/oracle_cpu.wav"
OUT_PT = "/scratch/csm/oracle_cpu.pt"
SEED = 1234


def main():
    torch.manual_seed(SEED)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = CsmForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32).eval()

    inputs = proc(TEXT, add_special_tokens=True, return_tensors="pt")
    print(f"[cpu] input_ids {tuple(inputs['input_ids'].shape)}")

    t0 = time.time()
    with torch.no_grad():
        audio = model.generate(**inputs, output_audio=True, do_sample=False,
                               max_new_tokens=128)
    dt = time.time() - t0

    # audio may be a list of tensors (one per batch item) or a tensor
    a = audio[0] if isinstance(audio, (list, tuple)) else audio
    a = a.detach().to(torch.float32).cpu().flatten()
    print(f"[cpu] generate {dt:.1f}s | audio samples={a.numel()} "
          f"mean={a.mean():.4e} std={a.std():.4e} absmax={a.abs().max():.4e}")

    torch.save({"audio": a, "text": TEXT, "n": a.numel(),
                "mean": a.mean().item(), "std": a.std().item(),
                "head": a[:256].clone()}, OUT_PT)
    try:
        proc.save_audio(audio, OUT_WAV)
        print(f"[cpu] saved {OUT_WAV}")
    except Exception as e:
        # fallback: write with soundfile at Mimi's 24kHz
        import soundfile as sf
        sf.write(OUT_WAV, a.numpy(), 24000)
        print(f"[cpu] saved {OUT_WAV} (soundfile fallback); save_audio err: {e}")
    print(f"[cpu] saved oracle -> {OUT_PT}")


if __name__ == "__main__":
    sys.exit(main())
