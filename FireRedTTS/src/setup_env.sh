#!/usr/bin/env bash
# Set up FireRedTTS v1 inside the native-PyTorch Neuron DLC container.
#
# Run this AFTER starting the container (see README). It clones the upstream v1 code,
# installs the deps against the container's native torch-neuronx (torch 2.11), and
# verifies the native "neuron" device. Every pin/flag below is load-bearing — see the
# inline notes (these were the actual blockers hit bringing FireRedTTS up on Neuron).
#
# Usage (inside container):  bash setup_env.sh
set -u

WORK=${WORK:-/root/firered}
mkdir -p "$WORK" && cd "$WORK"

echo "=== [1/5] clone FireRedTTS v1 (the 'main' branch — NOT the default) ==="
# The repo's DEFAULT branch is 'fireredtts-1s' (the newer streaming model, different
# architecture). The v1 checkpoints on HF (FireRedTeam/FireRedTTS) match the 'main' branch.
if [ ! -d FireRedTTS ]; then
  git clone --depth 1 --branch main https://github.com/FireRedTeam/FireRedTTS.git
fi

echo "=== [2/5] install fireredtts source (no deps) ==="
# NOTE: the top-level 'fireredtts' package has no __init__.py, so an editable install
# registers nothing. We import it via PYTHONPATH instead (see run commands / README).
# --no-deps so pip does NOT pull a CUDA torch that would clobber native torch-neuronx.
pip install --no-deps -e ./FireRedTTS || true

echo "=== [3/5] install runtime deps (torch is already provided by the container) ==="
pip install \
  "diffusers==0.27.2" "librosa==0.10.2" "soundfile==0.12.1" "einops==0.8.0" \
  "transformers==4.44.2" "tiktoken==0.7.0" "inflect==7.4.0" \
  "lingua-language-detector==2.0.2" "sentencex==0.6.1"

# diffusers 0.27.2 imports huggingface_hub.cached_download (removed in hub >= 0.26).
pip install "huggingface_hub==0.25.2"
# torchaudio must match the container torch (2.11); --no-deps so it won't swap torch.
pip install --no-deps "torchaudio==2.11.0"

echo "=== [4/5] WeTextProcessing (pynini) — OPTIONAL, expected to FAIL on py3.12 ==="
# pynini has no cp312 wheel and building from source needs OpenFst. If it fails, the
# runners auto-fall back to a lite text normalizer (--no-tn). This is expected.
pip install "WeTextProcessing==1.0.3" || \
  echo "WeTextProcessing/pynini install FAILED (expected on py3.12) — runners use --no-tn."

echo "=== [5/5] sanity: torch + native neuron device (expect 16.0) ==="
python - <<'PY'
import torch
try:
    import torch_neuronx  # registers the native "neuron" device
    d = torch.device("neuron")
    x = torch.ones(8, device=d)
    print("native neuron device OK, (x+x).sum() =", (x + x).sum().item())
except Exception as e:
    print("native neuron device check FAILED:", repr(e))
PY

echo
echo "Done. Run with:"
echo "  cd $WORK && PYTHONPATH=$WORK/FireRedTTS FIRERED_MODEL=$WORK/pretrained_models \\"
echo "    python run_fireredtts_neuron.py --prompt-wav FireRedTTS/examples/prompt_1.wav \\"
echo "      --text 'Hello from Trainium.' --lang en --no-tn --offload vocoder --out neuron.wav"
