#!/usr/bin/env bash
# Run the torch.compile throughput bench ONE config per process, so each is a clean
# single static compile (avoids in-process dynamic recompile, which the beta rejects).
source "$HOME/workspace/native_venv/bin/activate"
cd "$HOME/vmae"
for cfg in "bf16 1" "bf16 2" "bf16 4" "bf16 8" "fp32 4" "fp32 8"; do
  set -- $cfg
  echo "### $1 batch $2 (torch.compile) ###"
  python bench_pretrain.py --device neuron --dtypes "$1" --batches "$2" --compile
done
echo ALL_COMPILE_DONE
