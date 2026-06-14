#!/bin/bash
# Per-rank core assignment for the TP smoke test. Each rank gets its
# own pair of physical cores under LNC=2.
RANK=${LOCAL_RANK:-0}
if [ "$RANK" = "0" ]; then
    export NEURON_RT_VISIBLE_CORES=0-1
else
    export NEURON_RT_VISIBLE_CORES=2-3
fi
exec /opt/torch-neuronx/.venv/bin/python tp_smoke_test.py
