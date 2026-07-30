#!/bin/bash
# Launch the PUBLIC vLLM-Neuron container with all 16 Neuron devices + your model dir mounted.
IMAGE=public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04
MODELS_HOST=${MODELS_HOST:-$HOME/models}   # host dir containing gemma-4-31b/
docker run -d --name vllm_public \
  $(for i in $(seq 0 15); do echo --device /dev/neuron$i; done) \
  -v "$MODELS_HOST:/root/models" \
  -e NEURON_SKIP_EFA_AFFINITY=1 -p 8000:8000 --ipc=host \
  "$IMAGE" sleep infinity
echo "container 'vllm_public' up. exec in: docker exec -it vllm_public bash"
