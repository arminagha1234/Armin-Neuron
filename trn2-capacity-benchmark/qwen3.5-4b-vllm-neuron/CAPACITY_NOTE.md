# Serving 2000-in / 50-out at 500 RPS: a capacity note

This note is for a requester who asked for **500 requests/second** of Qwen3.5-4B
at **2000 input tokens / 50 output tokens**. The short version: that number is a
capacity decision, not a tuning target, and the cheapest way to move it is to
change the shape of the request — not the server config.

## What was asked vs. what one replica does

| | value |
|---|---|
| Requested throughput | 500 RPS |
| Shape | 2000 in / 50 out |
| Measured, one TP=4 replica | **0.157 RPS** (coherence 3/3) |
| Per `trn2.48xlarge` (16 replicas) | ~2.5 RPS |
| Floor to reach 500 RPS | **~200 × `trn2.48xlarge`** |

The measurement is in [`README.md`](README.md) and the raw run is
[`../results/raw/qwen3.5-4b-vllm-batching-ruled-out.json`](../results/raw/qwen3.5-4b-vllm-batching-ruled-out.json).

## Why more concurrency does not help

The instinct is to raise `max_num_seqs` and batch requests. We measured that and
it does not work on this model:

- 16 concurrent requests took **16 × 18.5 s — fully serial** (0.053 → 0.054 RPS,
  no throughput gain).
- Raising `max_num_seqs` from 4 to 16 made a **single** request *slower*
  (6.4 s → 19.0 s).

Qwen3.5-4B is a hybrid **GatedDeltaNet + attention** model. The linear-attention
layers carry a recurrent state that the current serving path steps one sequence
at a time; padding the decode graph to a batch dimension costs time without
running the sequences concurrently. Decode dominates the request — ~7.8 tok/s
means 50 output tokens cost **6.4 s**, against prefill's **0.12 s** (~50×). So the
request is decode-bound, and decode has no batching multiplier here.

## The levers, most effective first

| Option | Effect on box count | Cost to the requester |
|---|---|---|
| **1. Shorten the output** | Roughly linear. 50 → 10 tokens ≈ **5× fewer boxes**; decode is ~98% of the request. | Shorter completions. Often fine for extract/classify/score tasks. |
| **2. Use a dense-attention model** of similar size | Dense decode **does** batch. A dense 4B could give 10–50× the RPS/replica. | Different model, needs its own accuracy sign-off. |
| **3. Relax the latency SLA (offline/async)** | If requests need not be real-time, queue them; the box count tracks average, not peak. | Only works for batch/async use cases. |
| **4. Accept the capacity** | ~200 × `trn2.48xlarge` at the current shape. | Large, standing fleet. |

## Recommendation

If the 50-token output is a hard product requirement and it must be real-time,
this is a fleet-sizing conversation and the honest floor is ~200 boxes. Before
committing to that, we recommend pressure-testing **Option 1** — most 2000-in
workloads that want a 50-token answer are extract/score/classify tasks whose
answer is far shorter than 50 tokens, and every token cut off the output comes
straight off the box count. **Option 2** is the next-best structural fix if the
task tolerates a different model.

## What would change this answer

A decode kernel that runs the GatedDeltaNet recurrent state across the batch
dimension would restore a batching multiplier and cut the box count directly.
That does not exist in the shipped vLLM-Neuron 0.21 stack today; it is a kernel
project, not a config flag. Until then, the numbers above hold.
