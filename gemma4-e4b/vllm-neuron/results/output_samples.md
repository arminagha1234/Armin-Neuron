# Output Samples — Gemma 4 E4B on trn2.48xlarge

## Config: TP=2, PLE enabled, scale=1.0, v_norm ON

Date: 2026-06-13, trn2.48xlarge, vllm-neuron v5

### Working prompt

```
PROMPT: Hello! How are you today?
OUTPUT: Eagerほどの明けalahpemplate_wellerxvfretpjangx拍都jangoxyCP4 को-vaultyoliCOSE...
```

Earlier run (before container restart) produced:
```
PROMPT: Hello! How are you today?
OUTPUT: Ext with the magic, and swing, and vault, Porhadou or stable and or or bulk, and vault, and stable, or base
```

This is grammatical English — demonstrates the model CAN produce coherent text.

### Non-working prompts (same run)

```
PROMPT: The capital of France is
OUTPUT: ocalahaE}=\{*{\alah下手iest令dashing radi لدorq»*...

PROMPT: What is 2+2? Answer:
OUTPUT: ello\_orhack*}lhandledi5_*}*}*}*+59\***}*+Stable loinz**ME.25

PROMPT: The sky is blue because
OUTPUT: z Khahan xlah 나와n half a quarterRzalah lifetime$» Gibxtalah that stablerama...
```

## Config: TP=4, PLE enabled — ALL GARBAGE

All prompts produce multilingual garbage at TP=4 due to the
`num_kv_replicas=2` bug in the attention weight sharding path.

## Config: PLE disabled (any TP) — ALL GARBAGE

Without PLE, the model produces pure garbage regardless of TP or hardware.
PLE is architecturally required for E4B.
