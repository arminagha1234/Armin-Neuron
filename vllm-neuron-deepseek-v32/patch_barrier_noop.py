#!/usr/bin/env python3
"""Make tp_barrier and world_barrier no-ops.

Why: at 64 TP ranks, gloo's TCP barrier collective desyncs after
multiple short polled retries. All 64 workers reach the barrier,
but the barrier never returns. We've waited >6 hours.

The barrier was inserted for safety (sync compile completion across
all ranks). Skipping it is fine if the next operation naturally
synchronizes (which model.generate() effectively does via TP all-reduce).

Worst case: garbage on the first call (compile not done), but our
NEFF cache is hot, all workers reached the barrier (= all done compile),
so this should be safe.
"""
import sys

F = '/opt/conda/lib/python3.12/site-packages/vllm_neuron/parallel/neuron_parallel_state.py'
src = open(F).read()

if 'NOOP_BARRIER_FOR_DEEPSEEK' in src:
    print('PATCH_ALREADY_APPLIED')
    sys.exit(0)

# Replace the polling barrier impls with no-ops
old_chunked = '''def _chunked_barrier_wait(group, total_timeout: timedelta) -> None:'''
if old_chunked not in src:
    print('CHUNKED_NOT_FOUND')
    sys.exit(1)

# Find and replace the entire functions tp_barrier, world_barrier, and _chunked_barrier_wait
# Identify the start of _chunked_barrier_wait and the end of tp_barrier (= last def in this region)
start = src.index('def _chunked_barrier_wait')
# Find the end of tp_barrier function
tp_marker = src.index('def tp_barrier(', start)
# Look for the next def or class after tp_barrier
end_marker = src.find('\ndef ', tp_marker + 1)
if end_marker == -1:
    end_marker = src.find('\nclass ', tp_marker + 1)
if end_marker == -1:
    end_marker = len(src)

original = src[start:end_marker]

replacement = '''# NOOP_BARRIER_FOR_DEEPSEEK: gloo's 64-way TCP barrier desyncs under our
# polled retry loop. All ranks reach the barrier successfully but it
# never resolves. Replacing with no-ops since the next inference op
# (TP all-reduce in forward) naturally synchronizes the ranks.
def _chunked_barrier_wait(group, total_timeout: timedelta) -> None:
    return  # no-op


def world_barrier(timeout: timedelta = timedelta(seconds=43200)):
    return  # no-op (NOOP_BARRIER_FOR_DEEPSEEK)


def tp_barrier(timeout: timedelta = timedelta(seconds=43200)):
    return  # no-op (NOOP_BARRIER_FOR_DEEPSEEK)


'''

new_src = src[:start] + replacement + src[end_marker:]
open(F, 'w').write(new_src)
print('PATCH_APPLIED')
