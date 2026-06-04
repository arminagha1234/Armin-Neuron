#!/usr/bin/env python3
"""Apply the get_tensor_names shim directly. Run inside the container."""
import sys

F = '/opt/conda/lib/python3.12/site-packages/vllm_neuron/utils/checkpoints.py'
src = open(F).read()

if 'def get_tensor_names' in src:
    print('PATCH_ALREADY_PRESENT')
    sys.exit(0)

needle = '    def get_num_files(self) -> int:'
if needle not in src:
    print('NEEDLE_NOT_FOUND')
    sys.exit(1)

insertion = '''    def get_tensor_names(self) -> list[str]:
        """Return all tensor names in the checkpoint.

        Added by deepseek_v32 PR #2025 deploy shim.
        """
        self._ensure_indexed()
        return list(self._tensor_name_to_file.keys())

'''
new_src = src.replace(needle, insertion + needle, 1)
open(F, 'w').write(new_src)
print('PATCH_APPLIED')
