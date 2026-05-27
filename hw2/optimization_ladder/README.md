# Optimization Ladder Code Snapshots

Each file captures one step from the optimization ladder in `../Insights.md`.

These are reference snapshots for the loop changes only. The submitted runnable solution remains `../hw2_task.py`.

| File | Step | Main change |
| --- | --- | --- |
| `v0_naive_baseline.py` | v0 | Full sequence each step, `.item()`, growing `torch.cat`, fp32 |
| `v1_inference_mode.py` | v1 | Wrap generation in `torch.inference_mode()` |
| `v2_no_item_sync.py` | v2 | Keep token tensors on GPU during the loop; no per-step `.item()` |
| `v3_preallocate_full_sequence.py` | v3 | Preallocate the full token buffer instead of concatenating every step |
| `v4_kv_cache_fp32.py` | v4 | Use KV cache, one-token decode, `logits_to_keep=1`, fp32 |
| `v5_kv_cache_bf16.py` | v5 | Same KV-cache loop, but model loaded as bf16 |
