# Module 4 Learnings: Performance Engineering & GPU Inference

---

## Assignment 1: GPU Roofline Model — Memory-Bound vs Compute-Bound Kernels

### Arithmetic Intensity (AI)

- **Definition**: `AI = FLOPs / Bytes` — the ratio of compute to memory traffic for a kernel.
- **Low AI**: Kernel is **memory-bound** (limited by HBM bandwidth).
- **High AI**: Kernel is **compute-bound** (limited by peak FLOP/s throughput).
- **Ridge point**: The crossover on the roofline where `peak_compute = bandwidth × AI`. On H100 SXM (FP32), this is ~20 FLOP/Byte.

### Roofline Model Formula

```
achievable FLOP/s = min(peak_compute, bandwidth × AI)
```

- Plotting kernels on a log-log roofline diagram immediately reveals whether a kernel is hitting the memory ceiling or the compute ceiling.
- **Kernels left of the ridge point** are memory-bound; optimizing them requires reducing data movement or improving locality.
- **Kernels right of the ridge point** are compute-bound; optimizing them requires more FLOPs per unit time (e.g., better parallelism, tensor cores).

### Measuring GPU Performance with CUDA Events

- `torch.cuda.Event(enable_timing=True)` provides precise GPU-side timing, avoiding CPU launch overhead and synchronization delays.
- Benchmark workflow: warmup (to trigger `torch.compile` and warm caches) → CUDA event start → kernel execution → CUDA event end → synchronize → compute median elapsed time.
- Median over 100 repetitions is more robust than mean for GPU benchmarks due to occasional scheduling jitter.

### Impact of `torch.compile` and Kernel Fusion

- **Eager mode**: Each Python loop iteration launches separate GPU kernels (`mul`, `add`), materializing intermediates in global memory. Traffic grows with loop iterations.
- **Compiled mode (`torch.compile`)**: The compiler fuses the loop body into fewer kernels (often one), keeping intermediates in registers. Traffic stays constant (one read, one write), while FLOPs grow with `num_ops`.
- **Measured result**: For 128 ops, compiled AI reached **32 FLOP/Byte** (approaching compute-bound), while eager AI stayed flat at **0.083 FLOP/Byte** (deeply memory-bound).

### Byte-Traffic Models

| Variant | Assumed Traffic per Element | AI scaling |
|---------|----------------------------|------------|
| Eager | `num_ops × 6 × bytes_per_element` (separate kernels, intermediates) | Flat / very low |
| Compiled | `2 × bytes_per_element` (one read + one write at kernel boundary) | Linear with `num_ops` |

### Observed Roofline Data (H100)

| Operation | AI (FLOP/Byte) | Achieved TFLOP/s |
|-----------|---------------|------------------|
| `clone()` (lowest AI) | 0.01 | ~29 (bandwidth-bound) |
| 64 ops compiled | 16.0 | ~39,800 |
| 128 ops compiled | 32.0 | ~53,460 |
| Matmul 1024×1024 | 170.7 | ~32,435 |
| Matmul 4096×4096 | 682.7 | ~51,901 |

### Key Insight

A small matmul (1024×1024) can underperform a simple compiled element-wise kernel on a large GPU because the GEMM may not fully occupy all SMs, while the element-wise kernel exposes massive parallelism across vector elements.

---

## Assignment 2: LLM Inference Optimization

### Baseline Problems

The naive autoregressive loop had three major inefficiencies:
1. **Full sequence forward**: Passed the entire growing sequence every decode step.
2. **CPU-GPU sync**: `.item()` every step forced a device synchronization.
3. **Repeated allocation**: `torch.cat([generated_ids, next_token_id], dim=1)` reallocated and copied memory each step.

### Optimization Ladder (Measured on H100)

| Step | Change | Time (128 tokens) | Speedup vs Baseline |
|------|--------|------------------|---------------------|
| v0 | Naive baseline | 0.943 s | 1.00× |
| v1 | Add `torch.inference_mode()` | 0.929 s | 1.01× |
| v2 | Remove per-token `.item()` sync | 0.884 s | 1.07× |
| v3 | Preallocate full sequence buffer | 0.883 s | 1.07× |
| v4 | **KV cache + one-token decode** | **0.141 s** | **6.69×** |
| v5 | Same with bf16 weights | 0.167 s | 5.66× |

**Final submission** (fp32, KV cache): **0.17 s** → **5.49× speedup**.

### The Biggest Win: KV Cache

- **`past_key_values`**: After the initial prompt prefill, each decode step passes only the **latest token** (`input_ids=next_token_id`) while reusing cached keys and values from all previous positions.
- This eliminates repeated attention computation over the full prompt and all previously generated tokens.
- **`logits_to_keep=1`**: Tells Transformers to compute only the last-position logits, avoiding wasted FLOPs on intermediate positions during decode.

### Profiling with `torch.profiler`

- Produces a **summary table** (sorted by CPU/CUDA time) and a **Chrome trace** (viewable at `ui.perfetto.dev`).
- **Trace anatomy**:
  - CPU thread: nested `aten::` operator bars ending in `cudaLaunchKernel`.
  - GPU stream: actual kernel execution.
  - Healthy trace = both rows densely filled and overlapping.
- **What the trace revealed**:
  - Baseline: `aten::item` 12 times (~45 ms in profiled run), `aten::cat` 120 times, GPU kernels ~80 ms for 12 steps.
  - Optimized: `aten::item` count = 0, GPU kernel total ~12.8 ms, CUDA runtime events dropped from 1565 → 1158.

### Dtype Experiment: bf16 vs fp32

- On H100 with this tiny model, **bf16 was slightly slower than fp32**.
- Likely cause: the model is small enough that the extra dtype conversion overhead outweighs the memory-bandwidth savings of 16-bit.
- For large models or batch inference, bf16 typically wins; the lesson is to **measure, not assume**.

---

## Assignment 3: Mini Inference Engine (Optional, Conceptual)

### Core Design Philosophy

Real inference engines (vLLM, SGLang, TensorRT-LLM, TGI) are primarily about:
1. **Keeping expensive compute busy**
2. **Managing scarce KV memory efficiently**
3. **Serving many concurrent requests fairly**

### Paged KV Memory

- Instead of contiguous KV tensors per request, memory is split into fixed-size **physical blocks**.
- Each request has a `block_table` mapping its logical positions to physical block IDs.
- Benefits: no memory fragmentation, easy preemption/eviction, and prefix sharing.

### Prefix Caching

- Only **complete blocks** are cacheable (e.g., for block size 16, a 40-token prompt caches prefixes of 16 and 32 tokens).
- `match_prefix(tokens)` returns the longest cached prefix without pinning; `lock(handle)` pins blocks for live requests.
- Reusing cached prefix blocks avoids redundant prefill computation, dramatically improving TTFT (time-to-first-token) for repeated prompts.

### Continuous Batching

- Each engine step runs one **phase-pure** batch: either prefill or decode, never both.
- The set of active requests changes over time as new requests arrive, old ones finish, and memory pressure forces preemption.

### CacheManager Ownership Rules

| State | `_ref` | `_cache_ref` | Meaning |
|-------|--------|-------------|---------|
| Free | 0 | 0 | In free pool |
| Live only | 1 | 0 | Owned by one request |
| Cached only | 1 | >0 | Evictable by LRU |
| Pinned + cached | ≥2 | >0 | Do not evict |

### Scheduler Policies

- **PREFILL_FIRST**: Prefers prefill work; good for minimizing TTFT when many new requests are arriving.
- **DECODE_FIRST**: Prefers decode-ready running requests; good for maximizing throughput when decode work dominates.
- Admission is FIFO; if the front request cannot be admitted due to memory pressure, stop admitting for that step.

### Preemption Strategy

1. Try free blocks.
2. Let `CacheManager` evict LRU cached prefixes.
3. If still insufficient, **preempt** a running request (free its blocks, return to waiting queue).

---

## Cross-Cutting Themes

1. **Roofline thinking generalizes**: Any kernel can be classified by its arithmetic intensity. Before optimizing, know whether you are fighting memory bandwidth or compute throughput.
2. **Kernel fusion is transformative**: `torch.compile` can turn a memory-bound loop into a compute-bound fused kernel by keeping intermediates in registers.
3. **Profile first, optimize second**: The naive inference loop looked simple but hid massive repeated work. The trace and `time_generation` numbers, not intuition, revealed the real bottleneck.
4. **KV cache dominates autoregressive inference**: For long contexts, the difference between full-sequence and cached decode is orders of magnitude. Every production inference system builds around this principle.
5. **Measure dtype choices**: bf16 is not always faster than fp32, especially for small models where conversion overhead matters.
6. **Inference engines are memory managers first**: The scheduler, cache manager, and block allocator are as important as the model forward pass. Paged memory, prefix caching, and continuous batching are the core innovations in modern serving systems.
7. **Phase-pure batches**: Prefill and decode use different kernel shapes and attention patterns. Mixing them in one batch complicates kernel selection and hurts efficiency.
8. **Reference counting and LRU are foundational**: The cache manager's `_ref` / `_cache_ref` split ensures blocks are freed exactly once and cached prefixes can be safely evicted when unpinned.
