A good workflow for HW2 is:

  1. Get the baseline running
      - Run python3 hw2/hw2_task.py.
      - Confirm the slow baseline prints timing and
        creates v0_slow_trace.json.
  2. Read the slow loop first
      - Look for obvious per-token waste:
      - full sequence passed every step:
        model(input_ids=generated_ids)
      - CPU sync every step: next_token_id.item()
      - repeated allocation/copy: torch.cat(...)
      - no torch.inference_mode()
      - fp32 model loading
  3. Profile the baseline
      - Implement profile().
      - Open hw2/results/v0_slow_trace.json in Perfetto.
      - Look for GPU idle gaps, repeated large model
        forwards, aten::item, aten::cat, and lots of
        growing attention work.
  4. Make one optimization at a time
      - Add torch.inference_mode().
      - Remove .item() from inside the loop.
      - Use past_key_values / KV cache.
      - Stop concatenating the full generated sequence
        every step.
      - Try torch.bfloat16 or torch.float16 in
        generate_optimized().
  5. Measure after each change
      - Run the script and record the Optimized: timing.
      - Keep a small note like:
          - baseline: 0.95s
          - inference mode: ...
          - no .item(): ...
          - KV cache: ...
          - bf16: ...
  6. Profile the optimized version
      - Open v1_optimized_trace.json.
      - Confirm the trace changed in the way you
        expected:
      - fewer CPU-GPU sync points
      - less repeated full-sequence work
      - denser GPU stream
      - shorter per-token decode steps
  7. Fill in the writeup
      - Say what you changed.
      - Include speedup per fix if you measured it.
      - Explain the biggest win, usually KV cache,
        because it avoids recomputing the whole prompt
        every token.

  The main idea: read code, form a hypothesis, measure,
  change one thing, measure again, then use the trace to
  explain why the speedup happened.