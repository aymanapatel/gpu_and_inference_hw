"""v3: preallocate the full generated sequence buffer.

This avoids reallocating/copying the token ID tensor every step, but still
forwards the full current sequence each time, so most work remains.
"""

import torch
from utils import build_model, get_input_ids, time_generation


def loop(model, input_ids, n_steps):
    cur_len = input_ids.shape[1]
    generated_ids = torch.empty(
        (input_ids.shape[0], cur_len + n_steps),
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    generated_ids[:, :cur_len] = input_ids
    generated_tokens = torch.empty(n_steps, device=input_ids.device, dtype=input_ids.dtype)

    with torch.inference_mode():
        for step in range(n_steps):
            outputs = model(input_ids=generated_ids[:, :cur_len])
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated_tokens[step] = next_token_id
            generated_ids[:, cur_len] = next_token_id
            cur_len += 1

    return generated_tokens.tolist()


def build_optimized_model():
    return build_model(torch.float32)


if __name__ == "__main__":
    model = build_optimized_model()
    input_ids = get_input_ids()
    time_generation(loop, model, input_ids, "v3 preallocated full seq")
