"""v0: naive baseline loop.

This mirrors `utils.slow_loop`: every decode step forwards the full growing
sequence, synchronizes with `.item()`, and allocates a larger tensor with
`torch.cat`.
"""

import torch
from utils import build_model, get_input_ids, time_generation


def loop(model, input_ids, n_steps):
    generated_ids = input_ids.clone()
    generated_tokens = []
    for _ in range(n_steps):
        outputs = model(input_ids=generated_ids)
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        token_value = next_token_id.item()
        generated_tokens.append(token_value)
        generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
    return generated_tokens


def build_optimized_model():
    return build_model(torch.float32)


if __name__ == "__main__":
    model = build_optimized_model()
    input_ids = get_input_ids()
    time_generation(loop, model, input_ids, "v0 naive baseline")
