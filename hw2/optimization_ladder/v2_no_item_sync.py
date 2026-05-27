"""v2: remove the per-token `.item()` CPU sync.

Generated token tensors stay on GPU during the loop and are copied to a Python
list only once at the end. The model still sees the full growing sequence.
"""

import torch
from utils import build_model, get_input_ids, time_generation


def loop(model, input_ids, n_steps):
    generated_ids = input_ids.clone()
    generated_tokens = []
    with torch.inference_mode():
        for _ in range(n_steps):
            outputs = model(input_ids=generated_ids)
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            generated_tokens.append(next_token_id)
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
    return torch.stack(generated_tokens).squeeze(-1).tolist()


def build_optimized_model():
    return build_model(torch.float32)


if __name__ == "__main__":
    model = build_optimized_model()
    input_ids = get_input_ids()
    time_generation(loop, model, input_ids, "v2 no item sync")
