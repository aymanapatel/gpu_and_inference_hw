"""v4: KV cache plus one-token decode, using fp32.

The prompt is processed once. After that, each decode step passes only the
latest token and reuses `past_key_values`. `logits_to_keep=1` avoids producing
full prompt logits when only the final token distribution is needed.
"""

import torch
from utils import build_model, get_input_ids, time_generation


def loop(model, input_ids, n_steps):
    generated_tokens = []

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token_id)

        for _ in range(1, n_steps):
            outputs = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token_id)

    return torch.cat(generated_tokens, dim=1).squeeze(0).tolist()


def build_optimized_model():
    return build_model(torch.float32)


if __name__ == "__main__":
    model = build_optimized_model()
    input_ids = get_input_ids()
    time_generation(loop, model, input_ids, "v4 KV cache fp32")
