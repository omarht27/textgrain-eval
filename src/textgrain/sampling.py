"""Next-token distributions and the token-by-token generation loop."""

from collections.abc import Callable

import numpy as np
import torch
from transformers import PreTrainedModel

TokenChooser = Callable[[np.ndarray, list[int]], int]


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Takes logits and a temperature T > 0 and returns softmax(logits / T)."""
    return torch.softmax(logits / temperature, dim=-1)


def top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Takes a distribution and returns it restricted to the smallest set of tokens with mass >= top_p, renormalized."""
    if top_p >= 1.0:
        return probs
    sorted_probs, order = probs.sort(descending=True)
    keep = sorted_probs.cumsum(0) - sorted_probs < top_p
    filtered = torch.zeros_like(probs)
    filtered[order[keep]] = sorted_probs[keep]
    return filtered / filtered.sum()


def entropy(probs: torch.Tensor | np.ndarray) -> float:
    """Takes a distribution and returns its entropy in nats."""
    p = probs[probs > 0]
    return float(-(p * p.log()).sum()) if isinstance(p, torch.Tensor) else float(-(p * np.log(p)).sum())


def sample(probs: np.ndarray, rng: np.random.Generator) -> int:
    """Takes a distribution and a random generator and returns a token id drawn from it."""
    cdf = np.cumsum(probs)
    return int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))


def greedy(probs: np.ndarray, response: list[int]) -> int:
    """Takes a distribution and returns its most probable token id."""
    return int(np.argmax(probs))


def sampler(seed: int) -> TokenChooser:
    """Takes a seed and returns a chooser that draws tokens from P with its own random generator."""
    rng = np.random.default_rng(seed)
    return lambda probs, response: sample(probs, rng)


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    prompt_ids: torch.Tensor,
    choose: TokenChooser,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[int]:
    """Takes a prompt of shape (1, T) and a token chooser and returns the ids of the generated response."""
    stop_ids = model.generation_config.eos_token_id
    stop_ids = set(stop_ids) if isinstance(stop_ids, list) else {stop_ids}

    output = model(prompt_ids.to(model.device), use_cache=True)
    response = []
    for _ in range(max_new_tokens):
        probs = top_p_filter(apply_temperature(output.logits[0, -1].float(), temperature), top_p)
        probs = probs.cpu().double().numpy()
        token = choose(probs / probs.sum(), response)
        response.append(token)
        if token in stop_ids:
            break
        next_input = torch.tensor([[token]], device=model.device)
        output = model(next_input, past_key_values=output.past_key_values, use_cache=True)
    return response


@torch.no_grad()
def generate_batch(
    model: PreTrainedModel,
    prompts: list[torch.Tensor],
    choosers: list[TokenChooser],
    pad_id: int,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    ban_stop: bool = False,
    records: list[list[tuple[float, float]]] | None = None,
) -> list[list[int]]:
    """Takes prompts of shape (1, T_i) and one token chooser per prompt and returns the generated responses, computed in one left-padded batch.

    If ban_stop is True, end-of-text tokens get probability 0 so that every response has max_new_tokens tokens.
    If records is given, records[i] receives (H(P), probability of stopping) at each step of response i, before the ban.
    """
    stop_ids = model.generation_config.eos_token_id
    stop_ids = set(stop_ids) if isinstance(stop_ids, list) else {stop_ids}
    device = model.device
    width = max(p.shape[1] for p in prompts)
    input_ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(prompts), width), dtype=torch.long)
    for i, p in enumerate(prompts):
        input_ids[i, width - p.shape[1] :] = p[0]
        attention_mask[i, width - p.shape[1] :] = 1
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp_min(0)

    output = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=True)
    responses: list[list[int]] = [[] for _ in prompts]
    finished = [False] * len(prompts)
    for _ in range(max_new_tokens):
        probs = torch.softmax(output.logits[:, -1].float() / temperature, dim=-1)
        probs = probs.cpu().double().numpy()
        next_tokens = []
        for i in range(len(prompts)):
            if finished[i]:
                next_tokens.append(pad_id)
                continue
            p = probs[i] if top_p >= 1.0 else top_p_filter(torch.from_numpy(probs[i]), top_p).numpy()
            if records is not None:
                records[i].append((entropy(p), float(p[list(stop_ids)].sum())))
            if ban_stop:
                p = p.copy()
                p[list(stop_ids)] = 0.0
            token = choosers[i](p / p.sum(), responses[i])
            responses[i].append(token)
            finished[i] = token in stop_ids
            next_tokens.append(token)
        if all(finished):
            break
        attention_mask = torch.cat([attention_mask, torch.ones((len(prompts), 1), dtype=torch.long, device=device)], dim=1)
        position_ids = position_ids[:, -1:] + 1
        next_input = torch.tensor(next_tokens, device=device)[:, None]
        output = model(
            next_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=output.past_key_values,
            use_cache=True,
        )
    return responses
