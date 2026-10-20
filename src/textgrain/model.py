"""Model loading and next-token logits."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def get_device() -> torch.device:
    """Returns the best available device: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(name: str, device: torch.device) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Takes a Hugging Face model name and returns the model (float32, eval mode) and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).float().to(device)
    model.eval()
    return model, tokenizer


def encode_prompt(tokenizer: PreTrainedTokenizerBase, prompt: str) -> torch.Tensor:
    """Takes a user message and returns the token ids of the chat-formatted prompt, shape (1, T)."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt").input_ids


@torch.no_grad()
def next_token_logits(model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Takes token ids of shape (1, T) and returns the logits of the next token, shape (V,)."""
    return model(input_ids.to(model.device)).logits[0, -1].float()
