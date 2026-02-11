import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(
    model_name_or_path: str, is_eval: bool = False
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    device = get_device()
    print(f"Loading {model_name_or_path} on {device.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, dtype=torch.bfloat16
    ).to(get_device())
    if is_eval:
        model.eval()
    return model, tokenizer
