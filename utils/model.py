import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

CHAT_TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


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
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, dtype=torch.bfloat16
    ).to(get_device())
    if is_eval:
        model.eval()
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.bos_token = "<|im_start|>"
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|im_end|>"
    return model, tokenizer
