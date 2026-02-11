import argparse
import datetime as dt
from os.path import join

import torch
from datasets import load_dataset
from peft import OFTConfig, get_peft_model
from transformers import PreTrainedTokenizer
from trl import SFTConfig, SFTTrainer

from utils import console
from utils.misc import seed_everything
from utils.model import load_model_and_tokenizer

DATASET = "sahil2801/CodeAlpaca-20k"


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total, trainable = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return total, trainable


def formatting_func(
    example: dict[str, str], tokenizer: PreTrainedTokenizer
) -> str:
    user_content = example["instruction"].strip()
    inp = example["input"].strip()
    if len(inp) > 0:
        user_content += f"\n\nInput:\n{inp}"
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"].strip()},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def main(args: argparse.Namespace) -> None:
    effective_batch = args.batch_size * args.gradient_accumulation_steps
    console.print(f"""\
Model: {args.model_name_or_path}
Dataset: {DATASET}
Epochs: {args.epochs}
Learning Rate: {args.learning_rate}
Batch Size: {args.batch_size} (effective: {effective_batch})
OFT Block Size: {args.oft_block_size}
""")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)

    # Ensure tokenizer has pad token for batch processing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Apply OFT
    oft_config = OFTConfig(
        oft_block_size=args.oft_block_size,
        use_cayley_neumann=True,
        module_dropout=0.05,
        bias="oft_only",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, oft_config)
    total_params, trainable_params = count_params(model)
    console.print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({trainable_params / total_params * 100:.2f}%)\n"
    )

    # Configure training (optimizer defaults to AdamW)
    now = dt.datetime.now(dt.timezone.utc)
    output_dir = join("checkpoints", now.strftime(r"%Y%m%d-%H%M%S"))
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,
        bf16=True,
        max_length=tokenizer.model_max_length,
        packing=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=load_dataset(DATASET, split="train"),
        formatting_func=lambda example: formatting_func(example, tokenizer),
    )

    # Train
    console.print("[bold green]Training...[/]")
    trainer.train()

    # Save model and tokenizer
    trainer.model.merge_and_unload().save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    console.print(f"[bold green]Finetuned model saved to {output_dir}![/]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-name-or-path",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
    )
    parser.add_argument("-e", "--epochs", type=int, default=5)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3)
    parser.add_argument("-b", "--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--oft-block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)
