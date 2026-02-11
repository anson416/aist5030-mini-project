import argparse
import ast
import os
import random
from statistics import mean

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import evaluate
import numpy as np
import torch
from codebleu import calc_codebleu
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from utils.model import load_model_and_tokenizer
from utils.progress import progress_bar


def eval_oasst(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> float:
    def construct_conversation(text: str) -> list[dict[str, str]]:
        text = text.strip()
        messages = []
        while len(text) > 0:
            tmp = text.split("<|im_start|>", maxsplit=1)[1]
            msg, text = tmp.split("<|im_end|>", maxsplit=1)
            role, content = msg.split("\n", maxsplit=1)
            messages.append({"role": role.strip(), "content": content.strip()})
        assert len(messages) > 0 and messages[0]["role"] == "user"
        return messages

    dataset = load_dataset("OpenAssistant/oasst_top1_2023-08-25", split="test")
    losses: list[float] = []
    with progress_bar() as pbar:
        task_id = pbar.add_task("[bold]OASST[/]", total=len(dataset))

        for row in dataset:
            # Construct conversation and tokenize
            inputs = tokenizer.apply_chat_template(
                construct_conversation(row["text"]),
                truncation=True,
                max_length=4096,
                return_tensors="pt",
            ).to(model.device)

            # Store loss
            with torch.inference_mode():
                outputs = model(**inputs, labels=inputs["input_ids"])
                losses.append(outputs.loss.item())

            # Update progress bar
            pbar.update(task_id, advance=1)

    return torch.exp(torch.tensor(losses).mean()).item()


def eval_mbpp(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> dict[tuple[float, float], dict[str, float | dict[int, float]]]:
    TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9]
    TOP_PS = [0.1, 0.3, 0.5, 0.7, 0.9]
    K_VALUES = [1, 5, 10]

    code_eval = evaluate.load("code_eval")

    def extract_python_code(text: str) -> str:
        if "```python" in text:
            text = text.split("```python")[1]
            if "```" in text:
                text = text.split("```")[0]
        elif "```" in text:
            text = text.split("```")[1]
        return text.strip()

    def check_syntax(code: str) -> bool:
        try:
            ast.parse(code)
        except SyntaxError:
            return False
        return True

    def calculate_metrics(
        all_predictions: list[list[str]],
        first_predictions: list[str],
        all_references: list[str],
        pass_k_references: list[str],
    ) -> dict[str, float | dict[int, float]]:
        syntax_rate = mean([check_syntax(code) for code in first_predictions])
        pass_k_results = code_eval.compute(
            predictions=all_predictions,
            references=pass_k_references,
            k=K_VALUES,
        )
        codebleu_results = calc_codebleu(
            references=all_references,
            predictions=first_predictions,
            lang="python",
        )
        return dict(
            syntax_rate=syntax_rate,
            pass_k={
                k: pass_k_results[0][f"pass@{k}"].item() for k in K_VALUES
            },
            codebleu=codebleu_results["codebleu"],
        )

    dataset = load_dataset("mbpp", "sanitized", split="test")
    results = {}
    with progress_bar() as pbar:
        task_id_1 = pbar.add_task("[bold]MBPP[/]", total=len(dataset))
        task_id_2 = pbar.add_task("Temperature", total=len(TEMPERATURES))
        task_id_3 = pbar.add_task("Top-P", total=len(TOP_PS))

        for temperature in TEMPERATURES:
            pbar.reset(task_id_3)
            for top_p in TOP_PS:
                pbar.reset(task_id_1)

                all_predictions: list[list[str]] = []
                first_predictions: list[str] = []
                all_references: list[str] = []
                pass_k_references: list[str] = []
                for row in dataset:
                    # Construct conversation and tokenize
                    inputs = tokenizer.apply_chat_template(
                        [{"role": "user", "content": row["prompt"].strip()}],
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(model.device)

                    # Generate some candidates
                    with torch.inference_mode():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=4096,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                            num_return_sequences=max(K_VALUES),
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    # Process candidates for this problem
                    problem_candidates = []
                    for output in outputs:
                        response = tokenizer.decode(
                            output, skip_special_tokens=True
                        )
                        if "assistant\n" in response:
                            response = response.split("assistant\n")[-1]
                        problem_candidates.append(
                            extract_python_code(response)
                        )

                    # Store candidates and references
                    all_predictions.append(problem_candidates)
                    first_predictions.append(problem_candidates[0])
                    all_references.append(row["code"])
                    pass_k_references.append("\n".join(row["test_list"]))

                    # Update progress bar
                    pbar.update(task_id_1, advance=1)

                # Calculate and store metrics for this (temperature, top_p) pair
                results[(temperature, top_p)] = calculate_metrics(
                    all_predictions,
                    first_predictions,
                    all_references,
                    pass_k_references,
                )

                # Update progress bar
                pbar.update(task_id_3, advance=1)
            pbar.update(task_id_2, advance=1)
    return results


def main(args: argparse.Namespace) -> None:
    model, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path, is_eval=True
    )
    oasst_perplexity = eval_oasst(model, tokenizer)
    print(f"OASST Perplexity: {oasst_perplexity:.2f}\n")
    mbpp_results = eval_mbpp(model, tokenizer)
    print(mbpp_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-name-or-path",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Seeding
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    main(args)
