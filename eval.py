import argparse
import ast
import os
from statistics import mean

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import evaluate
import torch
from codebleu import calc_codebleu
from datasets import load_dataset
from rich.table import Table
from transformers import PreTrainedModel, PreTrainedTokenizer

from utils import console
from utils.misc import seed_everything
from utils.model import load_model_and_tokenizer
from utils.progress import progress_bar

TEMPERATURES = [0.1, 0.5, 0.9]
TOP_PS = [0.1, 0.5, 0.9]
K_VALUES = [1, 5, 10]

CodeGenEvalType = dict[
    tuple[float, float], dict[str, float | dict[int, float]]
]

code_eval = evaluate.load("code_eval")


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


def calculate_codegen_metrics(
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
        pass_k={k: pass_k_results[0][f"pass@{k}"].item() for k in K_VALUES},
        codebleu=codebleu_results["codebleu"],
    )


def eval_mbpp(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> CodeGenEvalType:
    def extract_func_name(test_list: list[str]) -> str:
        assert len(test_list) > 0
        code = test_list[0]
        assert code.startswith("assert ")
        return code.split("assert ")[1].split("(")[0].strip()

    dataset = load_dataset("mbpp", "sanitized", split="test")
    results = {}
    with progress_bar() as pbar:
        task_id_1 = pbar.add_task(
            "[bold]MBPP[/]",
            total=len(dataset) * len(TEMPERATURES) * len(TOP_PS),
        )
        task_id_2 = pbar.add_task("Temperature", total=len(TEMPERATURES))
        task_id_3 = pbar.add_task("Top-P", total=len(TOP_PS))

        for temperature in TEMPERATURES:
            pbar.update(task_id_2, description=f"Temperature: {temperature}")
            pbar.reset(task_id_3)
            for top_p in TOP_PS:
                pbar.update(task_id_3, description=f"Top-P: {top_p}")

                all_predictions: list[list[str]] = []
                first_predictions: list[str] = []
                all_references: list[str] = []
                pass_k_references: list[str] = []
                for row in dataset:
                    # Construct conversation and tokenize
                    func_name = extract_func_name(row["test_list"])
                    inputs = tokenizer.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": f"{row['prompt'].strip()}\nFunction name: {func_name}",
                            }
                        ],
                        add_generation_prompt=True,
                        return_tensors="pt",
                    ).to(model.device)

                    # Generate candidates
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
                    pbar.advance(task_id_1)

                # Calculate and store metrics for this (temperature, top_p) pair
                results[(temperature, top_p)] = calculate_codegen_metrics(
                    all_predictions,
                    first_predictions,
                    all_references,
                    pass_k_references,
                )

                # Update progress bar
                pbar.advance(task_id_3)
            pbar.advance(task_id_2)
    return results


def eval_humaneval(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> CodeGenEvalType:
    def combine_code(text: str, prompt: str) -> str:
        code = extract_python_code(text)
        if code.startswith("def "):
            return code
        return prompt + code

    dataset = load_dataset("openai/openai_humaneval", split="test")
    results = {}
    with progress_bar() as pbar:
        task_id_1 = pbar.add_task(
            "[bold]HumanEval[/]",
            total=len(dataset) * len(TEMPERATURES) * len(TOP_PS),
        )
        task_id_2 = pbar.add_task("Temperature", total=len(TEMPERATURES))
        task_id_3 = pbar.add_task("Top-P", total=len(TOP_PS))

        for temperature in TEMPERATURES:
            pbar.update(task_id_2, description=f"Temperature: {temperature}")
            pbar.reset(task_id_3)
            for top_p in TOP_PS:
                pbar.update(task_id_3, description=f"Top-P: {top_p}")

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

                    # Generate candidates
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
                            combine_code(response, row["prompt"])
                        )

                    # Store candidates and references
                    all_predictions.append(problem_candidates)
                    first_predictions.append(problem_candidates[0])
                    all_references.append(
                        row["prompt"] + row["canonical_solution"]
                    )
                    # Pass@k uses test field directly
                    pass_k_references.append(row["test"])

                    # Update progress bar
                    pbar.advance(task_id_1)

                # Calculate and store metrics for this (temperature, top_p) pair
                results[(temperature, top_p)] = calculate_codegen_metrics(
                    all_predictions,
                    first_predictions,
                    all_references,
                    pass_k_references,
                )

                # Update progress bar
                pbar.advance(task_id_3)
            pbar.advance(task_id_2)
    return results


def print_codegen_results_table(results: CodeGenEvalType, title: str) -> None:
    # Extract and sort unique temperature and top_p values
    temperatures = sorted(set(temp for temp, _ in results.keys()))
    top_ps = sorted(set(top_p for _, top_p in results.keys()))

    # Create table
    table = Table(title=title)

    # Add columns
    table.add_column("Temperature", style="green")
    for top_p in top_ps:
        table.add_column(f"top_p={top_p}", style="magenta")

    # Add rows
    for temp in temperatures:
        row = [f"{temp}"]
        for top_p in top_ps:
            metrics = results[(temp, top_p)]
            syntax_rate: float = metrics["syntax_rate"]
            pass_k: dict[int, float] = metrics["pass_k"]
            codebleu: float = metrics["codebleu"]
            cell = f"Syntax: {syntax_rate:.2f}\nPass@"
            cell += "/".join(str(k) for k in sorted(pass_k.keys()))
            cell += ": "
            cell += "/".join(f"{pass_k[k]:.2f}" for k in sorted(pass_k.keys()))
            cell += f"\nCodeBLEU: {codebleu:.2f}"
            row.append(cell)
        table.add_row(*row)

    # Print table
    console.print(table)


def print_mbpp_results_table(results: CodeGenEvalType) -> None:
    print_codegen_results_table(results, "MBPP Results")


def print_humaneval_results_table(results: CodeGenEvalType) -> None:
    print_codegen_results_table(results, "HumanEval Results")


def main(args: argparse.Namespace) -> None:
    model, tokenizer = load_model_and_tokenizer(
        args.model_name_or_path, is_eval=True
    )
    oasst_perplexity = eval_oasst(model, tokenizer)
    mbpp_results = eval_mbpp(model, tokenizer)
    humaneval_results = eval_humaneval(model, tokenizer)
    console.print(f"[bold]OASST Perplexity: {oasst_perplexity:.2f}[/]\n")
    print_mbpp_results_table(mbpp_results)
    console.print()
    print_humaneval_results_table(humaneval_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-name-or-path",
        type=str,
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)
