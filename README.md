# AIST5030 Mini-Project: Parameter-efficient Finetuning for Pretrained Foundation Models

Use orthogonal finetuning (OFT) to finetune a pretrained model (eg, Stable Diffusion, Llama, Qwen, or any pretrained models) for any downstream task and summarize your experimental results and findings.

## Installation

```bash
conda create -n aist5030 python=3.12 -y
conda activate aist5030
python -m pip install -r requirements.txt
python -m pip install tree-sitter==0.24.0  # See https://github.com/k4black/codebleu/issues/62
```

## Usage

### Finetuning

```text
python finetune.py \
  [-m MODEL_NAME_OR_PATH] \
  [-e EPOCHS] \
  [-lr LEARNING_RATE] \
  [-b BATCH_SIZE] \
  [--gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS] \
  [--oft-block-size OFT_BLOCK_SIZE] \
  [--seed SEED]

options:
  -m MODEL_NAME_OR_PATH, --model-name-or-path MODEL_NAME_OR_PATH
  -e EPOCHS, --epochs EPOCHS
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
  -b BATCH_SIZE, --batch-size BATCH_SIZE
  --gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS
  --oft-block-size OFT_BLOCK_SIZE
  --seed SEED
```

### Evaluation

```text
python eval.py [-m MODEL_NAME_OR_PATH] [--seed SEED]

options:
  -m MODEL_NAME_OR_PATH, --model-name-or-path MODEL_NAME_OR_PATH
  --seed SEED
```
