# Exponent-Aware Quantization

## Overview

An implementation of exponent-aware quantization (EXAQ) algorithm.
EXAQ is a pioneering approach to the exponent operation input quantization, based on analytical model that strategically shifts the focus towards minimizing the quantization error of subsequent to the exponent operation.


The substantial portion of the code was copied from https://github.com/EleutherAI/lm-evaluation-harness repository, whereas the main logic of EXAQ algorithm is concentrated in `lm_eval/experimental/utils.py`.


## Preparation Before Evaluation
The code was mainly tested on `nvcr.io/nvidia/pytorch:24.03-py3` image.

Before usage, install all dependencies.
```bash
pip install -r requirements.txt
```


## Evaluation
Basic script for evaluation:

```bash
PYTHONPATH=${path_to_current_repository} \
python __main__.py \
--model hf \
--model_args pretrained=${model} \
-tasks ${task} \
--device cuda:0 \
--batch_size 4 \
--dtype bfloat16 \
--replace-sdpa \
--quantize \
--cast-dtype float32 \
--bitwidth ${bitwidth} \
--clip-type ${clip_type} \
--calibrate
```

where:
- `model` is one of the llama models, i.e. any version and any size (Example: `huggyllama/llama-7b`).
- `task` is evaluation tasks: `boolq, piqa, hellaswag, winogrande, arc_challenge, arc_easy, openbookqa`
- `bitwidth` is one of the following: `2, 3, 4`
- `clip_type` is one of the following: `NONE, GAUSS`



