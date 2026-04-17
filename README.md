# EVE: Verifiable Self-Evolution of MLLMs via Executable Visual Transformations

Official implementation of **EVE**, a method enabling Multimodal Large Language Models (MLLMs) to self-evolve through executable visual transformations.

## Overview

EVE introduces a **Challenger-Solver dual-policy framework** where two policies co-evolve:

- **Challenger**: Generates executable Python image-editing programs with diverse parameter sets, creating verifiable visual transformation tasks.
- **Solver**: Learns to answer VQA questions derived from code execution results, receiving deterministic reward signals from the environment.

Unlike pseudo-label or template-based methods, EVE learns from an **external execution environment** — code execution provides deterministic, verifiable feedback independent of model confidence.


## Installation

```bash
git clone https://github.com/YOUR_USERNAME/eve.git
cd eve

# Create conda environment
conda env create -f env.yml
conda activate eve

# Install package
pip install -e .
```

**Requirements**: Python 3.10+, PyTorch 2.8+, vLLM 0.11+, transformers 4.57+, CUDA-capable GPU(s)

## Quick Start

### 1. Set environment variables

```bash
export BASEDIR=/path/to/eve
export OUTPUT_DIR=${BASEDIR}/output
export SAVE_CHECKPOINT_DIR=${OUTPUT_DIR}/checkpoints
export HF_HOME=/path/to/huggingface/cache

export BASE_MODEL=/path/to/Qwen3-VL-8B-Instruct
```

### 2. Run the full training loop

```bash
cd ${BASEDIR}
bash recipe/EVE/main.sh
```

This alternates between Challenger training and Solver training for `ITER_NUM` iterations.

## Training Details

### Challenger Training (`C_train.sh`)

Trains the Challenger policy to generate executable image-editing programs. The Challenger is rewarded for:
- Format validity and code executability
- Difficulty calibration (targeting ~50% Solver accuracy)
- Diversity of generated programs

```bash
# Called internally by main.sh; can also run standalone:
challenger_model=/path/to/model
solver_model=/path/to/model
experiment_name=my_experiment
bash recipe/EVE/C_train.sh
```

### Solver Training (`S_train.sh`)

Trains the Solver policy on VQA tasks derived from Challenger-generated code executions. Two question types:
- **Type-0**: Given parameters, identify which image was produced
- **Type-1**: Given an edited image, identify which parameters produced it

```bash
bash recipe/EVE/S_train.sh
```

### Key Configuration (`recipe/EVE/configs/grpo.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PQ_K` | 50 | Priority queue size for code examples |
| `TOTAL_TRAINING_STEPS` | 10 | Training steps per iteration |
| `ITER_NUM` | 5 | Number of evolution iterations |
| `SAMPLE_NUM` | 6 | Samples per prompt |

## Code Structure

```
EVE/
├── recipe/EVE/
│   ├── CS.py                  # Challenger-Solver dataset & reward logic
│   ├── main.sh                # Main training entry point
│   ├── C_train.sh             # Challenger training script
│   ├── S_train.sh             # Solver training script
│   ├── sandbox.py             # Code execution sandbox
│   ├── make_code_pq.py        # Priority queue construction
│   ├── gen_S_data_from_C.py   # Generate Solver data from Challenger output
│   ├── prepare_rl_data.py     # Data preprocessing pipeline
│   └── configs/grpo.yaml      # Training configuration
├── verl/
│   ├── trainer/main_ppo.py    # PPO/GRPO training entry point
│   └── utils/
│       ├── cs_sandbox.py      # Sandbox utilities
│       ├── cs_prompts.py      # System/user prompts
│       └── cs_tools.py        # Evaluation tools
└── VLMEvalKit/                # Evaluation suite
```

## Evaluation

We use [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for benchmark evaluation.

```bash
cd VLMEvalKit

# Configure model path, then run:
bash qwen3vl_8b_eval.sh
```


<!-- ## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{eve2026,
  title     = {EVE: Verifiable Self-Evolution of MLLMs via Executable Visual Transformations},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026}
}
``` -->

## Acknowledgments

This code builds upon:
- [VERL](https://github.com/volcengine/verl) — RL training infrastructure
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) — Multimodal evaluation suite
- [Vision-SR1-47K](https://huggingface.co/datasets/LMMs-Lab-Turtle/Vision-SR1-47K) - Image dataset

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
