---
name: burmese_code_eval
description: Evaluation suite for Burmese coding models. Supports functional correctness, reference metrics, and rubric-based judging.
version: 1.2.0
repository: https://github.com/WaiYanNyeinNaing/burmese-coding-eval
license: MIT
tags: [Burmese, Evaluation, HumanEval, Benchmarking]
---

# Burmese Code Evaluation Skill

| Field      | Value                                           |
| ---------- | ----------------------------------------------- |
| Identifier | `burmese_code_eval`                             |
| Version    | 1.2.0                                           |
| Repository | [burmese-coding-eval](https://github.com/WaiYanNyeinNaing/burmese-coding-eval) |
| Category   | Evaluation                                      |

Evaluation methodology developed to address the lack of specialized benchmarks for Burmese programming assistants, providing a standardized framework for measuring technical and linguistic quality.

## 🤖 Instructions

1.  **Requirement**: The Python `datasets` library must be installed to fetch the benchmark internally from `WYNN747/burmese-human-eval`. No local dataset files are required.
2.  **Inference**: Generate completions using `run_inference.py`.
3.  **Evaluation**: Execute `run_full_eval.sh` for multi-track reporting.

## 🚀 Usage

### 1. Setup
```bash
git clone https://github.com/WaiYanNyeinNaing/burmese-coding-eval.git
cd burmese-coding-eval
pip install sacrebleu google-genai openai datasets
```

### 2. Execution
```bash
# Inference
python3 src/run_inference.py --model <model_name>

# Evaluation
./src/run_full_eval.sh <model_name> results/completions_*.jsonl
```

## Metrics
- **Pass@1**: Functional correctness.
- **Rubric**: Expert linguistic scoring.
- **Mix Penalty**: Hallucination detection.
- **chrF**: Character n-gram similarity.
