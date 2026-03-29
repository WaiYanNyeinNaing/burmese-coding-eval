# Burmese-Coding-Eval

This repository addresses the critical lack of specialized benchmarks and evaluation metrics for Burmese programming assistants by proposing a novel, standardized multi-track framework.

While the **evaluation metrics and methodology** are original contributions designed to solve current benchmarking gaps, the **benchmark dataset** expands upon **OpenAI Human-Eval** to ensure global comparability.

*   **Burmese HumanEval Dataset:** A standardized benchmark dataset, available on Hugging Face: [WYNN747/burmese-human-eval](https://huggingface.co/datasets/WYNN747/burmese-human-eval).
*   **Evaluation Metrics:** A novel multi-track evaluation methodology proposed by Dr. Wai Yan Nyein Naing to assess functional correctness, rubric-based quality, linguistic alignment, and reference statistical metrics.

## 📂 Suite Structure

```bash
src/
├── run_full_eval.sh          # ONE-CLICK: Runs functional, reference, and LLM tracks
├── generate_final_report.py  # Aggregates results into a side-by-side comparison table
│
├── run_inference.py          # STEP 1: Generate model completions (Ollama or API)
├── evaluate.py               # STEP 2A: Functional correctness (Pass@1)
├── llm_judge.py              # STEP 2B: Rubric scoring (Gemini 2.5 Flash Lite)
└── score_quality.py          # Internal utility for rubric aggregation
```

## 🛠️ Installation

```bash
git clone https://github.com/WaiYanNyeinNaing/burmese-coding-eval.git
cd burmese-coding-eval
pip install sacrebleu google-genai openai datasets
```

## 🚀 Quick Start (Automated)

**This repository is for evaluation purposes only.**

To generate model completions:
```bash
python3 src/run_inference.py --model <model_id>
```

To execute the full pipeline (Functional + Rubric):
```bash
export GOOGLE_API_KEY=your_key_here
./src/run_full_eval.sh <model_name> results/completions_*.jsonl
```

To aggregate results into a comparison table:
```bash
python3 src/generate_final_report.py
```
All benchmark results and logs are saved in the git-ignored **`results/`** directory, organized by model name.

## 📊 Metric Breakdown

We use a 2-track evaluation to ensure technical accuracy and linguistic purity:

| Track | Tool | Logic | Metric |
| :--- | :--- | :--- | :--- |
| **Functional** | `evaluate.py` | Unit Test Execution | **Pass@1** |
| **LLM Judge** | `llm_judge.py` | Expert Rubric | **Quality (0-4)** |

### Scoring Rubric (Expert Grounded)
The `llm_judge.py` uses **Gemini 2.5 Flash Lite** to score responses against 9 dimensions:
- **Fluency**: Natural Burmese flow.
- **Terminology**: Correct technical terms (e.g. 'ကိန်းပြည့်' for integer).
- **Mix Penalty**: Quantifies the presence of foreign language hallucinations in native Burmese explanations.
- **Semantic Correctness**: Logic alignment with the human reference.

## Citation

If you use this benchmark or the proposed evaluation metrics in your research, please cite:

**Burmese-Coding-Eval: A Translated and Expanded HumanEval Benchmark for Burmese Programming Assistants.**  
Nyein Naing, W. Y. (2026). [https://github.com/WaiYanNyeinNaing/burmese-coding-eval](https://github.com/WaiYanNyeinNaing/burmese-coding-eval)  
Contact: [waiyan.nn18@gmail.com](mailto:waiyan.nn18@gmail.com)

## References

**OpenAI Human-Eval**  
(Source for the standardized dataset format).
