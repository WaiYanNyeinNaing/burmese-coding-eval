#!/usr/bin/env python3
"""
reference_eval.py — Reference-based metrics (chrF, BLEU) for Burmese HumanEval.

Since humaneval_burmese.jsonl has gold-standard `code` and `burmese_explanation`,
we can compute automatic reference-based metrics alongside LLM judging.

Metrics:
  - chrF  : Character n-gram F-score (best for Burmese — no tokenizer needed)
  - BLEU  : sacrebleu sentence BLEU (standard MT metric)
  - CodeBLEU (optional): structural code similarity

Requires:
  pip install nltk sacrebleu

Usage:
  python3 burmese_eval/reference_eval.py \\
      --completions results/completions_my_model_*.jsonl

Output:
  results/reference_eval_{model}_{timestamp}.jsonl
  Prints per-model summary table.
"""

from __future__ import annotations
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

BENCHMARK_FILE = Path(__file__).parent / "benchmark_dataset.jsonl"
RESULTS_DIR    = Path(__file__).parent.parent / "results"

# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #

def chrf_score(hypothesis: str, reference: str, char_order: int = 6) -> float:
    """
    chrF — character n-gram precision/recall/F-score.
    Perfect for Burmese: no word tokenizer required.
    Returns 0.0–100.0.
    """
    try:
        import sacrebleu
        score = sacrebleu.corpus_chrf(
            [hypothesis],
            [[reference]],
            char_order=char_order,
        )
        return round(score.score, 2)
    except ImportError:
        # Fallback: simple character overlap
        hyp_chars = set(hypothesis)
        ref_chars = set(reference)
        if not ref_chars:
            return 0.0
        overlap = len(hyp_chars & ref_chars)
        p = overlap / max(len(hyp_chars), 1)
        r = overlap / max(len(ref_chars), 1)
        f = 2 * p * r / max(p + r, 1e-9)
        return round(f * 100, 2)


def bleu_score(hypothesis: str, reference: str) -> float:
    """Sentence BLEU using sacrebleu. Returns 0.0–100.0."""
    try:
        import sacrebleu
        score = sacrebleu.sentence_bleu(hypothesis, [reference])
        return round(score.score, 2)
    except ImportError:
        return -1.0  # signal not available


def extract_python_code(text: str) -> str:
    """Extract code block from model output."""
    match = re.search(r"```(?:python|burmese)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_burmese_text(text: str) -> str:
    """
    Extract Burmese prose from model output.
    Strips code blocks, keeps the rest.
    """
    cleaned = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return cleaned.strip()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Reference-based evaluation (chrF, BLEU) for Burmese HumanEval"
    )
    parser.add_argument("--completions", nargs="+", required=True,
                        help="Completions JSONL from run_inference.py")
    parser.add_argument("--benchmark", default=str(BENCHMARK_FILE),
                        help="Path to humaneval_burmese.jsonl (ground truth)")
    args = parser.parse_args()

    # Load ground truth from Hugging Face
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package is required. Run 'pip install datasets'")
        return

    print("Loading benchmark dataset from Hugging Face (WYNN747/burmese-human-eval)...")
    dataset = load_dataset("WYNN747/burmese-human-eval", split="test")
    ground_truth: dict[str, dict] = {}
    for row in dataset:
        ground_truth[row["task_id"]] = row

    # Check sacrebleu
    try:
        import sacrebleu
        has_sacrebleu = True
    except ImportError:
        print("⚠ sacrebleu not installed — BLEU will be skipped. pip install sacrebleu")
        has_sacrebleu = False

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for comp_path_str in args.completions:
        comp_path = Path(comp_path_str)
        completions = load_jsonl(comp_path)
        if not completions:
            print(f"Empty: {comp_path}")
            continue

        model = completions[0].get("model", comp_path.stem)
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", model)
        out_path   = RESULTS_DIR / f"reference_eval_{safe_model}_{timestamp}.jsonl"

        results = []

        print(f"\n{'='*60}")
        print(f"Model   : {model}")
        print(f"Metric  : chrF (char n-gram F) + BLEU")
        print(f"Ground truth from: humaneval_burmese.jsonl")
        print('='*60)

        with out_path.open("w", encoding="utf-8") as out_f:
            for rec in completions:
                task_id = rec["task_id"]
                raw_completions = rec.get("completions", [])
                if not raw_completions:
                    continue

                gt = ground_truth.get(task_id, {})
                ref_code        = gt.get("code", "")
                ref_explanation = gt.get("burmese_explanation", "")

                # Use first completion
                model_output  = raw_completions[0]
                gen_code      = extract_python_code(model_output)
                gen_burmese   = extract_burmese_text(model_output)

                # --- Metrics ---
                # 1. chrF on Burmese explanation (vs reference burmese_explanation)
                chrf_explanation = chrf_score(gen_burmese, ref_explanation)

                # 2. chrF on code (vs reference code)
                chrf_code = chrf_score(gen_code, ref_code)

                # 3. BLEU on Burmese explanation
                bleu_explanation = bleu_score(gen_burmese, ref_explanation) if has_sacrebleu else -1.0

                # 4. BLEU on code
                bleu_code = bleu_score(gen_code, ref_code) if has_sacrebleu else -1.0

                record = {
                    "task_id":           task_id,
                    "model":             model,
                    "chrf_explanation":  chrf_explanation,
                    "chrf_code":         chrf_code,
                    "bleu_explanation":  bleu_explanation,
                    "bleu_code":         bleu_code,
                }
                results.append(record)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(
                    f"  {task_id:<20}"
                    f"  chrF_expl={chrf_explanation:5.1f}"
                    f"  chrF_code={chrf_code:5.1f}"
                    f"  BLEU_expl={bleu_explanation:5.1f}"
                )

        # --- Summary ---
        if results:
            metrics = ["chrf_explanation", "chrf_code"]
            if has_sacrebleu:
                metrics += ["bleu_explanation", "bleu_code"]

            print(f"\n{'='*60}")
            print(f"SUMMARY — {model}  (n={len(results)})")
            print(f"\n  Metric meaning:")
            print(f"  • chrF_explanation : similarity of Burmese prose vs reference (0–100)")
            print(f"  • chrF_code        : code similarity vs reference solution (0–100)")
            print(f"  • BLEU_explanation : BLEU score of Burmese explanation (0–100)")
            print(f"  • BLEU_code        : BLEU score of generated code (0–100)\n")
            print(f"{'Metric':<22} {'Mean':>7}  {'Std':>7}")
            print("-" * 40)
            for m in metrics:
                vals = [r[m] for r in results if r[m] >= 0]
                std  = pstdev(vals) if len(vals) > 1 else 0.0
                print(f"  {m:<20} {mean(vals):>7.2f}  {std:>7.2f}")
            print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
