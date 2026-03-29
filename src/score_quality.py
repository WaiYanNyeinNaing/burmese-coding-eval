#!/usr/bin/env python3
"""
score_quality.py — Aggregate rubric scores from annotation or llm_judge output.

Reads annotations.jsonl (human or LLM-generated) and prints a summary table.

Usage:
  python3 score_quality.py --input results/llm_judge_*.jsonl
  python3 score_quality.py --input my_annotations.jsonl
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

POSITIVE_FIELDS = [
    "fluency", "instruction_following", "semantic_correctness",
    "terminology", "clarity",
]
PENALTY_FIELDS = [
    "language_mixing_penalty", "script_pollution_penalty",
    "grammar_spelling_penalty", "hallucination_penalty",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_scores(row: dict) -> dict:
    base = (
        0.30 * row["fluency"] +
        0.25 * row["instruction_following"] +
        0.20 * row["semantic_correctness"] +
        0.15 * row["terminology"] +
        0.10 * row["clarity"]
    )
    penalty = (
        0.20 * row["language_mixing_penalty"] +
        0.10 * row.get("script_pollution_penalty", 0) +
        0.10 * row.get("grammar_spelling_penalty", 0) +
        0.10 * row.get("hallucination_penalty", 0)
    )
    row["base_score"]  = round(base, 4)
    row["penalty"]     = round(penalty, 4)
    row["final_score"] = round(max(0.0, base - penalty), 4)
    return row


def summarise(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        row = compute_scores(row)
        grouped[row.get("model", "unknown")].append(row)

    all_metrics = POSITIVE_FIELDS + PENALTY_FIELDS + ["base_score", "penalty", "final_score"]
    summary = {}
    for model, items in grouped.items():
        summary[model] = {"n": len(items)}
        for m in all_metrics:
            vals = [x[m] for x in items if m in x]
            if vals:
                summary[model][f"{m}_mean"] = round(mean(vals), 4)
                summary[model][f"{m}_std"]  = round(pstdev(vals) if len(vals) > 1 else 0.0, 4)
    return summary


def print_table(summary: dict) -> None:
    cols = ["fluency", "instruction_following", "semantic_correctness",
            "terminology", "clarity", "language_mixing_penalty", "final_score"]
    short = {
        "fluency": "Fluency",
        "instruction_following": "Instr.Follow",
        "semantic_correctness": "Semantic",
        "terminology": "Terminology",
        "clarity": "Clarity",
        "language_mixing_penalty": "Mix.Penalty",
        "final_score": "Final",
    }
    header = f"{'Model':<30} " + " ".join(f"{short[c]:>12}" for c in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for model, stats in summary.items():
        row_str = f"{model:<30} "
        row_str += " ".join(
            f"{stats.get(f'{c}_mean', 0.0):>12.3f}" for c in cols
        )
        print(row_str)
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Aggregate Burmese quality rubric scores")
    parser.add_argument("--input", nargs="+", required=True,
                        help="JSONL annotation file(s) from llm_judge.py or human annotators")
    parser.add_argument("--json-out", default=None,
                        help="Optional path to write JSON summary")
    args = parser.parse_args()

    all_rows: list[dict] = []
    for path_str in args.input:
        rows = load_jsonl(Path(path_str))
        all_rows.extend(rows)

    print(f"Loaded {len(all_rows)} annotations.")
    summary = summarise(all_rows)
    print_table(summary)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary written to: {args.json_out}")


if __name__ == "__main__":
    main()
