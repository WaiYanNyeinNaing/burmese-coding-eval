#!/usr/bin/env python3
"""
llm_judge.py — Burmese quality judge using Google Gemma-3-27B via Gemini API.

Uses the new google-genai client (v0.8+):
  client.models.generate_content(model=..., contents=..., config=...)

Requires only: GOOGLE_API_KEY env variable (or --api-key flag)

The judge evaluates generated output against:
  - The Burmese rubric in eval_criteria.md (9 dimensions)
  - The ground-truth reference from humaneval_burmese.jsonl (code + burmese_explanation)

Usage:
  export GOOGLE_API_KEY=your_key_here

  python3 burmese_eval/llm_judge.py \\
      --completions results/completions_my_model_*.jsonl

  # Use different judge model
  python3 burmese_eval/llm_judge.py \\
      --completions results/completions_*.jsonl \\
      --judge-model gemini-2.0-flash

Output:
  results/llm_judge_{model}_{judge_model}_{timestamp}.jsonl
  Compatible with score_quality.py
"""

from __future__ import annotations
import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Optional, List, Dict

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    pass

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BENCHMARK_FILE = Path(__file__).parent / "benchmark_dataset.jsonl"
RESULTS_DIR    = Path(__file__).parent.parent / "results"

# --------------------------------------------------------------------------- #
# Prompts — grounded in eval_criteria.md rubric
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """\
You are an expert evaluator of Burmese (မြန်မာဘာသာ) programming education content.
Your task is to score a model's response using a strict rubric.
Return ONLY valid JSON. Do not explain or add prose.
"""

JUDGE_PROMPT_TEMPLATE = """\
## Burmese Code Generation Evaluation

### Original Instruction (Burmese)
{instruction}

### Reference Answer (Ground Truth)
**Reference Code:**
```python
{reference_code}
```
**Reference Burmese Explanation:**
{reference_explanation}

### Model Response to Evaluate
{model_response}

---

## Scoring Rubric
Score EACH dimension as an integer 0–4:

**Positive dimensions (higher = better):**
- fluency: Natural Burmese language throughout (4=perfect, 0=unusable)
- instruction_following: Follows the Burmese coding prompt (4=fully, 0=fails)
- semantic_correctness: Meaning matches the reference explanation (4=correct, 0=wrong)
- terminology: Technical terms correct and consistent with reference (4=correct, 0=misleading)
- clarity: Easy for a Burmese reader to understand (4=very clear, 0=unclear)

**Penalty dimensions (lower = better, 0 = no penalty):**
- language_mixing_penalty: Unnecessary English mixed into Burmese text (0=none, 4=dominant)
- script_pollution_penalty: Romanised Burmese or mixed script (0=none, 4=severe)
- grammar_spelling_penalty: Burmese grammar or spelling errors (0=none, 4=severe)
- hallucination_penalty: False statements about the code behaviour (0=none, 4=severe)

Return ONLY this JSON:
{{
  "fluency": <int 0-4>,
  "instruction_following": <int 0-4>,
  "semantic_correctness": <int 0-4>,
  "terminology": <int 0-4>,
  "clarity": <int 0-4>,
  "language_mixing_penalty": <int 0-4>,
  "script_pollution_penalty": <int 0-4>,
  "grammar_spelling_penalty": <int 0-4>,
  "hallucination_penalty": <int 0-4>,
  "notes": "<one short English sentence about the key strength or weakness>"
}}
"""

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_json_from_text(text: str) -> dict | None:
    """Robustly extract JSON from model output (strips markdown fences)."""
    # Try fenced block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try bare JSON
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def compute_final_score(row: dict) -> dict:
    """Apply eval_criteria.md formula."""
    base = (
        0.30 * row.get("fluency", 0) +
        0.25 * row.get("instruction_following", 0) +
        0.20 * row.get("semantic_correctness", 0) +
        0.15 * row.get("terminology", 0) +
        0.10 * row.get("clarity", 0)
    )
    penalty = (
        0.20 * row.get("language_mixing_penalty", 0) +
        0.10 * row.get("script_pollution_penalty", 0) +
        0.10 * row.get("grammar_spelling_penalty", 0) +
        0.10 * row.get("hallucination_penalty", 0)
    )
    row["base_score"]  = round(base, 4)
    row["penalty"]     = round(penalty, 4)
    row["final_score"] = round(max(0.0, base - penalty), 4)
    return row


def error_scores(reason: str) -> dict:
    """Return zero scores on judge failure."""
    return {
        "fluency": 0, "instruction_following": 0,
        "semantic_correctness": 0, "terminology": 0, "clarity": 0,
        "language_mixing_penalty": 4, "script_pollution_penalty": 0,
        "grammar_spelling_penalty": 0, "hallucination_penalty": 0,
        "notes": f"[ERROR] {reason}",
    }

def call_judge_google(
    client,
    judge_model: str,
    instruction: str,
    ref_code: str,
    ref_explanation: str,
    model_response: str,
    max_output_tokens: int = 512,
    retries: int = 3,
) -> dict:
    """
    Call google judge with exponential backoff for 429 errors.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction=instruction[:2000],
        reference_code=ref_code[:1500],
        reference_explanation=ref_explanation[:1500],
        model_response=model_response[:2500],
    )

    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=judge_model,
                contents=full_prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=max_output_tokens,
                    temperature=0.0,
                ),
            )
            raw_text = response.text
            scores = parse_json_from_text(raw_text)
            if not scores:
                raise ValueError(f"Could not parse JSON from judge output:\n{raw_text[:300]}")
            return scores

        except Exception as exc:
            # Check for rate limit (429)
            exc_str = str(exc)
            if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                if attempt < retries:
                    wait = 20 * (attempt + 1)
                    print(f"  [WAIT] Rate limit hit (429). Sleeping {wait}s... (Attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                    continue
            raise exc

    return error_scores("Max retries exceeded")


def call_judge_deepseek(
    client,
    judge_model: str,
    instruction: str,
    ref_code: str,
    ref_explanation: str,
    model_response: str,
    max_output_tokens: int = 512,
    retries: int = 3,
) -> dict:
    """
    Call DeepSeek judge.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        instruction=instruction[:2000],
        reference_code=ref_code[:1500],
        reference_explanation=ref_explanation[:1500],
        model_response=model_response[:2500],
    )

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=max_output_tokens,
                stream=False
            )
            raw_text = response.choices[0].message.content.strip()
            scores = parse_json_from_text(raw_text)
            if not scores:
                raise ValueError(f"Could not parse JSON from judge output:\n{raw_text[:300]}")
            return scores

        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate limit" in exc_str.lower():
                if attempt < retries:
                    wait = 20 * (attempt + 1)
                    print(f"  [WAIT] Rate limit hit. Sleeping {wait}s... (Attempt {attempt+1}/{retries})")
                    time.sleep(wait)
                    continue
            raise exc

    return error_scores("Max retries exceeded")

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge (Gemma-3-27B) for Burmese HumanEval quality"
    )
    parser.add_argument(
        "--completions", nargs="+", required=True,
        help="Completions JSONL file(s) from run_inference.py",
    )
    parser.add_argument(
        "--judge-model", default="gemma-3-27b-it",
        help="Judge model ID (default: gemma-3-27b-it)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Google API key (or set GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=512,
        help="Max output tokens for judge (default: 512)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only judge first N problems (for testing)",
    )
    parser.add_argument(
        "--benchmark", default=str(BENCHMARK_FILE),
        help="Path to humaneval_burmese.jsonl (ground truth)",
    )
    parser.add_argument(
        "--sleep", type=float, default=1.0,
        help="Seconds to sleep between API calls (rate limiting)",
    )
    parser.add_argument(
        "--provider", choices=["google", "deepseek"], default="google",
        help="API provider (default: google)",
    )
    args = parser.parse_args()

    # --- Init client ---
    if args.provider == "google":
        api_key = args.api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            parser.error("No API key found. Export GOOGLE_API_KEY=your_key or use --api-key.")
        client = genai.Client(api_key=api_key)
    else:
        api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            parser.error("No API key found. Export DEEPSEEK_API_KEY=your_key or use --api-key.")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # --- Load ground truth ---
    ground_truth: dict[str, dict] = {}
    for row in load_jsonl(Path(args.benchmark)):
        ground_truth[row["task_id"]] = row

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for comp_path_str in args.completions:
        comp_path = Path(comp_path_str)
        completions = load_jsonl(comp_path)
        if not completions:
            print(f"Empty: {comp_path}")
            continue

        model = completions[0].get("model", comp_path.stem)
        if args.limit:
            completions = completions[: args.limit]

        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", model)
        safe_judge = re.sub(r"[^a-zA-Z0-9_.-]", "_", args.judge_model)
        out_path   = RESULTS_DIR / f"llm_judge_{safe_model}_{safe_judge}_{timestamp}.jsonl"

        all_scores: list[dict] = []
        print(f"\n{'='*60}")
        print(f"Model  : {model}")
        print(f"Judge  : {args.judge_model}")
        print(f"Tasks  : {len(completions)}")
        print(f"Output : {out_path}")
        print('='*60)

        with out_path.open("w", encoding="utf-8") as out_f:
            for i, rec in enumerate(completions, 1):
                task_id = rec["task_id"]
                raw_completions: list[str] = rec.get("completions", [])
                if not raw_completions:
                    continue

                gt = ground_truth.get(task_id, {})
                instruction    = gt.get("burmese_instruction", "")
                ref_code       = gt.get("code", "")
                ref_explanation = gt.get("burmese_explanation", "")

                # Judge only first completion (use --n 1 in run_inference for single sample)
                response_text = raw_completions[0]

                try:
                    if args.provider == "google":
                        scores = call_judge_google(
                            client,
                            judge_model=args.judge_model,
                            instruction=instruction,
                            ref_code=ref_code,
                            ref_explanation=ref_explanation,
                            model_response=response_text,
                            max_output_tokens=args.max_tokens,
                        )
                    else:
                        scores = call_judge_deepseek(
                            client,
                            judge_model=args.judge_model,
                            instruction=instruction,
                            ref_code=ref_code,
                            ref_explanation=ref_explanation,
                            model_response=response_text,
                            max_output_tokens=args.max_tokens,
                        )
                except Exception as exc:
                    print(f"  [{i:3d}] {task_id} ⚠ Judge error: {exc}")
                    scores = error_scores(str(exc)[:100])

                record = {
                    "task_id": task_id,
                    "model":   model,
                    "judge":   args.judge_model,
                    **scores,
                }
                record = compute_final_score(record)
                all_scores.append(record)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

                print(
                    f"  [{i:3d}/{len(completions)}] {task_id}"
                    f"  final={record['final_score']:.2f}"
                    f"  fluency={scores.get('fluency',0)}"
                    f"  mix_pen={scores.get('language_mixing_penalty',0)}"
                    f"  | {scores.get('notes','')[:55]}"
                )

                if i < len(completions):
                    time.sleep(args.sleep)

        # --- Summary ---
        if all_scores:
            metrics = [
                "fluency", "instruction_following", "semantic_correctness",
                "terminology", "clarity",
                "language_mixing_penalty", "hallucination_penalty",
                "base_score", "penalty", "final_score",
            ]
            print(f"\n{'='*60}")
            print(f"SUMMARY — {model}  (n={len(all_scores)})")
            print(f"{'Metric':<30} {'Mean':>6}  {'Std':>6}")
            print("-" * 46)
            for m in metrics:
                vals = [r[m] for r in all_scores if m in r]
                std  = pstdev(vals) if len(vals) > 1 else 0.0
                print(f"  {m:<28} {mean(vals):>6.3f}  {std:>6.3f}")
            print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
