#!/usr/bin/env python3
"""
evaluate.py — Functional correctness evaluator for Burmese HumanEval.

Reads model completions, extracts Python code, runs check() assertions,
and computes pass@1 / pass@k.

Usage:
  python3 evaluate.py --completions results/completions_my_model_*.jsonl
  python3 evaluate.py --completions results/completions_my_model_*.jsonl --k 1 3 5

Output:
  results/functional_eval_{model}_{timestamp}.jsonl  — per-problem results
  Prints summary table to stdout.
"""

from __future__ import annotations
import argparse
import contextlib
import io
import json
import math
import re
import signal
import textwrap
import traceback
from pathlib import Path
from datetime import datetime

BENCHMARK_FILE = Path(__file__).parent / "benchmark_dataset.jsonl"
RESULTS_DIR = Path(__file__).parent.parent / "results"


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


def extract_python_code(text: str) -> str:
    """
    Extract Python code from model output.
    Handles:
      - ```python ... ``` fenced blocks
      - ``` ... ``` fenced blocks
      - Raw code (fallback)
    """
    # Try fenced code blocks first
    pattern = r"```(?:python|burmese)?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: strip common prose lines and return rest
    lines = text.split("\n")
    code_lines = [l for l in lines if not l.strip().startswith("#") or "def " in l]
    return "\n".join(code_lines).strip()


class TimeoutError(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: int):
    def _handler(sig, frame):
        raise TimeoutError("Timed out")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def run_check(code: str, test_list: str, timeout: int = 10) -> tuple[bool, str]:
    """
    Execute candidate code + check() function.
    Returns (passed, error_message).
    """
    # Build the full program: candidate code + test harness
    full_code = textwrap.dedent(code) + "\n" + textwrap.dedent(test_list) + "\ncheck(globals().get(list(filter(lambda k: callable(globals()[k]) and k not in ['check'], globals()))[0]))"

    # Find the function name from code
    fn_match = re.search(r"^def\s+(\w+)\s*\(", code, re.MULTILINE)
    if not fn_match:
        return False, "No function definition found in code"
    fn_name = fn_match.group(1)

    exec_code = textwrap.dedent(code) + "\n" + textwrap.dedent(test_list) + f"\ncheck({fn_name})\n"

    try:
        with time_limit(timeout):
            namespace: dict = {}
            exec(exec_code, namespace)  # noqa: S102
        return True, ""
    except TimeoutError:
        return False, "Timeout"
    except AssertionError as e:
        return False, f"AssertionError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator from Chen et al. 2021.
    n = total samples, c = correct samples, k = k
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Evaluate Burmese HumanEval completions")
    parser.add_argument("--completions", nargs="+", required=True,
                        help="Path(s) to completions JSONL from run_inference.py")
    parser.add_argument("--k",  nargs="+", type=int, default=[1],
                        help="k values for pass@k (default: 1)")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Timeout per test case in seconds")
    parser.add_argument("--benchmark", default=str(BENCHMARK_FILE),
                        help="Path to humaneval_burmese.jsonl (answer keys)")
    args = parser.parse_args()

    # Load answer keys from Hugging Face
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package is required. Run 'pip install datasets'")
        return

    print("Loading benchmark dataset from Hugging Face (WYNN747/burmese-human-eval)...")
    dataset = load_dataset("WYNN747/burmese-human-eval", split="test")
    answer_key = {}
    for row in dataset:
        answer_key[row["task_id"]] = row

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for comp_path_str in args.completions:
        comp_path = Path(comp_path_str)
        completions = load_jsonl(comp_path)
        if not completions:
            print(f"Empty file: {comp_path}")
            continue

        model = completions[0].get("model", comp_path.stem)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", model)
        out_path = RESULTS_DIR / f"functional_eval_{safe_model}_{timestamp}.jsonl"

        results = []
        total = 0
        passed_at_1 = 0

        print(f"\nEvaluating: {model} ({len(completions)} problems)")
        print("-" * 55)

        for rec in completions:
            task_id = rec["task_id"]
            raw_completions: list[str] = rec.get("completions", [])
            if not raw_completions:
                continue

            key = answer_key.get(task_id)
            if not key:
                print(f"  SKIP {task_id} (not in answer key)")
                continue

            test_list = key.get("test", key.get("test_list", ""))
            n = len(raw_completions)
            c = 0
            per_completion = []

            for raw in raw_completions:
                code = extract_python_code(raw)
                passed, err = run_check(code, test_list, timeout=args.timeout)
                if passed:
                    c += 1
                per_completion.append({"passed": passed, "error": err})

            # pass@k
            passk = {f"pass@{k}": round(pass_at_k(n, c, k), 4)
                     for k in args.k if k <= n}

            total += 1
            if c > 0:
                passed_at_1 += 1

            status = "✓" if c > 0 else "✗"
            print(f"  {status} {task_id}  ({c}/{n} passed)  {passk}")

            results.append({
                "task_id": task_id,
                "model": model,
                "n_samples": n,
                "n_passed": c,
                **passk,
                "per_completion": per_completion,
            })

        # Write results
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Summary
        print("\n" + "=" * 55)
        print(f"Model    : {model}")
        print(f"Problems : {total}")
        for k in args.k:
            if k <= (results[0]["n_samples"] if results else 1):
                avg = sum(r.get(f"pass@{k}", 0) for r in results) / max(len(results), 1)
                print(f"pass@{k}  : {avg:.4f} ({avg*100:.1f}%)")
        print(f"Output   : {out_path}")


if __name__ == "__main__":
    main()
