#!/usr/bin/env python3
"""
run_inference.py — Run a model against all Burmese HumanEval benchmark problems.

Supports:
  - Ollama  (local, default)
  - OpenAI-compatible API (pass --api-base + --api-key)

Usage:
  # Ollama (local)
  python3 run_inference.py --model burmese-coder --n 1

  # OpenAI compatible (e.g. Together AI, DeepSeek, etc.)
  python3 run_inference.py --model gpt-4o \\
      --api-base https://api.openai.com/v1 \\
      --api-key sk-... \\
      --n 3

Output:
  results/completions_{model}_{timestamp}.jsonl
  Fields: task_id, model, completions (list[str])
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BENCHMARK_FILE = Path(__file__).parent / "benchmark_dataset.jsonl"
RESULTS_DIR = Path(__file__).parent.parent / "results"

SYSTEM_PROMPT = (
    "သင်သည် ကျွမ်းကျင်သော Python programmer တစ်ဦးဖြစ်သည်။ "
    "Burmese (မြန်မာဘာသာ) ဖြင့် ပေးထားသော coding question များကို ဖြေဆိုပါ။ "
    "Python code ကို markdown code block ထဲတွင်သာ ထုတ်ပေးပါ။ "
    "ရှင်းလင်းချက်များကို မြန်မာဘာသာဖြင့် ရေးပါ။"
)


def load_benchmark() -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")
    
    print("Loading benchmark dataset from Hugging Face (WYNN747/burmese-human-eval)...")
    dataset = load_dataset("WYNN747/burmese-human-eval", split="test")
    return list(dataset)


# --------------------------------------------------------------------------- #
# Inference backends
# --------------------------------------------------------------------------- #

def run_ollama(model: str, prompt: str, n: int, temperature: float) -> list[str]:
    """Call Ollama REST API."""
    import urllib.request

    completions = []
    for _ in range(n):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        completions.append(data["message"]["content"])
    return completions


def run_openai_compat(
    model: str,
    prompt: str,
    n: int,
    temperature: float,
    api_base: str,
    api_key: str,
) -> list[str]:
    """Call OpenAI-compatible API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(api_key=api_key, base_url=api_base)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        n=n,
        temperature=temperature,
    )
    return [choice.message.content for choice in resp.choices]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def build_prompt(row: dict) -> str:
    return row.get("prompt", "")


def main():
    parser = argparse.ArgumentParser(description="Run LLM inference on Burmese HumanEval")
    parser.add_argument("--model",       required=True, help="Model name")
    parser.add_argument("--n",           type=int, default=1,
                        help="Number of completions per problem (for pass@k)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--api-base",    default=None,
                        help="OpenAI-compat base URL (omit for Ollama)")
    parser.add_argument("--api-key",     default=None,
                        help="API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--limit",       type=int, default=None,
                        help="Only run first N problems (for testing)")
    parser.add_argument("--benchmark",   default=str(BENCHMARK_FILE),
                        help="Path to benchmark_dataset.jsonl")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    use_ollama = args.api_base is None

    rows = load_benchmark()
    if args.limit:
        rows = rows[: args.limit]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", args.model)
    out_path = RESULTS_DIR / f"completions_{safe_model}_{timestamp}.jsonl"

    print(f"Model     : {args.model}")
    print(f"Backend   : {'Ollama' if use_ollama else args.api_base}")
    print(f"Problems  : {len(rows)}")
    print(f"n per prob: {args.n}")
    print(f"Output    : {out_path}\n")

    with out_path.open("w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows, 1):
            prompt = build_prompt(row)
            try:
                if use_ollama:
                    completions = run_ollama(args.model, prompt, args.n, args.temperature)
                else:
                    completions = run_openai_compat(
                        args.model, prompt, args.n, args.temperature,
                        args.api_base, api_key,
                    )
            except Exception as exc:
                print(f"  [{i:3d}/{len(rows)}] {row['task_id']} ERROR: {exc}")
                completions = [""] * args.n

            record = {
                "task_id":     row["task_id"],
                "model":       args.model,
                "completions": completions,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"  [{i:3d}/{len(rows)}] {row['task_id']} OK")
            time.sleep(0.1)  # rate-limit courtesy

    print(f"\nDone! Completions saved to: {out_path}")


if __name__ == "__main__":
    main()
