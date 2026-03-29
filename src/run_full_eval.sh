#!/bin/bash

# Burmese Coder Evaluation Pipeline Automator
# Usage: ./burmese_eval/run_full_eval.sh <model_name> <completions_jsonl> [judge_model]

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <model_name> <completions_jsonl> [judge_model]"
    echo "Example: $0 burmese_coder_v4 results/completions_v4.jsonl gemini-2.5-flash-lite"
    exit 1
fi

MODEL_NAME=$1
COMPLETIONS_FILE=$2
JUDGE_MODEL=${3:-"gemini-2.5-flash-lite"}

if [ -z "$GOOGLE_API_KEY" ]; then
    echo "Error: GOOGLE_API_KEY environment variable is not set."
    exit 1
fi

# 1. Setup results directory
DIR_NAME=$(echo "$MODEL_NAME" | sed 's/:/_/g')
mkdir -p "results/$DIR_NAME/logs"

echo "-------------------------------------------------------"
echo "🚀 Starting Full Evaluation for: $MODEL_NAME"
echo "📂 Target Directory: results/$DIR_NAME"
echo "⚖️ Judge Model: $JUDGE_MODEL"
echo "-------------------------------------------------------"

# 2. Functional Evaluation
echo "🧪 Running Functional Evaluation (pass@k)..."
python3 burmese_eval/evaluate.py --completions "$COMPLETIONS_FILE" > "results/$DIR_NAME/logs/functional_eval.log" 2>&1
FUNC_JSONL=$(ls -t results/functional_eval_*.jsonl | head -n 1)
mv "$FUNC_JSONL" "results/$DIR_NAME/"

# 3. Reference-based Evaluation
echo "📏 Running Reference-based Evaluation (chrF/BLEU)..."
python3 burmese_eval/reference_eval.py --completions "$COMPLETIONS_FILE" > "results/$DIR_NAME/logs/reference_eval.log" 2>&1
REF_JSONL=$(ls -t results/reference_eval_*.jsonl | head -n 1)
mv "$REF_JSONL" "results/$DIR_NAME/"

# 4. LLM Judge Rubric Scoring
echo "⚖️ Running LLM-as-a-Judge (Rubric Scoring)..."
python3 burmese_eval/llm_judge.py --completions "$COMPLETIONS_FILE" --judge-model "$JUDGE_MODEL" --sleep 0.5 > "results/$DIR_NAME/logs/llm_judge.log" 2>&1
JUDGE_JSONL=$(ls -t results/llm_judge_*.jsonl | head -n 1)
mv "$JUDGE_JSONL" "results/$DIR_NAME/"

# 5. Summary Report
echo -e "\n✅ Evaluation Complete! Final Report:\n"
echo "--- Functional Correction ---"
tail -n 5 "results/$DIR_NAME/logs/functional_eval.log" | grep "pass@1" || echo "Check functional_eval.log"

echo -e "\n--- Quality Rubric ---"
python3 burmese_eval/score_quality.py --input "results/$DIR_NAME/$(basename $JUDGE_JSONL)"

echo -e "\n--- Reference Similarity ---"
python3 -c "import json; rows=list(map(json.loads, open('results/$DIR_NAME/$(basename $REF_JSONL)'))); print(f'chrF Expl: {sum(r[\"chrf_explanation\"] for r in rows)/len(rows):.2f}'); print(f'chrF Code: {sum(r[\"chrf_code\"] for r in rows)/len(rows):.2f}')"

echo -e "\n🔗 All detailed logs saved in: results/$DIR_NAME/logs/"
echo "-------------------------------------------------------"
