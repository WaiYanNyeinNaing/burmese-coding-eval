import os
import json
import glob

def get_stats(model_dir):
    stats = {
        "Model": os.path.basename(model_dir),
        "Pass@1": "N/A",
        "chrF": 0.0,
        "Judges": {}
    }
    
    # dimensions from score_quality.py
    dimensions = ["fluency", "instruction_following", "semantic_correctness", 
                  "terminology", "clarity", "language_mixing_penalty", 
                  "hallucination_penalty", "final_score"]

    # 1. Functional
    func_files = glob.glob(os.path.join(model_dir, "functional_eval_*.jsonl"))
    if func_files:
        with open(func_files[0]) as f:
            rows = [json.loads(l) for l in f if l.strip()]
            if rows:
                passed = sum(1 for r in rows if r.get('n_passed', 0) > 0)
                stats["Pass@1"] = f"{(passed/len(rows))*100:.1f}%"
    
    # 2. LLM Judge - Breakdown
    judge_files = glob.glob(os.path.join(model_dir, "llm_judge_*.jsonl"))
    for jf in judge_files:
        judge_name = "DeepSeek" if "deepseek" in jf.lower() else "Gemini"
        if judge_name not in stats["Judges"]:
            stats["Judges"][judge_name] = {d: 0.0 for d in dimensions}
            with open(jf) as f:
                rows = [json.loads(l) for l in f if l.strip()]
                if rows:
                    for d in dimensions:
                        stats["Judges"][judge_name][d] = sum(r.get(d, 0) for r in rows) / len(rows)

    # 3. Reference
    ref_files = glob.glob(os.path.join(model_dir, "reference_eval_*.jsonl"))
    if ref_files:
        with open(ref_files[0]) as f:
            rows = [json.loads(l) for l in f if l.strip()]
            if rows:
                stats["chrF"] = sum(r['chrf_explanation'] for r in rows)/len(rows)

    return stats

def main():
    results_dir = "results"
    model_dirs = sorted([d for d in glob.glob(os.path.join(results_dir, "*")) 
                 if os.path.isdir(d) and os.path.basename(d) not in ["archive", "logs"]])
    
    if not model_dirs:
      print("No model results found in results/ folder.")
      return

    report_lines = []
    report_lines.append("\n🚀 FINAL BURMESE CODER BENCHMARK COMPARISON\n")
    
    # Identify judges present
    judge_keys = set()
    model_stats = []
    for d in model_dirs:
        s = get_stats(d)
        model_stats.append(s)
        judge_keys.update(s["Judges"].keys())
    
    judge_keys = sorted(list(judge_keys))
    
    # 1. Summary Table
    rubric_headers = " | ".join([f"Rubric({j})" for j in judge_keys])
    header = f"{'Model':<25} | {'Pass@1':<8} | {rubric_headers} | {'chrF':<8}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    
    for s in model_stats:
        rubric_scores = []
        for j in judge_keys:
            score = s["Judges"].get(j, {}).get("final_score", 0.0)
            rubric_scores.append(f"{score:<{8+len(j)-6}.3f}")
        rubrics_str = " | ".join(rubric_scores)
        row = f"{s['Model']:<25} | {s['Pass@1']:<8} | {rubrics_str} | {s.get('chrF', 0.0):<8.2f}"
        report_lines.append(row)

    # 2. Rubric Breakdown
    report_lines.append("\n⚖️ LLM JUDGE DETAILED BREAKDOWN (0-4 Scope)")
    
    for judge in judge_keys:
        report_lines.append(f"\n--- JUDGE: {judge} ---")
        header2 = f"{'Model':<25} | {'Fluency':<7} | {'Instr.':<7} | {'Semantic':<8} | {'Term.':<7} | {'MixPen':<7}"
        report_lines.append(header2)
        report_lines.append("-" * len(header2))
        for s in model_stats:
            j_stats = s["Judges"].get(judge)
            if j_stats:
                row2 = f"{s['Model']:<25} | {j_stats['fluency']:<7.2f} | {j_stats['instruction_following']:<7.2f} | {j_stats['semantic_correctness']:<8.2f} | {j_stats['terminology']:<7.2f} | {j_stats['language_mixing_penalty']:<7.2f}"
                report_lines.append(row2)
            else:
                row2 = f"{s['Model']:<25} | {'N/A':<7} | {'N/A':<7} | {'N/A':<8} | {'N/A':<7} | {'N/A':<7}"
                report_lines.append(row2)

    # 3. Metric Glossary
    report_lines.append("\n📖 METRIC GLOSSARY")
    report_lines.append("-----------------")
    report_lines.append("• Pass@1: % of coding tasks that passed all unit tests (Strict Logic Accuracy).")
    report_lines.append("• Rubric Score: Overall quality based on the 9-dimension Burmese expert rubric.")
    report_lines.append("• chrF: Character-level similarity to human ground-truth (Burmese vocabulary).")
    report_lines.append("• Fluency: How natural the Burmese explanation sounds.")
    report_lines.append("• Mix Penalty: Points deducted when the model uses non-Burmese (English/Hallucinated) words.")
    report_lines.append("• Semantic Correctness: How accurately the explanation matches the actual code logic.")
    report_lines.append("• Terminology: Usage of correct Burmese technical terms (e.g. 'ကိန်းပြည့်' for integer).")
    
    report_content = "\n".join(report_lines)
    print(report_content)
    
    # Save to file
    model_names = "_vs_".join([os.path.basename(d) for d in model_dirs])
    if len(model_names) > 50: model_names = model_names[:47] + "..."
    save_path = os.path.join(results_dir, f"final_comparison_{model_names}.txt")
    with open(save_path, "w") as f:
        f.write(report_content)
        f.write(f"\n\nDetailed reports stored in: results/<model_name>/\n")

    print(f"\n✅ FULL REPORT SAVED TO: {save_path}")

if __name__ == "__main__":
    main()
