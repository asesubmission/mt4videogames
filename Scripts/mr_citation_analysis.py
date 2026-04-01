#!/usr/bin/env python3
"""
MR Citation Analysis for RQ2.

For each model, identifies "improved" clips (baseline=FN, props=TP under
game-specific MRs) and checks whether the VLM's reasoning cites a
game-specific MR ID (e.g., CP-002, RDR-001, FC-003).

Produces:
  - Per-model citation rates (Table 10a)
  - Cross-model MR frequency table (Table 10b)

Usage:
    python mr_citation_analysis.py \
        --rematched-dir ~/Desktop/rematched_csvs \
        --output mr_citation_report.txt
"""

import argparse
import os
import re
import json
import pandas as pd
from collections import defaultdict, Counter

MODEL_FILES = [
    ("G-Pro",   "vlm_eval_results_gemini-2_5-pro.csv",
                "vlm_eval_results_gemini-2_5-pro(new_60).csv"),
    ("G-Flash", "vlm_eval_results_gemini-2_5-flash.csv",
                "vlm_eval_results_gemini-2_5-flash(new_60).csv"),
    ("I-78B",   "vlm_eval_results_InternVL3-78B.csv",
                "vlm_eval_results_InternVL3-78B(new60).csv"),
    ("I-8B",    "vlm_eval_results_InternVL3-8B.csv",
                "vlm_eval_results_InternVL3-8B(new60).csv"),
    ("Q-32B",   "vlm_eval_results_Qwen2_5-VL-32B-Instruct.csv",
                "vlm_eval_results_Qwen2_5-VL-32B-Instruct(new_60).csv"),
    ("Q-7B",    "vlm_eval_results_Qwen2_5-VL-7B-Instruct.csv",
                "vlm_eval_results_Qwen2_5-VL-7B-Instruct(new60).csv"),
]

# Pattern to match game-specific MR IDs like CP-001, RDR-003, FC-002, etc.
MR_ID_PATTERN = re.compile(
    r'\b(CP|RDR|FC|GTA|JC|FO|SK|W3|F76|WD)-?\d{3}\b',
    re.IGNORECASE
)

# Mapping of MR ID prefixes to full game names
MR_PREFIX_TO_GAME = {
    "CP": "Cyberpunk 2077",
    "RDR": "Red Dead Redemption 2",
    "FC": "Far Cry 5",
    "GTA": "GTA V",
    "JC": "Just Cause 3",
    "FO": "Fallout 4",
    "SK": "Skyrim",
    "W3": "Witcher 3",
    "F76": "Fallout 76",
    "WD": "Watch Dogs 2",
}


def str_to_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().upper() == "TRUE"
    return False


def extract_mr_ids(text):
    """Extract all MR IDs from a text string."""
    if not text or pd.isna(text):
        return []
    # Normalize: sometimes models write CP002 without hyphen
    ids = MR_ID_PATTERN.findall(str(text))
    # The pattern captures the prefix only; re-extract full IDs
    full_ids = re.findall(
        r'\b((?:CP|RDR|FC|GTA|JC|FO|SK|W3|F76|WD)-?\d{3})\b',
        str(text), re.IGNORECASE
    )
    # Normalize to uppercase with hyphen
    normalized = []
    for mid in full_ids:
        # Ensure hyphen between prefix and number
        parts = re.match(r'([A-Za-z]+)-?(\d{3})', mid)
        if parts:
            normalized.append(f"{parts.group(1).upper()}-{parts.group(2)}")
    return list(set(normalized))


def is_tp(row, prefix):
    return (str_to_bool(row.get(f"{prefix}_glitch", False)) and
            str_to_bool(row.get(f"{prefix}_correct_match", False)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rematched-dir", required=True)
    parser.add_argument("--output", default="mr_citation_report.txt")
    args = parser.parse_args()

    all_improved = []  # List of (model, file_id, game, mr_ids_cited)
    model_stats = {}   # model -> {improved, citing, cite_rate}
    mr_counter = defaultdict(lambda: defaultdict(int))  # mr_id -> model -> count
    mr_total = Counter()  # mr_id -> total count across models
    mr_models = defaultdict(set)  # mr_id -> set of models that cite it

    for model_name, f210, f60 in MODEL_FILES:
        path_210 = os.path.join(args.rematched_dir, f210)
        path_60 = os.path.join(args.rematched_dir, f60)

        frames = []
        for p in [path_210, path_60]:
            if os.path.exists(p):
                frames.append(pd.read_csv(p))
            else:
                print(f"[WARN] Not found: {p}")

        if not frames:
            continue

        df = pd.concat(frames, ignore_index=True)
        buggy = df[df["gt_is_glitch"].astype(str).str.upper() == "TRUE"]

        improved_count = 0
        citing_count = 0

        for _, row in buggy.iterrows():
            baseline_tp = is_tp(row, "baseline")
            props_tp = is_tp(row, "props")

            # Improved = was FN under baseline, became TP under props
            if not baseline_tp and props_tp:
                improved_count += 1

                # Extract MR IDs from props_raw (the full VLM response)
                props_raw = str(row.get("props_raw", ""))
                mr_ids = extract_mr_ids(props_raw)

                if mr_ids:
                    citing_count += 1

                for mid in mr_ids:
                    mr_counter[mid][model_name] += 1
                    mr_total[mid] += 1
                    mr_models[mid].add(model_name)

                all_improved.append({
                    "model": model_name,
                    "file_id": row.get("file_id", ""),
                    "game": row.get("game", ""),
                    "mr_ids": mr_ids,
                })

        cite_rate = citing_count / improved_count * 100 if improved_count else 0
        model_stats[model_name] = {
            "improved": improved_count,
            "citing": citing_count,
            "cite_rate": cite_rate,
        }
        print(f"{model_name}: {improved_count} improved, {citing_count} citing ({cite_rate:.0f}%)")

    # ── Generate report ───────────────────────────────────────────
    lines = []
    lines.append("=" * 60)
    lines.append("MR CITATION ANALYSIS (RQ2)")
    lines.append("=" * 60)

    lines.append("\n--- Table 10a: Per-model citation rates ---")
    lines.append(f"{'Model':<10} {'Improved':>10} {'Cite any ID':>12} {'Cite rate':>10}")
    for model_name in [m[0] for m in MODEL_FILES]:
        s = model_stats[model_name]
        lines.append(f"{model_name:<10} {s['improved']:>10} {s['citing']:>12} {s['cite_rate']:>9.0f}%")

    # Sort MRs by number of models citing them, then by total count
    sorted_mrs = sorted(
        mr_total.keys(),
        key=lambda x: (-len(mr_models[x]), -mr_total[x])
    )

    lines.append("\n--- Table 10b: MRs cited by 3+ models ---")
    lines.append(f"{'MR ID':<10} {'# Models':>10} {'Total':>8}")
    for mid in sorted_mrs:
        n_models = len(mr_models[mid])
        if n_models >= 3:
            lines.append(f"{mid:<10} {n_models:>10} {mr_total[mid]:>8}")

    lines.append("\n--- Full MR citation table (all MRs) ---")
    lines.append(f"{'MR ID':<10} {'# Models':>10} {'Total':>8}  Models")
    for mid in sorted_mrs:
        model_list = ", ".join(sorted(mr_models[mid]))
        lines.append(f"{mid:<10} {len(mr_models[mid]):>10} {mr_total[mid]:>8}  {model_list}")

    lines.append("\n--- Per-model breakdown ---")
    for model_name in [m[0] for m in MODEL_FILES]:
        model_mrs = Counter()
        for mid, model_counts in mr_counter.items():
            if model_name in model_counts:
                model_mrs[mid] = model_counts[model_name]
        if model_mrs:
            lines.append(f"\n{model_name}:")
            for mid, count in model_mrs.most_common():
                lines.append(f"  {mid}: {count}")

    report = "\n".join(lines)
    print(f"\n{report}")

    with open(args.output, "w") as f:
        f.write(report)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
