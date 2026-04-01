import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_and_save_summary(
    rows: List[Dict[str, Any]],
    model_name: str,
    output_json_path: str,
    experiment_start_time: Optional[str] = None,
    experiment_end_time: Optional[str] = None,
    total_wall_time_s: Optional[float] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    gt_rows = [r for r in rows if r.get("has_gt", False)]
    buggy_rows = [r for r in gt_rows if r.get("gt_is_glitch") is True]
    clean_rows = [r for r in gt_rows if r.get("gt_is_glitch") is not True]

    n_total = len(rows)
    n_with_gt = len(gt_rows)
    n_no_gt = n_total - n_with_gt
    n_buggy = len(buggy_rows)
    n_clean = len(clean_rows)

    n_baseline_invalid = sum(
        1 for r in rows if not r.get("baseline_valid_json", True)
    )
    n_props_invalid = sum(
        1 for r in rows if not r.get("props_valid_json", True)
    )

    # ── Detection metrics (description-aware)
    #
    #   TP — Buggy clip + model detected + description matches GT
    #        (correctly caught AND correctly described the bug)
    #
    #   FN — Buggy clip + model missed OR detected but wrong description
    #        (failed to correctly identify the bug)
    #
    #   FP — Clean clip + model said "glitch detected"
    #        (hallucinated a bug on a normal clip)
    #
    #   TN — Clean clip + model said "no glitch"
    #        (correctly stayed silent)
    #
    #   Recall    = TP / (TP + FN)  — of all buggy clips, how many
    #               were correctly detected AND described
    #   Precision = TP / (TP + FP)  — of all positive predictions,
    #               how many were real bugs correctly described
    #

    def is_true(x: Any) -> bool:
        return x is True

    def is_correct_bug_report(r: Dict[str, Any], key_glitch: str, key_match: str) -> bool:
        return (
            r.get("gt_is_glitch") is True
            and r.get(key_glitch) is True
            and r.get(key_match) is True
        )

    def compute_detection(rows_gt, key_glitch, key_match):
        tp = fp = tn = fn = 0
        for r in rows_gt:
            pred = is_true(r.get(key_glitch))
            actual = is_true(r.get("gt_is_glitch"))
            match = is_true(r.get(key_match))

            if actual:
                if pred and match:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred:
                    fp += 1
                else:
                    tn += 1

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        accuracy = safe_div(tp + tn, tp + fp + tn + fn)
        specificity = safe_div(tn, tn + fp)
        fpr = safe_div(fp, fp + tn)

        return {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "specificity": round(specificity, 4),
            "fpr": round(fpr, 4),
        }

    baseline_det = compute_detection(
        gt_rows, "baseline_glitch", "baseline_correct_match"
    )
    props_det = compute_detection(
        gt_rows, "props_glitch", "props_correct_match"
    )

    

    baseline_pred_pos = [r for r in gt_rows if r.get("baseline_glitch") is True]
    props_pred_pos = [r for r in gt_rows if r.get("props_glitch") is True]

    baseline_correct = sum(
        1 for r in baseline_pred_pos if r.get("baseline_correct_match") is True
    )
    props_correct = sum(
        1 for r in props_pred_pos if r.get("props_correct_match") is True
    )

    baseline_desc_rate = safe_div(baseline_correct, len(baseline_pred_pos))
    props_desc_rate = safe_div(props_correct, len(props_pred_pos))

   

    improved = sum(
        1
        for r in buggy_rows
        if not is_correct_bug_report(
            r, "baseline_glitch", "baseline_correct_match"
        )
        and is_correct_bug_report(
            r, "props_glitch", "props_correct_match"
        )
    )

    regressed = sum(
        1
        for r in buggy_rows
        if is_correct_bug_report(
            r, "baseline_glitch", "baseline_correct_match"
        )
        and not is_correct_bug_report(
            r, "props_glitch", "props_correct_match"
        )
    )

   

    game_stats = defaultdict(
        lambda: {
            "n": 0,
            "n_buggy": 0,
            "n_clean": 0,
            "baseline": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
            "props": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        }
    )

    for r in gt_rows:
        g = r["game"]
        game_stats[g]["n"] += 1
        actual = is_true(r.get("gt_is_glitch"))

        if actual:
            game_stats[g]["n_buggy"] += 1
        else:
            game_stats[g]["n_clean"] += 1

        for method, key_glitch, key_match in [
            ("baseline", "baseline_glitch", "baseline_correct_match"),
            ("props", "props_glitch", "props_correct_match"),
        ]:
            pred = is_true(r.get(key_glitch))
            match = is_true(r.get(key_match))

            if actual:
                if pred and match:
                    game_stats[g][method]["tp"] += 1
                else:
                    game_stats[g][method]["fn"] += 1
            else:
                if pred:
                    game_stats[g][method]["fp"] += 1
                else:
                    game_stats[g][method]["tn"] += 1

    per_game = {}
    for g, s in sorted(game_stats.items()):
        per_game[g] = {
            "n": s["n"],
            "n_buggy": s["n_buggy"],
            "n_clean": s["n_clean"],
        }
        for method in ["baseline", "props"]:
            d = s[method]
            prec = safe_div(d["tp"], d["tp"] + d["fp"])
            rec = safe_div(d["tp"], d["tp"] + d["fn"])
            spec = safe_div(d["tn"], d["tn"] + d["fp"])
            per_game[g][method] = {
                "tp": d["tp"],
                "fp": d["fp"],
                "fn": d["fn"],
                "tn": d["tn"],
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "specificity": round(spec, 4),
            }



    def sum_field(fld):
        return sum(r.get(fld, 0) or 0 for r in rows)

    def mean_field(fld):
        vals = [r[fld] for r in rows if r.get(fld)]
        return round(sum(vals) / len(vals), 3) if vals else 0

    timing = {
        "total_baseline_time_s": round(sum_field("baseline_time_s"), 2),
        "total_props_time_s": round(sum_field("props_time_s"), 2),
        "mean_baseline_time_s": mean_field("baseline_time_s"),
        "mean_props_time_s": mean_field("props_time_s"),
        "total_wall_time_s": (
            round(total_wall_time_s, 2) if total_wall_time_s else None
        ),
    }

   

    tokens = {
        "total_baseline_prompt_tokens": sum_field("baseline_prompt_tokens"),
        "total_baseline_completion_tokens": sum_field(
            "baseline_completion_tokens"
        ),
        "total_props_prompt_tokens": sum_field("props_prompt_tokens"),
        "total_props_completion_tokens": sum_field(
            "props_completion_tokens"
        ),
    }
    tokens["total_prompt_tokens"] = (
        tokens["total_baseline_prompt_tokens"]
        + tokens["total_props_prompt_tokens"]
    )
    tokens["total_completion_tokens"] = (
        tokens["total_baseline_completion_tokens"]
        + tokens["total_props_completion_tokens"]
    )
    tokens["total_tokens"] = (
        tokens["total_prompt_tokens"] + tokens["total_completion_tokens"]
    )


    summary = {
        "model": model_name,
        "experiment_start": experiment_start_time,
        "experiment_end": experiment_end_time,
        "dataset": {
            "total_clips": n_total,
            "clips_with_gt": n_with_gt,
            "clips_without_gt": n_no_gt,
            "buggy_clips": n_buggy,
            "clean_clips": n_clean,
        },
        "invalid_json_responses": {
            "baseline": n_baseline_invalid,
            "with_properties": n_props_invalid,
        },
        "detection_metrics": {
            "baseline": baseline_det,
            "with_properties": props_det,
        },
        "description_correctness": {
            "baseline": {
                "correct": baseline_correct,
                "predicted_positives": len(baseline_pred_pos),
                "rate": round(baseline_desc_rate, 4),
            },
            "with_properties": {
                "correct": props_correct,
                "predicted_positives": len(props_pred_pos),
                "rate": round(props_desc_rate, 4),
            },
        },
        "improvement": {
            "improved": improved,
            "regressed": regressed,
            "net_gain": improved - regressed,
        },
        "per_game": per_game,
        "timing": timing,
        "tokens": tokens,
    }

    if extra_metadata:
        summary["metadata"] = extra_metadata

    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[INFO] Wrote summary JSON to {output_json_path}")

    print_summary(summary)

    return summary


def print_summary(s: Dict[str, Any]) -> None:
    """Print standardized console summary. Identical across all models."""

    print("\n" + "=" * 70)
    print(f"  EVALUATION SUMMARY — {s['model']}")
    print("=" * 70)

    d = s["dataset"]
    no_gt = d.get("clips_without_gt", 0)
    no_gt_str = f", {no_gt} no GT" if no_gt else ""
    print(
        f"\n  Dataset: {d['total_clips']} clips "
        f"({d['buggy_clips']} buggy, {d['clean_clips']} clean{no_gt_str})"
    )

    inv = s.get("invalid_json_responses", {})
    if inv.get("baseline", 0) or inv.get("with_properties", 0):
        print(
            f"  Invalid JSON responses: baseline={inv['baseline']}, "
            f"MRs={inv['with_properties']}"
        )

    
    print(
        f"\n  ┌──────────────────────────────────────────────────────────────┐"
    )
    print(
        f"  │  TP = Buggy clip + detected + description matches GT       │"
    )
    print(
        f"  │  FN = Buggy clip + missed OR wrong description             │"
    )
    print(
        f"  │  FP = Clean clip + model hallucinated a bug                │"
    )
    print(
        f"  │  TN = Clean clip + model correctly stayed silent           │"
    )
    print(
        f"  │                                                            │"
    )
    print(
        f"  │  Recall    = TP/(TP+FN) — % of buggy clips got right      │"
    )
    print(
        f"  │  Precision = TP/(TP+FP) — % of predictions that are real  │"
    )
    print(
        f"  └──────────────────────────────────────────────────────────────┘"
    )

   
    bl = s["detection_metrics"]["baseline"]
    pr = s["detection_metrics"]["with_properties"]
    print(
        f"\n  {'DETECTION METRICS':<28} {'Baseline':>10} {'With MRs':>10} {'Delta':>10}"
    )
    print(f"  {'-' * 60}")
    for m in ["accuracy", "precision", "recall", "f1", "specificity", "fpr"]:
        delta = pr[m] - bl[m]
        print(
            f"  {m.capitalize():<28} {bl[m]:>10.4f} {pr[m]:>10.4f} {delta:>+10.4f}"
        )
    print(
        f"  {'TP / FP / FN / TN':<28} "
        f"{bl['tp']:>3}/{bl['fp']:>3}/{bl['fn']:>3}/{bl['tn']:>3}     "
        f"{pr['tp']:>3}/{pr['fp']:>3}/{pr['fn']:>3}/{pr['tn']:>3}"
    )

    
    bld = s["description_correctness"]["baseline"]
    prd = s["description_correctness"]["with_properties"]
    print(
        f"\n  {'DESC. CORRECTNESS':<28} {'Baseline':>10} {'With MRs':>10}"
    )
    print(f"  {'-' * 60}")
    print(
        f"  {'Correct / Pred. Positives':<28} "
        f"{bld['correct']:>4}/{bld['predicted_positives']:<5}     "
        f"{prd['correct']:>4}/{prd['predicted_positives']:<5}"
    )
    print(f"  {'Rate':<28} {bld['rate']:>10.4f} {prd['rate']:>10.4f}")

    
    imp = s["improvement"]
    print(
        f"\n  Improved  (baseline not-TP → MRs TP):  {imp['improved']}"
    )
    print(
        f"  Regressed (baseline TP     → MRs not-TP): {imp['regressed']}"
    )
    print(
        f"  Net gain:                                {imp['net_gain']:+d}"
    )

   
    t = s["timing"]
    if t.get("total_wall_time_s"):
        print(
            f"\n  Total wall time: {t['total_wall_time_s']:.1f}s "
            f"({t['total_wall_time_s'] / 60:.1f} min)"
        )
    print(
        f"  Mean time per clip:  baseline={t['mean_baseline_time_s']:.2f}s  "
        f"MRs={t['mean_props_time_s']:.2f}s"
    )

    
    tok = s["tokens"]
    if tok["total_tokens"] > 0:
        print(
            f"\n  Total tokens: {tok['total_tokens']:,} "
            f"(prompt: {tok['total_prompt_tokens']:,}, "
            f"completion: {tok['total_completion_tokens']:,})"
        )

    print("=" * 70)