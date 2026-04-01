import argparse
import os
import pandas as pd
import numpy as np

try:
    from scipy.stats import binom_test, chi2
    from statsmodels.stats.contingency_tables import mcnemar
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARN] statsmodels not installed. Using scipy only.")
    print("       pip install statsmodels for exact McNemar test")


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


def str_to_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().upper() == "TRUE"
    return False


def is_tp(row, prefix):
    """A clip is TP if the model detected a bug AND the description matched GT."""
    return (str_to_bool(row.get(f"{prefix}_glitch", False)) and
            str_to_bool(row.get(f"{prefix}_correct_match", False)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rematched-dir", required=True,
                        help="Directory with rematched CSVs")
    args = parser.parse_args()

    print("=" * 70)
    print("McNEMAR'S TEST: Baseline vs Game-Specific MRs")
    print("=" * 70)
    print(f"\nH0: No difference in TP rate between baseline and MR-augmented")
    print(f"H1: MR-augmented condition has different TP rate")
    print(f"Test: McNemar's exact test (two-sided) on 135 paired buggy clips\n")

    results = []

    for model_name, f210, f60 in MODEL_FILES:
        path_210 = os.path.join(args.rematched_dir, f210)
        path_60 = os.path.join(args.rematched_dir, f60)

        frames = []
        for p in [path_210, path_60]:
            if os.path.exists(p):
                frames.append(pd.read_csv(p))

        if not frames:
            print(f"[ERROR] No files found for {model_name}")
            continue

        df = pd.concat(frames, ignore_index=True)
        buggy = df[df["gt_is_glitch"].astype(str).str.upper() == "TRUE"]

     
        a = 0  # baseline=TP, props=TP (both correct)
        b = 0  # baseline=TP, props=notTP (regression)
        c = 0  # baseline=notTP, props=TP (improvement)
        d = 0  # baseline=notTP, props=notTP (both missed)

        for _, row in buggy.iterrows():
            bl_tp = is_tp(row, "baseline")
            pr_tp = is_tp(row, "props")

            if bl_tp and pr_tp:
                a += 1
            elif bl_tp and not pr_tp:
                b += 1
            elif not bl_tp and pr_tp:
                c += 1
            else:
                d += 1

        n = a + b + c + d
        discordant = b + c

        print(f"--- {model_name} ---")
        print(f"  Contingency table (n={n} buggy clips):")
        print(f"                    MR=TP   MR=notTP")
        print(f"  Baseline=TP       {a:>5}    {b:>5}")
        print(f"  Baseline=notTP    {c:>5}    {d:>5}")
        print(f"  Discordant pairs: b={b}, c={c} (total={discordant})")

    
        if HAS_STATSMODELS:
           
            table = np.array([[a, b], [c, d]])
            if discordant <= 25:
                
                result = mcnemar(table, exact=True)
                test_type = "exact"
            else:
              
                result = mcnemar(table, exact=False, correction=True)
                test_type = "chi2 (corrected)"

            p_value = result.pvalue
            stat = result.statistic
        else:
      
        
            from scipy.stats import binom_test
            if discordant > 0:
                p_value = binom_test(b, b + c, 0.5)
                test_type = "exact (scipy)"
                stat = None
            else:
                p_value = 1.0
                test_type = "n/a (no discordant)"
                stat = None

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

        print(f"  McNemar ({test_type}): p = {p_value:.4f} {sig}")
        if stat is not None:
            print(f"  Statistic: {stat:.4f}")
        print()

        results.append({
            "model": model_name,
            "n": n,
            "stayed_tp": a,
            "regressed": b,
            "improved": c,
            "stayed_fn": d,
            "p_value": p_value,
            "significant": p_value < 0.05,
        })


    print("=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    print(f"\n{'Model':<10} {'Improved':>10} {'Regressed':>10} {'p-value':>10} {'Sig?':>6}")
    print("-" * 50)
    for r in results:
        sig = "Yes" if r["significant"] else "No"
        print(f"{r['model']:<10} {r['improved']:>10} {r['regressed']:>10} {r['p_value']:>10.4f} {sig:>6}")

    print(f"\nSignificance threshold: p < 0.05")
    sig_count = sum(1 for r in results if r["significant"])
    print(f"Significant: {sig_count} of {len(results)} models")

  
    print(f"\nFor the paper:")
    for r in results:
        p = r["p_value"]
        if p < 0.001:
            p_str = "$p < 0.001$"
        else:
            p_str = f"$p = {p:.3f}$"
        print(f"  {r['model']}: {p_str}")


if __name__ == "__main__":
    main()
