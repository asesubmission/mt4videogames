
import argparse
import glob
import os
import sys
import time

import pandas as pd
import requests

from shared_eval import compute_and_save_summary



SEMANTIC_MATCHER_URL = os.getenv(
    "SEMANTIC_MATCHER_URL", "http://127.0.0.1:8002"
)
DATA_DIR = os.getenv(
    "DATA_DIR", "."
)


CSV_PATTERNS = [
    os.path.join(DATA_DIR, "vlm_eval_results_*.csv"),
]



def descriptions_match(
    pred: str,
    gt1: str | None,
    gt2: str | None,
) -> dict:
   
    if not pred:
        return {"match": False, "reasoning": "empty prediction", "backend": "n/a"}
    if not gt1 and not gt2:
        return {"match": False, "reasoning": "no ground truth", "backend": "n/a"}

    try:
        response = requests.post(
            f"{SEMANTIC_MATCHER_URL}/match",
            json={
                "predicted_description": pred,
                "ground_truth_1": gt1,
                "ground_truth_2": gt2,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "match": bool(data.get("match", False)),
            "reasoning": str(data.get("reasoning", "")),
            "backend": str(data.get("backend", "unknown")),
        }
    except Exception as e:
        raise RuntimeError(
            f"Matcher server unreachable at {SEMANTIC_MATCHER_URL}: {e}"
        ) from e


#  Preflight

def preflight_check() -> dict:
    """Verify matcher server is running and return its info."""
    print(f"[PREFLIGHT] Pinging matcher at {SEMANTIC_MATCHER_URL}/health ...")
    resp = requests.get(f"{SEMANTIC_MATCHER_URL}/health", timeout=10)
    resp.raise_for_status()
    info = resp.json()
    print(
        f"[PREFLIGHT] OK. Backend: {info.get('backend')}, "
        f"Model: {info.get('model')}, "
        f"Version: {info.get('version')}"
    )
    return info


# Core re-matching logic

def rematch_csv(csv_path: str, dry_run: bool = False) -> str | None:
   
    print(f"\n{'='*70}")
    print(f"  Re-matching: {os.path.basename(csv_path)}")
    print(f"{'='*70}")

    df = pd.read_csv(csv_path)


    bool_cols = [
        "gt_is_glitch", "has_gt",
        "baseline_glitch", "props_glitch",
        "baseline_correct_match", "props_correct_match",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip().str.lower() == "true"
            )

    for col in ["baseline_valid_json", "props_valid_json"]:
        if col not in df.columns:
            df[col] = True
        else:
            df[col] = (
                df[col].astype(str).str.strip().str.lower() == "true"
            )

   
    buggy_mask = df["has_gt"] & df["gt_is_glitch"]

    baseline_needs_match = buggy_mask & df["baseline_glitch"]
    props_needs_match = buggy_mask & df["props_glitch"]

    n_baseline = baseline_needs_match.sum()
    n_props = props_needs_match.sum()
    n_total = n_baseline + n_props

    print(f"  Total rows: {len(df)}")
    print(f"  Baseline matches to evaluate: {n_baseline}")
    print(f"  MR matches to evaluate: {n_props}")
    print(f"  Total matcher calls: {n_total}")

    if dry_run:
        print("  [DRY RUN] Skipping actual matcher calls.")
        return None

    if n_total == 0:
        print("  Nothing to match. Skipping.")
        return None

 
    df["baseline_correct_match"] = False
    df["props_correct_match"] = False

  
    df["baseline_match_reasoning"] = ""
    df["props_match_reasoning"] = ""

 
    matched_count = 0
    t0 = time.time()

    for idx in df.index:
        row = df.loc[idx]

        gt1 = str(row["gt_descr1"]).strip() if pd.notna(row["gt_descr1"]) else None
        gt2 = str(row["gt_descr2"]).strip() if pd.notna(row["gt_descr2"]) else None
        if gt1 == "nan":
            gt1 = None
        if gt2 == "nan":
            gt2 = None

   
        if baseline_needs_match.loc[idx]:
            pred = str(row["baseline_desc"]).strip() if pd.notna(row["baseline_desc"]) else ""
            matched_count += 1
            result = descriptions_match(pred, gt1, gt2)
            df.at[idx, "baseline_correct_match"] = result["match"]
            df.at[idx, "baseline_match_reasoning"] = result["reasoning"]

            status = "MATCH" if result["match"] else "NO_MATCH"
            print(
                f"  [{matched_count:3d}/{n_total}] baseline | "
                f"{row['file_id']} | {status} | "
                f"{result['reasoning'][:60]}"
            )

  
        if props_needs_match.loc[idx]:
            pred = str(row["props_desc"]).strip() if pd.notna(row["props_desc"]) else ""
            matched_count += 1
            result = descriptions_match(pred, gt1, gt2)
            df.at[idx, "props_correct_match"] = result["match"]
            df.at[idx, "props_match_reasoning"] = result["reasoning"]

            status = "MATCH" if result["match"] else "NO_MATCH"
            print(
                f"  [{matched_count:3d}/{n_total}] with_mr  | "
                f"{row['file_id']} | {status} | "
                f"{result['reasoning'][:60]}"
            )

    elapsed = time.time() - t0
    print(f"\n  Matching done in {elapsed:.1f}s ({elapsed/max(n_total,1):.2f}s per call)")


    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_csv = os.path.join(DATA_DIR, f"{base_name}_rematched.csv")
    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")


    output_json = os.path.join(DATA_DIR, f"{base_name}_rematched.json")

   
    rows = df.to_dict("records")
    for r in rows:
        for col in bool_cols + ["baseline_valid_json", "props_valid_json"]:
            if col in r:
                r[col] = bool(r[col])

  
    model_name = base_name.replace("vlm_eval_results_", "")

    compute_and_save_summary(
        rows=rows,
        model_name=f"{model_name} (LLM-matched)",
        output_json_path=output_json,
    )

    return output_csv



def find_csvs() -> list[str]:
    
    csvs = []
    for pattern in CSV_PATTERNS:
        csvs.extend(glob.glob(pattern))
    
    csvs = [c for c in csvs if "_rematched" not in c]
    return sorted(csvs)


def main():
    parser = argparse.ArgumentParser(
        description="Re-match existing VLM eval CSVs with current matcher server"
    )
    parser.add_argument(
        "--csvs",
        nargs="+",
        default=None,
        help="Specific CSV paths to re-match (default: auto-discover)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be matched without calling the server",
    )
    args = parser.parse_args()

   
    if not args.dry_run:
        try:
            matcher_info = preflight_check()
        except Exception as e:
            print(f"[FATAL] Matcher server not reachable: {e}")
            print(
                "Start the Gemini matcher:\n"
                "  export GEMINI_API_KEY='your-key'\n"
                "  uvicorn server_semantic_matcher_gemini:app --port 8002"
            )
            sys.exit(1)
    else:
        matcher_info = {"backend": "dry-run"}

    
    if args.csvs:
        csv_paths = args.csvs
    else:
        csv_paths = find_csvs()

    if not csv_paths:
        print("[FATAL] No result CSVs found.")
        print(f"  Searched patterns: {CSV_PATTERNS}")
        sys.exit(1)

    print(f"\n[INFO] Found {len(csv_paths)} CSV(s) to re-match:")
    for p in csv_paths:
        print(f"  - {os.path.basename(p)}")

   
    results = []
    for csv_path in csv_paths:
        output = rematch_csv(csv_path, dry_run=args.dry_run)
        if output:
            results.append(output)

    
    print(f"\n{'='*70}")
    print(f"  DONE — {len(results)} CSV(s) re-matched")
    print(f"{'='*70}")
    for r in results:
        print(f"  → {r}")
    print(
        f"\n  Matcher backend: {matcher_info.get('backend', 'unknown')}"
    )


if __name__ == "__main__":
    main()