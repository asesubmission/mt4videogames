import argparse
import sys
import pandas as pd

try:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
except ImportError:
    print("pip install scikit-learn")
    sys.exit(1)


def normalize_label(val):
    
    if pd.isna(val):
        return None
    s = str(val).strip().upper()
    if s in ("MATCH", "TRUE", "1", "YES", "TP", "CORRECT"):
        return 1
    elif s in ("NO_MATCH", "NO MATCH", "FALSE", "0", "NO", "FP", "INCORRECT", "NOMATCH"):
        return 0
    if "MATCH" in s and "NO" not in s:
        return 1
    return 0


def compute_and_report(labels_a, labels_b, name_a, name_b):
    """Compute Cohen's kappa and print a report."""
    a = [normalize_label(x) for x in labels_a]
    b = [normalize_label(x) for x in labels_b]

    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        print(f"  {name_a} vs {name_b}: No valid paired labels found.")
        return None

    a_clean = [p[0] for p in pairs]
    b_clean = [p[1] for p in pairs]

    n = len(a_clean)
    agree = sum(1 for x, y in zip(a_clean, b_clean) if x == y)
    kappa = cohen_kappa_score(a_clean, b_clean)

    if kappa >= 0.81:
        interp = "Almost perfect"
    elif kappa >= 0.61:
        interp = "Substantial"
    elif kappa >= 0.41:
        interp = "Moderate"
    elif kappa >= 0.21:
        interp = "Fair"
    else:
        interp = "Slight/Poor"

    cm = confusion_matrix(a_clean, b_clean, labels=[0, 1])

    print(f"\n  {name_a} vs {name_b}:")
    print(f"    N = {n}")
    print(f"    Agreement = {agree}/{n} ({agree/n*100:.1f}%)")
    print(f"    Cohen's kappa = {kappa:.4f} ({interp})")
    print(f"    Confusion matrix (rows={name_a}, cols={name_b}):")
    print(f"                  {name_b}=NO_MATCH  {name_b}=MATCH")
    print(f"    {name_a}=NO_MATCH    {cm[0][0]:>8d}    {cm[0][1]:>8d}")
    print(f"    {name_a}=MATCH       {cm[1][0]:>8d}    {cm[1][1]:>8d}")

    return kappa


def load_and_merge_separate_files(path1, path2):
    
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    
    label_candidates = ["human_label_author1", "human_label_author2",
                        "human_label", "label", "match", "verdict"]
    merge_key_names = {"file_id", "model", "condition", "game", "vlm_description",
                       "gt_description_1", "gt_description_2", "matcher_label", "source_file"}

    def find_label_col(df, candidates):
        for col in candidates:
            if col in df.columns:
                return col
        
        for col in df.columns:
            if col not in merge_key_names:
                return col
        return None

    label_col_1 = find_label_col(df1, label_candidates)
    label_col_2 = find_label_col(df2, label_candidates)

    if not label_col_1 or not label_col_2:
        print("[ERROR] Cannot find label columns in author files")
        sys.exit(1)

    print(f"Author1 label column: '{label_col_1}' from {path1}")
    print(f"Author2 label column: '{label_col_2}' from {path2}")

    df1 = df1.rename(columns={label_col_1: "human_label_author1"})
    df2 = df2.rename(columns={label_col_2: "human_label_author2"})

    possible_keys = ["file_id", "model", "condition"]
    merge_keys = [k for k in possible_keys if k in df1.columns and k in df2.columns]

    if not merge_keys:
        print("[WARN] No common merge keys, merging by row position")
        merged = pd.DataFrame({
            "human_label_author1": df1["human_label_author1"].values,
            "human_label_author2": df2["human_label_author2"].values,
        })
        if "matcher_label" in df1.columns:
            merged["matcher_label"] = df1["matcher_label"].values
        return merged

    cols1 = merge_keys + ["human_label_author1"]
    if "matcher_label" in df1.columns:
        cols1.append("matcher_label")
    cols2 = merge_keys + ["human_label_author2"]

    merged = pd.merge(df1[cols1], df2[cols2], on=merge_keys, how="inner")
    print(f"Merged: {len(merged)} rows (author1: {len(df1)}, author2: {len(df2)})")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Compute Cohen's kappa for matcher validation."
    )
    parser.add_argument("--human-validation", type=str, default=None,
                        help="Single CSV with both authors' labels")
    parser.add_argument("--author1", type=str, default=None,
                        help="CSV with author1's labels")
    parser.add_argument("--author2", type=str, default=None,
                        help="CSV with author2's labels")
    args = parser.parse_args()

    if not args.human_validation and not (args.author1 and args.author2):
        print("Provide either --human-validation or both --author1 and --author2")
        parser.print_help()
        sys.exit(1)

    if args.author1 and args.author2:
        df = load_and_merge_separate_files(args.author1, args.author2)
    else:
        df = pd.read_csv(args.human_validation)

    print(f"\n{'='*60}")
    print(f"HUMAN VALIDATION")
    print(f"{'='*60}")

    a1_col = "human_label_author1"
    a2_col = "human_label_author2"
    matcher_col = "matcher_label"

    if a1_col not in df.columns or a2_col not in df.columns:
        print(f"\n[ERROR] Need columns '{a1_col}' and '{a2_col}'.")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    labeled = df[
        df[a1_col].notna() &
        (df[a1_col].astype(str).str.strip() != "") &
        df[a2_col].notna() &
        (df[a2_col].astype(str).str.strip() != "")
    ].copy()

    print(f"\nTotal rows: {len(df)}")
    print(f"Both authors labeled: {len(labeled)}")

    if len(labeled) == 0:
        print("[ERROR] No labeled rows found.")
        sys.exit(1)


    k_12 = compute_and_report(labeled[a1_col], labeled[a2_col], "Author1", "Author2")

  
    k_1m = k_2m = None
    if matcher_col in labeled.columns:
        k_1m = compute_and_report(labeled[a1_col], labeled[matcher_col], "Author1", "Matcher")
        k_2m = compute_and_report(labeled[a2_col], labeled[matcher_col], "Author2", "Matcher")

 
    print(f"\n{'='*60}")
    print(f"FOR THE PAPER (Section 4.2.1):")
    print(f"{'='*60}")
    if k_12 is not None:
        print(f"  Author1 vs Author2:  kappa = {k_12:.2f}")
    if k_1m is not None:
        print(f"  Author1 vs Matcher:  kappa = {k_1m:.2f}")
    if k_2m is not None:
        print(f"  Author2 vs Matcher:  kappa = {k_2m:.2f}")
    print(f"  Labeled sample size: {len(labeled)}")


if __name__ == "__main__":
    main()
