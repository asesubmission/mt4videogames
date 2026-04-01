import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", required=True)
    parser.add_argument("--output", default="clips_metadata.csv")
    args = parser.parse_args()

    frames = []
    for path in args.csvs:
        if not os.path.exists(path):
            print(f"[WARN] Not found: {path}")
            continue
        df = pd.read_csv(path)
        frames.append(df)
        print(f"Read {len(df)} rows from {os.path.basename(path)}")

    combined = pd.concat(frames, ignore_index=True)

    meta_cols = ["file_id", "game", "video_path", "gt_is_glitch", "has_gt",
                 "gt_descr1", "gt_descr2"]
    meta_cols = [c for c in meta_cols if c in combined.columns]

    meta = combined[meta_cols].drop_duplicates(subset=["file_id", "game"])
    meta = meta.sort_values(["game", "file_id"]).reset_index(drop=True)


    for col in ["gt_is_glitch", "has_gt"]:
        if col in meta.columns:
            meta[col] = meta[col].astype(str).str.upper()

    meta.to_csv(args.output, index=False)


    n_buggy = (meta["gt_is_glitch"] == "TRUE").sum() if "gt_is_glitch" in meta.columns else "?"
    n_clean = (meta["gt_is_glitch"] == "FALSE").sum() if "gt_is_glitch" in meta.columns else "?"
    games = meta["game"].nunique()

    print(f"\nSaved: {args.output}")
    print(f"Total clips: {len(meta)} ({n_buggy} buggy, {n_clean} clean)")
    print(f"Games: {games}")
    print(f"Columns: {list(meta.columns)}")


if __name__ == "__main__":
    main()
