import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from shared_data import (
    Example,
    descriptions_match,
    extract_detection,
    iter_examples,
)
from shared_eval import compute_and_save_summary
from shared_prompts import build_prompt_baseline, build_prompt_with_props


def run_evaluation(
    *,
    model_call_fn: Callable[[str, str], Dict[str, Any]],
    model_name: str,
    dataset_dir: str,
    output_csv: str,
    output_json: str,
    seed: int = 1234,
    max_examples: Optional[int] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    
    random.seed(seed)

   
    all_examples = list(iter_examples(dataset_dir))
    if not all_examples:
        print("[FATAL] No examples found. Check DATASET_DIR.")
        return

    if max_examples is not None:
        all_examples = all_examples[:max_examples]

    n_no_gt = sum(1 for ex in all_examples if ex.gt_is_glitch is None)
    if n_no_gt:
        print(
            f"[WARN] {n_no_gt} / {len(all_examples)} examples have NO gt.json. "
            f"These will be EXCLUDED from metric computation but still evaluated."
        )

    print(f"[INFO] Evaluating {len(all_examples)} clips with model: {model_name}\n")

    experiment_start_iso = datetime.now(timezone.utc).isoformat()
    experiment_start_wall = time.time()

    rows: List[Dict] = []

    for idx, ex in enumerate(all_examples, start=1):
        print(f"=== Example {idx}/{len(all_examples)} — {ex.file_id} | Game: {ex.game} ===")
        print(f"    Video: {ex.video_path}")

        
        baseline_prompt = build_prompt_baseline(ex.game)
        t0 = time.time()
        try:
            baseline_result = model_call_fn(ex.video_path, baseline_prompt)
        except Exception as e:
            print(f"[ERROR] Baseline call failed for {ex.file_id}: {e}")
            baseline_result = {
                "text": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        baseline_time = time.time() - t0
        baseline_output = baseline_result["text"]
        base_detect, base_desc, base_valid = extract_detection(baseline_output)
        print(
            f"    [BASELINE] glitch={base_detect}, valid_json={base_valid}, "
            f"desc={base_desc!r} ({baseline_time:.1f}s)"
        )

       
        props_prompt = build_prompt_with_props(ex.game)
        t0 = time.time()
        try:
            props_result = model_call_fn(ex.video_path, props_prompt)
        except Exception as e:
            print(f"[ERROR] MR call failed for {ex.file_id}: {e}")
            props_result = {
                "text": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        props_time = time.time() - t0
        props_output = props_result["text"]
        props_detect, props_desc, props_valid = extract_detection(props_output)
        print(
            f"    [WITH_MR]  glitch={props_detect}, valid_json={props_valid}, "
            f"desc={props_desc!r} ({props_time:.1f}s)"
        )

        

        has_gt = ex.gt_is_glitch is not None
        is_glitch = ex.gt_is_glitch

        base_correct_match = False
        props_correct_match = False

        if has_gt:
            if is_glitch:
                
                if base_detect:
                    base_correct_match = descriptions_match(
                        base_desc, ex.gt_descr1, ex.gt_descr2
                    )

                if props_detect:
                    props_correct_match = descriptions_match(
                        props_desc, ex.gt_descr1, ex.gt_descr2
                    )
            else:
                
                base_correct_match = not base_detect
                props_correct_match = not props_detect
        else:
            print(f"    [INFO] No GT for {ex.file_id} — excluded from metrics")

        
        rows.append({
            "file_id": ex.file_id,
            "game": ex.game,
            "video_path": ex.video_path,
            "gt_is_glitch": is_glitch,
            "has_gt": has_gt,
            "gt_descr1": ex.gt_descr1,
            "gt_descr2": ex.gt_descr2,
            "baseline_raw": baseline_output,
            "props_raw": props_output,
            "baseline_glitch": base_detect,
            "props_glitch": props_detect,
            "baseline_desc": base_desc,
            "props_desc": props_desc,
            "baseline_valid_json": base_valid,
            "props_valid_json": props_valid,
            "baseline_correct_match": base_correct_match,
            "props_correct_match": props_correct_match,
            "baseline_time_s": round(baseline_time, 2),
            "props_time_s": round(props_time, 2),
            "baseline_prompt_tokens": baseline_result["prompt_tokens"],
            "baseline_completion_tokens": baseline_result["completion_tokens"],
            "props_prompt_tokens": props_result["prompt_tokens"],
            "props_completion_tokens": props_result["completion_tokens"],
        })

        print()

 
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[INFO] Wrote detailed results to {output_csv}")
    except ImportError:
        
        import csv

        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        if rows:
            with open(output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"[INFO] Wrote detailed results to {output_csv}")
    except Exception as e:
        print(f"[WARN] Could not write CSV: {e}")

   
    experiment_end_iso = datetime.now(timezone.utc).isoformat()
    total_wall = time.time() - experiment_start_wall

    metadata = {
        "seed": seed,
        "n_examples_requested": max_examples,
        "n_examples_evaluated": len(all_examples),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    compute_and_save_summary(
        rows=rows,
        model_name=model_name,
        output_json_path=output_json,
        experiment_start_time=experiment_start_iso,
        experiment_end_time=experiment_end_iso,
        total_wall_time_s=total_wall,
        extra_metadata=metadata,
    )