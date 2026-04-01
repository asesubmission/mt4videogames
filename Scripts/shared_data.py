import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import requests


SEMANTIC_MATCHER_URL = os.getenv("SEMANTIC_MATCHER_URL", "http://127.0.0.1:8002")

@dataclass
class Example:
   
    game: str
    file_id: str
    clip_dir: str
    video_path: str
    meta: Dict
    gt_descr1: Optional[str] = None
    gt_descr2: Optional[str] = None
    gt_is_glitch: Optional[bool] = None

def find_video_file(clip_dir: str) -> Optional[str]:
    """Return path to the first video file in clip_dir, or None."""
    video_exts = (".mp4", ".webm", ".mkv", ".avi", ".mov")
    for fname in sorted(os.listdir(clip_dir)):
        if fname.startswith("."):
            continue
        if fname.lower().endswith(video_exts):
            return os.path.join(clip_dir, fname)
    return None


def iter_examples(dataset_dir: str) -> Iterator[Example]:
    
    if not os.path.isdir(dataset_dir):
        raise RuntimeError(
            f"DATASET_DIR does not exist or is not a directory: {dataset_dir}"
        )

    for game_name in sorted(os.listdir(dataset_dir)):
        if game_name.startswith("."):
            continue
        game_dir = os.path.join(dataset_dir, game_name)
        if not os.path.isdir(game_dir):
            continue

        for clip_folder in sorted(os.listdir(game_dir)):
            if clip_folder.startswith("."):
                continue
            clip_dir = os.path.join(game_dir, clip_folder)
            if not os.path.isdir(clip_dir):
                continue

           
            meta_path = os.path.join(clip_dir, "metadata.json")
            if not os.path.exists(meta_path):
                print(f"[WARN] Skipping {clip_dir}: no metadata.json")
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load metadata at {meta_path}: {e}")
                continue

            file_id = str(meta.get("File") or meta.get("file") or clip_folder)
            game_in_meta = str(
                meta.get("Game") or meta.get("game") or game_name
            )

            video_path = find_video_file(clip_dir)
            if not video_path:
                print(f"[WARN] Skipping {clip_dir}: no video file found")
                continue

            gt_descr1 = None
            gt_descr2 = None
            gt_is_glitch = None

            gt_path = os.path.join(clip_dir, "gt.json")
            if os.path.exists(gt_path):
                try:
                    with open(gt_path, "r", encoding="utf-8") as f:
                        gt = json.load(f)
                    d1 = str(gt.get("Descr1", "") or "").strip()
                    d2 = str(gt.get("Descr2", "") or "").strip()
                    bug_type = str(
                        gt.get("Bug Type", "") or ""
                    ).strip().lower()

                    gt_descr1 = d1 or None
                    gt_descr2 = d2 or None

                  
                    if bug_type == "no bug" and not d1 and not d2:
                        gt_is_glitch = False
                    else:
                        gt_is_glitch = True

                except Exception as e:
                    print(f"[WARN] Failed to load gt.json at {gt_path}: {e}")

            yield Example(
                game=game_in_meta,
                file_id=file_id,
                clip_dir=clip_dir,
                video_path=video_path,
                meta=meta,
                gt_descr1=gt_descr1,
                gt_descr2=gt_descr2,
                gt_is_glitch=gt_is_glitch,
            )


def extract_json_block(text: str) -> Optional[Dict]:
    
    if not text:
        return None
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
    except ValueError:
        return None

    candidate = text[start:end]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def extract_detection(text: str) -> Tuple[bool, str, bool]:
    
    obj = extract_json_block(text)
    if obj is None:
        print(f"[WARN] Could not parse JSON from model output: {text[:200]!r}")
        return False, "", False

    glitch = bool(obj.get("glitch_detected", False))
    desc = str(obj.get("description_of_glitch", "") or "").strip()
    return glitch, desc, True


def descriptions_match(
    pred: Optional[str],
    gt1: Optional[str],
    gt2: Optional[str],
) -> bool:
    
    if not pred:
        return False
    if not gt1 and not gt2:
        return False

    try:
        response = requests.post(
            f"{SEMANTIC_MATCHER_URL}/match",
            json={
                "predicted_description": pred,
                "ground_truth_1": gt1,
                "ground_truth_2": gt2,
            },
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        match_flag = bool(data.get("match", False))

        if "reasoning" in data:
            print(
                f"[SEMANTIC] Match={match_flag}, "
                f"Reasoning: {data['reasoning'][:120]}"
            )

        return match_flag

    except Exception as e:

        raise RuntimeError(
            f"Semantic matcher server unreachable at {SEMANTIC_MATCHER_URL}. "
            f"Start it with: uvicorn server_semantic_matcher:app --port 8002\n"
            f"Original error: {e}"
        ) from e