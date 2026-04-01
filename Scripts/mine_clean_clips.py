from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import string
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple



DEFAULT_TARGET_COUNTS: Dict[str, int] = {
    "Red Dead Redemption 2": 35,
    "GTA V": 30,
    "Cyberpunk 2077": 35,
    "The Witcher 3": 20,
    "Far Cry": 25,
    "The Elder Scrolls": 20,
    "Fallout": 25,
    "Just Cause": 20,
}


DEFAULT_GAME_DIR_MAP: Dict[str, str] = {
    "Red Dead Redemption 2": "RDR2",
    "GTA V": "GTAV",
    "Cyberpunk 2077": "Cyberpunk",
    "The Witcher 3": "Witcher",
    "Far Cry": "FarCry",
    "The Elder Scrolls": "ElderScrolls",
    "Fallout": "Fallout",
    "Just Cause": "JustCause",
}


DEFAULT_QUERIES: Dict[str, List[str]] = {
    "Red Dead Redemption 2": [
        "Red Dead Redemption 2 no commentary gameplay free roam -glitch -bug -fail -funny",
        "Red Dead Redemption 2 story mission gameplay no commentary -glitch -bug",
        "RDR2 4K gameplay no commentary -glitch -bug -physics",
    ],
    "GTA V": [
        "GTA V no commentary gameplay free roam -glitch -bug -fail",
        "GTA 5 4K gameplay no commentary -glitch -bug -mods",
        "Grand Theft Auto V walkthrough no commentary -glitch -bug",
    ],
    "Cyberpunk 2077": [
        "Cyberpunk 2077 no commentary gameplay free roam -glitch -bug -fail",
        "Cyberpunk 2077 4K gameplay no commentary -glitch -bug",
        "Cyberpunk 2077 story missions gameplay no commentary -glitch -bug",
    ],
    "The Witcher 3": [
        "Witcher 3 no commentary gameplay -glitch -bug",
        "The Witcher 3 4K gameplay no commentary -glitch -bug",
        "Witcher 3 walkthrough no commentary -glitch -bug",
    ],
    "Far Cry": [
        "Far Cry gameplay no commentary -glitch -bug",
        "Far Cry 5 no commentary gameplay -glitch -bug",
        "Far Cry 6 no commentary gameplay -glitch -bug",
    ],
    "The Elder Scrolls": [
        "Skyrim gameplay no commentary -glitch -bug",
        "Elder Scrolls Skyrim 4K gameplay no commentary -glitch -bug",
        "Skyrim walkthrough no commentary -glitch -bug",
    ],
    "Fallout": [
        "Fallout 4 gameplay no commentary -glitch -bug",
        "Fallout New Vegas gameplay no commentary -glitch -bug",
        "Fallout gameplay no commentary -glitch -bug",
    ],
    "Just Cause": [
        "Just Cause 3 gameplay no commentary -glitch -bug",
        "Just Cause 4 gameplay no commentary -glitch -bug",
        "Just Cause gameplay no commentary -glitch -bug",
    ],
}



@dataclass
class Candidate:
    url: str
    duration: Optional[float]
    title: str
    id: str


def run_cmd(cmd: List[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    if capture:
        return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return subprocess.run(cmd, check=check)


def which_or_die(bin_name: str) -> None:
    if shutil.which(bin_name) is None:
        print(f"ERROR: '{bin_name}' not found on PATH. Please install it first.", file=sys.stderr)
        sys.exit(2)


def ffprobe_duration_seconds(path: Path) -> Optional[float]:
    try:
        cp = run_cmd([
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ])
        s = (cp.stdout or "").strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def is_probably_clean_gt(gt_path: Path) -> bool:
    try:
        obj = json.loads(gt_path.read_text(encoding="utf-8"))
        bt = str(obj.get("Bug Type", "")).strip().lower()
        return bt in {"no bug", "clean", "none"}
    except Exception:
        return False


def list_buggy_clip_videos(game_dir: Path, clean_dir_name: str = "clean") -> List[Path]:

    videos: List[Path] = []
    for video in game_dir.rglob("video.mp4"):
        if clean_dir_name in video.parts:
            continue
        clip_dir = video.parent
        if not (clip_dir / "metadata.json").exists():
            continue
        gt = clip_dir / "gt.json"
        if gt.exists() and is_probably_clean_gt(gt):
            continue
        videos.append(video)
    return videos


def list_existing_clean_clip_dirs(clean_root: Path) -> List[Path]:
    if not clean_root.exists():
        return []
    return [p for p in clean_root.iterdir() if p.is_dir() and (p / "video.mp4").exists()]


def random_clip_id(k: int = 6) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(random.choice(alphabet) for _ in range(k))


def sanitize_game_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip())


def yt_search_candidates(query: str, limit: int = 25) -> List[Candidate]:

    search_url = f"ytsearch{limit}:{query}"
    cp = run_cmd([
        "yt-dlp",
        "--dump-json",
        "--skip-download",
        "--no-warnings",
        search_url,
    ])
    out = cp.stdout.splitlines()
    cands: List[Candidate] = []
    for line in out:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            url = obj.get("webpage_url") or obj.get("original_url")
            vid = obj.get("id") or ""
            title = obj.get("title") or ""
            dur = obj.get("duration")
            if url and vid:
                cands.append(Candidate(url=url, duration=float(dur) if dur is not None else None, title=title, id=str(vid)))
        except Exception:
            continue
    return cands


def choose_segment(duration_needed: float, source_duration: float, margin: float = 10.0) -> Tuple[float, float]:
    max_start = max(0.0, source_duration - duration_needed - margin)
    start = random.uniform(margin, max_start) if max_start > margin else 0.0
    end = start + duration_needed
    return (start, end)


def yt_download_section(url: str, start: float, end: float, out_video_path: Path, *, retries: int = 2) -> None:
    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    section = f"*{start:.3f}-{end:.3f}"

    out_tmpl = str(out_video_path.parent / "video.%(ext)s")

    last_err = None
    for _ in range(retries + 1):
        try:
            cp = run_cmd([
                "yt-dlp",
                "--no-warnings",
                "--download-sections",
                section,
                "--force-keyframes-at-cuts",
                "--merge-output-format",
                "mp4",
                "--remux-video",
                "mp4",
                "-o",
                out_tmpl,
                url,
            ], check=True, capture=True)

            mp4 = out_video_path.parent / "video.mp4"
            if mp4.exists():
                return
            for p in out_video_path.parent.glob("video.*"):
                if p.suffix.lower() == ".mp4":
                    p.rename(mp4)
                    return
            raise RuntimeError(f"Download succeeded but video file not found in {out_video_path.parent}")
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"Failed to download clip section after retries: {last_err}")


def write_metadata(clip_dir: Path, clip_id: str, game: str, source_url: str, start: float, end: float) -> None:
    meta = {"File": clip_id, "Game": game}
    (clip_dir / "metadata.json").write_text(json.dumps(meta, indent=4), encoding="utf-8")

    gt = {
        "File": clip_id,
        "Game": game,
        "Descr1": "",
        "Descr2": "",
        "Bug Type": "No bug",
        "Source": source_url,
        "Segment": {"start": round(start, 3), "end": round(end, 3)},
    }
    (clip_dir / "gt.json").write_text(json.dumps(gt, indent=4), encoding="utf-8")


def load_json_map(arg: Optional[str]) -> Optional[Dict[str, str]]:
    if not arg:
        return None
    try:
        return json.loads(arg)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON for --game-dir-map: {e}")


def clamp_clip_len(x: float, lo: float = 10.0, hi: float = 30.0) -> float:
    return max(lo, min(hi, x))


def build_duration_plan(buggy_videos: List[Path], target_n: int, *, seed: int) -> List[float]:
    random.seed(seed)

    durs: List[float] = []
    for v in buggy_videos:
        dur = ffprobe_duration_seconds(v)
        if dur is None:
            continue
        durs.append(clamp_clip_len(dur))

    if not durs:
        return [random.uniform(10.0, 30.0) for _ in range(target_n)]

    random.shuffle(durs)
    plan: List[float] = []
    i = 0
    while len(plan) < target_n:
        plan.append(durs[i % len(durs)])
        i += 1
    random.shuffle(plan)
    return plan


def mine_for_game(
    *,
    root: Path,
    game: str,
    game_dir_name: str,
    target_n: int,
    queries: List[str],
    clean_dir_name: str,
    seed: int,
    search_limit: int,
    max_tries_per_clip: int,
) -> None:
    game = sanitize_game_name(game)
    game_dir = root / game_dir_name
    game_dir.mkdir(parents=True, exist_ok=True)
    clean_root = game_dir / clean_dir_name
    clean_root.mkdir(parents=True, exist_ok=True)

    existing_clean = list_existing_clean_clip_dirs(clean_root)
    remaining = target_n - len(existing_clean)
    if remaining <= 0:
        print(f"[{game}] clean clips already present: {len(existing_clean)}/{target_n} (skipping)")
        return

    buggy_videos = list_buggy_clip_videos(game_dir, clean_dir_name=clean_dir_name)
    duration_plan = build_duration_plan(buggy_videos, target_n, seed=seed)
    duration_plan = duration_plan[len(existing_clean):]

    used_sources: Dict[str, int] = {}

    manifest_path = clean_root / "manifest.jsonl"
    mf = manifest_path.open("a", encoding="utf-8")

    print(f"[{game}] need to add {remaining} clean clips into: {clean_root}")

    try:
        for idx, clip_len in enumerate(duration_plan, start=1):
            clip_len = float(clip_len)
            clip_id = random_clip_id(6)
            clip_dir = clean_root / clip_id

            while clip_dir.exists():
                clip_id = random_clip_id(6)
                clip_dir = clean_root / clip_id

            success = False
            last_error = None

            for attempt in range(1, max_tries_per_clip + 1):
                q = random.choice(queries)
                try:
                    cands = yt_search_candidates(q, limit=search_limit)
                    good = [c for c in cands if c.duration and c.duration >= (clip_len + 25.0)]
                    good.sort(key=lambda c: used_sources.get(c.id, 0))
                    if not good:
                        raise RuntimeError("No suitable candidates returned by search")

                    pick_pool = good[: min(10, len(good))]
                    cand = random.choice(pick_pool)
                    used_sources[cand.id] = used_sources.get(cand.id, 0) + 1

                    start, end = choose_segment(clip_len, cand.duration or clip_len)

                    yt_download_section(cand.url, start, end, clip_dir / "video.mp4")

                    got = ffprobe_duration_seconds(clip_dir / "video.mp4")
                    if got is None:
                        raise RuntimeError("ffprobe failed on downloaded clip")
                    if abs(got - clip_len) > 0.75:

                        print(f"  warn: duration mismatch wanted={clip_len:.2f}s got={got:.2f}s for {clip_id}")

                    write_metadata(clip_dir, clip_id, game, cand.url, start, end)

                    rec = {
                        "clip_id": clip_id,
                        "game": game,
                        "source_url": cand.url,
                        "title": cand.title,
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "target_len": round(clip_len, 3),
                        "actual_len": round(got, 3),
                    }
                    mf.write(json.dumps(rec) + "\n")
                    mf.flush()

                    print(f"  + {idx}/{remaining}: {clip_id}  ({clip_len:.1f}s)  from {cand.id}")
                    success = True
                    break

                except Exception as e:
                    last_error = e
                    if clip_dir.exists():
                        shutil.rmtree(clip_dir, ignore_errors=True)

            if not success:
                raise RuntimeError(f"Failed to mine clip {idx}/{remaining} for {game}. Last error: {last_error}")

    finally:
        mf.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine clean YouTube clips and add them to your dataset.")
    ap.add_argument("--root", required=True, help="Dataset root containing per-game folders")
    ap.add_argument("--clean-dir-name", default="clean", help="Subfolder under each game dir for clean clips")
    ap.add_argument("--seed", type=int, default=1337, help="RNG seed")
    ap.add_argument("--search-limit", type=int, default=25, help="ytsearchN: number of results per query")
    ap.add_argument("--max-tries-per-clip", type=int, default=12, help="Retries (search+download) per clip")
    ap.add_argument(
        "--game-dir-map",
        default=None,
        help=(
            "JSON mapping from game name to folder name under --root. "
            "Example: '{\"GTA V\":\"GTA_V\",\"Red Dead Redemption 2\":\"RDR2\"}'"
        ),
    )
    ap.add_argument(
        "--counts",
        default=None,
        help=(
            "JSON mapping from game name to target clean count. "
            "If omitted, uses the counts you provided in chat."
        ),
    )
    ap.add_argument(
        "--queries",
        default=None,
        help=(
            "JSON mapping from game name to list of search queries. "
            "If omitted, uses built-in defaults."
        ),
    )

    args = ap.parse_args()

    which_or_die("yt-dlp")
    which_or_die("ffmpeg")
    which_or_die("ffprobe")

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    game_dir_map = load_json_map(args.game_dir_map) or DEFAULT_GAME_DIR_MAP
    counts = json.loads(args.counts) if args.counts else DEFAULT_TARGET_COUNTS

    if args.queries:
        queries_map = json.loads(args.queries)
    else:
        queries_map = DEFAULT_QUERIES

    counts = {sanitize_game_name(k): int(v) for k, v in counts.items()}
    game_dir_map = {sanitize_game_name(k): v for k, v in game_dir_map.items()}

    for game, target_n in counts.items():
        if game not in game_dir_map:
            print(f"WARN: No folder mapping for '{game}'. Skipping. Provide --game-dir-map.")
            continue
        game_dir_name = game_dir_map[game]
        queries = queries_map.get(game) or [f"{game} gameplay no commentary -glitch -bug -fail"]

        mine_for_game(
            root=root,
            game=game,
            game_dir_name=game_dir_name,
            target_n=target_n,
            queries=queries,
            clean_dir_name=args.clean_dir_name,
            seed=args.seed,
            search_limit=args.search_limit,
            max_tries_per_clip=args.max_tries_per_clip,
        )


if __name__ == "__main__":
    main()
