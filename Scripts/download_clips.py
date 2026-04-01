import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    sys.exit("ERROR: 'requests' is required.  Install with:  pip install requests")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


COOKIES_BROWSER: str | None = None


def has_video_file(folder: Path) -> bool:
    """Return True if folder already contains a video/gif file."""
    video_exts = {".mp4", ".webm", ".mkv", ".gif", ".gifv", ".avi", ".mov"}
    for f in folder.iterdir():
        if f.suffix.lower() in video_exts:
            return True
    return False


def classify_url(url: str) -> str:
    """Return one of: youtube, reddit_direct, reddit_post, imgur, unknown."""
    if not url:
        return "unknown"
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    if "v.redd.it" in url:
        return "reddit_direct"
    if "reddit.com" in url:
        return "reddit_post"
    if "imgur.com" in url:
        return "imgur"
    return "unknown"


def download_with_ytdlp(url: str, output_path: str) -> bool:
    """Use yt-dlp to download a video. Works for YouTube, Reddit, and many others."""
 
    strategies = [
        
        [
            "yt-dlp",
            "--no-playlist",
            "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
            "--merge-output-format", "mp4",
            "-o", output_path,
            "--no-overwrites",
            "--socket-timeout", "30",
            "--retries", "3",
            url,
        ],
       
        [
            "yt-dlp",
            "--no-playlist",
            "--extractor-args", "youtube:player_client=ios,web",
            "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
            "--merge-output-format", "mp4",
            "-o", output_path,
            "--no-overwrites",
            "--socket-timeout", "30",
            "--retries", "3",
            url,
        ],
        
        [
            "yt-dlp",
            "--no-playlist",
            "--extractor-args", "youtube:player_client=web_creator",
            "-f", "best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", output_path,
            "--no-overwrites",
            "--socket-timeout", "30",
            "--retries", "3",
            url,
        ],
    ]

    is_youtube = "youtube.com" in url or "youtu.be" in url

    try:
        for i, cmd in enumerate(strategies):
            
            if i > 0 and not is_youtube:
                break
            
            if COOKIES_BROWSER:
                cmd = cmd[:1] + ["--cookies-from-browser", COOKIES_BROWSER] + cmd[1:]
         
            if i > 0 and os.path.exists(output_path):
                os.remove(output_path)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode == 0:
                return True
            log.debug("Strategy %d failed: %s", i + 1,
                       result.stderr[-200:] if result.stderr else "")
      
        log.warning("yt-dlp all strategies failed: %s",
                     result.stderr[-300:] if result.stderr else "")
        return False
    except FileNotFoundError:
        log.error("yt-dlp not found. Install with:  pip install yt-dlp")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        log.warning("yt-dlp timed out for %s", url)
        return False


def download_reddit_direct(url: str, output_path: str) -> bool:
    """Download a direct v.redd.it MP4 link."""
    try:
        resp = requests.get(url, headers=HEADERS, stream=True, timeout=30)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                f.write(chunk)
        return os.path.getsize(output_path) > 1000
    except Exception as e:
        log.warning("Direct download failed for %s: %s", url, e)
        return False


def download_reddit_post(url: str, output_path: str) -> bool:
    
    
    if download_with_ytdlp(url, output_path):
        return True

   
    try:
        json_url = url.rstrip("/") + ".json"
        resp = requests.get(json_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        post_data = data[0]["data"]["children"][0]["data"]

        if "media" in post_data and post_data["media"]:
            media = post_data["media"]
            if "reddit_video" in media:
                video_url = media["reddit_video"].get("fallback_url", "")
                if video_url:
                    return download_reddit_direct(video_url, output_path)

        if "crosspost_parent_list" in post_data:
            for xp in post_data["crosspost_parent_list"]:
                if "media" in xp and xp["media"] and "reddit_video" in xp["media"]:
                    video_url = xp["media"]["reddit_video"].get("fallback_url", "")
                    if video_url:
                        return download_reddit_direct(video_url, output_path)

        
        ext_url = post_data.get("url", "")
        if ext_url and ext_url != url:
            url_type = classify_url(ext_url)
            if url_type == "imgur":
                return download_imgur(ext_url, output_path)
            elif url_type == "reddit_direct":
                return download_reddit_direct(ext_url, output_path)
            else:
                return download_with_ytdlp(ext_url, output_path)

    except Exception as e:
        log.warning("Reddit JSON fallback failed for %s: %s", url, e)
    return False


def download_imgur(url: str, output_path: str) -> bool:
    
    mp4_url = url.replace(".gifv", ".mp4").replace(".gif", ".mp4")
    if not mp4_url.endswith(".mp4"):
        mp4_url += ".mp4"
    
    mp4_url = mp4_url.replace("http://", "https://")

    try:
        resp = requests.get(mp4_url, headers=HEADERS, stream=True, timeout=30)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 64):
                f.write(chunk)
        return os.path.getsize(output_path) > 1000
    except Exception as e:
        log.warning("Imgur download failed for %s: %s", mp4_url, e)
        
        return download_with_ytdlp(url, output_path)


def process_folder(folder: Path, stats: dict):
    
    gt_file = folder / "gt.json"
    if not gt_file.exists():
        return

    try:
        with open(gt_file) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        log.warning("Invalid JSON: %s", gt_file)
        stats["errors"] += 1
        return

    source = data.get("Source", "").strip()
    if not source:
        log.info("No Source URL in %s – skipping", gt_file)
        stats["no_source"] += 1
        return

   
    if has_video_file(folder):
        log.info("SKIP (already exists): %s", folder.name)
        stats["skipped"] += 1
        return

    clip_name = "video.mp4"
    output_path = str(folder / clip_name)
    url_type = classify_url(source)

    log.info("Downloading [%s] %s -> %s", url_type, source[:80], folder.name)

    success = False
    if url_type == "youtube":
        success = download_with_ytdlp(source, output_path)
    elif url_type == "reddit_direct":
        success = download_reddit_direct(source, output_path)
        if not success:
            success = download_with_ytdlp(source, output_path)
    elif url_type == "reddit_post":
        success = download_reddit_post(source, output_path)
    elif url_type == "imgur":
        success = download_imgur(source, output_path)
    else:
      
        success = download_with_ytdlp(source, output_path)

    if success:
        log.info("  OK  %s", clip_name)
        stats["downloaded"] += 1
    else:
        
        if os.path.exists(output_path) and os.path.getsize(output_path) < 1000:
            os.remove(output_path)
        log.warning("  FAIL  %s", source)
        stats["errors"] += 1

   
    if url_type == "youtube":
        time.sleep(2)
    else:
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Download game bug video clips from gt.json Source URLs.")
    parser.add_argument(
        "--root",
        type=str,
        default="Games_json",
        help="Path to the Games_json folder (default: ./Games_json)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        default=None,
        metavar="BROWSER",
        help="Pass browser name (chrome, firefox, safari, edge) to let yt-dlp "
             "use your logged-in YouTube cookies. Helps bypass 403 errors.",
    )
    args = parser.parse_args()

    global COOKIES_BROWSER
    COOKIES_BROWSER = args.cookies_from_browser

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"ERROR: Directory not found: {root}")

    stats = {"downloaded": 0, "skipped": 0, "errors": 0, "no_source": 0}

    
    gt_files = sorted(root.rglob("gt.json"))
    total = len(gt_files)
    log.info("Found %d gt.json files under %s", total, root)

    for i, gt_file in enumerate(gt_files, 1):
        folder = gt_file.parent
        log.info("[%d/%d] %s", i, total, folder.relative_to(root))
        process_folder(folder, stats)

    log.info("=" * 50)
    log.info("DONE.  Downloaded: %d | Skipped: %d | Failed: %d | No source: %d",
             stats["downloaded"], stats["skipped"], stats["errors"], stats["no_source"])


if __name__ == "__main__":
    main()