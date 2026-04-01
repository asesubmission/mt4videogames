import argparse
import json
import os
import re
import sys
import time

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("pip install requests beautifulsoup4")
    sys.exit(1)


# ── Game metadata ─────────────────────────────────────────────────
GAMES = {
    "Cyberpunk 2077": {
        "steam_appid": 1091500,
        "gamefaqs_board": "292195-cyberpunk-2077",
    },
    "Fallout 4": {
        "steam_appid": 377160,
        "gamefaqs_board": "164592-fallout-4",
    },
    "Fallout 76": {
        "steam_appid": 1151340,
        "gamefaqs_board": "248373-fallout-76",
    },
    "Far Cry 5": {
        "steam_appid": 552520,
        "gamefaqs_board": "210403-far-cry-5",
    },
    "Grand Theft Auto V": {
        "steam_appid": 271590,
        "gamefaqs_board": "805602-grand-theft-auto-v",
    },
    "Just Cause 3": {
        "steam_appid": 225540,
        "gamefaqs_board": "168538-just-cause-3",
    },
    "Red Dead Redemption 2": {
        "steam_appid": 1174180,
        "gamefaqs_board": "248479-red-dead-redemption-2",
    },
    "The Elder Scrolls V Skyrim": {
        "steam_appid": 489830,  # Special Edition
        "gamefaqs_board": "615803-the-elder-scrolls-v-skyrim-special-edition",
    },
    "The Witcher 3 Wild Hunt": {
        "steam_appid": 292030,
        "gamefaqs_board": "168653-the-witcher-3-wild-hunt",
    },
    "Watch Dogs 2": {
        "steam_appid": 447040,
        "gamefaqs_board": "190431-watch-dogs-2",
    },
}


BUG_KEYWORDS = [
    "clipping",
    "floating",
    "T-pose",
    "ragdoll",
    "physics bug",
    "physics glitch",
    "visual glitch",
    "visual bug",
    "animation bug",
    "animation glitch",
    "stuck in ground",
    "falling through",
    "invisible",
    "despawn",
    "teleport",
    "launch into sky",
    "flying car",
    "flying horse",
    "stretching",
    "jitter",
    "vibrating",
]


EXCLUDE_KEYWORDS = [
    "feature request",
    "balance",
    "nerf",
    "buff",
    "dlc",
    "price",
    "sale",
    "refund",
    "crash to desktop",
    "ctd",
    "won't launch",
    "black screen",
    "fps drop",
    "performance",
    "stuttering",
    "mod",
    "save file",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def search_steam_discussions(appid, keyword, max_pages=3, delay=2.0):
    """Search Steam Community Discussions for a keyword."""
    results = []
    for page in range(1, max_pages + 1):
        url = (f"https://steamcommunity.com/app/{appid}/discussions/"
               f"search/?q={keyword.replace(' ', '+')}&p={page}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"      Steam HTTP {resp.status_code} for '{keyword}' page {page}")
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            threads = soup.select(".forum_topic_name")

            if not threads:
                break

            for thread in threads:
                link = thread.get("href", "")
                title = thread.get_text(strip=True)

               
                title_lower = title.lower()
                if any(ex in title_lower for ex in EXCLUDE_KEYWORDS):
                    continue

                results.append({
                    "title": title,
                    "url": link,
                    "source": "steam",
                    "keyword": keyword,
                    "body": "",  
                })

            time.sleep(delay)

        except Exception as e:
            print(f"      Steam error for '{keyword}': {e}")
            break

    return results


def fetch_steam_thread_body(url, delay=1.5):
    """Fetch the body text of a Steam discussion thread."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
       
        op = soup.select_one(".forum_op .content")
        if op:
            return op.get_text(separator="\n", strip=True)
        return ""
    except Exception:
        return ""
    finally:
        time.sleep(delay)


def search_gamefaqs_board(board_id, keyword, max_pages=2, delay=3.0):
    """Search GameFAQs message board for a keyword."""
    results = []
    for page in range(max_pages):
        url = (f"https://gamefaqs.gamespot.com/boards/{board_id}"
               f"?search={keyword.replace(' ', '+')}&page={page}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 403:
                print(f"      GameFAQs blocked (403) for '{keyword}' — skipping")
                break
            if resp.status_code != 200:
                print(f"      GameFAQs HTTP {resp.status_code} for '{keyword}'")
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            topics = soup.select("td.topic a.link_title")

            if not topics:
                break

            for topic in topics:
                title = topic.get_text(strip=True)
                href = topic.get("href", "")
                full_url = f"https://gamefaqs.gamespot.com{href}" if href.startswith("/") else href

                title_lower = title.lower()
                if any(ex in title_lower for ex in EXCLUDE_KEYWORDS):
                    continue

                results.append({
                    "title": title,
                    "url": full_url,
                    "source": "gamefaqs",
                    "keyword": keyword,
                    "body": "",
                })

            time.sleep(delay)

        except Exception as e:
            print(f"      GameFAQs error for '{keyword}': {e}")
            break

    return results


def mine_game(game_name, game_info, fetch_bodies=True, max_body_fetch=20):
    """Mine bug reports for a single game from both platforms."""
    appid = game_info["steam_appid"]
    board_id = game_info.get("gamefaqs_board", "")

    all_posts = []
    seen_urls = set()

    print(f"\n  Mining Steam (appid={appid})...")
    for kw in BUG_KEYWORDS:
        posts = search_steam_discussions(appid, kw)
        for p in posts:
            if p["url"] not in seen_urls:
                seen_urls.add(p["url"])
                all_posts.append(p)
        if posts:
            print(f"    '{kw}': {len(posts)} threads")

    if board_id:
        print(f"  Mining GameFAQs (board={board_id})...")
        for kw in BUG_KEYWORDS:
            posts = search_gamefaqs_board(board_id, kw)
            for p in posts:
                if p["url"] not in seen_urls:
                    seen_urls.add(p["url"])
                    all_posts.append(p)
            if posts:
                print(f"    '{kw}': {len(posts)} threads")

    
    if fetch_bodies and all_posts:
        steam_posts = [p for p in all_posts if p["source"] == "steam" and p["url"]]
        n_fetch = min(len(steam_posts), max_body_fetch)
        print(f"  Fetching body text for {n_fetch} Steam threads...")
        for i, post in enumerate(steam_posts[:n_fetch]):
            body = fetch_steam_thread_body(post["url"])
            post["body"] = body
            if (i + 1) % 5 == 0:
                print(f"    Fetched {i+1}/{n_fetch}")

    return all_posts


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Mine community bug reports from Steam and GameFAQs."
    )
    parser.add_argument("--output-dir", default="./mined_reports",
                        help="Output directory for JSON files")
    parser.add_argument("--games", nargs="*", default=None,
                        help="Specific game names to mine (default: all 10)")
    parser.add_argument("--no-bodies", action="store_true",
                        help="Skip fetching thread body text (faster)")
    parser.add_argument("--max-bodies", type=int, default=20,
                        help="Max thread bodies to fetch per game")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    target_games = args.games if args.games else list(GAMES.keys())

    print("=" * 60)
    print("STEP 1: Community Data Mining")
    print("=" * 60)
    print(f"Games: {len(target_games)}")
    print(f"Keywords: {len(BUG_KEYWORDS)}")
    print(f"Platforms: Steam Community, GameFAQs")
    print(f"Output: {args.output_dir}/")

    total_posts = 0

    for game_name in target_games:
        if game_name not in GAMES:
            
            matches = [g for g in GAMES if game_name.lower() in g.lower()]
            if matches:
                game_name = matches[0]
            else:
                print(f"\n[WARN] Unknown game: '{game_name}'")
                print(f"  Available: {', '.join(GAMES.keys())}")
                continue

        print(f"\n{'='*60}")
        print(f"Mining: {game_name}")
        print(f"{'='*60}")

        posts = mine_game(
            game_name, GAMES[game_name],
            fetch_bodies=not args.no_bodies,
            max_body_fetch=args.max_bodies,
        )

       
        safe_name = game_name.lower().replace(" ", "_").replace(":", "")
        output_path = os.path.join(args.output_dir, f"{safe_name}_reports.json")

        output_data = {
            "game": game_name,
            "steam_appid": GAMES[game_name]["steam_appid"],
            "keywords_used": BUG_KEYWORDS,
            "total_posts": len(posts),
            "posts": posts,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n  Saved: {output_path} ({len(posts)} posts)")
        total_posts += len(posts)

    print(f"\n{'='*60}")
    print(f"DONE: {total_posts} total posts across {len(target_games)} games")
    print(f"Output directory: {args.output_dir}/")
    print(f"{'='*60}")
    print(f"\nNext step: run generate_mrs.py on the mined reports")


if __name__ == "__main__":
    main()
