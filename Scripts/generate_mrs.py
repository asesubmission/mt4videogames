import argparse
import json
import os
import sys
import time
import glob

try:
    import google.generativeai as genai
except ImportError:
    print("pip install google-generativeai")
    sys.exit(1)



GAME_ID_PREFIX = {
    "Cyberpunk 2077":               "CP",
    "Fallout 4":                    "FO",
    "Fallout 76":                   "F76",
    "Far Cry 5":                    "FC",
    "Grand Theft Auto V":           "GTA",
    "Just Cause 3":                 "JC",
    "Red Dead Redemption 2":        "RDR",
    "The Elder Scrolls V Skyrim":   "SK",
    "The Witcher 3 Wild Hunt":      "W3",
    "Watch Dogs 2":                 "WD",
}




GAME_SPECIFIC_PROMPT = """You are a game testing expert. I will provide community bug reports for the video game "{game}".

Your task: analyze these bug reports and generate structured Metamorphic Relations (MRs) that a Vision-Language Model (VLM) can use as a checklist to detect bugs in gameplay videos.

Each MR must follow this exact JSON format:
{{
    "id": "{prefix}-001",
    "name": "Short descriptive name",
    "check_question": "A yes/no question a VLM can answer by watching a gameplay video. Use domain-specific terms for this game (e.g., 'stagecoach' instead of 'vehicle' for a Western game).",
    "example": "A concrete example of this bug occurring in the game.",
    "category": "One of: collision, character_physics, character_animation, vehicle_physics, environment_physics, rendering, camera, projectile, ui, object, character_model, audio"
}}

Rules:
1. Each MR must describe a VISUAL pattern detectable from video alone (no audio, no game state).
2. Check questions must be specific enough to distinguish bugs from intentional mechanics.
3. Use game-appropriate terminology (e.g., "horse" for RDR2, "motorcycle" for Cyberpunk).
4. Merge similar reports into a single MR (e.g., "car through wall" and "NPC in floor" both become a clipping MR).
5. Include a "not_a_bug" list of known intentional behaviors that visually resemble glitches.
6. Generate IDs sequentially: {prefix}-001, {prefix}-002, etc.

Additionally, generate a "not_a_bug" list:
{{
    "not_a_bug": [
        "Description of intentional mechanic that looks like a bug"
    ]
}}

BUG REPORTS:
{reports}

Respond with ONLY a JSON object containing:
{{
    "game": "{game}",
    "mrs": [ ... list of MR objects ... ],
    "not_a_bug": [ ... list of strings ... ]
}}"""

GENERIC_PROMPT = """You are a game testing expert. I will provide community bug reports from multiple video games.

Your task: analyze these reports and generate GENERIC Metamorphic Relations (MRs) that apply to ANY video game. These should encode universal physical invariants whose violations indicate bugs.

Each MR must follow this exact JSON format:
{{
    "id": "MR-001",
    "name": "Short descriptive name",
    "check_question": "A yes/no question a VLM can answer by watching gameplay video from any game.",
    "example": "A concrete example from any game.",
    "category": "One of: collision, character_physics, character_animation, vehicle_physics, environment_physics, rendering, camera, projectile, ui, object, character_model, audio"
}}

Rules:
1. Each MR must describe a VISUAL pattern detectable from video alone.
2. Check questions must be general enough to apply across games and genres.
3. Use generic terms (e.g., "vehicle" not "car", "character" not "NPC").
4. Focus on physical invariants: collision, gravity, animation consistency, object permanence.
5. Generate IDs sequentially: MR-001, MR-002, etc.

BUG REPORTS FROM MULTIPLE GAMES:
{reports}

Respond with ONLY a JSON object containing:
{{
    "mrs": [ ... list of MR objects ... ]
}}"""


def load_reports(path):
    """Load mined reports from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    game = data.get("game", "Unknown")
    posts = data.get("posts", [])

   
    report_texts = []
    for i, post in enumerate(posts[:100]):  
        title = post.get("title", "").strip()
        body = post.get("body", "").strip()
        text = f"[Report {i+1}] {title}"
        if body:
            text += f"\n{body[:500]}"  
        report_texts.append(text)

    return game, "\n\n".join(report_texts), len(posts)


def call_gemini(prompt, api_key, model_name="gemini-2.5-pro", max_retries=3):
    """Call Gemini API and return parsed JSON."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=8192,
                ),
            )

            text = response.text.strip()
    
            text = text.replace("```json", "").replace("```", "").strip()

            return json.loads(text)

        except json.JSONDecodeError as e:
            print(f"    JSON parse error (attempt {attempt+1}): {e}")
            print(f"    Raw response: {text[:200]}...")
            time.sleep(2)

        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            time.sleep(5)

    return None


def generate_game_specific_mrs(report_path, api_key, output_dir):
    """Generate game-specific MRs from a single game's mined reports."""
    game, report_text, n_posts = load_reports(report_path)

    if n_posts == 0:
        print(f"  [SKIP] No reports found for {game}")
        return


    prefix = "XX"
    for gname, pref in GAME_ID_PREFIX.items():
        if gname.lower() in game.lower() or game.lower() in gname.lower():
            prefix = pref
            break

    print(f"  Game: {game} (prefix={prefix}, {n_posts} reports)")

    prompt = GAME_SPECIFIC_PROMPT.format(
        game=game, prefix=prefix, reports=report_text
    )

    print(f"  Calling Gemini 2.5 Pro...")
    result = call_gemini(prompt, api_key)

    if result is None:
        print(f"  [ERROR] Failed to generate MRs for {game}")
        return

    safe_name = game.lower().replace(" ", "_").replace(":", "")
    output_path = os.path.join(output_dir, f"{safe_name}_candidate_mrs.json")

    result["_metadata"] = {
        "game": game,
        "source_file": os.path.basename(report_path),
        "n_input_reports": n_posts,
        "generator": "gemini-2.5-pro",
        "note": "Candidate MRs — requires manual review (Step 3)"
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    n_mrs = len(result.get("mrs", []))
    n_nab = len(result.get("not_a_bug", []))
    print(f"  Generated: {n_mrs} candidate MRs, {n_nab} not-a-bug entries")
    print(f"  Saved: {output_path}")
    return result


def generate_generic_mrs(input_dir, api_key, output_dir):
    """Generate generic MRs from all games' mined reports."""
    all_reports = []

    json_files = glob.glob(os.path.join(input_dir, "*_reports.json"))
    for path in sorted(json_files):
        game, report_text, n_posts = load_reports(path)
        if n_posts > 0:
      
            lines = report_text.split("\n\n")[:15]
            all_reports.append(f"=== {game} ===\n" + "\n\n".join(lines))
            print(f"  Loaded {min(n_posts, 15)} reports from {game}")

    if not all_reports:
        print("[ERROR] No reports found in input directory")
        return

    combined_text = "\n\n".join(all_reports)

    print(f"\n  Calling Gemini 2.5 Pro for generic MRs...")
    prompt = GENERIC_PROMPT.format(reports=combined_text)
    result = call_gemini(prompt, api_key)

    if result is None:
        print("[ERROR] Failed to generate generic MRs")
        return

    output_path = os.path.join(output_dir, "generic_candidate_mrs.json")
    result["_metadata"] = {
        "n_games": len(all_reports),
        "generator": "gemini-2.5-pro",
        "note": "Candidate generic MRs — requires manual review (Step 3)"
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    n_mrs = len(result.get("mrs", []))
    print(f"  Generated: {n_mrs} candidate generic MRs")
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Generate candidate MRs from mined bug reports using Gemini 2.5 Pro."
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Single game report JSON file")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Directory with all game report JSONs")
    parser.add_argument("--output-dir", default="./candidate_mrs",
                        help="Output directory for candidate MR JSONs")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Google AI API key (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--generic", action="store_true",
                        help="Generate generic MRs (aggregates all games)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] Set GOOGLE_API_KEY env var or use --api-key")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 2: LLM-Based MR Generation")
    print("=" * 60)
    print(f"Model: Gemini 2.5 Pro")
    print(f"Output: {args.output_dir}/")

    if args.generic:
        input_dir = args.input_dir or "."
        generate_generic_mrs(input_dir, api_key, args.output_dir)

    elif args.input:
        generate_game_specific_mrs(args.input, api_key, args.output_dir)

    elif args.input_dir:
        json_files = sorted(glob.glob(os.path.join(args.input_dir, "*_reports.json")))
        if not json_files:
            print(f"[ERROR] No *_reports.json files found in {args.input_dir}")
            sys.exit(1)

        print(f"Found {len(json_files)} game report files\n")

        for path in json_files:
            print(f"\n{'='*60}")
            generate_game_specific_mrs(path, api_key, args.output_dir)
            time.sleep(2)  

        print(f"\n{'='*60}")
        print("Generating generic MRs from all games...")
        generate_generic_mrs(args.input_dir, api_key, args.output_dir)

    else:
        print("[ERROR] Provide --input (single file) or --input-dir (directory)")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"Candidate MRs saved to: {args.output_dir}/")
    print(f"\nNext step: Manual review and negotiated agreement (Step 3)")
    print(f"Two authors should independently review each candidate MR.")


if __name__ == "__main__":
    main()
