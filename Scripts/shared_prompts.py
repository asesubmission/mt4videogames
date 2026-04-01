import json
import os
from typing import Dict

MR_DIR = os.getenv(
    "MR_DIR", "./Metamorphic Relations/Generic"
)

#game specific:

# GAME_TO_MR_FILE: Dict[str, str] = {
#     # Canonical names
#     "Red Dead Redemption 2": "rdr2_mr.json",
#     "Grand Theft Auto V": "gtav_mr.json",
#     "Cyberpunk 2077": "cyberpunk2077_mr.json",
#     "Far Cry 5": "farcry5_mr.json",
#     "Just Cause 3": "justcause3_mr.json",
#     "Fallout 4": "fallout4_mr.json",
#     "The Elder Scrolls V - Skyrim": "skyrim_mr.json",
#     "The Witcher 3 - Wild Hunt": "witcher3_mr.json",

#     #New Games
#     "Assassin's Creed Unity": "assassins_creed_unity_mr.json",
#     "Fallout 76": "fallout_76_mr.json",
#     "Watch Dogs 2": "watch_dogs_2_mr.json",

#     # Aliases
#     "Far Cry": "farcry5_mr.json",
#     "Fallout": "fallout4_mr.json",
#     "fallout 4": "fallout4_mr.json",
#     "GTA V": "gtav_mr.json",
#     "Just Cause": "justcause3_mr.json",
#     "The Elder Scrolls": "skyrim_mr.json",
#     "The Witcher 3": "witcher3_mr.json",
#     "Assassin's Creed": "assassins_creed_unity_mr.json",
#     "Watch Dogs": "watch_dogs_2_mr.json",
# }



#generic:

GAME_TO_MR_FILE: Dict[str, str] = {
    # Canonical names
    "Red Dead Redemption 2": "generic_mr.json",
    "Grand Theft Auto V": "generic_mr.json",
    "Cyberpunk 2077": "generic_mr.json",
    "Far Cry 5": "generic_mr.json",
    "Just Cause 3": "generic_mr.json",
    "Fallout 4": "generic_mr.json",
    "The Elder Scrolls V - Skyrim": "generic_mr.json",
    "The Witcher 3 - Wild Hunt": "generic_mr.json",

    #New Games
    "Assassin's Creed Unity": "generic_mr.json",
    "Fallout 76": "generic_mr.json",
    "Watch Dogs 2": "generic_mr.json",

    # Aliases
    "Far Cry": "generic_mr.json",
    "Fallout": "generic_mr.json",
    "fallout 4": "generic_mr.json",
    "GTA V": "generic_mr.json",
    "Just Cause": "generic_mr.json",
    "The Elder Scrolls": "generic_mr.json",
    "The Witcher 3": "generic_mr.json",
    "Assassin's Creed": "generic_mr.json",
    "Watch Dogs": "generic_mr.json",
}


def load_mr_checklist(game: str) -> str:
   
    fname = GAME_TO_MR_FILE.get(game)
    if not fname:
        print(
            f"[WARN] No MR file mapping for game '{game}'. "
            f"Running without properties."
        )
        return ""

    path = os.path.join(MR_DIR, fname)
    if not os.path.exists(path):
        print(f"[WARN] MR file does not exist for game '{game}': {path}")
        return ""

    try:
        with open(path, "r", encoding="utf-8") as f:
            mr = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load MR file for game '{game}': {e}")
        return ""

    relations = mr.get("metamorphic_relations", [])
    if not relations:
        return ""

    lines = []
    for r in relations:
        rid = r.get("id", "?")
        name = r.get("name", "")
        check = r.get("check") or r.get("violation_hint", "")
        if check:
            lines.append(f"  {rid} ({name}): {check}")
        else:
            lines.append(f"  {rid}: {name}")

    return "\n".join(lines)


BASE_PROMPT_TEMPLATE = """You are a helpful assistant analyzing short video clips from the video game "{game}".

You may or may not see a bug or glitch in the clip (for example physics violations, animation problems, rendering artifacts, clipping, or other clearly abnormal behavior).

Your job is:

1. Decide whether the clip contains ANY visible bug or glitch based only on what you can actually see in the video.
2. If you think there IS a clear glitch, briefly describe what is wrong in one short sentence.
3. If you think there is NO clear glitch, briefly explain in one short sentence that the behavior looks normal or plausible for this kind of game.

If the behavior is ambiguous and could reasonably be normal gameplay, treat it as **no glitch**.

Then respond strictly in the following JSON format:

{{
  "reasoning": "very short explanation of why you think there is or is not a glitch",
  "glitch_detected": true or false,
  "description_of_glitch": "very short description of the glitch or abnormal behavior; empty string if no glitch"
}}

Do NOT include any extra text outside the JSON.
"""


WITH_PROPS_PROMPT_TEMPLATE = """You are a helpful assistant analyzing short video clips from the video game "{game}".

Below is a checklist of known glitch patterns for this game. Use it to systematically scan the video. For each item, briefly consider whether the video shows that specific violation.

GLITCH CHECKLIST:
{checklist}

INSTRUCTIONS:
1. Scan the video carefully, keeping the checklist in mind.
2. If you see something that matches ANY checklist item, or any other clearly abnormal visual behavior, report it as a glitch.
3. If nothing in the video matches any checklist item and everything looks like plausible normal gameplay, report no glitch.
4. Do NOT hallucinate glitches. Only report what you can actually see. Ambiguous or borderline cases should be treated as no glitch.

Respond strictly in the following JSON format:

{{
  "reasoning": "which checklist items you considered and why you think there is or is not a glitch; cite checklist IDs here if applicable",
  "glitch_detected": true or false,
  "description_of_glitch": "describe IN YOUR OWN WORDS what you actually see going wrong in the video (e.g. 'a car clips through a concrete barrier' or 'the NPC floats into the sky'). Do NOT just write checklist IDs or category names. Empty string if no glitch."
}}

Do NOT include any extra text outside the JSON.
"""


def build_prompt_baseline(game: str) -> str:
    """Build the baseline prompt (no checklist)."""
    return BASE_PROMPT_TEMPLATE.format(game=game)


def build_prompt_with_props(game: str) -> str:
    """Build the with-properties prompt (game-specific checklist)."""
    checklist = load_mr_checklist(game)
    if not checklist:
        print(
            f"[WARN] No checklist for '{game}', "
            f"falling back to baseline prompt."
        )
        return BASE_PROMPT_TEMPLATE.format(game=game)
    return WITH_PROPS_PROMPT_TEMPLATE.format(game=game, checklist=checklist)