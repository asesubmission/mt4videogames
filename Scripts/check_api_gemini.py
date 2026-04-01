import json
import os
import sys
import time
from typing import Optional

import requests
from google import genai
from google.genai import types

from shared_runner import run_evaluation



GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable not set. "
        "Get your API key from: https://aistudio.google.com/apikey"
    )

SEMANTIC_MATCHER_URL = os.getenv("SEMANTIC_MATCHER_URL", "http://127.0.0.1:8002")

DATASET_DIR = "."
OUTPUT_DIR = "."


GEMINI_MODEL = "gemini-2.5-pro"

SEED = 1234

client = genai.Client(api_key=GEMINI_API_KEY)



def make_call_gemini(model_name: str):
    """Create a call function bound to a specific Gemini model."""

    def call_gemini(video_path: str, prompt: str) -> dict:
        attempt = 0
        while True:
            try:
                uploaded_file = client.files.upload(file=video_path)

                while uploaded_file.state.name == "PROCESSING":
                    print(".", end="", flush=True)
                    time.sleep(2)
                    uploaded_file = client.files.get(name=uploaded_file.name)

                if uploaded_file.state.name == "FAILED":
                    print(f"\n[ERROR] Video processing failed for {video_path}")
                    return {"text": "", "prompt_tokens": 0, "completion_tokens": 0}

                response = client.models.generate_content(
                    model=model_name,
                    contents=[uploaded_file, prompt],
                    config=types.GenerateContentConfig(temperature=0.0),
                )

                try:
                    client.files.delete(name=uploaded_file.name)
                except Exception:
                    pass

                prompt_tokens = 0
                completion_tokens = 0
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    completion_tokens = getattr(usage, "candidates_token_count", 0) or 0

                return {
                    "text": response.text or "",
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }

            except Exception as e:
                error_str = str(e)
                if ("503" in error_str or "UNAVAILABLE" in error_str or
                    "429" in error_str or "RESOURCE_EXHAUSTED" in error_str):
                    attempt += 1
                    wait = min(2 ** attempt * 10, 300)  # cap at 5 min
                    print(f"\n[RETRY {attempt}] {error_str[:80]}... waiting {wait}s")
                    time.sleep(wait)
                    continue
                else:
                    print(f"\n[ERROR] Gemini call failed for {video_path}: {e}")
                    return {"text": "", "prompt_tokens": 0, "completion_tokens": 0}

    return call_gemini



def preflight_checks() -> None:
    

    print(f"[PREFLIGHT] Testing Gemini API key ...")
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL, contents="Say hello in one short sentence."
        )
        print(f"[PREFLIGHT] Gemini OK. Reply: {response.text!r}")
    except Exception as e:
        raise RuntimeError(f"Gemini API test failed: {e}") from e

    print(
        f"[PREFLIGHT] Pinging SBERT matcher at "
        f"{SEMANTIC_MATCHER_URL}/health ..."
    )
    resp = requests.get(f"{SEMANTIC_MATCHER_URL}/health", timeout=10)
    resp.raise_for_status()
    info = resp.json()
    print(
        f"[PREFLIGHT] SBERT OK. Backend: {info.get('backend')}, "
        f"Model: {info.get('model')}, Threshold: {info.get('threshold')}"
    )




def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Gemini models")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit number of clips (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=GEMINI_MODEL,
        help=f"Gemini model name (default: {GEMINI_MODEL})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model

    GEMINI_MODEL = model_name

    try:
        preflight_checks()
    except Exception as e:
        print(f"[FATAL] Preflight failed: {e}")
        sys.exit(1)

    safe_name = model_name.replace("/", "_").replace(".", "_")
    output_csv = os.path.join(OUTPUT_DIR, f"vlm_eval_results_{safe_name}.csv")
    output_json = os.path.join(OUTPUT_DIR, f"vlm_eval_summary_{safe_name}.json")

    run_evaluation(
        model_call_fn=make_call_gemini(model_name),
        model_name=model_name,
        dataset_dir=DATASET_DIR,
        output_csv=output_csv,
        output_json=output_json,
        seed=SEED,
        max_examples=args.max_examples,
        extra_metadata={
            "gemini_model": model_name,
            "temperature": 0.0,
        },
    )