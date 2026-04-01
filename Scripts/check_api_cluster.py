

import argparse
import base64
import io
import os
import sys

import requests
from openai import OpenAI

from shared_runner import run_evaluation


SERVER_BASE_URL = os.getenv("CLUSTER_VLM_SERVER", "http://127.0.0.1:8000/v1")

SEMANTIC_MATCHER_URL = os.getenv("SEMANTIC_MATCHER_URL", "http://127.0.0.1:8002")

LOCAL_DATASET_DIR = os.getenv(
    "DATASET_DIR",
    "."
)

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    "."
)

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
SEED = 1234
TARGET_FPS = 2.0      # sample 
MAX_FRAMES = 25       # cap of 45 frames 
MAX_DIMENSION = 1280   # resize long edge to 672px 
JPEG_QUALITY = 95

CLIENT = OpenAI(
    base_url=SERVER_BASE_URL,
    api_key="NOT_USED",
    timeout=1800.0,
)



def extract_frames_base64(video_path: str) -> list:
    """
    Extract frames from a local video at TARGET_FPS, resize, return
    as list of {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    interval = max(1, int(fps / TARGET_FPS))
    frame_indices = list(range(0, total_frames, interval))

    if len(frame_indices) > MAX_FRAMES:
        step = len(frame_indices) / MAX_FRAMES
        frame_indices = [frame_indices[int(i * step)] for i in range(MAX_FRAMES)]

    frames_b64 = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        if max(h, w) > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

        frames_b64.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    cap.release()
    print(f"    [FRAMES] {video_path.split('/')[-2]}: {len(frames_b64)} frames "
          f"from {duration:.1f}s video ({fps:.0f}fps source)")
    return frames_b64



def make_call_cluster(model_id: str):

    def call_vlm(video_path: str, prompt: str) -> dict:
        frames = extract_frames_base64(video_path)

        if not frames:
            return {
                "text": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        content = [{"type": "text", "text": prompt}] + frames

        response = CLIENT.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            max_tokens=512,
            temperature=0.0,
            top_p=1.0,
        )

        usage = response.usage
        return {
            "text": response.choices[0].message.content,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        }

    return call_vlm



def preflight_checks(model_id: str) -> None:
    print(f"[PREFLIGHT] Pinging cluster vLLM at {SERVER_BASE_URL} ...")
    try:
        resp = requests.get(f"{SERVER_BASE_URL}/models", timeout=30)
        resp.raise_for_status()
        available = [m["id"] for m in resp.json().get("data", [])]
        print(f"[PREFLIGHT] vLLM OK. Models: {available}")
        if model_id not in available:
            print(f"[WARN] Requested '{model_id}' not in {available}")
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach vLLM at {SERVER_BASE_URL}.\n"
            f"Is the SSH tunnel running? Error: {e}"
        ) from e

    print(f"[PREFLIGHT] Testing inference ...")
    try:
        resp = CLIENT.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=16,
            temperature=0.0,
        )
        print(f"[PREFLIGHT] Inference OK. Reply: {resp.choices[0].message.content!r}")
    except Exception as e:
        raise RuntimeError(f"Inference test failed: {e}") from e

    print(f"[PREFLIGHT] Pinging matcher at {SEMANTIC_MATCHER_URL}/health ...")
    try:
        resp = requests.get(f"{SEMANTIC_MATCHER_URL}/health", timeout=10)
        resp.raise_for_status()
        info = resp.json()
        print(
            f"[PREFLIGHT] Matcher OK. Backend: {info.get('backend')}, "
            f"Model: {info.get('model')}"
        )
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach matcher at {SEMANTIC_MATCHER_URL}. "
            f"Start it with: uvicorn server_semantic_matcher_gemini:app --port 8002\n"
            f"Error: {e}"
        ) from e



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLM on remote GPU cluster via SSH tunnel"
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Limit number of clips (default: all)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model ID (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()
    model_id = args.model

    try:
        preflight_checks(model_id)
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    safe_name = model_id.split("/")[-1].replace(".", "_")
    output_csv = os.path.join(OUTPUT_DIR, f"vlm_eval_results_{safe_name}.csv")
    output_json = os.path.join(OUTPUT_DIR, f"vlm_eval_summary_{safe_name}.json")

    print(f"\n[INFO] Evaluating with model: {model_id}")
    print(f"[INFO] Frames: {TARGET_FPS}fps, max {MAX_FRAMES}, resize {MAX_DIMENSION}px")
    print(f"[INFO] Local GT/metadata from:   {LOCAL_DATASET_DIR}")
    print(f"[INFO] Output: {output_csv}\n")

    run_evaluation(
        model_call_fn=make_call_cluster(model_id),
        model_name=safe_name,
        dataset_dir=LOCAL_DATASET_DIR,
        output_csv=output_csv,
        output_json=output_json,
        seed=SEED,
        max_examples=args.max_examples,
        extra_metadata={
            "server_url": SERVER_BASE_URL,
            "cluster": "NCSA Delta",
            "model_id": model_id,
            "target_fps": TARGET_FPS,
            "max_frames": MAX_FRAMES,
            "max_dimension": MAX_DIMENSION,
        },
    )