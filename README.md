# MetaGlitch: Physics-Grounded Metamorphic Testing for Video Game Testing

**Replication Package — ASE 2026**

<p align="center">
  <img src="/Assets/metaglitch_demo.gif" alt="MetaGlitch detecting a physics bug in Far Cry 5" width="640">
</p>

 
---

## Overview

MetaGlitch is a metamorphic testing methodology that improves VLM-based video game bug detection by injecting physics-grounded metamorphic relations (MRs) into VLM prompts. MRs encode visual invariants of correct gameplay whose violations signal bugs.

**Key results:**
- Gemini 2.5 Flash with MRs (**39.3%**) surpasses unguided Gemini 2.5 Pro (**29.6%**)
- Game-specific MRs improve recall by **+5.2 to +14.8 pp** across all six VLMs
- Precision remains high (**76.8–100%**) with near-zero false positive rates


---

## Package Contents

```
Replication_Package_ASE_2026/
│
├── REPLICATION_README.md               # This file
│
├── Games_json.zip                      # Dataset archive (unzip before use)
│   └── Games_json/                     # 270 clips across 10 game subfolders
│       ├── Cyberpunk_2077_Dataset/     #   Each clip has its own subfolder:
│       │   ├── kd3oxq/                 #     gt.json      — ground truth + source URL
│       │   │   └── gt.json             #     metadata.json — additional metadata
│       │   └── .../                    #     video.mp4    — video (after download)
│       ├── Fallout_4/
│       ├── Fallout_76/
│       ├── Far_Cry_5_Dataset/
│       ├── GTAV_Dataset/
│       ├── Just_Cause_3/
│       ├── RDR2_Dataset/
│       ├── The_Elder_scrolls_V_Skyrim/
│       ├── The_Witcher_3_Dataset/
│       └── Watch_Dogs/
│
├── Metamorphic Relations/
│   ├── Generic/
│   │   └── generic_mr.json             # 32 generic MRs (applicable to any game)
│   └── Game_Specific/                  # 120 game-specific MRs (10 folders, one per game)
│       ├── cyberpunk_2077/cyberpunk_2077_mr.json
│       ├── fallout_4/fallout_4_mr.json
│       └── .../
│
├── Prompts/
│   ├── baseline_prompt.txt             # Baseline prompt (no MRs)
│   ├── mr_augmented_prompt.txt         # MR-augmented prompt template
│   ├── matcher_prompt.txt              # Claude Sonnet 4 semantic matcher prompt
│   └── mr_generation_prompt.txt        # Gemini 2.5 Pro MR generation prompts (Step 2)
│
├── Raw Data/
│   ├── With Game-Specific MRs - 210 clips/   # 6 CSVs: 8 original games (210 clips each)
│   ├── With Game-Specific MRs - New 60 clips/ # 6 CSVs: Fallout 76 + Watch Dogs 2 (60 clips each)
│   ├── With Generic MRs - full 270 clips/     # 6 CSVs: all 10 games with generic MRs
│   ├── Summaries/
│   │   ├── Summaries of Run with Game-Specific MRs/  # 6 JSON evaluation summaries
│   │   └── Summaries of Run with Generic MRs/        # 6 JSON evaluation summaries
│   └── Manual Matcher Validation/
│       ├── matcher_sample-author1.csv  # Author 1 labels (256 samples)
│       └── matcher_sample-author2.csv  # Author 2 labels (256 samples)
│
└── Scripts/
    ├── download_clips.py               # Download video clips from source URLs in gt.json
    ├── mine_community_bugs.py          # Step 1: Crawl Steam & GameFAQs for bug reports
    ├── generate_mrs.py                 # Step 2: Generate MRs via Gemini 2.5 Pro
    ├── mine_clean_clips.py             # Collect clean clips from YouTube
    ├── shared_runner.py                # VLM experiment runner
    ├── shared_eval.py                  # Core evaluation (TP/FN/FP/TN computation)
    ├── shared_data.py                  # Dataset loading utilities
    ├── shared_prompts.py               # Prompt templates
    ├── check_api_gemini.py             # Run experiments with Gemini models
    ├── check_api_cluster.py            # Run experiments on HPC cluster (vLLM)
    ├── rematch_with_llm.py             # Re-run semantic matcher with Claude Sonnet 4
    ├── server_semantic_matcher_gemini.py # Gemini matcher server (original, replaced by Claude)
    ├── compute_kappa.py                # Cohen's kappa for human evaluation
    ├── mcnemar_test.py                 # McNemar's test for statistical significance
    ├── mr_citation_analysis.py         # RQ2: MR citation rates in improved clips
    ├── extract_clips_metadata.py       # Extract clips_metadata.csv from experiment CSVs
    └── env/
        └── qwenvl_gpu_environment.yml  # Conda environment for open-weight models
```

**Note on the two game-specific data directories:** The experiment was initially run on 8 games (210 clips). Two additional games — Fallout 76 and Watch Dogs 2 — were added later (60 clips: 30 buggy, 30 clean). All paper results combine both directories to produce the full 270-clip dataset. Scripts that accept `--rematched-dir` can be pointed at either directory individually; to reproduce the paper's aggregate numbers, run on both and combine.

---

## Requirements

**Full environment (includes VLM inference + all analysis scripts):**

```bash
conda env create -f Scripts/env/qwenvl_gpu_environment.yml
conda activate qwenvl_gpu
```

This single environment includes all dependencies: pandas, scipy, scikit-learn, statsmodels, beautifulsoup4, google-generativeai, anthropic, vllm, and torch. No additional `pip install` steps are needed.

**Analysis-only (no GPU required):** If you only want to reproduce metrics from the pre-computed CSVs, a lightweight install suffices:

```bash
pip install pandas scipy scikit-learn statsmodels anthropic
```

---

## Downloading the Dataset

The dataset includes 270 ground-truth annotation files (`gt.json`) organized by game, but **video clips are not bundled** due to size and licensing constraints. A download script fetches the clips from their original sources (YouTube and Reddit).

```bash
# 1. Unzip the dataset (creates Games_json/ folder)
unzip Games_json.zip

# 2. Install download dependencies
pip install yt-dlp requests

# 3. Download all 270 clips (--cookies-from-browser recommended for YouTube)
cd Scripts/
python download_clips.py --root ../Games_json --cookies-from-browser chrome

# Preview what would be downloaded (no actual downloads)
python download_clips.py --root ../Games_json
```

The script walks every subfolder in `Games_json/`, reads `gt.json`, and downloads the video as `video.mp4` in the same folder. It automatically handles YouTube (with segment timestamps), Reddit post URLs, direct `v.redd.it` links, and imgur URLs. Already-downloaded clips are skipped.

**Note:** YouTube requires browser cookies to avoid HTTP 403 errors — use `--cookies-from-browser` with your browser name (chrome, firefox, safari, edge). Some Reddit video links may have expired since data collection; this is a known limitation of the GamePhysics dataset. The script reports failed downloads at the end.

---

## Reproducing Results

### From pre-computed data (fastest)

All experiment outputs are included in `Raw Data/`. The 18 CSVs contain per-clip VLM predictions with Claude Sonnet 4 matcher verdicts. The 12 JSON summaries contain all aggregate metrics reported in the paper.

To recompute evaluation metrics from the CSVs:

```bash
cd Scripts/

# Game-specific MRs (combines both directories for full 270-clip results)
python shared_eval.py \
    --input-dirs "../Raw Data/With Game-Specific MRs - 210 clips/" \
                 "../Raw Data/With Game-Specific MRs - New 60 clips/"

# Generic MRs
python shared_eval.py \
    --input-dirs "../Raw Data/With Generic MRs - full 270 clips/"
```

To recompute inter-rater agreement:

```bash
python compute_kappa.py \
    --author1 "../Raw Data/Manual Matcher Validation/matcher_sample-author1.csv" \
    --author2 "../Raw Data/Manual Matcher Validation/matcher_sample-author2.csv"
```

To recompute statistical significance (McNemar's test):

```bash
# Must include both directories to cover all 270 clips
python mcnemar_test.py \
    --rematched-dir "../Raw Data/With Game-Specific MRs - 210 clips/" \
                    "../Raw Data/With Game-Specific MRs - New 60 clips/"
```

To recompute MR citation analysis (RQ2):

```bash
# Must include both directories to cover all 270 clips
python mr_citation_analysis.py \
    --rematched-dir "../Raw Data/With Game-Specific MRs - 210 clips/" \
                    "../Raw Data/With Game-Specific MRs - New 60 clips/"
```

### Full pipeline reproduction

#### Step 1: Mine community bug reports

```bash
python mine_community_bugs.py --output-dir ./mined_reports/
```

Crawls Steam Community Discussions and GameFAQs (via Google search) for visual/physics bug reports across all 10 games.

#### Step 2: Generate candidate MRs

```bash
export GOOGLE_API_KEY="your-key"

python generate_mrs.py \
    --input-dir ./mined_reports/ \
    --output-dir ./candidate_mrs/
```

Feeds mined reports to Gemini 2.5 Pro to produce structured candidate MRs.

#### Step 3: Manual review (human process)

Two authors independently reviewed each candidate MR against three criteria: (1) detectable from video alone, (2) broad enough to justify inclusion, (3) distinguishes bugs from intentional design. Disagreements were resolved through negotiated agreement. The validated MRs are in `Metamorphic Relations/`.

#### Step 4: Run VLM experiments

**Proprietary models (Gemini):**

```bash
export GEMINI_API_KEY="your-key"
python check_api_gemini.py                            # Gemini 2.5 Pro
python check_api_gemini.py --model gemini-2.5-flash   # Gemini 2.5 Flash
```

**Open-weight models (via vLLM on GPU cluster):**

Our open-weight experiments ran on an HPC cluster with multi-GPU nodes via SLURM. The example below shows the setup for InternVL3-78B (4×GPU); adjust `--gpus-per-node`, `--tensor-parallel-size`, and `--model` for other models.

```bash
# 1. Submit SLURM job to start vLLM server
cat > run_vllm_server.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=vllm-internvl-78b
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4          # 4 GPUs for 78B/32B; 1 GPU for 8B/7B
#SBATCH --time=08:00:00
#SBATCH --mem=256G

module load cuda/12.6

source /path/to/vllm_env/bin/activate    # venv with: pip install vllm qwen-vl-utils

python3 -m vllm.entrypoints.openai.api_server \
    --model OpenGVLab/InternVL3-78B \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"video": 1}' \
    --tensor-parallel-size 4 \
    --disable-custom-all-reduce
EOF

sbatch run_vllm_server.sh

# 2. Once the server is running, set up SSH tunnel from local machine
ssh -N -L 8000:<node>:8000 user@cluster-login

# 3. Run experiment (from local machine or cluster)
python check_api_cluster.py
```

Model-specific adjustments:

| Model | `--gpus-per-node` | `--tensor-parallel-size` | `--model` |
|-------|-------------------|--------------------------|-----------|
| InternVL3-78B | 4 | 4 | `OpenGVLab/InternVL3-78B` |
| InternVL3-8B | 1 | 1 | `OpenGVLab/InternVL3-8B` |
| Qwen2.5-VL-32B | 4 | 4 | `Qwen/Qwen2.5-VL-32B-Instruct` |
| Qwen2.5-VL-7B | 1 | 1 | `Qwen/Qwen2.5-VL-7B-Instruct` |

#### Step 5: Semantic matching

```bash
export ANTHROPIC_API_KEY="your-key"

python rematch_with_llm.py \
    --csvs "Raw Data/With Game-Specific MRs - 210 clips/"*.csv \
           "Raw Data/With Game-Specific MRs - New 60 clips/"*.csv \
    --output-dir ./rematched/ \
    --model claude-sonnet-4-20250514
```

---

## Dataset

| Component | Count | Source |
|-----------|-------|--------|
| Buggy clips | 135 | GamePhysics dataset + YouTube compilations |
| Clean clips | 135 | "No commentary" YouTube walkthroughs |
| Games | 10 | Open-world titles, 6 engines, 2011–2020 |
| Generic MRs | 32 | Mined from community forums, human-validated |
| Game-specific MRs | 120 | Mined per-title, human-validated |

The evaluation clips are independent of the community posts used for MR mining, ensuring no overlap between the two datasets.

**gt.json fields:**

| Field | Description |
|-------|-------------|
| `File` | Unique clip identifier (also the subfolder name) |
| `Game` | Game title |
| `Descr1`, `Descr2` | Human-annotated bug descriptions (empty for clean clips) |
| `Bug Type` | Bug category or `"No bug"` for clean clips |
| `Source` | URL to download the video (YouTube, Reddit, or direct link) |
| `Segment` | Start/end timestamps in seconds (YouTube clean clips only) |

## Models

| Model | Type | Input | Deployment |
|-------|------|-------|------------|
| Gemini 2.5 Pro | Proprietary | Native video | Google AI API |
| Gemini 2.5 Flash | Proprietary | Native video | Google AI API |
| InternVL3-78B | Open-weight | ≤25 frames | vLLM (4×A100) |
| InternVL3-8B | Open-weight | ≤25 frames | vLLM (1×A100) |
| Qwen2.5-VL-32B | Open-weight | ≤25 frames | vLLM (4×A100) |
| Qwen2.5-VL-7B | Open-weight | ≤25 frames | vLLM (1×A100) |

## Evaluation

We use a **description-aware** evaluation where a detection counts as TP only when the VLM correctly identifies a bug AND its description semantically matches the ground truth. Claude Sonnet 4 serves as the semantic matcher, validated by two authors on 256 stratified samples (Cohen's κ = 0.90 and 0.88 between each author and the matcher; κ = 0.96 between authors).

---

## CSV Column Reference

| Column | Description |
|--------|-------------|
| `file_id` | Unique clip identifier |
| `game` | Game title |
| `gt_is_glitch` | Ground truth (TRUE/FALSE) |
| `gt_descr1`, `gt_descr2` | Human-annotated bug descriptions |
| `baseline_glitch` | Baseline prediction (TRUE/FALSE) |
| `baseline_desc` | Baseline bug description |
| `baseline_correct_match` | Matcher verdict for baseline |
| `props_glitch` | MR-augmented prediction |
| `props_desc` | MR-augmented bug description |
| `props_correct_match` | Matcher verdict for MR-augmented |
| `baseline_raw`, `props_raw` | Full VLM JSON responses |

**Note:** The generic MR CSVs (`With Generic MRs/`) do not contain `baseline_*` columns, as the baseline is identical across conditions and is stored in the game-specific CSVs.
