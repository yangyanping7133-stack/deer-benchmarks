# DEER Benchmark Results v3.0-full

## Environment
- **Model**: Qwen3-32B (bfloat16)
- **Framework**: vLLM-Ascend (8x Ascend 910B4, TP=8)
- **Server**: max_model_len=65536, dtype=bfloat16
- **Concurrency**: 256
- **Temperature**: 0.6

## Datasets (1270 total)
| Dataset | Samples | Type |
|---------|---------|------|
| GSM8K | 500 | math |
| MATH500 | 500 | math |
| GPQA | 200 | multiple choice |
| AMC | 40 | math |
| AIME | 30 | math |

---

## Baseline Results

| Dataset | Accuracy | Avg E2E (s) | Avg Completion Tokens | Avg Think Tokens (est) |
|---------|----------|-------------|----------------------|----------------------|
| GSM8K | 480/500 (96.0%) | 290 | 1650 | 1144 |
| MATH500 | 444/500 (88.8%) | 629 | 4590 | 2800 |
| GPQA | 188/200 (94.0%) | 191 | 952 | 651 |
| AMC | 19/40 (47.5%) | 920 | 6993 | 4240 |
| AIME | 6/30 (20.0%) | 1564 | 13286 | 7986 |
| **Overall** | **1137/1270 (89.5%)** | **539** | **4173** | **2609** |

---

## DEER Iteration History

### iter_v1: Original DEER (threshold=0.95, max_steps=4, think_ratio=0.8)

| Dataset | BL Acc | DEER Acc | Avg E2E (s) | Speedup | Think Ratio | Acc Diff |
|---------|--------|----------|-------------|---------|-------------|----------|
| GSM8K | 480/500 (96%) | 433/500 (87%) | 105 | 64% | 26% | -47 |
| MATH500 | 444/500 (89%) | 386/500 (77%) | 189 | 70% | 15% | -58 |
| GPQA | 188/200 (94%) | 194/200 (97%) | 122 | 36% | 60% | +6 |
| AMC | 19/40 (48%) | 23/40 (58%) | 316 | 66% | 23% | +4 |
| AIME | 6/30 (20%) | 4/30 (13%) | 242 | 85% | 12% | -2 |
| **Overall** | **1137/1270** | **1040/1270** | | **64%** | **22%** | **-97** |

**Result**: High speedup but significant accuracy drop (-97 cases). Root cause: max_steps=4 limits thinking to ~1200 tokens, far below what complex problems need.

**125 regression cases identified** (baseline correct, DEER wrong).

---

### Optimization Experiments (on 125 regression cases)

| Experiment | Config | Fixed | Key Finding |
|------------|--------|-------|-------------|
| v2 | threshold=0.98, steps=8, think_ratio=0.9 | 41/125 | More steps helps but still truncates |
| v3 | no_stop_wait, steps=12, think_ratio=1.0 | 53/125 | No Wait = no early exit, near-zero speedup |
| v4a | threshold=0.95, steps=20, think_ratio=0.8 | 41/125 | More steps with Wait, moderate |
| v4b | threshold=0.98, steps=20, think_ratio=0.8 | ~51/125 | Higher threshold = fewer exits = less speedup |
| v5 | Adaptive per-dataset + fallback, thresh=0.90 | 40/125 | Confidence check unreliable (41% false exit rate) |
| v5b | Adaptive per-dataset + fallback, thresh=0.98 | 51/125 | Only 11 deer_exit, rest natural/fallback = no speedup |
| **P1** | **Progressive Budget [0.5x, 0.8x, 1.0x]** | **48/125** | **31% speedup, no confidence check needed** |

---

### iter_v2 (candidate): Progressive Budget P1

**Config**: Per-dataset progressive think budget [0.5x avg → 0.8x avg → 1.0x avg]

| Dataset | Budget Levels (tokens) | L1 Finish | L2 Finish | L3 Finish |
|---------|----------------------|-----------|-----------|-----------|
| GSM8K | 572 → 915 → 1144 | 29/50 | 21/50 | 0/50 |
| MATH500 | 1400 → 2239 → 2799 | 23/58 | 31/58 | 4/58 |
| GPQA | 326 → 521 → 651 | 2/4 | 2/4 | 0/4 |
| AMC | 2120 → 3392 → 4240 | 4/7 | 3/7 | 0/7 |
| AIME | 3993 → 6389 → 7986 | 5/6 | 1/6 | 0/6 |

**Regression test results (125 cases)**:
- Fixed: 48/125 (38%)
- Speedup vs baseline (on regression cases): 31%
- Budget distribution: L1=63, L2=58, L3=4
- Almost all finish at L1 or L2 (97%)

**Projected full-run accuracy**: ~83.5% (vs baseline 89.5%, drop ~6%)
- Still has accuracy gap vs baseline
- 77 regression cases remain unfixed even with progressive budget

---

## Key Findings

### 1. Confidence-based early exit is unreliable
- Logprobs confidence check has 41-64% false positive rate
- High confidence ≠ correct answer
- The check measures "can the model continue fluently" not "is the reasoning correct"

### 2. "Wait" segmentation is not the core problem
- Wait-based segmentation works as natural thinking breakpoints
- The real issue is max_steps limiting total thinking budget
- With enough steps (20+), Wait segmentation works fine

### 3. Speedup-accuracy tradeoff is fundamental
- High speedup requires truncating thinking → accuracy drops
- Full accuracy requires full thinking → no speedup
- Progressive budget offers a middle ground

### 4. Per-dataset adaptation matters
- Easy datasets (GPQA, GSM8K): large speedup possible
- Hard datasets (AIME, AMC): very limited speedup room
- One-size-fits-all config doesn't work

---

## Files

### Scripts
- `run_full_mixed.py` — Main benchmark runner (baseline + DEER, 256 concurrency)
- `run_deer_optimize.py` — DEER parameter optimizer (tests configs on regressions)
- `run_deer_adaptive.py` — Adaptive per-dataset DEER with fallback
- `run_deer_progressive.py` — Progressive budget DEER (P1)

### Results
- `*_baseline.json` / `*_baseline_judged.json` — Baseline results
- `*_deer.json` / `*_deer_judged.json` — DEER v1 results
- `regressions.json` — 125 regression cases (BL correct, DEER wrong)
- `adaptive_v5_*.json` — Adaptive DEER test results
- `progressive_P1_*.json` — Progressive budget test results

### Logs
- `optimize_v2.log` — v2 optimization log
- `optimize_v3.log` — v3 optimization log
- `optimize_v4.log` — v4 optimization log
- `adaptive_v5.log` / `adaptive_v5b.log` — Adaptive DEER logs
- `progressive_P1.log` — Progressive budget log
