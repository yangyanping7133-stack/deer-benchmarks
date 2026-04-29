# DEER Mid-Scale Evaluation Report

## 1. Environment

| Item | Value |
|------|-------|
| Hardware | Ascend NPU × 8 (tensor-parallel-size=8) |
| Model | Qwen3-32B |
| Serving | vLLM-Ascend, http://127.0.0.1:8000 |
| max_model_len | 65536 |
| dtype | bfloat16 |
| Concurrency | 12 (requests), 4 (judge) |
| Date | 2026-04-29 |

## 2. DEER Configuration (v2.1-final)

| Parameter | Value |
|-----------|-------|
| threshold | 0.90 |
| prob_check_tokens | 10 |
| think_ratio | 0.6 |
| max_judge_steps | 2 |
| min_think_tokens | 300 |
| stop | ["Wait"] |
| answer_max_tokens | 512 |
| temperature | 0.6 |
| answer_phase | /no_think (3-message structure) |

## 3. Test Cases

- **Total**: 18 cases (3 per dataset × 6 datasets)
- **Datasets**: GSM8K, MATH-500, AMC, GPQA, AIME, LiveCodeBench
- **Selection**: Random sampling (seed=42) from full dataset
- **File**: `/root/benchmarks/data/CRC-QAD/test_cases_v1.0.json`

## 4. Results

### 4.1 Performance Comparison

| Dataset | Method | N | Avg Time (s) | Avg Tokens | Avg Think Tokens | Accuracy |
|---------|--------|---|-------------|-----------|-----------------|----------|
| GSM8K | Baseline | 3 | 179.9 | 2,980 | 2,387 | 2/3 (67%) |
| GSM8K | DEER | 3 | 16.1 | 264 | 148 | 2/3 (67%) |
| MATH-500 | Baseline | 3 | 283.6 | 4,614 | 2,486 | 2/3 (67%) |
| MATH-500 | DEER | 3 | 118.4 | 1,959 | 1,102 | 2/3 (67%) |
| AMC | Baseline | 3 | 755.3 | 12,137 | 6,960 | 0/3 (0%) |
| AMC | DEER | 3 | 76.3 | 1,239 | 743 | 0/3 (0%) |
| GPQA | Baseline | 3 | 45.9 | 749 | 481 | 3/3 (100%) |
| GPQA | DEER | 3 | 23.3 | 388 | 315 | 3/3 (100%) |
| AIME | Baseline | 3 | 571.6 | 9,214 | 6,545 | 1/3 (33%) |
| AIME | DEER | 3 | 53.2 | 872 | 562 | 1/3 (33%) |
| LiveCodeBench | Baseline | 3 | 642.0 | 10,612 | 8,353 | 2/3 (67%) |
| LiveCodeBench | DEER | 3 | 36.4 | 561 | 322 | 0/3 (0%) |

### 4.2 Speedup Summary

| Dataset | Time Speedup | Think Token Reduction | Accuracy Δ |
|---------|-------------|----------------------|-----------|
| GSM8K | **+91.1%** | +93.8% | 0pp |
| MATH-500 | **+58.2%** | +55.7% | 0pp |
| AMC | **+89.9%** | +89.3% | 0pp |
| GPQA | **+49.2%** | +34.4% | 0pp |
| AIME | **+90.7%** | +91.4% | 0pp |
| LiveCodeBench | **+94.3%** | +96.1% | -67pp |
| **AVERAGE** | **+78.9%** | **+76.8%** | **-11pp** |

### 4.3 DEER Early Exit Statistics

| Dataset | Avg Steps | Early Exits | Stop Reasons |
|---------|-----------|-------------|-------------|
| GSM8K | 2.0 | 0/3 | max_steps (3) |
| MATH-500 | 1.0 | 3/3 | deer_exit (3) |
| AMC | 1.3 | 2/3 | deer_exit (2), max_steps (1) |
| GPQA | 1.3 | 0/3 | natural (1), max_steps (2) |
| AIME | 1.7 | 1/3 | deer_exit (1), max_steps (2) |
| LiveCodeBench | 1.3 | 2/3 | deer_exit (2), max_steps (1) |

## 5. Verdict

### Target: Speed ≥ 20% ↑, Accuracy not degraded

| Criterion | Result | Status |
|-----------|--------|--------|
| Speed improvement ≥ 20% | **+78.9%** (average) | ✅ PASS |
| Per-dataset speed ≥ 20% | All ≥ +49.2% | ✅ PASS |
| Accuracy not degraded (per dataset) | 5/6 datasets unchanged, LiveCodeBench -67pp | ❌ FAIL |
| Overall accuracy not degraded | Baseline 55.6% → DEER 44.4% (-11pp) | ❌ FAIL |

**Conclusion: Speed target exceeded, but accuracy degraded on LiveCodeBench (2/3 → 0/3).**

## 6. Error Analysis (LiveCodeBench)

LiveCodeBench DEER lost 2 correct cases:

| Case | Baseline | DEER | Issue |
|------|----------|------|-------|
| livecodebench-001 | ✅ correct | ❌ wrong | DEER early exit (step 2, max_steps), answer=`\boxed{7}`, gt=easy — answer format mismatch |
| livecodebench-002 | ✅ correct | ❌ wrong | DEER early exit (step 1, deer_exit), incomplete reasoning, code not fully generated |

**Root cause**: Code generation tasks require longer reasoning chains. DEER's early stopping truncates the code output before completion. The `/no_think` answer phase cannot recover the lost code.

## 7. Recommendations

1. **LiveCodeBench-specific tuning**: Increase `min_think_tokens` or `think_ratio` for code tasks to allow more reasoning before early exit
2. **Answer extraction**: Improve post-processing to extract code blocks from truncated DEER output
3. **Larger test set**: 3 cases per dataset has high variance — expand to 10+ for statistical significance
4. **AMC accuracy**: Both baseline and DEER scored 0/3, suggesting these particular samples are very difficult; not a DEER-specific issue
