# DEER v3.0 Demo Report — End-to-End Verification

Generated: 2026-04-30 16:13:16

## Configuration
- Model: Qwen3-32B on vLLM-Ascend (8×Ascend 910B4)
- Datasets: GSM8K, MATH-500, AMC, GPQA, AIME (1 case each = 5 cases)
- LiveCodeBench: EXCLUDED
- Temperature: 0.6
- Max Tokens: 32768
- Concurrency: 12

## DEER Parameters (v2.1-final)
- threshold: 0.9
- prob_check_tokens: 10
- think_ratio: 0.6
- max_judge_steps: 2
- temperature: 0.6
- min_think_tokens: 300

## Raw Results

| Method | Accuracy | Avg E2E (s) | Avg TTFT (s) | Think Tokens | Completion Tokens |
|--------|----------|-------------|--------------|-------------|-------------------|
| baseline | 3/5 | 98.5 | 0.43 | 918 | 1644 |
| deer | 4/5 | 59.7 | 42.62 | 643 | 967 |

## Per-Dataset Breakdown

| Dataset | Method | Correct | E2E (s) | Think Tok | Comp Tok | Stop |
|---------|--------|---------|---------|-----------|----------|------|
| aime | baseline | False | 155.0 | 1091 | 2611 | stop |
| amc | baseline | True | 107.9 | 1093 | 1807 | stop |
| gpqa | baseline | True | 31.7 | 427 | 522 | stop |
| gsm8k | baseline | False | 96.2 | 878 | 1590 | stop |
| math500 | baseline | True | 102.0 | 1103 | 1688 | stop |
| aime | deer | True | 98.3 | 1091 | 1597 | deer_exit |
| amc | deer | True | 109.0 | 1165 | 1778 | natural |
| gpqa | deer | True | 48.7 | 565 | 790 | natural |
| gsm8k | deer | False | 20.8 | 238 | 325 | max_steps |
| math500 | deer | True | 22.0 | 155 | 347 | max_steps |

## Comparison: Baseline vs DEER

| Metric | Value |
|--------|-------|
| Time Speedup | **39.4%** |
| Token Reduction | 41.1% |
| Think Token Reduction | 30.0% |
| Baseline Accuracy | 3/5 |
| DEER Accuracy | 4/5 |

## Verdict
- Speed target (≥20%): **PASS** (39.4%)
- Accuracy (no degradation): **PASS** (BL=3/5, DEER=4/5)
- Overall: **PASS**