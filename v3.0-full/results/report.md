# DEER CoT Compression Full Benchmark Report v3.0
Generated: 2026-05-04 18:56:19

## Environment
- Model: Qwen3-32B (bfloat16)
- Framework: vLLM-Ascend (8x Ascend 910B4)
- Concurrency: 256
- Temperature: 0.6, Max Tokens: 32768

## DEER Parameters
- threshold: 0.95
- prob_check_tokens: 10
- think_ratio: 0.8
- max_judge_steps: 4
- temperature: 0.6
- min_think_tokens: 1000

## Comparison: Baseline vs DEER

| Dataset | Samples | BL Acc | DEER Acc | BL E2E(s) | DEER E2E(s) | Speedup | BL Tok | DEER Tok | Tok Red | Acc Diff |
|---------|---------|--------|----------|-----------|-------------|---------|--------|----------|---------|----------|
| gsm8k | 500 | 480/500 | 443/500 | 289.6 | 104.5 | 63.9% | 1650 | 702 | 57.5% | -37 |
| math500 | 500 | 444/500 | 398/500 | 628.7 | 189.2 | 69.9% | 4590 | 1332 | 71.0% | -46 |
| amc | 40 | 19/40 | 23/40 | 919.8 | 316.3 | 65.6% | 6993 | 2478 | 64.6% | +4 |
| gpqa | 200 | 188/200 | 192/200 | 190.7 | 121.6 | 36.2% | 948 | 828 | 12.6% | +4 |
| aime | 30 | 6/30 | 4/30 | 1564.0 | 241.7 | 84.5% | 13286 | 1811 | 86.4% | -2 |

- **Avg Time Speedup: 64.0%**
- **Min Accuracy Diff: -46**

## Conclusion: NEEDS OPTIMIZATION
- Speedup >= 20%: MET (64.0%)
- Accuracy no drop: NOT MET (min: -46)