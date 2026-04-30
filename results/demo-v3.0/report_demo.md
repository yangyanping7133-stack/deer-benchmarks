# DEER CoT Compression - Demo Pilot Report

Generated: 2026-04-30 15:53:20

## Environment
- Model: Qwen3-32B
- Framework: vLLM-Ascend (8x Ascend 910B4)
- Temperature: 0.6
- Max Tokens: 32768
- Concurrency: 12
- Scale: 6 test cases (1 per dataset, mixed mode)

## DEER Parameters
- threshold: 0.90
- prob_check_tokens: 10
- think_ratio: 0.8
- max_judge_steps: 4
- min_think_tokens: 1500

## Results

| Dataset | BL Correct | DEER Correct | Delta | BL Time | DEER Time | Speedup | Think Token Red |
|---------|-----------|-------------|-------|---------|-----------|---------|----------------|
| gsm8k           | ❌        | ❌          | SAME     |  103.9s |    59.8s |  +42.5% |   +43.7% |
| math500         | ✅        | ✅          | SAME     |  101.2s |    41.9s |  +58.6% |   +48.0% |
| amc             | ✅        | ✅          | SAME     |  116.6s |   116.5s |   +0.1% |   +11.0% |
| gpqa            | ✅        | ✅          | SAME     |   19.4s |    54.3s | -180.4% |  -169.2% |
| aime            | ❌        | ❌          | SAME     |  153.0s |    86.8s |  +43.3% |   +29.3% |
| livecodebench   | ✅        | ❌          | REGRESS  |  162.1s |    66.3s |  +59.1% |   +51.9% |

## Summary
- **Avg Speedup**: +35.1% (BL 656.2s → DEER 425.6s)
- **Accuracy**: BL 4/6 (67%) → DEER 3/6 (50%) | Delta: **-17pp**
- **Target (speedup ≥20%, accuracy loss = 0pp)**: ❌ FAIL

## Key Findings
1. **GPQA**: DEER is **slower** (-180% speedup) — min_think_tokens=1500 forces excessive thinking on a short-reasoning MCQ (BL only needed 250 think tokens)
2. **AMC**: Near-zero speedup — baseline already produces long reasoning that DEER doesn't truncate
3. **LiveCodeBench**: Accuracy regression — early exit truncated code reasoning
4. **GSM8K & AIME**: Both wrong in BL and DEER — no regression from DEER
5. **MATH-500**: Best case — +58.6% speedup with accuracy preserved

## Recommendations
- Lower min_think_tokens (1500→300-500) to avoid forcing excessive thinking on short-reasoning tasks
- Consider per-dataset adaptive min_think_tokens
- LiveCodeBench may need special handling (code generation differs from math reasoning)
