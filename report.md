# DEER CoT Compression Benchmark Report v2.0

Generated: 2026-04-29 09:49:14

## Environment
- Model: Qwen3-32B
- Framework: vLLM-Ascend (8x Ascend 910B4)
- API: http://127.0.0.1:8000
- Temperature: 0.6
- Max Tokens: 32768
- Concurrency: 12

## DEER Parameters
- threshold: 0.9
- prob_check_tokens: 10
- think_ratio: 0.6
- max_judge_steps: 2
- temperature: 0.6
- min_think_tokens: 300

## Results Summary

| Dataset | Method | Accuracy | Avg E2E (s) | Avg TTFT (s) | Think Tokens | Completion Tokens | DEER Steps | Speedup |
|---------|--------|----------|-------------|--------------|-------------|-------------------|------------|---------|
| gsm8k | baseline | 0/1 | 112.8 | 0.28 | 1065 | 1934 | 0.0 | - |
| gsm8k | deer | 0/1 | 49.6 | 19.71 | 257 | 852 | 2.0 | - |
| math500 | baseline | 1/1 | 140.2 | 0.22 | 1601 | 2417 | 0.0 | - |
| math500 | deer | 1/1 | 52.1 | 21.52 | 311 | 882 | 1.0 | - |
| amc | baseline | 1/1 | 85.4 | 0.22 | 951 | 1479 | 0.0 | - |
| amc | deer | 1/1 | 62.4 | 31.86 | 485 | 1064 | 1.0 | - |
| gpqa | baseline | 1/1 | 34.1 | 0.21 | 328 | 590 | 0.0 | - |
| gpqa | deer | 1/1 | 22.8 | 22.82 | 306 | 392 | 0.0 | - |
| aime | baseline | 1/1 | 130.2 | 0.23 | 943 | 2255 | 0.0 | - |
| aime | deer | 0/1 | 110.9 | 73.52 | 953 | 1886 | 2.0 | - |
| livecodebench | baseline | 1/1 | 175.1 | 0.24 | 1661 | 3019 | 0.0 | - |
| livecodebench | deer | 1/1 | 54.9 | 24.36 | 367 | 929 | 1.0 | - |

## Comparison: Baseline vs DEER

| Dataset | BL Acc | DEER Acc | Time Speedup | Token Reduction | Think Reduction | DEER Steps |
|---------|--------|----------|-------------|-----------------|-----------------|------------|
| gsm8k | 0/1 | 0/1 | 56.0% | 55.9% | 75.9% | 2.0 |
| math500 | 1/1 | 1/1 | 62.8% | 63.5% | 80.6% | 1.0 |
| amc | 1/1 | 1/1 | 27.0% | 28.1% | 49.0% | 1.0 |
| gpqa | 1/1 | 1/1 | 33.2% | 33.6% | 6.7% | 0.0 |
| aime | 1/1 | 0/1 | 14.8% | 16.4% | -1.1% | 2.0 |
| livecodebench | 1/1 | 1/1 | 68.7% | 69.2% | 77.9% | 1.0 |

## Overall Averages
- Avg Time Speedup: 43.8%
- Avg Token Reduction: 44.4%
- Avg Think Token Reduction: 48.2%

## Conclusion
- Target (speedup >= 20%): **MET**