# DEER Benchmarks

Benchmark data, scripts, and results for evaluating the [DEER](https://github.com/yangyanping7133-stack/DEER) inference optimization on Qwen3-32B.

## Structure

```
.
├── v2.0-6sets/          # v2.0 results (6 datasets, small scale)
├── v3.0-full/
│   ├── scripts/         # Benchmark scripts
│   └── results/         # Baseline + DEER results, optimization logs
├── data/                # Dataset files (JSON)
└── reports/             # Generated reports
```

## Datasets (1270 total)

| Dataset | Samples | Type |
|---------|---------|------|
| GSM8K | 500 | math |
| MATH500 | 500 | math |
| GPQA | 200 | multiple choice |
| AMC | 40 | math |
| AIME | 30 | math |

## Key Scripts

| Script | Description |
|--------|-------------|
| `run_full_mixed.py` | Main benchmark runner (256 concurrency, baseline + DEER) |
| `run_deer_optimize.py` | DEER parameter optimizer (tests on regression cases) |
| `run_deer_adaptive.py` | Adaptive per-dataset DEER with fallback |
| `run_deer_progressive.py` | Progressive budget DEER (latest approach) |

## Full Report

See [v3.0-full/results/report.md](v3.0-full/results/report.md) for detailed benchmark results and analysis.
