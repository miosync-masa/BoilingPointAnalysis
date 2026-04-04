<p align="center">
  <img src="https://www.miosync.link/github/0_4.jpg" alt="Lambda³" width="400"/>
</p>

# GETTER One

**Geometric Event-driven Tensor-based Time-series Extraction & Recognition**

*Omnidimensional Network Engine*

[![PyPI](https://img.shields.io/pypi/v/getter-one)](https://pypi.org/project/getter-one/)
[![Python](https://img.shields.io/pypi/pyversions/getter-one)](https://pypi.org/project/getter-one/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---
## v0.1.2 (2026-04) — 📄 Paper Submission Version

> **This version corresponds to the manuscript submitted to Information Science.**
>
> ```
> pip install getter-one==0.1.2
> ```


### GETTER One is a **discrete geometric framework** for structural event detection and causal network extraction in N-dimensional time series.

It is **not** a forecasting model. It detects *what is changing*, *when*, and *which dimensions are driving the change* — using closed-form geometric computations with no MCMC, no gradient descent, and no threshold tuning.

## Key Features

- **Lambda³ (Λ³) structural analysis** — Computes displacement vectors (ΛF), structural tension (ρT), cross-dimension synchrony (σₛ), and cooperative events (ΔΛC) in closed form
- **Causal network extraction** — Identifies directed, time-lagged causal relationships between dimensions
- **Statistical confidence** — Permutation tests, bootstrap confidence intervals, and effect sizes (no Bayesian inference required)
- **Domain-agnostic** — Works on any N-dimensional time series: weather, finance, sensors, biology
- **Minimal dependencies** — Core requires only `numpy`, `scipy`, `pandas`
- **GPU-ready** — Optional CUDA acceleration for large-scale datasets via CuPy

## Installation

```bash
pip install getter-one
```

Optional extras:

```bash
pip install getter-one[gpu]       # CUDA/CuPy acceleration
pip install getter-one[viz]       # matplotlib + plotly
pip install getter-one[full]      # everything
```

## Quick Start

```python
from getter_one.pipeline import run

# One line — full pipeline
result = run("weather.csv", target="precipitation")
print(result.report)
```

### Step by Step

```python
from getter_one.data import load
from getter_one.structures import LambdaStructuresCore
from getter_one.analysis import NetworkAnalyzerCore, assess_confidence

# 1. Load data
dataset = load("weather.csv", target="precipitation", normalize="range")

# 2. Compute Λ³ structures
core = LambdaStructuresCore()
structures = core.compute_lambda_structures(
    dataset.state_vectors, window_steps=24,
    dimension_names=dataset.dimension_names,
)

# 3. Extract causal network
analyzer = NetworkAnalyzerCore(sync_threshold=0.3, causal_threshold=0.25, max_lag=12)
network = analyzer.analyze(dataset.state_vectors, dimension_names=dataset.dimension_names)

# 4. Assess confidence
confidence = assess_confidence(
    state_vectors=dataset.state_vectors,
    lambda_structures=structures,
    network_result=network,
    dimension_names=dataset.dimension_names,
)
```

## CLI

```bash
# Run full pipeline
getter-one run weather.csv --target precipitation --report report.md

# Data preparation
getter-one-loader load weather.csv --target precipitation -o prepared.csv
getter-one-loader merge weather.csv air_quality.json --time date -o merged.csv
getter-one-loader info data.csv

# System info
getter-one info
getter-one check-gpu
```

## Pipeline Architecture

```
Input Data (csv/json/parquet/xlsx/npy)
  │
  ▼
┌─────────────────────────────────┐
│  Lambda³ Structures (Λ³)        │  Closed-form geometric computation
│  ΛF, ΛFF, ρT, σₛ, Q_Λ, ΔΛC   │  No parameters to tune
└──────────────┬──────────────────┘
               │
  ┌────────────┼────────────┐
  ▼            ▼            ▼
┌──────┐  ┌──────┐  ┌──────────┐
│Bound │  │Topo  │  │Anomaly   │  Detection modules
│ary   │  │Break │  │Detection │  (GPU-accelerated)
└──┬───┘  └──┬───┘  └────┬─────┘
   └─────────┼───────────┘
             ▼
┌─────────────────────────────────┐
│  Causal Network Analysis        │  Sync + directed causal links
│  Sync matrix, lag estimation    │  Hub/driver/follower detection
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  Confidence Assessment          │  Permutation test → p-values
│  No MCMC, no Bayesian inference │  Bootstrap → confidence intervals
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│  Report Generation              │  Markdown report with all results
└─────────────────────────────────┘
```

## What GETTER One Is (and Isn't)

| | GETTER One | Traditional Models |
|---|---|---|
| **Purpose** | Structural event detection | Forecasting / prediction |
| **Method** | Discrete geometry (closed-form) | Regression / deep learning |
| **Parameters** | Window size only | Thresholds, hyperparameters |
| **Output** | Events, boundaries, causal network | Predicted values |
| **Relationship** | Complements prediction models | Competes with each other |

GETTER One detects **what changed and why**, not **what will happen next**. It can be used as a feature extractor for prediction models — experiments show that combining raw data with GETTER One features improves Transformer event detection F1 by +5.7% over raw data alone, and outperforms adding 7 additional weather variables (+4.3% vs +6.1%).

## Benchmark Results

### Standard Benchmark

Causal inference benchmark against 5 established methods on synthetic data with ground truth (6 scenarios, 20 repeats each):

| Method | Composite | F1 (dir) | Lag MAE | Sign Acc | Spurious |
|--------|-----------|----------|---------|----------|----------|
| PCMCI+ | 0.877 | 0.740 ± 0.364 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| VAR Granger | 0.853 | 0.697 ± 0.353 | 0.010 ± 0.100 | 0.990 ± 0.100 | 0.050 ± 0.154 |
| **GETTER One** | **0.821** | **0.792 ± 0.366** | **0.000 ± 0.000** | **1.000 ± 0.000** | 0.500 ± 0.000 |
| Event XCorr | 0.537 | 0.534 ± 0.403 | 0.244 ± 0.825 | — | 0.300 ± 0.340 |
| Transfer Entropy | 0.446 | 0.432 ± 0.368 | 0.413 ± 1.396 | — | 0.525 ± 0.112 |
| Graphical Lasso | 0.120 | — | — | — | 1.000 ± 0.000 |

**Per-scenario F1 (directed):**

| Scenario | GETTER One | PCMCI+ | VAR Granger | TE | EventXCorr |
|----------|-----------|--------|-------------|-----|------------|
| S0 null | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| S1 delayed | **1.000 ± 0.000** | 0.825 ± 0.183 | 0.662 ± 0.141 | 0.645 ± 0.233 | 0.450 ± 0.446 |
| S2 asymmetric | **0.950 ± 0.122** | 0.883 ± 0.163 | 0.658 ± 0.037 | 0.662 ± 0.177 | 0.520 ± 0.434 |
| S5 confounder | 0.800 ± 0.000 | **0.990 ± 0.045** | 0.960 ± 0.082 | 0.734 ± 0.092 | 0.623 ± 0.213 |
| S7 event delayed | **1.000 ± 0.000** | 0.825 ± 0.183 | 0.917 ± 0.148 | 0.342 ± 0.417 | 0.842 ± 0.206 |
| S8 event asym | **1.000 ± 0.000** | 0.917 ± 0.148 | 0.983 ± 0.075 | 0.212 ± 0.356 | 0.767 ± 0.262 |

GETTER One achieves **F1 = 1.000 with σ = 0.000** (perfectly deterministic) on all event-driven scenarios, with **zero lag error** and **perfect sign accuracy** across all 20 trials. Its closed-form computation produces identical results regardless of noise realization — a direct consequence of the displacement-based (diff) approach inherited from molecular dynamics trajectory analysis.

### Hell Mode Robustness Benchmark

Causal detection under non-stationary, non-linear, non-Gaussian conditions (8 scenarios × 5 repeats, ground truth: X0→X1 lag=2, X0→X2 lag=3):

| Method | F1 | Precision | Recall | Lag MAE | Detected | Spurious |
|--------|-----|-----------|--------|---------|----------|----------|
| **GETTER One** | **0.394 ± 0.406** | **0.471** | 0.362 | **0.000** | **1.4** | **0.118** |
| PCMCI+ | 0.376 ± 0.271 | 0.279 | 0.738 | 0.306 | 8.6 | 0.142 |
| VAR Granger | 0.283 ± 0.308 | 0.217 | 0.475 | 0.630 | 7.5 | 0.260 |
| Transfer Entropy | 0.164 ± 0.240 | 0.119 | 0.887 | 0.770 | 84.3 | 0.188 |

**Per-scenario F1 (Hell Mode):**

| Scenario | GETTER One | PCMCI+ | VAR Granger | TE |
|----------|-----------|--------|-------------|-----|
| H1 pulse | 0.513 | 0.494 | 0.431 | 0.385 |
| H2 phase jump | **0.547** | 0.183 | 0.093 | 0.025 |
| H3 bifurcation | 0.000 | 0.142 | 0.040 | 0.025 |
| H4 cascade | 0.133 | **0.649** | 0.581 | 0.668 |
| H5 resonance | **0.960** | 0.774 | 0.000 | 0.043 |
| H6 decay | 0.533 | 0.428 | **0.813** | 0.044 |
| H7 multi-hell | **0.467** | 0.238 | 0.304 | 0.029 |
| H8 progressive | 0.000 | 0.100 | 0.000 | 0.092 |

GETTER One achieves the **highest F1 and precision** under non-ideal conditions, with the **fewest detected links** (1.4 avg vs ground truth of 2) — demonstrating conservative, high-confidence detection rather than over-reporting. Each method has characteristic strengths: GETTER One excels at phase jumps and resonance, PCMCI+ at cascades, and VAR Granger at structural decay.

## Real-World Application: Weather Network

Analysis of 7 cities around Tokyo (hourly data, October 2024) reveals physically correct causal structures:

**Precipitation causal network:**
- Nagano → Gunma (lag=1h) → Saitama (lag=1h) — inland-to-plains propagation
- Yamanashi → Gunma (lag=2h) — basin-to-plains flow
- Drivers: Nagano, Yamanashi (mountain sources)
- Followers: Saitama, Tochigi (downstream receivers)

**Cross-variable causality (42D full analysis):**
- Temperature changes → Surface pressure changes (lag=2-3h) — thermodynamic law rediscovered from data alone

## Supported Data Formats

| Format | Extension | Read | Write |
|--------|-----------|------|-------|
| CSV / TSV | `.csv`, `.tsv` | ✅ | ✅ |
| JSON | `.json` | ✅ | — |
| Parquet | `.parquet` | ✅ | — |
| Excel | `.xlsx`, `.xls` | ✅ | — |
| NumPy | `.npy`, `.npz` | ✅ | ✅ |

## Project Structure

```
getter_one/
├── data/           Data loading & merging (multi-format)
├── structures/     Lambda³ structural computation (CPU)
├── core/           GPU infrastructure (CuPy/CUDA)
├── detection/      Boundary, topology, anomaly, phase-space detection
├── analysis/       Network analysis, confidence assessment, report generation
├── pipeline.py     Full pipeline orchestration
└── cli.py          Command-line interface
```

## Related Projects

- [**BANKAI-MD**](https://github.com/miosync-masa/bankai) — GPU-accelerated molecular dynamics analysis built on the same Lambda³ theory. GETTER One is the domain-agnostic generalization; BANKAI-MD is the MD-specialized implementation.

## Citation

Paper in preparation. If you use GETTER One in your research, please cite this repository.

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with 💕 by Masamichi & Tamaki*

*CHANGE EAGLE // Aerial Type // Structure Detection*
