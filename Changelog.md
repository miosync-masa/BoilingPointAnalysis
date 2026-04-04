# Changelog

All notable changes to GETTER One will be documented in this file.

## v0.1.9 (2026-04-04)

### Added
- Partial correlation (precision matrix + multi-lag residual) for confound removal
- Common ancestor filter (`_filter_spurious_edges`) for spurious edge elimination
- Adaptive network parameters (data-driven threshold tuning)
- Pipeline integration for `extended_detection_gpu` and `phase_space_gpu`
- `check_gpu_features()` utility function
- `CascadeTracker`, `InverseChecker`, `Pipeline` lazy imports

### Performance
- Synthetic benchmark (9 scenarios × 5 repeats): **#1** (Composite 0.869)
- Hell Mode benchmark (8 scenarios × 5 repeats): **#1** (Composite 0.658)

## v0.1.8 (2026-04-04)

- `enable_extended`, `enable_phase_space`, `adaptive_network` pipeline parameters
- `InverseChecker` and `gpu_inverse.py` (GPU-only structural verification)
- Maintainer name typo fix

## v0.1.7 (2026-04-03)

- Extended detection GPU module (periodic, gradual, drift)
- Phase space GPU analysis (attractor, recurrence, Lyapunov)

## v0.1.3 – v0.1.6 (2026-04)

- Cascade tracker implementation
- Confidence assessment kit
- Network analyzer improvements
- GPU/CPU automatic fallback enhancements

## v0.1.2 (2026-04) — 📄 Paper Submission Version

> **This version corresponds to the manuscript submitted to Information Science.**
>
> ```
> pip install getter-one==0.1.2
> ```

- Lambda structures core (Λ, ΛF, ρT, σₛ, ΔΛC)
- Structural boundary detection (GPU/CPU)
- Topological break detection (GPU/CPU)
- Multi-scale anomaly detection (GPU/CPU)
- Network analyzer core (sync + causal, pairwise correlation)
- Data loader with multi-format support
- CLI tools (`getter-one`, `getter-one-loader`)

## v0.1.1 (2026-04)

- Initial release
