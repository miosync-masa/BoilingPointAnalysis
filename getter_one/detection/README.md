## Detection Module — CUDA Kernel Reference

BANKAI's `detection/` module implements 5 specialized detectors with **20+ custom CUDA kernels** (CuPy RawKernel + Numba CUDA JIT). Each kernel is hand-written C/CUDA — not CuPy wrapper calls — with `atomicAdd`, shared memory awareness, and coalesced access for maximum throughput on sub-picosecond trajectory data.

All kernels include automatic CPU fallback when GPU is unavailable.

---

### Architecture Overview

```
detection/
├── anomaly_detection_gpu.py      ── Layer 3: Multi-scale anomaly scoring
├── boundary_detection_gpu.py     ── Layer 2: Structural boundary (ΔΛC) detection
├── topology_breaks_gpu.py        ── Layer 2: Topological flow break detection
├── extended_detection_gpu.py     ── Layer 3+: Long-range pattern detection
└── phase_space_gpu.py            ── Layer 4: Phase space attractor analysis (13 kernels)
```

---

### 1. Anomaly Detection (`anomaly_detection_gpu.py`)

**Purpose:** Multi-scale anomaly scoring — combines global (topological break-derived), local (boundary-focused), and extended (drift/periodic/phase) anomaly signals into a unified score.

| Kernel | Type | Purpose |
|--------|------|---------|
| `apply_boundary_emphasis_kernel` | RawKernel | Applies Gaussian-weighted emphasis around detected boundary locations. Each thread handles one boundary point and uses `atomicAdd` to accumulate weighted contributions across the surrounding window (σ=20, window=50 frames). |
| `anomaly_detection_kernel` | Imported (gpu_kernels) | Adaptive z-score based local anomaly detection with sliding window. |

**Pipeline:**
1. Global anomalies — weighted sum of ΛF, ΛFF, ρT, and Q_λ break signals
2. Local anomalies — Gaussian emphasis around structural boundaries (RawKernel)
3. Extended anomalies — periodic (FFT), gradual (multi-scale trend), drift, Rg transitions, phase space
4. Final integration — `0.4 × global + 0.3 × local + 0.3 × extended`

---

### 2. Boundary Detection (`boundary_detection_gpu.py`)

**Purpose:** Detects structural boundaries (ΔΛC — "meaning crystallization moments") by combining fractal dimension changes, structural coherence drops, coupling weakness, and information entropy barriers.

| Kernel | Type | Purpose |
|--------|------|---------|
| `shannon_entropy_kernel` | RawKernel | Computes local Shannon entropy within a sliding window over the tension field ρT. Normalizes local values into probability distributions and calculates H = -Σ p·log(p) per frame. |
| `detect_jumps_kernel` | RawKernel | Detects sudden jumps in tension field by comparing frame-to-frame differences against local mean amplitude. Threshold-based with adaptive local normalization. |
| `compute_gradient_kernel` | Imported (gpu_kernels) | GPU-accelerated numerical gradient for fractal/entropy gradient computation. |
| `compute_local_fractal_dimension_kernel` | Imported (gpu_kernels) | Local fractal dimension estimation via variance-based scaling in sliding windows. |

**Boundary Score Formula:**
```
boundary_score = (2.0 × fractal_gradient + 1.5 × coherence_drop
                + 1.0 × coupling_weakness + 1.0 × entropy_gradient) / 5.5
```

Peak detection uses `scipy.signal.find_peaks` on CPU (stability over speed) with adaptive thresholding (`mean + σ`, fallback to `mean + 0.5σ`).

---

### 3. Topology Breaks Detection (`topology_breaks_gpu.py`)

**Purpose:** Detects topological flow disruptions — irreversible structural changes where the ΛF flow field undergoes qualitative transformation. Includes phase coherence breaks and structural singularity detection.

| Kernel | Type | Purpose |
|--------|------|---------|
| `local_extrema_kernel` | RawKernel | Identifies local maxima and minima within a window by brute-force comparison. Uses early termination when a point is confirmed neither max nor min. |
| `anomaly_detection_kernel` | Imported (gpu_kernels) | Adaptive z-score for ΛF and ΛFF magnitude anomalies. |
| `gaussian_filter1d_gpu` | CuPy scipy | Multi-scale Gaussian smoothing for tension field jump detection at 3 sigma scales. |

**7-Component Detection:**
1. **ΛF flow anomaly** — sudden changes in structural flow magnitude
2. **ΛFF acceleration anomaly** — sign changes and magnitude jumps in flow acceleration
3. **ρT tension jumps** — multi-scale smoothing residual analysis (3 Gaussian scales)
4. **Q_λ topological charge breaks** — phase difference thresholding + cumulative deviation from linear growth
5. **Phase coherence breaks** — negative gradient emphasis (coherence drops weighted 2×)
6. **Structural singularities** — vector field divergence (3σ threshold) + phase space correlation anomalies + local extrema
7. **Combined score** — weighted integration with weights [1.0, 0.8, 0.6, 1.2, 0.9, 1.1]

---

### 4. Extended Detection (`extended_detection_gpu.py`)

**Purpose:** Long-range pattern detection across multiple timescales — periodic transitions (FFT), gradual structural drift, Rg-based size changes, and phase space embedding anomalies.

| Kernel | Type | Purpose |
|--------|------|---------|
| `_drift_kernel` | Numba `@cuda.jit` | Per-frame structural drift calculation. Each thread computes local ρT mean within a reference window and measures deviation from initial reference values (both ρT and Q_cumulative). |
| `_change_rate_kernel` | Numba `@cuda.jit` | Local change rate calculation for Radius of Gyration transitions. Normalizes absolute gradient by local Rg mean. |
| `_knn_anomaly_kernel` | Numba `@cuda.jit` | k-nearest neighbor anomaly detection in phase space. Uses `cuda.local.array` for per-thread distance storage and iterative min-finding for k-NN (k=20, max 100 neighbors). |

**Sub-Detectors:**
- **Periodic transitions** — FFT power spectrum analysis with MAD-based peak detection; periodic contribution mapped as sinusoidal score modulation
- **Gradual transitions** — multi-scale trend extraction (windows: 500, 1000, 2000) with sustained gradient detection
- **Structural drift** — reference window comparison with Gaussian smoothing (σ=100)
- **Rg transitions** — gradient-based with 2× emphasis on contraction (aggregation detection)
- **Phase space anomalies** — Takens embedding + k-NN density-based outlier scoring

---

### 5. Phase Space Analysis (`phase_space_gpu.py`) — The Beast

**Purpose:** Comprehensive phase space reconstruction, attractor characterization, recurrence quantification, and trajectory anomaly detection. Contains **13 custom CUDA RawKernels** — the largest kernel collection in BANKAI.

| # | Kernel | Purpose |
|---|--------|---------|
| 1 | `compute_rqa_features_kernel` | Recurrence Quantification Analysis — extracts determinism (diagonal line density), laminarity (vertical line density), and trapping time from the recurrence matrix. Uses `atomicAdd` for parallel accumulation across diagonal/vertical structures. |
| 2 | `compute_attractor_volume_kernel` | Maps phase space points to a 3D voxel grid (128³ default) and marks occupied voxels. Attractor volume = fraction of occupied voxels, characterizing the geometric extent of the dynamical attractor. |
| 3 | `compute_fractal_dimension_kernel` | Correlation integral computation for Grassberger-Procaccia dimension estimation. Each thread evaluates one distance pair against 30 log-spaced radii, using `atomicAdd` to build the correlation integral. |
| 4 | `compute_complexity_measures_kernel` | 5-in-1 kernel computing: (a) local divergence rate, (b) prediction error, (c) local variance/complexity, (d) entropy contribution, (e) orbital stability — all via `atomicAdd` accumulation. |
| 5 | `knn_trajectory_anomaly_kernel` | Per-point k-NN anomaly scoring with fixed-size distance array (256 elements). Iterative min-extraction for k nearest neighbors. Points far from their neighbors = anomalous trajectory segments. |
| 6 | `compute_diagonal_distribution_kernel` | Extracts the distribution of diagonal line lengths from the recurrence matrix for entropy calculation and determinism assessment. |
| 7 | `compute_curvature_anomaly_kernel` | 3-point trajectory curvature via cosine angle between consecutive velocity vectors. High curvature (1 - cos θ → 2) indicates sharp trajectory turns = potential structural events. |
| 8 | `compute_velocity_anomaly_kernel` | Phase space velocity magnitude (Euclidean norm of consecutive point differences). Velocity spikes indicate rapid structural transitions. |
| 9 | `compute_acceleration_anomaly_kernel` | Second-order finite difference in phase space. Acceleration anomalies reveal forces driving structural change. |
| 10 | `compute_lyapunov_kernel` | Local Lyapunov exponent estimation. Each thread tracks a reference point and its nearest neighbor through time evolution, computing log(divergence_rate) averaged over up to 10 timesteps. Positive = chaos, negative = stability. |
| 11 | `map_transition_scores_kernel` | Maps windowed transition scores back to the original frame timeline using `atomicAdd` for overlapping window contributions. |
| 12 | `compute_pairwise_distances_kernel` | O(n²) pairwise Euclidean distance computation from flattened phase space. Index recovery from linear pair index via quadratic formula. Used for correlation dimension and recurrence matrix. |
| 13 | `compute_fnn_kernel` | False Nearest Neighbors test for optimal embedding dimension selection. Kennel et al. criterion: if distance ratio in (d+1)-th dimension exceeds 10×, the neighbor is "false." |

**Full Pipeline:**
1. **Optimal embedding estimation** — mutual information (delay τ) + FNN (dimension d)
2. **Phase space reconstruction** — Takens time-delay embedding with broadcast indexing
3. **Attractor analysis** — correlation dimension, Lyapunov exponent, voxel volume, fractal measure, information dimension
4. **Recurrence quantification** — recurrence matrix → RQA features (determinism, laminarity, entropy, trapping time)
5. **Anomaly detection** — k-NN distance + curvature + velocity + acceleration (weighted integration)
6. **Dynamics characterization** — complexity measures (chaos, predictability, entropy rate, stability, periodicity)
7. **Score integration** — `0.7 × local_anomaly + 0.3 × (0.5 × attractor_anomaly + 0.5 × recurrence_anomaly)`

---

### Kernel Count Summary

| Module | RawKernels | Numba JIT | Imported | Total |
|--------|-----------|-----------|----------|-------|
| anomaly_detection | 1 | — | 1 | 2 |
| boundary_detection | 2 | — | 2 | 4 |
| topology_breaks | 1 | — | 1+ | 2+ |
| extended_detection | — | 3 | — | 3 |
| phase_space | 13 | — | — | 13 |
| **Total** | **17** | **3** | **4** | **24+** |

All RawKernels target PTX 8.4 (CUDA 12.x compatible). All modules include comprehensive CPU fallback paths.
