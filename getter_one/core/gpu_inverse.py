"""
CUDA Kernels for Inverse Problem & Network Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

2系統のGPU検証カーネルを統合管理 🔥

Detection系統 (逆問題 3軸):
  1. Reconstruction Error — Λ³構造でデータを再構成できるか
  2. Topo Preservation   — 位相構造の連続性が保たれているか
  3. Jump Integrity      — 変化が構造的に有意か
  → inverse_verify_all()

Network系統 (統計テスト 4軸):
  1. Surrogate   — 局所的に見て本当に異常か（ランク検定）
  2. ρT整合性    — テンション蓄積後の発火か
  3. 持続性      — 構造変化が持続してるか（不可逆性）
  4. 協調性      — 複数次元で協調的に発火してるか
  → network_verify_all()

⚠️ GPU ONLY — CPUフォールバック無し
by 環ちゃん
"""

from __future__ import annotations

import logging

import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError as _err:
    raise ImportError(
        "InverseChecker requires CUDA.\n"
        "Install CuPy: pip install cupy-cuda12x\n"
        "Or: pip install getter-one[gpu]"
    ) from _err

logger = logging.getLogger("getter_one.core.gpu_inverse")


# =====================================================================
# CUDA Kernels — Detection (逆問題 3軸)
# =====================================================================

INVERSE_RECONSTRUCTION_KERNEL = r"""
extern "C" __global__
void inverse_reconstruction_kernel(
    const float* __restrict__ lambda_matrix,
    const float* __restrict__ events_gram,
    float* __restrict__ output_errors,
    const int n_paths,
    const int n_events
) {
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_events) return;

    float row_error = 0.0f;
    for (int j = 0; j < n_events; j++) {
        float recon_ij = 0.0f;
        for (int k = 0; k < n_paths; k++) {
            recon_ij += lambda_matrix[k * n_events + eid]
                      * lambda_matrix[k * n_events + j];
        }
        float diff = events_gram[eid * n_events + j] - recon_ij;
        row_error += diff * diff;
    }
    output_errors[eid] = sqrtf(row_error);
}
"""

TOPO_PRESERVATION_KERNEL = r"""
extern "C" __global__
void topo_preservation_kernel(
    const float* __restrict__ lambda_matrix,
    float* __restrict__ output_topo_scores,
    const int n_paths,
    const int n_events
) {
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_events) return;

    if (eid == 0 || eid == n_events - 1) {
        output_topo_scores[eid] = 0.5f;
        return;
    }

    float topo_score = 0.0f;
    int valid_paths = 0;

    for (int k = 0; k < n_paths; k++) {
        float v_prev = lambda_matrix[k * n_events + (eid - 1)];
        float v_curr = lambda_matrix[k * n_events + eid];
        float v_next = lambda_matrix[k * n_events + (eid + 1)];

        float phase_before = atan2f(v_curr, v_prev);
        float phase_after  = atan2f(v_next, v_curr);
        float phase_diff   = phase_after - phase_before;

        if (phase_diff > 3.14159265f) phase_diff -= 6.28318530f;
        if (phase_diff < -3.14159265f) phase_diff += 6.28318530f;

        topo_score += phase_diff * phase_diff;
        valid_paths++;
    }

    output_topo_scores[eid] = (valid_paths > 0)
        ? topo_score / (float)valid_paths : 0.5f;
}
"""

JUMP_INTEGRITY_KERNEL = r"""
extern "C" __global__
void jump_integrity_kernel(
    const float* __restrict__ lambda_matrix,
    const float* __restrict__ path_means,
    const float* __restrict__ path_stds,
    float* __restrict__ output_jump_scores,
    const int n_paths,
    const int n_events,
    const float jump_sigma
) {
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_events || eid == 0) {
        if (eid < n_events) output_jump_scores[eid] = 0.0f;
        return;
    }

    float jump_score = 0.0f;
    int jump_count = 0;

    for (int k = 0; k < n_paths; k++) {
        float v_prev = lambda_matrix[k * n_events + (eid - 1)];
        float v_curr = lambda_matrix[k * n_events + eid];
        float delta = fabsf(v_curr - v_prev);
        float threshold = path_means[k] + jump_sigma * path_stds[k];

        if (delta > threshold) {
            jump_score += (delta - threshold) / (path_stds[k] + 1e-8f);
            jump_count++;
        }
    }

    output_jump_scores[eid] = (jump_count > 0)
        ? jump_score / (float)n_paths : 0.0f;
}
"""

HYBRID_SCORE_KERNEL = r"""
extern "C" __global__
void hybrid_score_kernel(
    const float* __restrict__ recon_errors,
    const float* __restrict__ topo_scores,
    const float* __restrict__ jump_scores,
    float* __restrict__ output_hybrid,
    float* __restrict__ output_verdict,
    const int n_events,
    const float w_recon,
    const float w_topo,
    const float w_jump,
    const float recon_mean, const float recon_std,
    const float topo_mean,  const float topo_std,
    const float jump_mean,  const float jump_std
) {
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_events) return;

    float z_recon = (recon_std > 1e-8f)
        ? (recon_errors[eid] - recon_mean) / recon_std : 0.0f;
    float z_topo  = (topo_std > 1e-8f)
        ? (topo_scores[eid] - topo_mean) / topo_std : 0.0f;
    float z_jump  = (jump_std > 1e-8f)
        ? (jump_scores[eid] - jump_mean) / jump_std : 0.0f;

    float hybrid = -w_recon * z_recon - w_topo * z_topo + w_jump * z_jump;
    output_hybrid[eid] = hybrid;
    output_verdict[eid] = (hybrid > 0.0f) ? 1.0f : 0.0f;
}
"""


# =====================================================================
# CUDA Kernels — Network (統計テスト 4軸)
# =====================================================================

SURROGATE_KERNEL = r"""
extern "C" __global__
void surrogate_kernel(
    const float* __restrict__ dlc_scores,
    const int*   __restrict__ event_frames,
    float*       __restrict__ output_scores,
    const int n_diff, const int n_events,
    const int surrogate_window
) {
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_events) return;

    const int frame = event_frames[eid];
    if (frame < 0 || frame >= n_diff) { output_scores[eid] = 0.0f; return; }

    const float event_score = dlc_scores[frame];
    const int w_start = max(0, frame - surrogate_window);
    const int w_end   = min(n_diff, frame + surrogate_window + 1);

    int n_total = 0, n_lower = 0;
    for (int t = w_start; t < w_end; t++) {
        if (abs(t - frame) <= 2) continue;
        n_total++;
        if (dlc_scores[t] < event_score) n_lower++;
    }

    output_scores[eid] = (n_total > 0) ? (float)n_lower / (float)n_total : 0.5f;
}
"""

RHO_T_CONSISTENCY_KERNEL = r"""
extern "C" __global__
void rho_t_consistency_kernel(
    const float* __restrict__ rho_t_mean,
    const int*   __restrict__ event_frames,
    float*       __restrict__ output_scores,
    const int n_frames, const int n_events,
    const int lookback_window
) {
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_events) return;

    const int frame = event_frames[eid];
    if (frame < 0 || frame >= n_frames) { output_scores[eid] = 0.0f; return; }

    const float rho_at_event = rho_t_mean[frame];
    const int pre_start = max(0, frame - lookback_window);
    const int pre_end   = max(0, frame - 2);

    float pre_sum = 0.0f; int pre_count = 0;
    for (int t = pre_start; t < pre_end; t++) { pre_sum += rho_t_mean[t]; pre_count++; }

    if (pre_count == 0) { output_scores[eid] = 0.5f; return; }

    const float ratio = rho_at_event / (pre_sum / (float)pre_count + 1e-8f);
    output_scores[eid] = fminf(ratio, 1.0f);
}
"""

PERSISTENCE_KERNEL = r"""
extern "C" __global__
void persistence_kernel(
    const float* __restrict__ local_std_mean,
    const int*   __restrict__ event_frames,
    float*       __restrict__ output_scores,
    const int n_frames, const int n_events,
    const int persistence_window
) {
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_events) return;

    const int frame = event_frames[eid];
    if (frame < 0 || frame >= n_frames) { output_scores[eid] = 0.0f; return; }

    const int pre_start = max(0, frame - persistence_window);
    float pre_sum = 0.0f; int pre_count = 0;
    for (int t = pre_start; t < frame; t++) { pre_sum += local_std_mean[t]; pre_count++; }

    const int post_end = min(n_frames, frame + persistence_window + 1);
    float post_sum = 0.0f; int post_count = 0;
    for (int t = frame + 1; t < post_end; t++) { post_sum += local_std_mean[t]; post_count++; }

    if (pre_count == 0 || post_count == 0) { output_scores[eid] = 0.5f; return; }

    const float pre_mean  = pre_sum  / (float)pre_count;
    const float post_mean = post_sum / (float)post_count;
    const float change = fabsf(post_mean - pre_mean) / (pre_mean + 1e-8f);
    output_scores[eid] = fminf(change, 1.0f);
}
"""

COORDINATION_KERNEL = r"""
extern "C" __global__
void coordination_kernel(
    const float* __restrict__ dlc_per_dim,
    const int*   __restrict__ event_frames,
    float*       __restrict__ output_scores,
    const int n_diff, const int n_dims, const int n_events,
    const int coord_window
) {
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_events) return;

    const int frame = event_frames[eid];
    if (frame < 0 || frame >= n_diff) { output_scores[eid] = 0.0f; return; }

    int n_active_dims = 0;
    for (int d = 0; d < n_dims; d++) {
        const int w_start = max(0, frame - coord_window);
        const int w_end   = min(n_diff, frame + coord_window + 1);
        for (int t = w_start; t < w_end; t++) {
            if (dlc_per_dim[t * n_dims + d] > 0.0f) { n_active_dims++; break; }
        }
    }

    output_scores[eid] = (float)n_active_dims / (float)n_dims;
}
"""

GENUINENESS_KERNEL = r"""
extern "C" __global__
void genuineness_kernel(
    const float* __restrict__ surrogate_scores,
    const float* __restrict__ rho_t_scores,
    const float* __restrict__ persistence_scores,
    const float* __restrict__ coordination_scores,
    float*       __restrict__ output_genuineness,
    const int n_events,
    const float w_surrogate, const float w_rho_t,
    const float w_persistence, const float w_coordination
) {
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_events) return;

    output_genuineness[eid] =
          w_surrogate    * surrogate_scores[eid]
        + w_rho_t        * rho_t_scores[eid]
        + w_persistence  * persistence_scores[eid]
        + w_coordination * coordination_scores[eid];
}
"""


# =====================================================================
# Kernel Manager (両系統統合)
# =====================================================================


class InverseKernels:
    _instance: InverseKernels | None = None

    def __new__(cls) -> InverseKernels:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._compiled = False
        return cls._instance

    def __init__(self) -> None:
        if not self._compiled:
            self._compile_kernels()
            self._compiled = True

    def _compile_kernels(self) -> None:
        logger.info("🔧 Compiling CUDA kernels (Detection 3-axis + Network 4-axis)...")
        self._kernels = {}
        try:
            # Detection 3軸
            self._kernels["reconstruction"] = cp.RawKernel(
                INVERSE_RECONSTRUCTION_KERNEL, "inverse_reconstruction_kernel")
            self._kernels["topo_preservation"] = cp.RawKernel(
                TOPO_PRESERVATION_KERNEL, "topo_preservation_kernel")
            self._kernels["jump_integrity"] = cp.RawKernel(
                JUMP_INTEGRITY_KERNEL, "jump_integrity_kernel")
            self._kernels["hybrid_score"] = cp.RawKernel(
                HYBRID_SCORE_KERNEL, "hybrid_score_kernel")

            # Network 4軸
            self._kernels["surrogate"] = cp.RawKernel(
                SURROGATE_KERNEL, "surrogate_kernel")
            self._kernels["rho_t_consistency"] = cp.RawKernel(
                RHO_T_CONSISTENCY_KERNEL, "rho_t_consistency_kernel")
            self._kernels["persistence"] = cp.RawKernel(
                PERSISTENCE_KERNEL, "persistence_kernel")
            self._kernels["coordination"] = cp.RawKernel(
                COORDINATION_KERNEL, "coordination_kernel")
            self._kernels["genuineness"] = cp.RawKernel(
                GENUINENESS_KERNEL, "genuineness_kernel")

            logger.info("✅ All kernels compiled (Detection 4 + Network 5 = 9 total)")
        except Exception:
            logger.exception("❌ Kernel compilation failed")
            raise

    def get_kernel(self, name: str) -> cp.RawKernel:
        if name not in self._kernels:
            raise KeyError(f"Unknown kernel: {name}")
        return self._kernels[name]


# =====================================================================
# Helpers — Detection
# =====================================================================


def compute_gram_matrix(events: np.ndarray) -> cp.ndarray:
    events_gpu = cp.asarray(events, dtype=cp.float32)
    gram = events_gpu @ events_gpu.T
    gram_norm = cp.sqrt(cp.trace(gram @ gram))
    if gram_norm > 1e-8:
        gram = gram / gram_norm
    return gram


def compute_path_statistics(lambda_matrix: np.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    lm = np.asarray(lambda_matrix, dtype=np.float32)
    deltas = np.abs(np.diff(lm, axis=1))
    means = np.mean(deltas, axis=1).astype(np.float32)
    stds = np.std(deltas, axis=1).astype(np.float32)
    return cp.asarray(means), cp.asarray(stds)


# =====================================================================
# Helpers — Network
# =====================================================================


def _compute_dlc_scores(local_lambda_f: np.ndarray) -> cp.ndarray:
    scores = np.sum(np.abs(local_lambda_f), axis=1).astype(np.float32)
    return cp.asarray(scores)


def _compute_rho_t_mean(local_rho_t: np.ndarray) -> cp.ndarray:
    return cp.asarray(np.mean(local_rho_t, axis=1).astype(np.float32))


def _compute_local_std_mean(local_std: np.ndarray) -> cp.ndarray:
    return cp.asarray(np.mean(local_std, axis=1).astype(np.float32))


def _compute_dlc_per_dim(local_lambda_f: np.ndarray) -> cp.ndarray:
    return cp.asarray((np.abs(local_lambda_f) > 0).astype(np.float32))


# =====================================================================
# Entry Point 1: Detection — 逆問題 3軸
# =====================================================================


def inverse_verify_all(
    lambda_matrix: np.ndarray,
    events: np.ndarray,
    *,
    w_recon: float = 0.4,
    w_topo: float = 0.3,
    w_jump: float = 0.3,
    jump_sigma: float = 2.5,
    block_size: int = 256,
) -> dict[str, np.ndarray]:
    """Detection逆問題: 3軸GPU並列検証"""
    lm = np.asarray(lambda_matrix, dtype=np.float32)
    ev = np.asarray(events, dtype=np.float32)
    n_paths, n_events = lm.shape

    logger.info(f"🔺 Inverse verification: {n_events} events × {n_paths} paths (GPU)")

    kernels = InverseKernels()
    grid_size = (n_events + block_size - 1) // block_size

    lm_gpu = cp.asarray(lm)
    gram_gpu = compute_gram_matrix(ev)
    path_means, path_stds = compute_path_statistics(lm)

    recon_errors = cp.zeros(n_events, dtype=cp.float32)
    topo_scores = cp.zeros(n_events, dtype=cp.float32)
    jump_scores = cp.zeros(n_events, dtype=cp.float32)

    kernels.get_kernel("reconstruction")(
        (grid_size,), (block_size,),
        (lm_gpu, gram_gpu, recon_errors, n_paths, n_events))

    kernels.get_kernel("topo_preservation")(
        (grid_size,), (block_size,),
        (lm_gpu, topo_scores, n_paths, n_events))

    kernels.get_kernel("jump_integrity")(
        (grid_size,), (block_size,),
        (lm_gpu, path_means, path_stds, jump_scores,
         n_paths, n_events, np.float32(jump_sigma)))

    hybrid_scores = cp.zeros(n_events, dtype=cp.float32)
    verdicts = cp.zeros(n_events, dtype=cp.float32)

    r_mean, r_std = float(cp.mean(recon_errors)), float(cp.std(recon_errors))
    t_mean, t_std = float(cp.mean(topo_scores)), float(cp.std(topo_scores))
    j_mean, j_std = float(cp.mean(jump_scores)), float(cp.std(jump_scores))

    kernels.get_kernel("hybrid_score")(
        (grid_size,), (block_size,),
        (recon_errors, topo_scores, jump_scores,
         hybrid_scores, verdicts, n_events,
         np.float32(w_recon), np.float32(w_topo), np.float32(w_jump),
         np.float32(r_mean), np.float32(r_std),
         np.float32(t_mean), np.float32(t_std),
         np.float32(j_mean), np.float32(j_std)))

    results = {
        "reconstruction_errors": cp.asnumpy(recon_errors),
        "topo_scores": cp.asnumpy(topo_scores),
        "jump_scores": cp.asnumpy(jump_scores),
        "hybrid_scores": cp.asnumpy(hybrid_scores),
        "verdicts": cp.asnumpy(verdicts),
    }

    n_structural = int(np.sum(results["verdicts"] > 0.5))
    logger.info(f"✅ Inverse complete: {n_structural} genuine / {n_events - n_structural} spurious")
    return results


# =====================================================================
# Entry Point 2: Network — 統計テスト 4軸
# =====================================================================


def network_verify_all(
    event_frames: np.ndarray,
    state_vectors: np.ndarray,
    local_lambda_f: np.ndarray,
    local_rho_t: np.ndarray,
    local_std: np.ndarray | None,
    *,
    w_surrogate: float = 0.3,
    w_rho_t: float = 0.25,
    w_persistence: float = 0.25,
    w_coordination: float = 0.2,
    surrogate_window: int = 100,
    persistence_window: int = 20,
    coordination_window: int = 3,
    block_size: int = 256,
) -> dict[str, np.ndarray]:
    """Network検証: 4軸GPU並列検証"""
    n_events = len(event_frames)
    n_frames, n_dims = state_vectors.shape
    n_diff = local_lambda_f.shape[0]

    logger.info(f"🔺 Network verification: {n_events} events, {n_dims} dims (GPU 4-axis)")

    kernels = InverseKernels()
    grid_size = (n_events + block_size - 1) // block_size

    ef_gpu = cp.asarray(event_frames.astype(np.int32))
    dlc_scores_gpu = _compute_dlc_scores(local_lambda_f)
    rho_t_mean_gpu = _compute_rho_t_mean(local_rho_t)
    dlc_per_dim_gpu = _compute_dlc_per_dim(local_lambda_f)
    local_std_mean_gpu = (
        _compute_local_std_mean(local_std) if local_std is not None
        else rho_t_mean_gpu
    )

    surrogate_out = cp.zeros(n_events, dtype=cp.float32)
    rho_t_out = cp.zeros(n_events, dtype=cp.float32)
    persistence_out = cp.zeros(n_events, dtype=cp.float32)
    coordination_out = cp.zeros(n_events, dtype=cp.float32)
    genuineness_out = cp.zeros(n_events, dtype=cp.float32)

    kernels.get_kernel("surrogate")(
        (grid_size,), (block_size,),
        (dlc_scores_gpu, ef_gpu, surrogate_out,
         np.int32(n_diff), np.int32(n_events), np.int32(surrogate_window)))

    lookback = max(persistence_window, 10)
    kernels.get_kernel("rho_t_consistency")(
        (grid_size,), (block_size,),
        (rho_t_mean_gpu, ef_gpu, rho_t_out,
         np.int32(n_frames), np.int32(n_events), np.int32(lookback)))

    kernels.get_kernel("persistence")(
        (grid_size,), (block_size,),
        (local_std_mean_gpu, ef_gpu, persistence_out,
         np.int32(n_frames), np.int32(n_events), np.int32(persistence_window)))

    kernels.get_kernel("coordination")(
        (grid_size,), (block_size,),
        (dlc_per_dim_gpu, ef_gpu, coordination_out,
         np.int32(n_diff), np.int32(n_dims),
         np.int32(n_events), np.int32(coordination_window)))

    kernels.get_kernel("genuineness")(
        (grid_size,), (block_size,),
        (surrogate_out, rho_t_out, persistence_out, coordination_out,
         genuineness_out, np.int32(n_events),
         np.float32(w_surrogate), np.float32(w_rho_t),
         np.float32(w_persistence), np.float32(w_coordination)))

    results = {
        "surrogate_scores": cp.asnumpy(surrogate_out),
        "rho_t_scores": cp.asnumpy(rho_t_out),
        "persistence_scores": cp.asnumpy(persistence_out),
        "coordination_scores": cp.asnumpy(coordination_out),
        "genuineness": cp.asnumpy(genuineness_out),
    }

    logger.info(f"✅ Network verification complete (mean genuineness={np.mean(results['genuineness']):.3f})")
    return results
