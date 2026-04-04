"""
CUDA Kernels for Inverse Problem Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Λ³構造の逆問題による構造的検証カーネル 🔥
検出されたイベントが「構造的に説明可能か」を
GPU並列で一気に検証するよ！

⚠️ GPU ONLY — CPUフォールバック無し
pip install getter-one[gpu] で使ってね！

by 環ちゃん
"""
from __future__ import annotations

import logging

import numpy as np

from ..models import NDArray

# ===============================
# GPU ONLY — No Fallback
# ===============================

try:
    import cupy as cp
    from cupy import cuda

    HAS_GPU = True
except ImportError as _err:
    raise ImportError(
        "InverseChecker requires CUDA.\n"
        "Install CuPy: pip install cupy-cuda12x\n"
        "Or: pip install getter-one[gpu]"
    ) from _err

logger = logging.getLogger("getter_one.core.gpu_inverse")


# ===============================
# CUDA Kernel Templates
# ===============================

# 逆問題 再構成誤差カーネル
# 1スレッド = 1イベントの再構成誤差を計算
INVERSE_RECONSTRUCTION_KERNEL = r"""
extern "C" __global__
void inverse_reconstruction_kernel(
    const float* __restrict__ lambda_matrix,   // (n_paths, n_events) row-major
    const float* __restrict__ events_gram,     // (n_events, n_events) 正規化済み
    float* __restrict__ output_errors,         // (n_events,)
    const int n_paths,
    const int n_events
) {
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_events) return;

    // このイベントの再構成Gram行の計算
    // recon_gram[eid][j] = Σ_k lambda[k][eid] * lambda[k][j]
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

# QΛ トポロジカル保存カーネル
# 1スレッド = 1イベントのトポロジー整合性を計算
TOPO_PRESERVATION_KERNEL = r"""
extern "C" __global__
void topo_preservation_kernel(
    const float* __restrict__ lambda_matrix,   // (n_paths, n_events)
    float* __restrict__ output_topo_scores,    // (n_events,)
    const int n_paths,
    const int n_events
) {
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_events) return;

    // 位相差の連続性をチェック
    // イベント eid の前後での位相不整合を計算
    if (eid == 0 || eid == n_events - 1) {
        // 端点は位相評価不可 → 中立スコア
        output_topo_scores[eid] = 0.5f;
        return;
    }

    float topo_score = 0.0f;
    int valid_paths = 0;

    for (int k = 0; k < n_paths; k++) {
        float v_prev = lambda_matrix[k * n_events + (eid - 1)];
        float v_curr = lambda_matrix[k * n_events + eid];
        float v_next = lambda_matrix[k * n_events + (eid + 1)];

        // 位相角の計算
        float phase_before = atan2f(v_curr, v_prev);
        float phase_after  = atan2f(v_next, v_curr);
        float phase_diff   = phase_after - phase_before;

        // 2π正規化
        if (phase_diff > 3.14159265f) phase_diff -= 6.28318530f;
        if (phase_diff < -3.14159265f) phase_diff += 6.28318530f;

        topo_score += phase_diff * phase_diff;
        valid_paths++;
    }

    // 正規化: 低い = 整合的（良い）、高い = 不整合（怪しい）
    if (valid_paths > 0) {
        output_topo_scores[eid] = topo_score / (float)valid_paths;
    } else {
        output_topo_scores[eid] = 0.5f;
    }
}
"""

# ジャンプ整合性カーネル
# 1スレッド = 1イベントのジャンプが構造的に正当かを計算
JUMP_INTEGRITY_KERNEL = r"""
extern "C" __global__
void jump_integrity_kernel(
    const float* __restrict__ lambda_matrix,   // (n_paths, n_events)
    const float* __restrict__ path_means,      // (n_paths,) 各pathのdelta平均
    const float* __restrict__ path_stds,       // (n_paths,) 各pathのdelta標準偏差
    float* __restrict__ output_jump_scores,    // (n_events,)
    const int n_paths,
    const int n_events,
    const float jump_sigma                     // ジャンプ判定倍率（default 2.5）
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
            // ジャンプ検出: 正規化されたジャンプ強度
            jump_score += (delta - threshold) / (path_stds[k] + 1e-8f);
            jump_count++;
        }
    }

    // 高い = 多くのpathで大きなジャンプ = 構造的に有意なイベント
    output_jump_scores[eid] = (jump_count > 0)
        ? jump_score / (float)n_paths
        : 0.0f;
}
"""

# ハイブリッド統合カーネル
# 3つのスコアを重み付け統合
HYBRID_SCORE_KERNEL = r"""
extern "C" __global__
void hybrid_score_kernel(
    const float* __restrict__ recon_errors,     // (n_events,)
    const float* __restrict__ topo_scores,      // (n_events,)
    const float* __restrict__ jump_scores,      // (n_events,)
    float* __restrict__ output_hybrid,          // (n_events,)
    float* __restrict__ output_verdict,         // (n_events,) 0.0=noise, 1.0=structural
    const int n_events,
    const float w_recon,                        // 再構成誤差の重み
    const float w_topo,                         // トポロジーの重み
    const float w_jump,                         // ジャンプの重み
    const float recon_mean,                     // 正規化用
    const float recon_std,
    const float topo_mean,
    const float topo_std,
    const float jump_mean,
    const float jump_std
) {
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_events) return;

    // Z-score正規化
    float z_recon = (recon_std > 1e-8f)
        ? (recon_errors[eid] - recon_mean) / recon_std : 0.0f;
    float z_topo  = (topo_std > 1e-8f)
        ? (topo_scores[eid] - topo_mean) / topo_std : 0.0f;
    float z_jump  = (jump_std > 1e-8f)
        ? (jump_scores[eid] - jump_mean) / jump_std : 0.0f;

    // ハイブリッドスコア
    // recon: 低い = 構造的に説明可能 → 反転して使う
    // topo:  低い = 位相整合的 → 反転して使う
    // jump:  高い = 構造的に有意なジャンプ → そのまま使う
    float hybrid = -w_recon * z_recon - w_topo * z_topo + w_jump * z_jump;
    output_hybrid[eid] = hybrid;

    // 判定: hybrid > 0 なら構造的イベント（仮）
    // jump が高く、recon が低く、topo が低い = 本物
    output_verdict[eid] = (hybrid > 0.0f) ? 1.0f : 0.0f;
}
"""


# ===============================
# Kernel Manager
# ===============================

class InverseKernels:
    """逆問題CUDA カーネル管理クラス"""

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
        """全カーネルをコンパイル"""
        logger.info("🔧 Compiling inverse problem CUDA kernels...")
        self._kernels = {}
        try:
            self._kernels["reconstruction"] = cp.RawKernel(
                INVERSE_RECONSTRUCTION_KERNEL, "inverse_reconstruction_kernel"
            )
            self._kernels["topo_preservation"] = cp.RawKernel(
                TOPO_PRESERVATION_KERNEL, "topo_preservation_kernel"
            )
            self._kernels["jump_integrity"] = cp.RawKernel(
                JUMP_INTEGRITY_KERNEL, "jump_integrity_kernel"
            )
            self._kernels["hybrid_score"] = cp.RawKernel(
                HYBRID_SCORE_KERNEL, "hybrid_score_kernel"
            )
            logger.info("✅ All inverse kernels compiled successfully")
        except Exception:
            logger.exception("❌ Kernel compilation failed")
            raise

    def get_kernel(self, name: str) -> cp.RawKernel:
        """コンパイル済みカーネルを取得"""
        if name not in self._kernels:
            raise KeyError(f"Unknown kernel: {name}")
        return self._kernels[name]


# ===============================
# Wrapper Functions
# ===============================

def compute_gram_matrix(events: np.ndarray) -> cp.ndarray:
    """
    観測データのGram行列を計算（GPU上）

    Parameters
    ----------
    events : np.ndarray
        (n_events, n_features) イベント特徴量

    Returns
    -------
    cp.ndarray
        (n_events, n_events) 正規化済みGram行列
    """
    events_gpu = cp.asarray(events, dtype=cp.float32)
    gram = events_gpu @ events_gpu.T

    # Frobenius正規化（スケール不変性）
    gram_norm = cp.sqrt(cp.trace(gram @ gram))
    if gram_norm > 1e-8:
        gram = gram / gram_norm

    return gram


def compute_path_statistics(
    lambda_matrix: np.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    各pathのdelta統計量を計算（ジャンプ検出用）

    Parameters
    ----------
    lambda_matrix : np.ndarray
        (n_paths, n_events)

    Returns
    -------
    tuple[cp.ndarray, cp.ndarray]
        (path_means, path_stds) 各 (n_paths,)
    """
    lm = np.asarray(lambda_matrix, dtype=np.float32)
    n_paths = lm.shape[0]

    deltas = np.abs(np.diff(lm, axis=1))
    means = np.mean(deltas, axis=1).astype(np.float32)
    stds = np.std(deltas, axis=1).astype(np.float32)

    return cp.asarray(means), cp.asarray(stds)


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
    """
    全イベントの逆問題検証を1回のGPUパスで実行

    3つの検証軸を同時計算:
    1. 再構成誤差: Λ³構造でデータを再構成できるか
    2. トポロジー保存: 位相構造の連続性が保たれているか
    3. ジャンプ整合性: 変化が構造的に有意か

    Parameters
    ----------
    lambda_matrix : np.ndarray
        (n_paths, n_events) Λ³パス行列
    events : np.ndarray
        (n_events, n_features) イベント特徴量
    w_recon : float
        再構成誤差の重み (default: 0.4)
    w_topo : float
        トポロジー保存の重み (default: 0.3)
    w_jump : float
        ジャンプ整合性の重み (default: 0.3)
    jump_sigma : float
        ジャンプ判定の閾値σ倍率 (default: 2.5)
    block_size : int
        CUDAブロックサイズ (default: 256)

    Returns
    -------
    dict[str, np.ndarray]
        - reconstruction_errors: (n_events,) 再構成誤差
        - topo_scores: (n_events,) トポロジー不整合度
        - jump_scores: (n_events,) ジャンプ強度
        - hybrid_scores: (n_events,) 統合スコア
        - verdicts: (n_events,) 判定 (1.0=structural, 0.0=noise)
    """
    lm = np.asarray(lambda_matrix, dtype=np.float32)
    ev = np.asarray(events, dtype=np.float32)
    n_paths, n_events = lm.shape

    logger.info(
        f"🔺 Inverse verification: {n_events} events × {n_paths} paths (GPU)"
    )

    # カーネル取得
    kernels = InverseKernels()
    grid_size = (n_events + block_size - 1) // block_size

    # GPU転送（1回のみ）
    lm_gpu = cp.asarray(lm)
    gram_gpu = compute_gram_matrix(ev)
    path_means, path_stds = compute_path_statistics(lm)

    # 出力バッファ確保
    recon_errors = cp.zeros(n_events, dtype=cp.float32)
    topo_scores = cp.zeros(n_events, dtype=cp.float32)
    jump_scores = cp.zeros(n_events, dtype=cp.float32)

    # === Kernel 1: 再構成誤差 ===
    kernels.get_kernel("reconstruction")(
        (grid_size,),
        (block_size,),
        (lm_gpu, gram_gpu, recon_errors, n_paths, n_events),
    )

    # === Kernel 2: トポロジー保存 ===
    kernels.get_kernel("topo_preservation")(
        (grid_size,),
        (block_size,),
        (lm_gpu, topo_scores, n_paths, n_events),
    )

    # === Kernel 3: ジャンプ整合性 ===
    kernels.get_kernel("jump_integrity")(
        (grid_size,),
        (block_size,),
        (
            lm_gpu,
            path_means,
            path_stds,
            jump_scores,
            n_paths,
            n_events,
            np.float32(jump_sigma),
        ),
    )

    # === Kernel 4: ハイブリッド統合 ===
    hybrid_scores = cp.zeros(n_events, dtype=cp.float32)
    verdicts = cp.zeros(n_events, dtype=cp.float32)

    # 正規化パラメータ計算
    r_mean = float(cp.mean(recon_errors))
    r_std = float(cp.std(recon_errors))
    t_mean = float(cp.mean(topo_scores))
    t_std = float(cp.std(topo_scores))
    j_mean = float(cp.mean(jump_scores))
    j_std = float(cp.std(jump_scores))

    kernels.get_kernel("hybrid_score")(
        (grid_size,),
        (block_size,),
        (
            recon_errors,
            topo_scores,
            jump_scores,
            hybrid_scores,
            verdicts,
            n_events,
            np.float32(w_recon),
            np.float32(w_topo),
            np.float32(w_jump),
            np.float32(r_mean),
            np.float32(r_std),
            np.float32(t_mean),
            np.float32(t_std),
            np.float32(j_mean),
            np.float32(j_std),
        ),
    )

    # GPU → CPU 一括転送
    results = {
        "reconstruction_errors": cp.asnumpy(recon_errors),
        "topo_scores": cp.asnumpy(topo_scores),
        "jump_scores": cp.asnumpy(jump_scores),
        "hybrid_scores": cp.asnumpy(hybrid_scores),
        "verdicts": cp.asnumpy(verdicts),
    }

    n_structural = int(np.sum(results["verdicts"] > 0.5))
    n_noise = n_events - n_structural
    logger.info(
        f"✅ Verification complete: "
        f"{n_structural} structural / {n_noise} noise"
    )

    return results
