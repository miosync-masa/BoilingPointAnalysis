"""
CUDA Kernels for Lambda³ GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

高速なCUDAカーネル集だよ〜！💕
めちゃくちゃ速い処理を実現しちゃう！

by 環ちゃん
"""
from __future__ import annotations

import logging

from ..models import NDArray

# GPU imports
try:
    import cupy as cp
    from cupy import cuda

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None
    cuda = None

logger = logging.getLogger("getter_one.core.kernels")

# ===============================
# Kernel Code Templates
# ===============================

# 残基COM計算カーネル
RESIDUE_COM_KERNEL = r"""
extern "C" __global__
void compute_residue_com_kernel(
    const float* __restrict__ trajectory,      // (n_frames, n_atoms, 3)
    const int* __restrict__ atom_indices,      // 各残基の原子インデックス
    const int* __restrict__ residue_starts,    // 各残基の開始インデックス
    const int* __restrict__ residue_sizes,     // 各残基の原子数
    float* __restrict__ com_output,           // (n_frames, n_residues, 3)
    const int n_frames,
    const int n_atoms,
    const int n_residues
) {
    // グリッド・ストライド・ループ
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int total_work = n_frames * n_residues;
    
    for (int i = idx; i < total_work; i += stride) {
        const int frame = i / n_residues;
        const int res_id = i % n_residues;
        
        // 残基の原子範囲
        const int start = residue_starts[res_id];
        const int size = residue_sizes[res_id];
        
        // COM計算（共有メモリ使用で高速化）
        float com_x = 0.0f, com_y = 0.0f, com_z = 0.0f;
        
        for (int j = 0; j < size; j++) {
            const int atom_idx = atom_indices[start + j];
            const int traj_idx = (frame * n_atoms + atom_idx) * 3;
            
            com_x += trajectory[traj_idx + 0];
            com_y += trajectory[traj_idx + 1];
            com_z += trajectory[traj_idx + 2];
        }
        
        // 平均を計算して出力
        const float inv_size = 1.0f / size;
        const int out_idx = (frame * n_residues + res_id) * 3;
        com_output[out_idx + 0] = com_x * inv_size;
        com_output[out_idx + 1] = com_y * inv_size;
        com_output[out_idx + 2] = com_z * inv_size;
    }
}
"""

# テンション場計算カーネル（高速版）
TENSION_FIELD_KERNEL = r"""
extern "C" __global__
void compute_tension_field_kernel(
    const float* __restrict__ positions,       // (n_frames, 3)
    float* __restrict__ rho_T,                // (n_frames,)
    const int n_frames,
    const int window_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_frames) return;
    
    const int start = max(0, idx - window_size);
    const int end = min(n_frames, idx + window_size + 1);
    const int local_size = end - start;
    const float inv_size = 1.0f / local_size;
    
    // 局所平均計算
    float mean_x = 0.0f, mean_y = 0.0f, mean_z = 0.0f;
    
    for (int i = start; i < end; i++) {
        const int pos_idx = i * 3;
        mean_x += positions[pos_idx + 0];
        mean_y += positions[pos_idx + 1];
        mean_z += positions[pos_idx + 2];
    }
    
    mean_x *= inv_size;
    mean_y *= inv_size;
    mean_z *= inv_size;
    
    // 共分散行列のトレース計算
    float cov_xx = 0.0f, cov_yy = 0.0f, cov_zz = 0.0f;
    
    for (int i = start; i < end; i++) {
        const int pos_idx = i * 3;
        const float dx = positions[pos_idx + 0] - mean_x;
        const float dy = positions[pos_idx + 1] - mean_y;
        const float dz = positions[pos_idx + 2] - mean_z;
        
        cov_xx += dx * dx;
        cov_yy += dy * dy;
        cov_zz += dz * dz;
    }
    
    rho_T[idx] = (local_size > 1)
        ? (cov_xx + cov_yy + cov_zz) / (float)(local_size - 1)
        : 0.0f;
}
"""

# 異常検出カーネル（適応的z-score）
ANOMALY_DETECTION_KERNEL = r"""
extern "C" __global__
void detect_anomalies_kernel(
    const float* __restrict__ series,
    float* __restrict__ anomaly_scores,
    const int n_points,
    const int window_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_points) return;
    
    const int start = max(0, idx - window_size);
    const int end = min(n_points, idx + window_size + 1);
    const int local_size = end - start;
    const float inv_size = 1.0f / local_size;
    
    // 局所統計量計算（Welford's algorithm使用）
    float local_mean = 0.0f;
    float local_m2 = 0.0f;
    
    for (int i = start; i < end; i++) {
        const float delta = series[i] - local_mean;
        local_mean += delta * inv_size;
        local_m2 += delta * (series[i] - local_mean);
    }
    
    const float local_var = local_m2 * inv_size;
    const float local_std = sqrtf(local_var);
    
    // Z-score計算
    if (local_std > 1e-10f) {
        anomaly_scores[idx] = fabsf(series[idx] - local_mean) / local_std;
    } else {
        anomaly_scores[idx] = 0.0f;
    }
}
"""

# 距離行列計算カーネル（最適化版）
DISTANCE_MATRIX_KERNEL = r"""
extern "C" __global__
void compute_distance_matrix_kernel(
    const float* __restrict__ positions,      // (n_points, 3)
    float* __restrict__ distances,           // (n_points, n_points)
    const int n_points
) {
    // 共有メモリを使用して高速化
    extern __shared__ float shared_pos[];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int block_size = blockDim.x;
    
    const int row = by * block_size + ty;
    const int col = bx * block_size + tx;
    
    // タイルごとに処理
    float sum = 0.0f;
    
    for (int tile = 0; tile < (n_points + block_size - 1) / block_size; tile++) {
        // 共有メモリにロード
        const int idx_a = row;
        const int idx_b = tile * block_size + tx;
        
        if (idx_a < n_points && idx_b < n_points) {
            shared_pos[ty * block_size * 3 + tx * 3 + 0] = positions[idx_b * 3 + 0];
            shared_pos[ty * block_size * 3 + tx * 3 + 1] = positions[idx_b * 3 + 1];
            shared_pos[ty * block_size * 3 + tx * 3 + 2] = positions[idx_b * 3 + 2];
        }
        
        __syncthreads();
        
        // 距離計算
        if (row < n_points && col < n_points) {
            const int shared_idx = ty * block_size * 3 + tx * 3;
            const float dx = positions[row * 3 + 0] - shared_pos[shared_idx + 0];
            const float dy = positions[row * 3 + 1] - shared_pos[shared_idx + 1];
            const float dz = positions[row * 3 + 2] - shared_pos[shared_idx + 2];
            sum = dx * dx + dy * dy + dz * dz;
        }
        
        __syncthreads();
    }
    
    // 結果を書き込み
    if (row < n_points && col < n_points) {
        distances[row * n_points + col] = sqrtf(sum);
    }
}
"""

# トポロジカルチャージ計算カーネル（double精度内部演算版）
TOPOLOGICAL_CHARGE_KERNEL = r"""
extern "C" __global__
void compute_topological_charge_kernel(
    const float* __restrict__ lambda_F,        // (n_steps, 3)
    const float* __restrict__ lambda_F_mag,    // (n_steps,)
    float* __restrict__ Q_lambda,             // (n_steps,)
    const int n_steps
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_steps - 1 || idx == 0) {
        if (idx < n_steps) Q_lambda[idx] = 0.0f;
        return;
    }
    
    const float mag_curr = lambda_F_mag[idx];
    const float mag_prev = lambda_F_mag[idx - 1];
    
    if (mag_curr > 1e-10f && mag_prev > 1e-10f) {
        // double精度で内部演算（cross_z ≈ 0 での符号反転を防止）
        const double inv_mag_prev = 1.0 / (double)mag_prev;
        const double inv_mag_curr = 1.0 / (double)mag_curr;
        
        const double v1_x = (double)lambda_F[(idx - 1) * 3 + 0] * inv_mag_prev;
        const double v1_y = (double)lambda_F[(idx - 1) * 3 + 1] * inv_mag_prev;
        const double v1_z = (double)lambda_F[(idx - 1) * 3 + 2] * inv_mag_prev;
        
        const double v2_x = (double)lambda_F[idx * 3 + 0] * inv_mag_curr;
        const double v2_y = (double)lambda_F[idx * 3 + 1] * inv_mag_curr;
        const double v2_z = (double)lambda_F[idx * 3 + 2] * inv_mag_curr;
        
        // 内積（double精度）
        const double cos_angle = v1_x * v2_x + v1_y * v2_y + v1_z * v2_z;
        const double clamped_cos = fmax(-1.0, fmin(1.0, cos_angle));
        const double angle = acos(clamped_cos);
        
        // 2D回転方向（double精度で符号判定）
        const double cross_z = v1_x * v2_y - v1_y * v2_x;
        const double signed_angle = (cross_z >= 0.0) ? angle : -angle;
        
        Q_lambda[idx] = (float)(signed_angle / (2.0 * 3.14159265358979323846));
    } else {
        Q_lambda[idx] = 0.0f;
    }
}
"""

# フラクタル次元計算カーネル（新規追加）
FRACTAL_DIMENSION_KERNEL = r"""
extern "C" __global__
void compute_local_fractal_dimension_kernel(
    const float* __restrict__ q_cumulative,    // (n_points,)
    float* __restrict__ dimensions,            // (n_points,)
    const int window_size,
    const int n_points
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_points) return;
    
    const int start = max(0, idx - window_size);
    const int end = min(n_points, idx + window_size + 1);
    const int local_size = end - start;
    
    if (local_size < 3) {
        dimensions[idx] = 1.0f;
        return;
    }
    
    // 簡易ボックスカウント法
    float min_val = 1e30f;
    float max_val = -1e30f;
    
    // 局所範囲の最小・最大値
    for (int i = start; i < end; i++) {
        min_val = fminf(min_val, q_cumulative[i]);
        max_val = fmaxf(max_val, q_cumulative[i]);
    }
    
    float range = max_val - min_val;
    if (range < 1e-10f) {
        dimensions[idx] = 1.0f;
        return;
    }
    
    // 複数のボックスサイズでカウント
    float log_n_sum = 0.0f;
    float log_eps_sum = 0.0f;
    int n_scales = 0;
    
    for (int scale = 2; scale <= 10 && scale < local_size; scale++) {
        float eps = range / scale;
        int box_count = 0;
        
        // ボックスカウント
        for (int i = start; i < end; i++) {
            int box_id = (int)((q_cumulative[i] - min_val) / eps);
            // 簡易的な重複チェック（完全ではないが高速）
            if (i == start || box_id != (int)((q_cumulative[i-1] - min_val) / eps)) {
                box_count++;
            }
        }
        
        if (box_count > 0) {
            log_n_sum += logf((float)box_count);
            log_eps_sum += logf(eps);
            n_scales++;
        }
    }
    
    // 線形回帰で次元推定
    if (n_scales > 1) {
        dimensions[idx] = -log_n_sum / log_eps_sum;
        dimensions[idx] = fmaxf(1.0f, fminf(3.0f, dimensions[idx])); // クランプ
    } else {
        dimensions[idx] = 1.0f;
    }
}
"""

# 勾配計算カーネル（新規追加）
GRADIENT_KERNEL = r"""
extern "C" __global__
void compute_gradient_kernel(
    const float* __restrict__ input,           // (n_points,)
    float* __restrict__ gradient,              // (n_points,)
    const int n_points
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_points) return;
    
    if (idx == 0) {
        // 前方差分
        gradient[idx] = input[1] - input[0];
    } else if (idx == n_points - 1) {
        // 後方差分
        gradient[idx] = input[idx] - input[idx - 1];
    } else {
        // 中心差分
        gradient[idx] = (input[idx + 1] - input[idx - 1]) * 0.5f;
    }
}
"""

# ===============================
# Kernel Manager Class
# ===============================


class CUDAKernels:
    """
    CUDAカーネル管理クラス
    カーネルのコンパイルと実行を管理するよ〜！
    """

    def __init__(self):
        self.kernels = {}
        self.is_initialized = False

        if HAS_GPU and cp.cuda.is_available():
            self._compile_kernels()

    def _compile_kernels(self):
        """全カーネルをコンパイル"""
        try:
            # 残基COM計算
            self.kernels["residue_com"] = cp.RawKernel(
                RESIDUE_COM_KERNEL, "compute_residue_com_kernel"
            )

            # テンション場
            self.kernels["tension_field"] = cp.RawKernel(
                TENSION_FIELD_KERNEL, "compute_tension_field_kernel"
            )

            # 異常検出
            self.kernels["anomaly_detection"] = cp.RawKernel(
                ANOMALY_DETECTION_KERNEL, "detect_anomalies_kernel"
            )

            # 距離行列
            self.kernels["distance_matrix"] = cp.RawKernel(
                DISTANCE_MATRIX_KERNEL, "compute_distance_matrix_kernel"
            )

            # トポロジカルチャージ
            self.kernels["topological_charge"] = cp.RawKernel(
                TOPOLOGICAL_CHARGE_KERNEL, "compute_topological_charge_kernel"
            )

            # フラクタル次元（新規追加）
            self.kernels["fractal_dimension"] = cp.RawKernel(
                FRACTAL_DIMENSION_KERNEL, "compute_local_fractal_dimension_kernel"
            )

            # 勾配計算（新規追加）
            self.kernels["gradient"] = cp.RawKernel(
                GRADIENT_KERNEL, "compute_gradient_kernel"
            )

            self.is_initialized = True
            logger.info("CUDA kernels compiled successfully!")

        except Exception as e:
            logger.error(f"Failed to compile CUDA kernels: {e}")
            self.is_initialized = False

    def get_kernel(self, name: str):
        """カーネルを取得"""
        if not self.is_initialized:
            raise RuntimeError("CUDA kernels not initialized")

        if name not in self.kernels:
            raise KeyError(f"Kernel '{name}' not found")

        return self.kernels[name]


# ===============================
# Kernel Wrapper Functions
# ===============================


def residue_com_kernel(
    trajectory: NDArray, residue_mapping: dict, block_size: int = 256
) -> NDArray:
    """
    残基COM計算カーネルのラッパー

    Parameters
    ----------
    trajectory : cp.ndarray or np.ndarray
        トラジェクトリ (n_frames, n_atoms, 3)
    residue_mapping : dict
        残基ID -> 原子インデックスリスト
    block_size : int
        CUDAブロックサイズ

    Returns
    -------
    cp.ndarray
        残基COM (n_frames, n_residues, 3)
    """
    if not HAS_GPU:
        raise RuntimeError("GPU not available")

    # ⚡ GPU配列に変換（NumPy配列が来ても安全！）
    if not isinstance(trajectory, cp.ndarray):
        trajectory = cp.asarray(trajectory, dtype=cp.float32)

    n_frames, n_atoms, _ = trajectory.shape
    n_residues = len(residue_mapping)

    # 残基情報を準備
    all_indices = []
    residue_starts = []
    residue_sizes = []

    current_start = 0
    for res_id in sorted(residue_mapping.keys()):
        atoms = residue_mapping[res_id]
        all_indices.extend(atoms)
        residue_starts.append(current_start)
        residue_sizes.append(len(atoms))
        current_start += len(atoms)

    # GPU配列に変換
    atom_indices_gpu = cp.array(all_indices, dtype=cp.int32)
    residue_starts_gpu = cp.array(residue_starts, dtype=cp.int32)
    residue_sizes_gpu = cp.array(residue_sizes, dtype=cp.int32)

    # 出力配列
    com_output = cp.zeros((n_frames, n_residues, 3), dtype=cp.float32)

    # カーネル実行
    kernel_manager = CUDAKernels()
    kernel = kernel_manager.get_kernel("residue_com")

    grid_size = (n_frames * n_residues + block_size - 1) // block_size

    kernel(
        (grid_size,),
        (block_size,),
        (
            trajectory.ravel(),
            atom_indices_gpu,
            residue_starts_gpu,
            residue_sizes_gpu,
            com_output,
            n_frames,
            n_atoms,
            n_residues,
        ),
    )

    return com_output


def tension_field_kernel(
    positions: NDArray, window_size: int, block_size: int = 256
) -> NDArray:
    """
    テンション場計算カーネルのラッパー
    環ちゃんがミスってたところ、修正したよ！💦
    """
    if not HAS_GPU:
        raise RuntimeError("GPU not available")

    # ⚡ GPU配列に変換（これが抜けてたのがエラーの原因だった！）
    if not isinstance(positions, cp.ndarray):
        positions = cp.asarray(positions, dtype=cp.float32)

    n_frames = positions.shape[0]
    rho_T = cp.zeros(n_frames, dtype=cp.float32)

    kernel_manager = CUDAKernels()
    kernel = kernel_manager.get_kernel("tension_field")

    grid_size = (n_frames + block_size - 1) // block_size

    kernel(
        (grid_size,), (block_size,), (positions.ravel(), rho_T, n_frames, window_size)
    )

    return rho_T


def anomaly_detection_kernel(
    series: NDArray, window_size: int, block_size: int = 256
) -> NDArray:
    """
    異常検出カーネルのラッパー
    """
    if not HAS_GPU:
        raise RuntimeError("GPU not available")

    # ⚡ GPU配列に変換
    if not isinstance(series, cp.ndarray):
        series = cp.asarray(series, dtype=cp.float32)

    n_points = len(series)
    anomaly_scores = cp.zeros(n_points, dtype=cp.float32)

    kernel_manager = CUDAKernels()
    kernel = kernel_manager.get_kernel("anomaly_detection")

    grid_size = (n_points + block_size - 1) // block_size

    kernel((grid_size,), (block_size,), (series, anomaly_scores, n_points, window_size))

    return anomaly_scores


def distance_matrix_kernel(positions: NDArray, block_size: int = 16) -> NDArray:
    """
    距離行列計算カーネルのラッパー
    共有メモリ使用で高速化！
    """
    if not HAS_GPU:
        raise RuntimeError("GPU not available")

    # ⚡ GPU配列に変換
    if not isinstance(positions, cp.ndarray):
        positions = cp.asarray(positions, dtype=cp.float32)

    n_points = positions.shape[0]
    distances = cp.zeros((n_points, n_points), dtype=cp.float32)

    kernel_manager = CUDAKernels()
    kernel = kernel_manager.get_kernel("distance_matrix")

    grid_size = (
        (n_points + block_size - 1) // block_size,
        (n_points + block_size - 1) // block_size,
    )

    # 共有メモリサイズ
    shared_mem_size = block_size * block_size * 3 * 4  # float32

    kernel(
        grid_size,
        (block_size, block_size),
        (positions.ravel(), distances, n_points),
        shared_mem=shared_mem_size,
    )

    return distances


def topological_charge_kernel(
    lambda_F: NDArray, lambda_F_mag: NDArray, block_size: int = 256
) -> NDArray:
    """
    トポロジカルチャージ計算カーネルのラッパー
    ここは元々ちゃんとやってたよ！えらい環ちゃん！✨
    """
    if not HAS_GPU:
        raise RuntimeError("GPU not available")

    # ⚡ 入力をGPU配列に変換！（これは元々あった！）
    if not isinstance(lambda_F, cp.ndarray):
        lambda_F = cp.asarray(lambda_F, dtype=cp.float32)
    if not isinstance(lambda_F_mag, cp.ndarray):
        lambda_F_mag = cp.asarray(lambda_F_mag, dtype=cp.float32)

    n_steps = len(lambda_F_mag)
    Q_lambda = cp.zeros(n_steps, dtype=cp.float32)

    kernel_manager = CUDAKernels()
    kernel = kernel_manager.get_kernel("topological_charge")

    grid_size = (n_steps + block_size - 1) // block_size

    kernel(
        (grid_size,), (block_size,), (lambda_F.ravel(), lambda_F_mag, Q_lambda, n_steps)
    )

    return Q_lambda


def compute_local_fractal_dimension_kernel(
    q_cumulative: NDArray, window_size: int, block_size: int = 256
) -> NDArray:
    """
    局所フラクタル次元計算カーネルのラッパー（新規追加）
    """
    if not HAS_GPU:
        raise RuntimeError("GPU not available")

    # ⚡ GPU配列に変換
    if not isinstance(q_cumulative, cp.ndarray):
        q_cumulative = cp.asarray(q_cumulative, dtype=cp.float32)

    n_points = len(q_cumulative)
    dimensions = cp.ones(n_points, dtype=cp.float32)

    kernel_manager = get_kernel_manager()
    kernel = kernel_manager.get_kernel("fractal_dimension")

    grid_size = (n_points + block_size - 1) // block_size

    kernel(
        (grid_size,), (block_size,), (q_cumulative, dimensions, window_size, n_points)
    )

    return dimensions


def compute_gradient_kernel(input_array: NDArray, block_size: int = 256) -> NDArray:
    """
    勾配計算カーネルのラッパー（新規追加）
    """
    if not HAS_GPU:
        raise RuntimeError("GPU not available")

    # ⚡ GPU配列に変換
    if not isinstance(input_array, cp.ndarray):
        input_array = cp.asarray(input_array, dtype=cp.float32)

    n_points = len(input_array)
    gradient = cp.zeros(n_points, dtype=cp.float32)

    kernel_manager = get_kernel_manager()
    kernel = kernel_manager.get_kernel("gradient")

    grid_size = (n_points + block_size - 1) // block_size

    kernel((grid_size,), (block_size,), (input_array, gradient, n_points))

    return gradient


# ===============================
# Utility Kernels
# ===============================

# 要素ごとの演算カーネル（汎用）
ELEMENTWISE_KERNEL_TEMPLATE = """
extern "C" __global__
void {kernel_name}(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n_elements,
    const float param1,
    const float param2
) {{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {{
        {operation}
    }}
}}
"""


def create_elementwise_kernel(name: str, operation: str):
    """
    要素ごとの演算カーネルを動的に生成

    使用例:
        sigmoid_kernel = create_elementwise_kernel(
            'sigmoid',
            'output[idx] = 1.0f / (1.0f + expf(-input[idx]))'
        )
    """
    if not HAS_GPU:
        return None

    kernel_code = ELEMENTWISE_KERNEL_TEMPLATE.format(
        kernel_name=name, operation=operation
    )

    return cp.RawKernel(kernel_code, name)


# ===============================
# Performance Testing
# ===============================


def benchmark_kernels(
    n_frames: int = 10000, n_atoms: int = 1000, n_residues: int = 100
):
    """
    カーネルのベンチマーク
    """
    if not HAS_GPU:
        logger.warning("GPU not available for benchmarking")
        return

    import time

    logger.info(f"\n{'=' * 60}")
    logger.info("CUDA Kernel Benchmarks")
    logger.info(f"{'=' * 60}")
    logger.info(f"Frames: {n_frames}, Atoms: {n_atoms}, Residues: {n_residues}")

    # ダミーデータ
    trajectory = cp.random.randn(n_frames, n_atoms, 3, dtype=cp.float32)
    positions = cp.random.randn(n_frames, 3, dtype=cp.float32)
    residue_mapping = {
        i: list(range(i * (n_atoms // n_residues), (i + 1) * (n_atoms // n_residues)))
        for i in range(n_residues)
    }

    # 1. 残基COM計算
    start = time.time()
    residue_com_kernel(trajectory[:100], residue_mapping)  # 100フレームのみ
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start
    logger.info(f"\nResidue COM: {elapsed:.4f} seconds")
    logger.info(f"  Throughput: {100 * n_atoms / elapsed:.2f} atoms/sec")

    # 2. テンション場
    start = time.time()
    tension_field_kernel(positions, window_size=50)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start
    logger.info(f"\nTension Field: {elapsed:.4f} seconds")
    logger.info(f"  Throughput: {n_frames / elapsed:.2f} frames/sec")

    # 3. 異常検出
    series = cp.random.randn(n_frames, dtype=cp.float32)
    start = time.time()
    anomaly_detection_kernel(series, window_size=50)
    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start
    logger.info(f"\nAnomaly Detection: {elapsed:.4f} seconds")
    logger.info(f"  Throughput: {n_frames / elapsed:.2f} points/sec")

    logger.info(f"\n{'=' * 60}\n")


# グローバルカーネルマネージャー
_kernel_manager = None


def get_kernel_manager() -> CUDAKernels:
    """グローバルカーネルマネージャーを取得"""
    global _kernel_manager
    if _kernel_manager is None:
        _kernel_manager = CUDAKernels()
    return _kernel_manager


# ===============================
# Export Kernels
# ===============================

__all__ = [
    "CUDAKernels",
    "residue_com_kernel",
    "tension_field_kernel",
    "anomaly_detection_kernel",
    "distance_matrix_kernel",
    "topological_charge_kernel",
    "compute_local_fractal_dimension_kernel",  # 新規追加
    "compute_gradient_kernel",  # 新規追加
    "create_elementwise_kernel",
    "benchmark_kernels",
    "get_kernel_manager",
]
