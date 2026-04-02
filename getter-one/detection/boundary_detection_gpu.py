"""
Lambda³ GPU版構造境界検出モジュール
構造境界（ΔΛC）検出のGPU最適化実装
CuPy RawKernelベース（PTX 8.4対応）
完全修正版 - 全てのcp直接参照をself.xpに置換
"""

from typing import Any

import numpy as np

try:
    import logging

    import cupy as cp
    from scipy.signal import find_peaks  # cupyx.scipyではなくscipyを使用

    HAS_CUDA = True
except ImportError:
    cp = None
    find_peaks = None
    HAS_CUDA = False

from ..core.gpu_kernels import (
    compute_gradient_kernel,
    compute_local_fractal_dimension_kernel,
)
from ..core.gpu_utils import GPUBackend
from ..models import NDArray

logger = logging.getLogger(__name__)

# ===============================
# CuPy RawKernel定義
# ===============================

SHANNON_ENTROPY_KERNEL_CODE = r"""
extern "C" __global__
void shannon_entropy_kernel(
    const float* rho_t,
    float* entropy,
    const int window,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= window && idx < n - window) {
        int start = idx - window;
        int end = idx + window;
        
        // 正規化して確率分布を作成
        float local_sum = 0.0f;
        for (int i = start; i < end; i++) {
            local_sum += rho_t[i];
        }
        
        if (local_sum > 1e-10f) {
            // シャノンエントロピー計算
            float h = 0.0f;
            for (int i = start; i < end; i++) {
                if (rho_t[i] > 1e-10f) {
                    float p = rho_t[i] / local_sum;
                    h -= p * logf(p + 1e-10f);
                }
            }
            entropy[idx] = h;
        } else {
            entropy[idx] = 0.0f;
        }
    }
}
"""

DETECT_JUMPS_KERNEL_CODE = r"""
extern "C" __global__
void detect_jumps_kernel(
    const float* data,
    float* jumps,
    const float threshold,
    const int window,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx > 0 && idx < n - 1) {
        // 前後の差分
        float diff_prev = fabsf(data[idx] - data[idx-1]);
        float diff_next = fabsf(data[idx+1] - data[idx]);
        
        // ローカル平均
        float local_sum = 0.0f;
        int count = 0;
        int start = max(0, idx - window);
        int end = min(n, idx + window + 1);
        
        for (int i = start; i < end; i++) {
            local_sum += fabsf(data[i]);
            count++;
        }
        
        float local_mean = (count > 0) ? local_sum / count : 1.0f;
        
        // ジャンプ検出
        if (diff_prev > threshold * local_mean || 
            diff_next > threshold * local_mean) {
            jumps[idx] = fmaxf(diff_prev, diff_next) / (local_mean + 1e-10f);
        } else {
            jumps[idx] = 0.0f;
        }
    }
}
"""


class BoundaryDetectorGPU(GPUBackend):
    """構造境界検出のGPU実装（CuPy RawKernel版）"""

    def __init__(self, force_cpu=False):
        super().__init__(force_cpu)
        self.boundary_cache = {}

        # xpの初期設定
        self._setup_xp()

        # CuPy RawKernelをコンパイル
        if HAS_CUDA and not force_cpu:
            try:
                self.shannon_entropy_kernel = cp.RawKernel(
                    SHANNON_ENTROPY_KERNEL_CODE, "shannon_entropy_kernel"
                )
                self.detect_jumps_kernel = cp.RawKernel(
                    DETECT_JUMPS_KERNEL_CODE, "detect_jumps_kernel"
                )
                logger.info("✅ CuPy RawKernels compiled successfully (PTX 8.4)")
            except Exception as e:
                logger.warning(f"Failed to compile CuPy kernels: {e}")
                self.shannon_entropy_kernel = None
                self.detect_jumps_kernel = None
        else:
            self.shannon_entropy_kernel = None
            self.detect_jumps_kernel = None

    def _setup_xp(self):
        """numpy/cupyの切り替え設定"""
        if self.is_gpu and HAS_CUDA:
            import cupy as cp

            self.xp = cp
        else:
            import numpy as np

            self.xp = np

    def detect_structural_boundaries(
        self, structures: dict[str, np.ndarray], window_steps: int
    ) -> dict[str, Any]:
        """
        構造境界検出（ΔΛC - 意味の結晶化モーメント）

        Parameters
        ----------
        structures : Dict[str, np.ndarray]
            Lambda構造辞書
        window_steps : int
            ウィンドウサイズ

        Returns
        -------
        Dict[str, Any]
            境界情報
        """
        print("\n🔍 Detecting structural boundaries (ΔΛC) on GPU...")

        # xpの確認
        if not hasattr(self, "xp") or self.xp is None:
            self._setup_xp()

        n_steps = len(structures["rho_T"])

        # GPUメモリコンテキスト
        with self.memory_manager.temporary_allocation(n_steps * 4 * 8, "boundaries"):
            # 各指標をGPUで計算
            fractal_dims = self._compute_fractal_dimensions_gpu(
                structures.get("Q_cumulative", np.zeros(n_steps)), window_steps
            )

            coherence = self._get_coherence_gpu(structures)

            coupling = self._compute_coupling_strength_gpu(
                structures.get("Q_cumulative", np.zeros(n_steps)), window_steps
            )

            entropy = self._compute_structural_entropy_gpu(
                structures["rho_T"], window_steps
            )

            # 境界スコア計算
            boundary_score = self._compute_boundary_score_gpu(
                fractal_dims, coherence, coupling, entropy
            )

            # ピーク検出
            peaks, properties = self._detect_peaks_gpu(boundary_score, n_steps)

        # CPU形式で返す
        return {
            "boundary_score": self.to_cpu(boundary_score),
            "boundary_locations": self.to_cpu(peaks),
            "boundary_strengths": self.to_cpu(boundary_score[peaks])
            if len(peaks) > 0
            else np.array([]),
            "fractal_dimension": self.to_cpu(fractal_dims),
            "structural_coherence": self.to_cpu(coherence),
            "coupling_strength": self.to_cpu(coupling),
            "structural_entropy": self.to_cpu(entropy),
        }

    def _compute_fractal_dimensions_gpu(
        self, q_cumulative: np.ndarray, window: int
    ) -> NDArray:
        """局所フラクタル次元の計算（GPU版）"""
        if len(q_cumulative) == 0:
            return self.zeros(len(q_cumulative))

        # xpの確認
        if not hasattr(self, "xp") or self.xp is None:
            self._setup_xp()
            print(f"    ⚠️ Emergency xp setup in fractal_dimensions: {self.xp.__name__}")

        q_cum_gpu = self.to_gpu(q_cumulative)

        # CUDAカーネルでフラクタル次元計算
        if compute_local_fractal_dimension_kernel is not None:
            try:
                dims = compute_local_fractal_dimension_kernel(q_cum_gpu, window)
            except Exception:
                dims = None

            if dims is not None:
                return dims

        # フォールバック実装（self.xpを使う）
        dims = self.ones(len(q_cum_gpu))
        for i in range(window, len(q_cum_gpu) - window):
            local = q_cum_gpu[i - window : i + window]
            var = self.xp.var(local)
            if var > 1e-10:
                dims[i] = 1.0 + self.xp.log(var) / self.xp.log(window)

        return dims

    def _get_coherence_gpu(self, structures: dict) -> NDArray:
        """構造的コヒーレンスを取得"""
        if "structural_coherence" in structures:
            return self.to_gpu(structures["structural_coherence"])
        elif "lambda_F" in structures and len(structures["lambda_F"].shape) > 1:
            # lambda_Fから計算
            lambda_f = self.to_gpu(structures["lambda_F"])
            coherence = self._compute_coherence_from_lambda_f(lambda_f)
            return coherence
        else:
            # なければ1の配列
            return self.ones(len(structures["rho_T"]))

    def _compute_coherence_from_lambda_f(self, lambda_f: NDArray) -> NDArray:
        """Lambda_Fから一貫性を計算"""
        # xpの確認
        if not hasattr(self, "xp") or self.xp is None:
            self._setup_xp()

        n_frames = len(lambda_f)
        coherence = self.ones(n_frames)

        window = 50
        for i in range(window, n_frames - window):
            local_f = lambda_f[i - window : i + window]
            # 方向の一貫性を評価
            mean_dir = self.xp.mean(local_f, axis=0)
            mean_norm = self.xp.linalg.norm(mean_dir)
            if mean_norm > 1e-10:
                mean_dir /= mean_norm
                # 各ベクトルとの内積
                dots = self.xp.sum(local_f * mean_dir[None, :], axis=1)
                norms = self.xp.linalg.norm(local_f, axis=1)
                valid = norms > 1e-10
                if self.xp.any(valid):
                    coherence[i] = self.xp.mean(dots[valid] / norms[valid])

        return coherence

    def _compute_coupling_strength_gpu(
        self, q_cumulative: np.ndarray, window: int
    ) -> NDArray:
        """結合強度の計算（GPU版）"""
        # xpの確認
        if not hasattr(self, "xp") or self.xp is None:
            self._setup_xp()

        q_cum_gpu = self.to_gpu(q_cumulative)
        n = len(q_cum_gpu)
        coupling = self.ones(n)

        # 並列で局所分散を計算
        for i in range(window, n - window):
            local_q = q_cum_gpu[i - window : i + window]
            var = self.xp.var(local_q)

            if var > 1e-10:
                coupling[i] = 1.0 / (1.0 + var)

        return coupling

    def _compute_structural_entropy_gpu(
        self, rho_t: np.ndarray, window: int
    ) -> NDArray:
        """
        構造エントロピーの計算（CuPy RawKernel版）
        PTX 8.4対応
        """
        # xpの確認
        if not hasattr(self, "xp") or self.xp is None:
            self._setup_xp()

        rho_t_gpu = self.to_gpu(rho_t).astype(self.xp.float32)
        n = len(rho_t_gpu)

        if self.is_gpu and self.shannon_entropy_kernel is not None:
            # CuPy RawKernelを使用
            entropy = self.xp.zeros(n, dtype=self.xp.float32)

            threads = 256
            blocks = (n + threads - 1) // threads

            # カーネル実行
            self.shannon_entropy_kernel(
                (blocks,), (threads,), (rho_t_gpu, entropy, window, n)
            )

            self.xp.cuda.Stream.null.synchronize()
            return entropy
        else:
            # フォールバック：self.xpで直接計算
            entropy = self.xp.zeros(n)
            for i in range(window, n - window):
                local_data = rho_t_gpu[i - window : i + window]
                local_sum = self.xp.sum(local_data)
                if local_sum > 1e-10:
                    p = local_data / local_sum
                    valid = p > 1e-10
                    if self.xp.any(valid):
                        entropy[i] = -self.xp.sum(p[valid] * self.xp.log(p[valid]))

            return entropy

    def _compute_boundary_score_gpu(
        self,
        fractal_dims: NDArray,
        coherence: NDArray,
        coupling: NDArray,
        entropy: NDArray,
    ) -> NDArray:
        """統合境界スコアの計算"""
        # xpの確認
        if not hasattr(self, "xp") or self.xp is None:
            self._setup_xp()

        # 長さを揃える
        min_len = min(len(fractal_dims), len(coupling), len(entropy))
        if len(coherence) > 0:
            min_len = min(min_len, len(coherence))

        # 各成分の計算
        if self.is_gpu:
            # compute_gradient_kernelが使えるか確認
            if compute_gradient_kernel is not None:
                fractal_gradient = self.xp.abs(
                    compute_gradient_kernel(fractal_dims[:min_len])
                )
                entropy_gradient = self.xp.abs(
                    compute_gradient_kernel(entropy[:min_len])
                )
            else:
                fractal_gradient = self.xp.abs(self.xp.gradient(fractal_dims[:min_len]))
                entropy_gradient = self.xp.abs(self.xp.gradient(entropy[:min_len]))

            coherence_drop = (
                1 - coherence[:min_len]
                if len(coherence) > 0
                else self.xp.zeros(min_len)
            )
            coupling_weakness = 1 - coupling[:min_len]
        else:
            fractal_gradient = np.abs(np.gradient(fractal_dims[:min_len]))
            coherence_drop = (
                1 - coherence[:min_len] if len(coherence) > 0 else np.zeros(min_len)
            )
            coupling_weakness = 1 - coupling[:min_len]
            entropy_gradient = np.abs(np.gradient(entropy[:min_len]))

        # 重み付き統合
        boundary_score = (
            2.0 * fractal_gradient  # フラクタル次元の変化
            + 1.5 * coherence_drop  # 構造的一貫性の低下
            + 1.0 * coupling_weakness  # 結合の弱まり
            + 1.0 * entropy_gradient  # 情報障壁
        ) / 5.5

        return boundary_score

    def _detect_peaks_gpu(
        self, boundary_score: NDArray, n_steps: int
    ) -> tuple[NDArray, dict]:
        """ピーク検出（GPU版）"""
        # xpの確認
        if not hasattr(self, "xp") or self.xp is None:
            self._setup_xp()

        if len(boundary_score) > 10:
            min_distance_steps = max(50, n_steps // 30)

            if self.is_gpu:
                # NaN値のチェックと除去
                if self.xp.any(self.xp.isnan(boundary_score)):
                    logger.warning("NaN values detected in boundary_score, cleaning...")
                    boundary_score = self.xp.nan_to_num(boundary_score, nan=0.0)

                # 統計値の計算
                mean_val = float(self.xp.mean(boundary_score))
                std_val = float(self.xp.std(boundary_score))

                # 閾値設定
                height_threshold = mean_val + std_val

                # デバッグ情報
                print(
                    f"    Peak detection (GPU): mean={mean_val:.3f}, std={std_val:.3f}, threshold={height_threshold:.3f}"
                )

                # CPUに転送してscipy.signal.find_peaksを使う（安定性重視）
                boundary_score_cpu = self.xp.asnumpy(boundary_score)
                peaks, properties = find_peaks(
                    boundary_score_cpu,
                    height=height_threshold,
                    distance=min_distance_steps,
                )
                # GPU配列として返す
                peaks = self.xp.array(peaks)

            else:
                # CPU版
                # NaN処理
                boundary_score = np.nan_to_num(boundary_score, nan=0.0)
                mean_val = np.mean(boundary_score)
                std_val = np.std(boundary_score)

                height_threshold = mean_val + std_val

                print(
                    f"    Peak detection (CPU): mean={mean_val:.3f}, std={std_val:.3f}, threshold={height_threshold:.3f}"
                )

                peaks, properties = find_peaks(
                    boundary_score, height=height_threshold, distance=min_distance_steps
                )

            # フォールバック：ピークが見つからない場合
            if len(peaks) == 0 and np.max(self.to_cpu(boundary_score)) > 0:
                print("    No peaks found, trying with lower threshold...")
                height_threshold = mean_val + 0.5 * std_val

                if self.is_gpu:
                    boundary_score_cpu = self.xp.asnumpy(boundary_score)
                else:
                    boundary_score_cpu = boundary_score

                peaks, properties = find_peaks(
                    boundary_score_cpu,
                    height=height_threshold,
                    distance=min_distance_steps // 2,
                )

                if self.is_gpu:
                    peaks = self.xp.array(peaks)

        else:
            peaks = self.xp.array([]) if self.is_gpu else np.array([])
            properties = {}

        logger.info(f"   Found {len(peaks)} structural boundaries")

        return peaks, properties

    def _simple_peak_detection_gpu(
        self, array: NDArray, threshold: float, min_distance: int
    ) -> NDArray:
        """シンプルなピーク検出実装（フォールバック用）"""
        # xpの確認
        if not hasattr(self, "xp") or self.xp is None:
            self._setup_xp()

        peaks = []

        # GPU配列をCPUに転送して処理（安定性重視）
        array_cpu = self.xp.asnumpy(array) if self.is_gpu else array

        for i in range(1, len(array_cpu) - 1):
            if (
                array_cpu[i] > threshold
                and array_cpu[i] > array_cpu[i - 1]
                and array_cpu[i] > array_cpu[i + 1]
            ):
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)

        # GPU配列として返す
        if self.is_gpu:
            return self.xp.array(peaks, dtype=self.xp.int64)
        else:
            return np.array(peaks, dtype=np.int64)

    # ===============================
    # 追加のメソッド（トポロジカル破れ検出など）
    # ===============================

    def _detect_lambda_anomalies_gpu(
        self, lambda_mag: np.ndarray, window: int
    ) -> NDArray:
        """Lambda異常検出"""
        # xpの確認
        if not hasattr(self, "xp") or self.xp is None:
            self._setup_xp()

        lambda_gpu = self.to_gpu(lambda_mag)
        n = len(lambda_gpu)
        anomalies = self.zeros(n)

        # 移動平均と標準偏差
        for i in range(window, n - window):
            local = lambda_gpu[i - window : i + window]
            mean = self.xp.mean(local)
            std = self.xp.std(local)
            if std > 1e-10:
                anomalies[i] = self.xp.abs(lambda_gpu[i] - mean) / std

        return anomalies

    def _detect_tension_jumps_gpu(self, rho_t: np.ndarray, window: int) -> NDArray:
        """テンション場ジャンプ検出（CuPy RawKernel版）"""
        # xpの確認
        if not hasattr(self, "xp") or self.xp is None:
            self._setup_xp()

        rho_t_gpu = self.to_gpu(rho_t).astype(self.xp.float32)
        n = len(rho_t_gpu)

        if self.is_gpu and self.detect_jumps_kernel is not None:
            jumps = self.xp.zeros(n, dtype=self.xp.float32)

            threads = 256
            blocks = (n + threads - 1) // threads

            # CuPy RawKernel実行
            self.detect_jumps_kernel(
                (blocks,),
                (threads,),
                (rho_t_gpu, jumps, 2.0, window, n),  # threshold=2.0
            )

            self.xp.cuda.Stream.null.synchronize()
            return jumps
        else:
            # フォールバック
            jumps = self.zeros(n)
            for i in range(1, n - 1):
                diff = abs(rho_t_gpu[i] - rho_t_gpu[i - 1])
                local_mean = self.xp.mean(
                    self.xp.abs(rho_t_gpu[max(0, i - window) : min(n, i + window)])
                )

                if local_mean > 1e-10 and diff > 2.0 * local_mean:
                    jumps[i] = diff / local_mean

            return jumps

    def _detect_phase_breaks_gpu(self, q_lambda: np.ndarray) -> NDArray:
        """位相破れ検出"""
        # xpの確認
        if not hasattr(self, "xp") or self.xp is None:
            self._setup_xp()

        q_gpu = self.to_gpu(q_lambda)
        n = len(q_gpu)
        breaks = self.zeros(n)

        # 位相変化を検出
        phase_diff = self.xp.abs(self.xp.diff(q_gpu))
        threshold = 0.1
        breaks[1:] = self.xp.where(phase_diff > threshold, phase_diff, 0)

        return breaks

    def detect_topological_breaks(
        self, structures: dict[str, np.ndarray], window_steps: int
    ) -> dict[str, np.ndarray]:
        """
        トポロジカル破れの検出（GPU版）

        Returns
        -------
        Dict[str, np.ndarray]
            各種破れ検出結果
        """
        print("\n💥 Detecting topological breaks on GPU...")

        # xpの確認
        if not hasattr(self, "xp") or self.xp is None:
            self._setup_xp()

        with self.memory_manager.temporary_allocation(
            len(structures["rho_T"]) * 4 * 5, "topology"
        ):
            # 1. ΛF異常
            lambda_f_anomaly = self._detect_lambda_anomalies_gpu(
                structures.get("lambda_F_mag", np.zeros(len(structures["rho_T"]))),
                window_steps,
            )

            # 2. ΛFF異常
            lambda_ff_anomaly = self._detect_lambda_anomalies_gpu(
                structures.get("lambda_FF_mag", np.zeros(len(structures["rho_T"]))),
                window_steps // 2,
            )

            # 3. テンション場ジャンプ
            rho_t_breaks = self._detect_tension_jumps_gpu(
                structures["rho_T"], window_steps
            )

            # 4. トポロジカルチャージ異常
            q_breaks = self._detect_phase_breaks_gpu(
                structures.get("Q_lambda", np.zeros(len(structures["rho_T"]) - 1))
            )

            # 5. 統合破れスコア
            combined = self._combine_topological_breaks(
                lambda_f_anomaly, lambda_ff_anomaly, rho_t_breaks, q_breaks
            )

        return {
            "lambda_F_anomaly": self.to_cpu(lambda_f_anomaly),
            "lambda_FF_anomaly": self.to_cpu(lambda_ff_anomaly),
            "rho_T_breaks": self.to_cpu(rho_t_breaks),
            "Q_breaks": self.to_cpu(q_breaks),
            "combined_breaks": self.to_cpu(combined),
        }

    def _combine_topological_breaks(self, *breaks) -> NDArray:
        """破れスコアの統合"""
        # 長さを揃える
        min_len = min(len(b) for b in breaks if len(b) > 0)

        weights = [1.0, 0.8, 1.2, 1.0]  # 各破れの重み
        combined = self.zeros(min_len)

        for break_score, weight in zip(breaks, weights):
            if len(break_score) >= min_len:
                combined += weight * break_score[:min_len]

        return combined


# 追加のGPU最適化関数
def compute_structural_boundaries_batch_gpu(
    structures_list: list[dict[str, np.ndarray]],
    window_steps_list: list[int],
    gpu_backend: GPUBackend = None,
) -> list[dict[str, Any]]:
    """
    複数の構造に対してバッチで境界検出を実行

    Parameters
    ----------
    structures_list : List[Dict]
        構造辞書のリスト
    window_steps_list : List[int]
        各構造のウィンドウサイズ
    gpu_backend : GPUBackend, optional
        共有GPUバックエンド

    Returns
    -------
    List[Dict]
        境界検出結果のリスト
    """
    if gpu_backend is None:
        detector = BoundaryDetectorGPU()
    else:
        detector = BoundaryDetectorGPU()
        detector.device = gpu_backend.device
        detector.memory_manager = gpu_backend.memory_manager

    results = []

    # バッチ処理で効率化
    total_frames = sum(len(s["rho_T"]) for s in structures_list)
    with detector.memory_manager.temporary_allocation(total_frames * 4 * 8, "batch"):
        for structures, window_steps in zip(structures_list, window_steps_list):
            result = detector.detect_structural_boundaries(structures, window_steps)
            results.append(result)

    return results
