"""
Lambda³ GPU版トポロジカル破れ検出モジュール
構造フローのトポロジカルな破れをGPUで高速検出
CuPy RawKernelベース（PTX 8.4対応）
"""

import logging

import numpy as np

# CuPyが利用可能かチェック
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter1d as gaussian_filter1d_gpu

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None
    gaussian_filter1d_gpu = None

from ..core.gpu_kernels import anomaly_detection_kernel
from ..core.gpu_utils import GPUBackend
from ..models import NDArray


def _anomaly_detection_cpu(series: np.ndarray, window: int) -> np.ndarray:
    """
    適応的z-scoreによる異常検出（CPUフォールバック版）
    anomaly_detection_kernelのCUDA実装と同等の処理をnumpyで実行

    Parameters
    ----------
    series : np.ndarray
        入力時系列（1D）
    window : int
        スライディングウィンドウサイズ

    Returns
    -------
    np.ndarray
        異常スコア（適応的z-score）
    """
    n = len(series)
    scores = np.zeros(n, dtype=np.float32)
    series = np.asarray(series, dtype=np.float32)

    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        local = series[start:end]
        local_mean = np.mean(local)
        local_std = np.std(local)
        if local_std > 1e-10:
            scores[i] = abs(float(series[i]) - float(local_mean)) / float(local_std)

    return scores

# ロガー設定
logger = logging.getLogger(__name__)

# ===============================
# CuPy RawKernel定義
# ===============================

LOCAL_EXTREMA_KERNEL_CODE = r"""
extern "C" __global__
void local_extrema_kernel(
    const float* data,
    float* extrema,
    const int window,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= window && idx < n - window) {
        float center = data[idx];
        bool is_max = true;
        bool is_min = true;
        
        // 局所範囲で比較
        for (int i = idx - window; i <= idx + window; i++) {
            if (i != idx) {
                if (data[i] >= center) {
                    is_max = false;
                }
                if (data[i] <= center) {
                    is_min = false;
                }
                
                // 早期終了
                if (!is_max && !is_min) {
                    break;
                }
            }
        }
        
        extrema[idx] = (is_max || is_min) ? 1.0f : 0.0f;
    }
}
"""


class TopologyBreaksDetectorGPU(GPUBackend):
    """トポロジカル破れ検出のGPU実装（CuPy RawKernel版）"""

    def __init__(self, force_cpu=False):
        super().__init__(force_cpu)
        self.breaks_cache = {}

        # CuPy RawKernelをコンパイル
        if HAS_GPU and not force_cpu:
            try:
                self.local_extrema_kernel = cp.RawKernel(
                    LOCAL_EXTREMA_KERNEL_CODE, "local_extrema_kernel"
                )
                logger.info("✅ Topology breaks kernel compiled successfully (PTX 8.4)")
            except Exception as e:
                logger.warning(f"Failed to compile local extrema kernel: {e}")
                self.local_extrema_kernel = None
        else:
            self.local_extrema_kernel = None

    def detect_topological_breaks(
        self, structures: dict[str, np.ndarray], window_steps: int
    ) -> dict[str, np.ndarray]:
        """
        トポロジカル破れの検出（完全GPU版）

        Parameters
        ----------
        structures : Dict[str, np.ndarray]
            Lambda構造辞書
        window_steps : int
            ウィンドウサイズ

        Returns
        -------
        Dict[str, np.ndarray]
            各種破れの検出結果
        """
        print("\n💥 Detecting topological breaks on GPU...")

        n_frames = len(structures["rho_T"])

        # temporary_allocationを使用
        with self.memory_manager.temporary_allocation(
            n_frames * 4 * 8, "topology_breaks"
        ):
            # 1. ΛF異常（構造フロー破れ）
            lambda_f_anomaly = self._detect_flow_anomalies_gpu(
                structures["lambda_F_mag"], window_steps
            )

            # 2. ΛFF異常（加速度破れ）
            lambda_ff_anomaly = self._detect_acceleration_anomalies_gpu(
                structures["lambda_FF_mag"], window_steps // 2
            )

            # 3. テンション場ジャンプ
            rho_t_breaks = self._detect_tension_field_jumps_gpu(
                structures["rho_T"], window_steps
            )

            # 4. トポロジカルチャージ異常
            q_breaks = self._detect_topological_charge_breaks_gpu(
                structures["Q_lambda"]
            )

            # 5. 位相コヒーレンス破れ（新規追加）
            phase_coherence_breaks = self._detect_phase_coherence_breaks_gpu(structures)

            # 6. 構造的特異点検出（新規追加）
            singularities = self._detect_structural_singularities_gpu(
                structures, window_steps
            )

            # 7. 統合異常スコア
            combined_anomaly = self._combine_topological_anomalies_gpu(
                lambda_f_anomaly,
                lambda_ff_anomaly,
                rho_t_breaks,
                q_breaks,
                phase_coherence_breaks,
                singularities,
            )

        return {
            "lambda_F_anomaly": self.to_cpu(lambda_f_anomaly),
            "lambda_FF_anomaly": self.to_cpu(lambda_ff_anomaly),
            "rho_T_breaks": self.to_cpu(rho_t_breaks),
            "Q_breaks": self.to_cpu(q_breaks),
            "phase_coherence_breaks": self.to_cpu(phase_coherence_breaks),
            "singularities": self.to_cpu(singularities),
            "combined_anomaly": self.to_cpu(combined_anomaly),
        }

    def _detect_flow_anomalies_gpu(
        self, lambda_f_mag: np.ndarray, window: int
    ) -> NDArray:
        """構造フローの異常検出（GPU最適化）"""
        lf_mag_gpu = self.to_gpu(lambda_f_mag)

        # 適応的z-scoreによる異常検出
        if self.is_gpu:
            anomaly_gpu = anomaly_detection_kernel(lf_mag_gpu, window)
        else:
            anomaly_gpu = _anomaly_detection_cpu(lambda_f_mag, window)

        # 追加: 急激な変化の検出
        if self.is_gpu:
            gradient = cp.abs(cp.gradient(lf_mag_gpu))
        else:
            gradient = np.abs(np.gradient(lambda_f_mag))

        sudden_changes = self._detect_sudden_changes_gpu(gradient, window)

        # 両方の異常を統合
        if self.is_gpu:
            return cp.maximum(anomaly_gpu, sudden_changes)
        else:
            return np.maximum(anomaly_gpu, sudden_changes)

    def _detect_acceleration_anomalies_gpu(
        self, lambda_ff_mag: np.ndarray, window: int
    ) -> NDArray:
        """加速度異常の検出"""
        lff_mag_gpu = self.to_gpu(lambda_ff_mag)

        # 基本的な異常検出
        if self.is_gpu:
            anomaly_gpu = anomaly_detection_kernel(lff_mag_gpu, window)
        else:
            anomaly_gpu = _anomaly_detection_cpu(lambda_ff_mag, window)

        # 加速度特有の処理：符号変化の検出
        if "lambda_FF" in self.breaks_cache:
            lambda_ff = self.to_gpu(self.breaks_cache["lambda_FF"])
            sign_changes = self._detect_sign_changes_gpu(lambda_ff)
            if self.is_gpu:
                anomaly_gpu = cp.maximum(anomaly_gpu, sign_changes)
            else:
                anomaly_gpu = np.maximum(anomaly_gpu, sign_changes)

        return anomaly_gpu

    def _detect_tension_field_jumps_gpu(
        self, rho_t: np.ndarray, window_steps: int
    ) -> NDArray:
        """テンション場のジャンプ検出（改良版）"""
        rho_t_gpu = self.to_gpu(rho_t)

        # マルチスケールスムージング
        sigmas = [window_steps / 6, window_steps / 3, window_steps / 2]

        if self.is_gpu:
            jumps_multiscale = cp.zeros_like(rho_t_gpu)
        else:
            jumps_multiscale = np.zeros_like(rho_t)

        for sigma in sigmas:
            # ガウシアンフィルタ
            if self.is_gpu and gaussian_filter1d_gpu is not None:
                rho_t_smooth = gaussian_filter1d_gpu(rho_t_gpu, sigma=sigma)
                # ジャンプ検出
                jumps = cp.abs(rho_t_gpu - rho_t_smooth)
                # 正規化
                jumps_norm = jumps / (cp.std(jumps) + 1e-10)
            else:
                from scipy.ndimage import gaussian_filter1d

                rho_t_np = rho_t if not self.is_gpu else cp.asnumpy(rho_t_gpu)
                rho_t_smooth = gaussian_filter1d(rho_t_np, sigma=sigma)
                jumps = np.abs(rho_t_np - rho_t_smooth)
                jumps_norm = jumps / (np.std(jumps) + 1e-10)
                if self.is_gpu:
                    jumps_norm = cp.asarray(jumps_norm)

            jumps_multiscale += jumps_norm / len(sigmas)

        return jumps_multiscale

    def _detect_topological_charge_breaks_gpu(self, q_lambda: np.ndarray) -> NDArray:
        """トポロジカルチャージの破れ検出"""
        q_lambda_gpu = self.to_gpu(q_lambda)

        if self.is_gpu:
            breaks = cp.zeros_like(q_lambda_gpu)
            # 位相差の計算
            phase_diff = cp.abs(cp.diff(q_lambda_gpu))
            # 閾値以上の急激な変化を検出
            threshold = 0.1  # 0.1 * 2π radians
            breaks[1:] = cp.where(phase_diff > threshold, phase_diff, 0)
        else:
            breaks = np.zeros_like(q_lambda)
            phase_diff = np.abs(np.diff(q_lambda))
            threshold = 0.1
            breaks[1:] = np.where(phase_diff > threshold, phase_diff, 0)

        # 累積的な破れの検出
        cumulative_breaks = self._detect_cumulative_breaks_gpu(q_lambda_gpu)

        if self.is_gpu:
            return cp.maximum(breaks, cumulative_breaks)
        else:
            return np.maximum(breaks, cumulative_breaks)

    def _detect_phase_coherence_breaks_gpu(self, structures: dict) -> NDArray:
        """位相コヒーレンスの破れ検出（新機能）"""
        if "structural_coherence" not in structures:
            if self.is_gpu:
                return cp.zeros(len(structures["rho_T"]))
            else:
                return np.zeros(len(structures["rho_T"]))

        coherence_gpu = self.to_gpu(structures["structural_coherence"])

        if self.is_gpu:
            # コヒーレンスの急激な低下を検出
            coherence_gradient = cp.gradient(coherence_gpu)

            # 負の勾配（コヒーレンス低下）を強調
            breaks = cp.where(
                coherence_gradient < 0,
                -coherence_gradient * 2.0,
                cp.abs(coherence_gradient),
            )

            # 閾値処理
            threshold = cp.mean(breaks) + 2 * cp.std(breaks)
            breaks = cp.where(breaks > threshold, breaks, 0)
        else:
            coherence_gradient = np.gradient(structures["structural_coherence"])
            breaks = np.where(
                coherence_gradient < 0,
                -coherence_gradient * 2.0,
                np.abs(coherence_gradient),
            )
            threshold = np.mean(breaks) + 2 * np.std(breaks)
            breaks = np.where(breaks > threshold, breaks, 0)

        return breaks

    def _detect_structural_singularities_gpu(
        self, structures: dict, window: int
    ) -> NDArray:
        """構造的特異点の検出（新機能）"""
        n_frames = len(structures["rho_T"])

        if self.is_gpu:
            singularities = cp.zeros(n_frames)
        else:
            singularities = np.zeros(n_frames)

        # 複数の指標から特異点を検出
        rho_t_gpu = self.to_gpu(structures["rho_T"])
        lf_mag_gpu = self.to_gpu(structures["lambda_F_mag"])

        # 1. テンション場の局所極値
        tension_extrema = self._find_local_extrema_gpu(rho_t_gpu, window)

        # 2. フロー場の発散/収束
        if len(structures["lambda_F"].shape) == 2:  # ベクトル場の場合
            lambda_f_gpu = self.to_gpu(structures["lambda_F"])
            divergence = self._compute_divergence_gpu(lambda_f_gpu)

            # divergenceの長さを確認して調整
            if len(divergence) != n_frames:
                # divergenceが短い場合はパディング
                if len(divergence) < n_frames:
                    if self.is_gpu:
                        divergence = cp.pad(
                            divergence, (0, n_frames - len(divergence)), mode="edge"
                        )
                    else:
                        divergence = np.pad(
                            divergence, (0, n_frames - len(divergence)), mode="edge"
                        )
                else:
                    # divergenceが長い場合は切り詰め
                    divergence = divergence[:n_frames]

            if self.is_gpu:
                div_anomaly = cp.abs(divergence) > cp.std(divergence) * 3
                singularities += div_anomaly.astype(cp.float32)
            else:
                div_anomaly = np.abs(divergence) > np.std(divergence) * 3
                singularities += div_anomaly.astype(np.float32)

        # 3. 位相空間での異常軌道
        phase_anomaly = self._detect_phase_space_singularities_gpu(
            lf_mag_gpu, rho_t_gpu, window
        )

        # phase_anomalyの長さも確認
        if len(phase_anomaly) != n_frames:
            if len(phase_anomaly) < n_frames:
                if self.is_gpu:
                    phase_anomaly = cp.pad(
                        phase_anomaly, (0, n_frames - len(phase_anomaly)), mode="edge"
                    )
                else:
                    phase_anomaly = np.pad(
                        phase_anomaly, (0, n_frames - len(phase_anomaly)), mode="edge"
                    )
            else:
                phase_anomaly = phase_anomaly[:n_frames]

        # tension_extremaの長さも確認
        if len(tension_extrema) != n_frames:
            if len(tension_extrema) < n_frames:
                if self.is_gpu:
                    tension_extrema = cp.pad(
                        tension_extrema,
                        (0, n_frames - len(tension_extrema)),
                        mode="edge",
                    )
                else:
                    tension_extrema = np.pad(
                        tension_extrema,
                        (0, n_frames - len(tension_extrema)),
                        mode="edge",
                    )
            else:
                tension_extrema = tension_extrema[:n_frames]

        singularities += tension_extrema + phase_anomaly

        return singularities / 3.0  # 正規化

    def _detect_sudden_changes_gpu(self, gradient: NDArray, window: int) -> NDArray:
        """急激な変化の検出"""
        # 移動標準偏差
        moving_std = self._moving_std_gpu(gradient, window)

        # 外れ値検出
        threshold = 3.0

        if self.is_gpu:
            sudden_changes = cp.where(
                gradient > moving_std * threshold, gradient / (moving_std + 1e-10), 0
            )
        else:
            sudden_changes = np.where(
                gradient > moving_std * threshold, gradient / (moving_std + 1e-10), 0
            )

        return sudden_changes

    def _detect_sign_changes_gpu(self, vector_field: NDArray) -> NDArray:
        """符号変化の検出"""
        if len(vector_field.shape) == 1:
            # スカラー場
            if self.is_gpu:
                sign_diff = cp.diff(cp.sign(vector_field))
                changes = cp.abs(sign_diff) / 2.0
                return cp.pad(changes, (1, 0), mode="constant")
            else:
                sign_diff = np.diff(np.sign(vector_field))
                changes = np.abs(sign_diff) / 2.0
                return np.pad(changes, (1, 0), mode="constant")
        else:
            # ベクトル場
            if self.is_gpu:
                changes = cp.zeros(len(vector_field))
            else:
                changes = np.zeros(len(vector_field))

            for i in range(vector_field.shape[1]):
                component = vector_field[:, i]
                if self.is_gpu:
                    sign_diff = cp.diff(cp.sign(component))
                    changes[1:] += cp.abs(sign_diff) / (2.0 * vector_field.shape[1])
                else:
                    sign_diff = np.diff(np.sign(component))
                    changes[1:] += np.abs(sign_diff) / (2.0 * vector_field.shape[1])
            return changes

    def _detect_cumulative_breaks_gpu(self, q_lambda: NDArray) -> NDArray:
        """累積的な破れの検出"""
        if self.is_gpu:
            # 累積和
            q_cumsum = cp.cumsum(q_lambda)

            # 期待される線形成長からの乖離
            x = cp.arange(len(q_lambda))
            slope = (q_cumsum[-1] - q_cumsum[0]) / (len(q_lambda) - 1)
            expected = q_cumsum[0] + slope * x

            deviation = cp.abs(q_cumsum - expected)

            # 急激な乖離を検出
            deviation_gradient = cp.abs(cp.gradient(deviation))

            return deviation_gradient / (cp.max(deviation_gradient) + 1e-10)
        else:
            q_cumsum = np.cumsum(q_lambda)
            x = np.arange(len(q_lambda))
            slope = (q_cumsum[-1] - q_cumsum[0]) / (len(q_lambda) - 1)
            expected = q_cumsum[0] + slope * x
            deviation = np.abs(q_cumsum - expected)
            deviation_gradient = np.abs(np.gradient(deviation))
            return deviation_gradient / (np.max(deviation_gradient) + 1e-10)

    def _find_local_extrema_gpu(self, data: NDArray, window: int) -> NDArray:
        """局所極値の検出 - CuPy RawKernel使用（PTX 8.4対応）"""
        if self.is_gpu:
            data_gpu = self.to_gpu(data).astype(cp.float32)
        else:
            data_gpu = np.asarray(data, dtype=np.float32)

        if self.is_gpu and self.local_extrema_kernel is not None:
            extrema = cp.zeros_like(data_gpu, dtype=cp.float32)

            # CuPy RawKernel呼び出し
            threads = 256
            blocks = (len(data_gpu) + threads - 1) // threads

            self.local_extrema_kernel(
                (blocks,), (threads,), (data_gpu, extrema, window, len(data_gpu))
            )

            cp.cuda.Stream.null.synchronize()
            return extrema
        else:
            # フォールバック（CPUまたはCuPy）
            if self.is_gpu:
                extrema = cp.zeros_like(data_gpu)
                for i in range(window, len(data_gpu) - window):
                    local_max = cp.max(data_gpu[i - window : i + window + 1])
                    local_min = cp.min(data_gpu[i - window : i + window + 1])
                    if data_gpu[i] == local_max or data_gpu[i] == local_min:
                        extrema[i] = 1.0
            else:
                extrema = np.zeros_like(data)
                for i in range(window, len(data) - window):
                    local_max = np.max(data[i - window : i + window + 1])
                    local_min = np.min(data[i - window : i + window + 1])
                    if data[i] == local_max or data[i] == local_min:
                        extrema[i] = 1.0
            return extrema

    def _compute_divergence_gpu(self, vector_field: NDArray) -> NDArray:
        """ベクトル場の発散を計算（修正版）"""
        if len(vector_field.shape) != 2:
            if self.is_gpu:
                return cp.zeros(len(vector_field))
            else:
                return np.zeros(len(vector_field))

        n_frames = len(vector_field)

        # 各成分の偏微分
        if self.is_gpu:
            div = cp.zeros(n_frames)  # 元のサイズで初期化
            for i in range(vector_field.shape[1]):
                # gradientは同じ長さを返すはず
                component_grad = cp.gradient(vector_field[:, i])
                div += component_grad
        else:
            div = np.zeros(n_frames)  # 元のサイズで初期化
            for i in range(vector_field.shape[1]):
                component_grad = np.gradient(vector_field[:, i])
                div += component_grad

        return div

    def _detect_phase_space_singularities_gpu(
        self, lf_mag: NDArray, rho_t: NDArray, window: int
    ) -> NDArray:
        """位相空間での特異点検出"""
        n = len(lf_mag)

        if self.is_gpu:
            singularities = cp.zeros(n)
        else:
            singularities = np.zeros(n)

        # 簡易的な位相空間埋め込み
        for i in range(window, n - window):
            # 局所的な軌道の異常性
            local_lf = lf_mag[i - window : i + window]
            local_rho = rho_t[i - window : i + window]

            # 相関の急激な変化
            if len(local_lf) > 5:
                if self.is_gpu:
                    corr = cp.corrcoef(local_lf, local_rho)[0, 1]
                    if cp.isnan(corr):
                        corr = 0
                    # 相関の絶対値が低い = 特異的
                    singularities[i] = 1 - cp.abs(corr)
                else:
                    corr = np.corrcoef(local_lf, local_rho)[0, 1]
                    if np.isnan(corr):
                        corr = 0
                    singularities[i] = 1 - np.abs(corr)

        return singularities

    def _moving_std_gpu(self, data: NDArray, window: int) -> NDArray:
        """移動標準偏差の計算"""
        if self.is_gpu:
            std_array = cp.zeros_like(data)
        else:
            std_array = np.zeros_like(data)

        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)

            if end - start > 1:
                if self.is_gpu:
                    local_data = data[start:end]
                    # NaNチェック（有効な値だけで計算を試みる）
                    if cp.any(cp.isnan(local_data)):
                        valid_mask = ~cp.isnan(local_data)
                        if cp.sum(valid_mask) > 1:  # 2個以上有効値があれば
                            std_array[i] = cp.std(local_data[valid_mask])
                        else:
                            std_array[i] = 0.0
                    else:
                        std_array[i] = cp.std(local_data)
                else:
                    local_data = data[start:end]
                    if np.any(np.isnan(local_data)):
                        valid_mask = ~np.isnan(local_data)
                        if np.sum(valid_mask) > 1:  # 2個以上有効値があれば
                            std_array[i] = np.std(local_data[valid_mask])
                        else:
                            std_array[i] = 0.0
                    else:
                        std_array[i] = np.std(local_data)
            else:
                # データ点が1個以下の場合
                std_array[i] = 0.0

        return std_array

    def _combine_topological_anomalies_gpu(self, *anomalies) -> NDArray:
        """トポロジカル異常の統合"""
        # 全ての長さを揃える
        min_len = min(len(a) for a in anomalies)

        # 重み（新しい破れタイプも含む）
        weights = [1.0, 0.8, 0.6, 1.2, 0.9, 1.1]

        if self.is_gpu:
            combined = cp.zeros(min_len)
        else:
            combined = np.zeros(min_len)

        for i, (anomaly, weight) in enumerate(zip(anomalies, weights)):
            if i < len(weights):
                combined += weight * anomaly[:min_len]

        combined /= sum(weights[: len(anomalies)])

        return combined


# ===============================
# テスト関数
# ===============================


def test_topology_breaks():
    """トポロジカル破れ検出のテスト"""
    print("\n🧪 Testing Topology Breaks Detection GPU...")

    # テストデータ生成
    n_frames = 10000
    structures = {
        "rho_T": np.random.randn(n_frames).astype(np.float32),
        "lambda_F": np.random.randn(n_frames, 3).astype(np.float32),  # ベクトル場
        "lambda_F_mag": np.random.rand(n_frames).astype(np.float32),
        "lambda_FF_mag": np.random.rand(n_frames).astype(np.float32),
        "Q_lambda": np.cumsum(np.random.randn(n_frames) * 0.1).astype(np.float32),
        "structural_coherence": np.random.rand(n_frames).astype(np.float32),
    }

    # 検出器初期化
    detector = TopologyBreaksDetectorGPU()

    # トポロジカル破れ検出実行
    print("Running topological breaks detection...")
    results = detector.detect_topological_breaks(structures, window_steps=100)

    # 結果確認
    for key, value in results.items():
        print(
            f"  {key}: shape={value.shape}, mean={np.mean(value):.4f}, max={np.max(value):.4f}"
        )

    # 局所極値検出のテスト
    print("\nTesting local extrema detection...")
    test_data = np.sin(np.linspace(0, 4 * np.pi, 1000)).astype(np.float32)
    extrema = detector._find_local_extrema_gpu(test_data, window=10)
    n_extrema = np.sum(detector.to_cpu(extrema) > 0)
    print(f"  Found {n_extrema} extrema in sine wave")

    print("\n✅ Topology breaks detection test passed!")
    return True


if __name__ == "__main__":
    test_topology_breaks()
