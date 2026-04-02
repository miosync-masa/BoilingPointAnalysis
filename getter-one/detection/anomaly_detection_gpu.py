"""
Lambda³ GPU版異常検出モジュール
異常検出アルゴリズムの完全GPU実装
CuPy RawKernelベース（PTX 8.4対応）
"""

import logging
import warnings

import numpy as np

# CuPyの条件付きインポート
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

from ..core.gpu_kernels import anomaly_detection_kernel
from ..core.gpu_utils import GPUBackend
from ..models import NDArray

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ===============================
# CuPy RawKernel定義
# ===============================

BOUNDARY_EMPHASIS_KERNEL_CODE = r"""
extern "C" __global__
void apply_boundary_emphasis_kernel(
    float* local_score,
    const int* boundary_locs,
    const int n_frames,
    const int window,
    const float sigma,
    const int n_boundaries
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_boundaries) {
        int loc = boundary_locs[idx];
        
        // 境界位置の範囲チェック
        if (loc >= 0 && loc < n_frames) {
            // ガウシアンウィンドウを適用
            int start = max(0, loc - window);
            int end = min(n_frames, loc + window);
            
            for (int i = start; i < end; i++) {
                float dist = fabsf((float)(i - loc));
                float weight = expf(-0.5f * powf(dist / sigma, 2.0f));
                atomicAdd(&local_score[i], weight);
            }
        }
    }
}
"""

# ===============================
# 異常検出GPUクラス
# ===============================


class AnomalyDetectorGPU(GPUBackend):
    """異常検出のGPU実装（CuPy RawKernel版）"""

    def __init__(self, force_cpu=False):
        super().__init__(force_cpu)
        self.anomaly_cache = {}

        # CuPy RawKernelをコンパイル
        if HAS_CUPY and not force_cpu:
            try:
                self.boundary_emphasis_kernel = cp.RawKernel(
                    BOUNDARY_EMPHASIS_KERNEL_CODE, "apply_boundary_emphasis_kernel"
                )
                logger.info(
                    "✅ Anomaly detection kernel compiled successfully (PTX 8.4)"
                )
            except Exception as e:
                logger.warning(f"Failed to compile boundary emphasis kernel: {e}")
                self.boundary_emphasis_kernel = None
        else:
            self.boundary_emphasis_kernel = None

    def detect_local_anomalies(self, series: np.ndarray, window: int) -> np.ndarray:
        """
        局所異常検出（CUDAカーネル使用）

        Parameters
        ----------
        series : np.ndarray
            時系列データ
        window : int
            ウィンドウサイズ

        Returns
        -------
        np.ndarray
            異常スコア
        """
        # GPUに転送
        series_gpu = self.to_gpu(series)

        # anomaly_detection_kernelを使用
        anomaly_gpu = anomaly_detection_kernel(series_gpu, window)

        # CPUに戻す
        return self.to_cpu(anomaly_gpu)

    def compute_multiscale_anomalies(
        self,
        lambda_structures: dict[str, np.ndarray],
        boundaries: dict[str, any],
        breaks: dict[str, np.ndarray],
        md_features: dict[str, np.ndarray],
        config: any,
    ) -> dict[str, np.ndarray]:
        """
        マルチスケール異常検出（GPU並列処理）

        Returns
        -------
        Dict[str, np.ndarray]
            各種異常スコア
        """
        print("\n🚀 GPU Multi-scale Anomaly Detection...")

        n_frames = len(lambda_structures["rho_T"])

        # GPUメモリ管理
        if hasattr(self, "memory_manager"):
            # temporary_allocationを使用（batch_contextの代わり）
            with self.memory_manager.temporary_allocation(n_frames * 4 * 10, "anomaly"):
                return self._compute_anomalies_impl(
                    lambda_structures, boundaries, breaks, md_features, config, n_frames
                )
        else:
            return self._compute_anomalies_impl(
                lambda_structures, boundaries, breaks, md_features, config, n_frames
            )

    def _compute_anomalies_impl(
        self, lambda_structures, boundaries, breaks, md_features, config, n_frames
    ):
        """異常検出の実装"""
        # 1. Global anomalies
        global_score = self._compute_global_anomalies_gpu(breaks, config, n_frames)

        # 2. Local anomalies (boundary-focused)
        local_score = self._compute_local_anomalies_gpu(boundaries, n_frames)

        # 3. Extended anomalies (新しい検出ロジック)
        extended_scores = self._compute_extended_anomalies_gpu(
            lambda_structures, md_features, config
        )

        # 4. スコアの正規化
        global_score_norm = self._normalize_scores_gpu(global_score)
        local_score_norm = self._normalize_scores_gpu(local_score)

        # 5. 統合スコア計算
        final_combined = self._compute_final_combined_gpu(
            global_score_norm, local_score_norm, extended_scores
        )

        return {
            "global": self.to_cpu(global_score_norm),
            "local": self.to_cpu(local_score_norm),
            "combined": self.to_cpu((global_score_norm + local_score_norm) / 2),
            **{k: self.to_cpu(v) for k, v in extended_scores.items()},
            "final_combined": self.to_cpu(final_combined),
        }

    def _compute_global_anomalies_gpu(
        self, breaks: dict, config: any, n_frames: int
    ) -> NDArray:
        """グローバル異常スコア計算（GPU）"""
        global_score = cp.zeros(n_frames) if self.is_gpu else np.zeros(n_frames)

        # 各種異常を重み付けして加算
        if "lambda_F_anomaly" in breaks:
            lf_anomaly = self.to_gpu(breaks["lambda_F_anomaly"])
            # NaNチェック付き加算
            if self.is_gpu:
                lf_anomaly = cp.nan_to_num(lf_anomaly, nan=0.0)
            else:
                lf_anomaly = np.nan_to_num(lf_anomaly, nan=0.0)
            global_score[: len(lf_anomaly)] += config.w_lambda_f * lf_anomaly

        if "lambda_FF_anomaly" in breaks:
            lff_anomaly = self.to_gpu(breaks["lambda_FF_anomaly"])
            if self.is_gpu:
                lff_anomaly = cp.nan_to_num(lff_anomaly, nan=0.0)
            else:
                lff_anomaly = np.nan_to_num(lff_anomaly, nan=0.0)
            global_score[: len(lff_anomaly)] += config.w_lambda_ff * lff_anomaly

        if "rho_T_breaks" in breaks:
            rho_breaks = self.to_gpu(breaks["rho_T_breaks"])
            if self.is_gpu:
                rho_breaks = cp.nan_to_num(rho_breaks, nan=0.0)
            else:
                rho_breaks = np.nan_to_num(rho_breaks, nan=0.0)
            global_score[: len(rho_breaks)] += config.w_rho_t * rho_breaks

        if "Q_breaks" in breaks:
            q_breaks = self.to_gpu(breaks["Q_breaks"])
            if self.is_gpu:
                q_breaks = cp.nan_to_num(q_breaks, nan=0.0)
            else:
                q_breaks = np.nan_to_num(q_breaks, nan=0.0)
            global_score[: len(q_breaks)] += config.w_topology * q_breaks

        # 最終NaNチェック
        if self.is_gpu:
            global_score = cp.nan_to_num(global_score, nan=0.0)
        else:
            global_score = np.nan_to_num(global_score, nan=0.0)

        return global_score

    def _compute_local_anomalies_gpu(self, boundaries: dict, n_frames: int) -> NDArray:
        """ローカル異常スコア計算（境界周辺を強調）- CuPy RawKernel版"""
        local_score = (
            cp.zeros(n_frames, dtype=cp.float32)
            if self.is_gpu
            else np.zeros(n_frames, dtype=np.float32)
        )

        if "boundary_locations" in boundaries:
            # 境界位置をGPUに転送
            boundary_locs = self.to_gpu(boundaries["boundary_locations"])

            if self.is_gpu and len(boundary_locs) > 0:
                # CuPy RawKernelを使用
                if self.boundary_emphasis_kernel is not None:
                    # int32型に変換
                    boundary_locs_int = boundary_locs.astype(cp.int32)

                    threads = 256
                    blocks = (len(boundary_locs_int) + threads - 1) // threads

                    # CuPy RawKernel実行
                    self.boundary_emphasis_kernel(
                        (blocks,),
                        (threads,),
                        (
                            local_score,
                            boundary_locs_int,
                            n_frames,
                            50,
                            cp.float32(20.0),
                            len(boundary_locs_int),
                        ),  # window=50, sigma=20
                    )

                    # 同期
                    cp.cuda.Stream.null.synchronize()
                else:
                    # CuPyフォールバック（カーネルがコンパイルできなかった場合）
                    logger.warning("Using CuPy fallback for boundary emphasis")
                    for loc in boundary_locs:
                        loc = int(loc)
                        window = 50
                        sigma = 20.0
                        start = max(0, loc - window)
                        end = min(n_frames, loc + window)

                        for i in range(start, end):
                            dist = abs(i - loc)
                            weight = cp.exp(-0.5 * (dist / sigma) ** 2)
                            local_score[i] += weight
            else:
                # CPU版フォールバック
                for loc in boundary_locs:
                    loc = int(loc)
                    window = 50
                    sigma = 20
                    for i in range(max(0, loc - window), min(n_frames, loc + window)):
                        dist = abs(i - loc)
                        weight = np.exp(-0.5 * (dist / sigma) ** 2)
                        local_score[i] += weight

        # boundary_scoreを加算（NaNチェック付き）
        if "boundary_score" in boundaries:
            bs = self.to_gpu(boundaries["boundary_score"])
            if self.is_gpu:
                bs = cp.nan_to_num(bs, nan=0.0)
            else:
                bs = np.nan_to_num(bs, nan=0.0)
            local_score[: len(bs)] += bs

        # 最終NaNチェック
        if self.is_gpu:
            local_score = cp.nan_to_num(local_score, nan=0.0)
        else:
            local_score = np.nan_to_num(local_score, nan=0.0)

        return local_score

    def _compute_extended_anomalies_gpu(
        self, structures: dict, md_features: dict, config: any
    ) -> dict[str, NDArray]:
        """拡張異常検出（GPU最適化）"""
        extended_scores = {}

        # 各種拡張検出をGPUで実行
        if hasattr(config, "use_periodic") and config.use_periodic:
            # FFTベースの周期検出はCuPyのFFTを使用
            extended_scores["periodic"] = self._detect_periodic_gpu(structures)

        if hasattr(config, "use_gradual") and config.use_gradual:
            extended_scores["gradual"] = self._detect_gradual_gpu(structures)

        if hasattr(config, "use_drift") and config.use_drift:
            extended_scores["drift"] = self._detect_drift_gpu(structures)

        if hasattr(config, "radius_of_gyration") and config.radius_of_gyration:
            if md_features and "radius_of_gyration" in md_features:
                extended_scores["rg_based"] = self._detect_rg_transitions_gpu(
                    md_features
                )

        if hasattr(config, "use_phase_space") and config.use_phase_space:
            extended_scores["phase_space"] = self._detect_phase_space_gpu(structures)

        # 統合スコア
        if extended_scores:
            weights = {
                "periodic": 0.2,
                "gradual": 0.3,
                "drift": 0.3,
                "phase_space": 0.2,
                "rg_based": 0.3,
            }

            combined_extended = (
                cp.zeros(len(structures["rho_T"]))
                if self.is_gpu
                else np.zeros(len(structures["rho_T"]))
            )

            for key, score in extended_scores.items():
                if key in weights and len(score) > 0:
                    weight = weights[key]
                    combined_extended[: len(score)] += weight * score

            extended_scores["combined_extended"] = combined_extended

        return extended_scores

    def _normalize_scores_gpu(self, scores: NDArray) -> NDArray:
        """スコアの正規化"""
        if self.is_gpu:
            # NaN値を除去して統計計算
            valid_mask = ~cp.isnan(scores)
            if cp.sum(valid_mask) == 0:
                # 全部NaNの場合はゼロを返す
                return cp.zeros_like(scores)

            valid_scores = scores[valid_mask]
            mean_score = cp.mean(valid_scores)
            std_score = cp.std(valid_scores)

            # NaNを平均値で埋める
            scores_filled = cp.where(cp.isnan(scores), mean_score, scores)
        else:
            # CPU版も同様に
            valid_mask = ~np.isnan(scores)
            if np.sum(valid_mask) == 0:
                return np.zeros_like(scores)

            valid_scores = scores[valid_mask]
            mean_score = np.mean(valid_scores)
            std_score = np.std(valid_scores)
            scores_filled = np.where(np.isnan(scores), mean_score, scores)

        # 正規化（NaN処理済みのscores_filledを使用）
        if std_score > 1e-10:
            normalized = (scores_filled - mean_score) / std_score
            # 外れ値のクリッピング
            if self.is_gpu:
                normalized = cp.clip(normalized, -5, 5)
            else:
                normalized = np.clip(normalized, -5, 5)
        else:
            normalized = scores_filled - mean_score

        return normalized

    def _compute_final_combined_gpu(
        self,
        global_score: NDArray,
        local_score: NDArray,
        extended_scores: dict[str, NDArray],
    ) -> NDArray:
        """最終統合スコア計算"""
        # 基本スコア
        combined = 0.4 * global_score + 0.3 * local_score

        # 拡張スコアの追加
        if "combined_extended" in extended_scores:
            extended = extended_scores["combined_extended"]
            combined[: len(extended)] += 0.3 * extended

        return combined

    # ===== 各種検出メソッド =====

    def _detect_periodic_gpu(self, structures: dict) -> NDArray:
        """周期的遷移検出（CuPy FFT使用）"""
        rho_t_gpu = self.to_gpu(structures["rho_T"])

        if self.is_gpu:
            # FFT実行
            rho_t_centered = rho_t_gpu - cp.mean(rho_t_gpu)
            yf = cp.fft.rfft(rho_t_centered)
            power = cp.abs(yf) ** 2

            # 周期成分の強さをスコア化（簡略版）
            # 高周波成分を除去
            cutoff = len(power) // 4
            power_low = power[:cutoff]

            # パワーの変動を異常スコアとして使用
            scores = cp.zeros_like(rho_t_gpu)
            # 簡単のため、パワーの移動平均を使用
            window = 50
            for i in range(window, len(scores) - window):
                local_power = cp.mean(power_low[max(0, i // 10 - 5) : i // 10 + 5])
                scores[i] = local_power / (cp.mean(power_low) + 1e-10)
        else:
            # NumPy版
            rho_t_centered = rho_t_gpu - np.mean(rho_t_gpu)
            yf = np.fft.rfft(rho_t_centered)
            power = np.abs(yf) ** 2

            cutoff = len(power) // 4
            power_low = power[:cutoff]

            scores = np.zeros_like(rho_t_gpu)
            window = 50
            for i in range(window, len(scores) - window):
                local_power = np.mean(power_low[max(0, i // 10 - 5) : i // 10 + 5])
                scores[i] = local_power / (np.mean(power_low) + 1e-10)

        return scores

    def _detect_gradual_gpu(self, structures: dict) -> NDArray:
        """緩やかな遷移検出"""
        rho_t_gpu = self.to_gpu(structures["rho_T"])

        # GPU上でガウシアンフィルタ
        window_sizes = [500, 1000, 2000]
        gradual_scores = (
            cp.zeros_like(rho_t_gpu) if self.is_gpu else np.zeros_like(rho_t_gpu)
        )

        for window in window_sizes:
            if window < len(rho_t_gpu):
                # 簡略化：移動平均で代用
                smoothed = self._moving_average_gpu(
                    rho_t_gpu, min(window, len(rho_t_gpu))
                )
                gradient = (
                    cp.gradient(smoothed) if self.is_gpu else np.gradient(smoothed)
                )
                gradual_scores += (
                    cp.abs(gradient) if self.is_gpu else np.abs(gradient)
                ) / len(window_sizes)

        return self._normalize_scores_gpu(gradual_scores)

    def _detect_drift_gpu(self, structures: dict) -> NDArray:
        """構造ドリフト検出"""
        rho_t_gpu = self.to_gpu(structures["rho_T"])
        reference_window = min(1000, len(rho_t_gpu) // 4)

        # 参照値
        ref_value = (
            cp.mean(rho_t_gpu[:reference_window])
            if self.is_gpu
            else np.mean(rho_t_gpu[:reference_window])
        )

        # ドリフト計算
        if self.is_gpu:
            drift_scores = cp.abs(rho_t_gpu - ref_value) / (ref_value + 1e-10)
        else:
            drift_scores = np.abs(rho_t_gpu - ref_value) / (ref_value + 1e-10)

        return drift_scores

    def _detect_rg_transitions_gpu(self, md_features: dict) -> NDArray:
        """Radius of Gyration変化検出"""
        rg_gpu = self.to_gpu(md_features["radius_of_gyration"])

        # 勾配計算
        gradient = cp.gradient(rg_gpu) if self.is_gpu else np.gradient(rg_gpu)

        # 収縮を強調
        if self.is_gpu:
            contraction_score = cp.where(
                gradient < 0, cp.abs(gradient) * 2.0, cp.abs(gradient)
            )
        else:
            contraction_score = np.where(
                gradient < 0, np.abs(gradient) * 2.0, np.abs(gradient)
            )

        return contraction_score

    def _detect_phase_space_gpu(self, structures: dict) -> NDArray:
        """位相空間異常検出（簡略版）"""
        rho_t_gpu = self.to_gpu(structures["rho_T"])

        # 簡略化：局所的な変動を異常とする
        window = 50
        local_std = self._local_std_gpu(rho_t_gpu, window)

        return local_std

    # ===== ユーティリティメソッド =====

    def _moving_average_gpu(self, data: NDArray, window: int) -> NDArray:
        """GPU上での移動平均"""
        if window >= len(data):
            window = len(data) - 1

        if self.is_gpu:
            # CuPyのconvolveを使用
            kernel = cp.ones(window) / window
            # パディングして同じサイズを維持
            padded = cp.pad(data, (window // 2, window // 2), mode="edge")
            result = cp.convolve(padded, kernel, mode="valid")
            # サイズ調整
            if len(result) > len(data):
                result = result[: len(data)]
            elif len(result) < len(data):
                result = cp.pad(result, (0, len(data) - len(result)), mode="edge")
            return result
        else:
            kernel = np.ones(window) / window
            padded = np.pad(data, (window // 2, window // 2), mode="edge")
            result = np.convolve(padded, kernel, mode="valid")
            if len(result) > len(data):
                result = result[: len(data)]
            elif len(result) < len(data):
                result = np.pad(result, (0, len(data) - len(result)), mode="edge")
            return result

    def _local_std_gpu(self, data: NDArray, window: int) -> NDArray:
        """局所標準偏差"""
        n = len(data)
        if self.is_gpu:
            std_array = cp.zeros_like(data)
        else:
            std_array = np.zeros_like(data)

        for i in range(n):
            start = max(0, i - window // 2)
            end = min(n, i + window // 2 + 1)

            if end - start > 1:
                if self.is_gpu:
                    std_array[i] = cp.std(data[start:end])
                else:
                    std_array[i] = np.std(data[start:end])

        return std_array


# ===============================
# テスト関数
# ===============================


def test_anomaly_detection():
    """異常検出モジュールのテスト"""
    print("\n🧪 Testing Anomaly Detection GPU...")

    # テストデータ生成
    n_frames = 10000
    lambda_structures = {
        "rho_T": np.random.randn(n_frames).astype(np.float32),
        "lambda_F_mag": np.random.randn(n_frames).astype(np.float32),
        "lambda_FF_mag": np.random.randn(n_frames).astype(np.float32),
        "Q_lambda": np.cumsum(np.random.randn(n_frames - 1)).astype(np.float32),
    }

    boundaries = {
        "boundary_locations": np.array([100, 500, 900, 2000, 5000], dtype=np.int32),
        "boundary_score": np.random.rand(n_frames).astype(np.float32),
    }

    breaks = {
        "lambda_F_anomaly": np.random.rand(n_frames).astype(np.float32),
        "lambda_FF_anomaly": np.random.rand(n_frames).astype(np.float32),
        "rho_T_breaks": np.random.rand(n_frames).astype(np.float32),
        "Q_breaks": np.random.rand(n_frames - 1).astype(np.float32),
    }

    md_features = {"radius_of_gyration": np.random.rand(n_frames).astype(np.float32)}

    # 設定オブジェクト
    class Config:
        w_lambda_f = 1.0
        w_lambda_ff = 0.8
        w_rho_t = 1.2
        w_topology = 1.0
        use_periodic = True
        use_gradual = True
        use_drift = True
        radius_of_gyration = True
        use_phase_space = True

    config = Config()

    # 異常検出器初期化
    detector = AnomalyDetectorGPU()

    # 異常検出実行
    print("Running multi-scale anomaly detection...")
    results = detector.compute_multiscale_anomalies(
        lambda_structures, boundaries, breaks, md_features, config
    )

    # 結果確認
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, mean={np.mean(value):.4f}")

    print("\n✅ Anomaly detection test passed!")
    return True


if __name__ == "__main__":
    test_anomaly_detection()
