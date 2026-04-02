"""
Lambda³ GPU版拡張異常検出モジュール
周期的遷移、緩やかな遷移、構造ドリフトなどの長期的異常パターンをGPUで検出
完全修正版 - 全てのcp直接参照をself.xpに置換
"""

import numpy as np
from numba import cuda

try:
    import cupy as cp
    from scipy.signal import find_peaks  # cupyxではなくscipyを使用

    HAS_CUDA = True
except ImportError:
    cp = None
    find_peaks = None
    HAS_CUDA = False

from ..core.gpu_utils import GPUBackend
from ..models import NDArray


class ExtendedDetectorGPU(GPUBackend):
    """拡張異常検出アルゴリズムのGPU実装"""

    def __init__(self, force_cpu=False):
        super().__init__(force_cpu)
        self.fft_cache = {}

        # xpの初期設定
        self._setup_xp()

    def _setup_xp(self):
        """numpy/cupyの切り替え設定"""
        if self.is_gpu and HAS_CUDA:
            import cupy as cp

            self.xp = cp
        else:
            import numpy as np

            self.xp = np

    def _ensure_xp(self):
        """xpの存在確認と再設定"""
        if not hasattr(self, "xp") or self.xp is None:
            self._setup_xp()

    def detect_periodic_transitions(
        self,
        structures: dict[str, np.ndarray],
        min_period: int = 1000,
        max_period: int = 10000,
    ) -> dict[str, NDArray]:
        """
        FFTベースの長期周期検出（GPU高速化）

        Parameters
        ----------
        structures : Dict[str, np.ndarray]
            Lambda構造体
        min_period : int
            検出する最小周期
        max_period : int
            検出する最大周期

        Returns
        -------
        Dict[str, NDArray]
            周期的異常スコアと検出された周期
        """
        print("\n🌊 Detecting periodic transitions on GPU...")

        # xpの確認
        self._ensure_xp()

        # 入力検証
        if "rho_T" not in structures or len(structures["rho_T"]) == 0:
            print("   ⚠️ Warning: rho_T not found or empty.")
            return {"scores": self.xp.zeros(1), "detected_periods": []}

        rho_t_gpu = self.to_gpu(structures["rho_T"])
        n_frames = len(rho_t_gpu)

        # 周期範囲の調整
        min_period = min(min_period, n_frames // 10)
        max_period = min(max_period, n_frames)

        # GPU上でFFT解析
        periodic_scores, detected_periods = self._analyze_periodicity_gpu(
            rho_t_gpu, min_period, max_period
        )

        # 他の信号も解析（オプション）
        if "sigma_s" in structures and len(structures["sigma_s"]) == n_frames:
            sigma_s_gpu = self.to_gpu(structures["sigma_s"])
            sigma_scores, sigma_periods = self._analyze_periodicity_gpu(
                sigma_s_gpu, min_period, max_period, weight=0.8
            )
            periodic_scores += sigma_scores
            detected_periods.extend(sigma_periods)

        # スコアの最終調整
        periodic_scores = self._finalize_periodic_scores_gpu(periodic_scores)

        return {
            "scores": periodic_scores,
            "detected_periods": detected_periods,
            "metadata": {
                "n_frames": n_frames,
                "frequency_range": (1 / max_period, 1 / min_period)
                if max_period > 0
                else (0, 0),
            },
        }

    def detect_gradual_transitions(
        self, structures: dict[str, np.ndarray], window_sizes: list[int] = None
    ) -> dict[str, NDArray]:
        """
        複数時間スケールでの緩やかな遷移検出（GPU版）
        """
        print("\n🌅 Detecting gradual transitions on GPU...")

        # xpの確認
        self._ensure_xp()

        if window_sizes is None:
            window_sizes = [500, 1000, 2000]

        rho_t_gpu = self.to_gpu(structures["rho_T"])
        n_frames = len(rho_t_gpu)
        gradual_scores = self.xp.zeros(n_frames)

        # マルチスケール解析
        for window in window_sizes:
            if window < n_frames:
                # 長期トレンド抽出
                trend = self._extract_trend_gpu(rho_t_gpu, window)

                # 勾配計算
                gradient = self.xp.gradient(trend)

                # 持続的な勾配を検出
                sustained_gradient = self._detect_sustained_gradient_gpu(
                    gradient, window
                )

                # 正規化して加算
                if self.xp.std(sustained_gradient) > 1e-10:
                    normalized = (
                        sustained_gradient - self.xp.mean(sustained_gradient)
                    ) / self.xp.std(sustained_gradient)
                    gradual_scores += normalized / len(window_sizes)

        # σsの変化も考慮
        if "sigma_s" in structures:
            sigma_s_gpu = self.to_gpu(structures["sigma_s"])
            sigma_gradient = self._compute_sigma_gradient_gpu(sigma_s_gpu)
            gradual_scores += 0.5 * sigma_gradient

        print(
            f"   Score range: {self.xp.min(gradual_scores):.2f} to {self.xp.max(gradual_scores):.2f}"
        )

        return {"scores": gradual_scores, "window_sizes": window_sizes}

    def detect_structural_drift(
        self, structures: dict[str, np.ndarray], reference_window: int = 1000
    ) -> dict[str, NDArray]:
        """
        構造的ドリフト検出（GPU最適化）
        """
        print("\n🌀 Detecting structural drift on GPU...")

        # xpの確認
        self._ensure_xp()

        rho_t_gpu = self.to_gpu(structures["rho_T"])
        n_frames = len(rho_t_gpu)

        # Q_cumulativeの処理
        if "Q_cumulative" in structures:
            q_cumulative_gpu = self.to_gpu(structures["Q_cumulative"])
            q_cumulative_len = len(q_cumulative_gpu)
        else:
            q_cumulative_gpu = self.xp.zeros(n_frames - 1)
            q_cumulative_len = n_frames - 1

        # 参照ウィンドウのサイズ調整
        ref_window_size = min(reference_window, n_frames, q_cumulative_len)

        # ドリフトスコア計算
        drift_scores = self._compute_drift_scores_gpu(
            rho_t_gpu, q_cumulative_gpu, ref_window_size, n_frames
        )

        # スムージング（gaussian_filter1dを自前実装）
        drift_scores = self._gaussian_smooth_gpu(drift_scores, sigma=100)

        print(f"   Maximum drift: {self.xp.max(drift_scores):.2f}")

        return {"scores": drift_scores, "reference_window": reference_window}

    def detect_rg_transitions(
        self, md_features: dict[str, np.ndarray], window_size: int = 100
    ) -> dict[str, NDArray]:
        """
        Radius of Gyration変化検出（GPU版）
        """
        # xpの確認
        self._ensure_xp()

        if "radius_of_gyration" not in md_features:
            return {"scores": self.xp.zeros(1), "type": "size_change"}

        rg_gpu = self.to_gpu(md_features["radius_of_gyration"])
        len(rg_gpu)

        # 勾配計算
        rg_gradient = self.xp.gradient(rg_gpu)

        # 局所変化率
        rg_change_rate = self._compute_local_change_rate_gpu(
            rg_gpu, rg_gradient, window_size
        )

        # 収縮を強調（凝集検出用）
        contraction_score = self.xp.where(
            rg_gradient < 0,
            rg_change_rate * 2.0,  # 収縮は2倍
            rg_change_rate,
        )

        return {
            "scores": contraction_score,
            "type": "size_change",
            "raw_gradient": rg_gradient,
        }

    def detect_phase_space_anomalies(
        self, structures: dict[str, np.ndarray], embedding_dim: int = 3, delay: int = 50
    ) -> dict[str, NDArray]:
        """
        位相空間埋め込みによる異常検出（GPU高速化）
        """
        print("\n🔄 Detecting phase space anomalies on GPU...")

        # xpの確認
        self._ensure_xp()

        rho_t_gpu = self.to_gpu(structures["rho_T"])
        n_frames = len(rho_t_gpu)

        # 埋め込み次元の検証
        embed_length = n_frames - (embedding_dim - 1) * delay
        if embed_length <= 0:
            return {"scores": self.xp.zeros(n_frames)}

        # 位相空間構築と異常検出
        anomaly_scores = self._phase_space_analysis_gpu(
            rho_t_gpu, embedding_dim, delay, embed_length
        )

        print(
            f"   Anomaly range: {self.xp.min(anomaly_scores):.2f} to {self.xp.max(anomaly_scores):.2f}"
        )

        return {
            "scores": anomaly_scores,
            "embedding_dim": embedding_dim,
            "delay": delay,
        }

    # === Private GPU methods ===

    def _gaussian_smooth_gpu(self, data: NDArray, sigma: float) -> NDArray:
        """ガウシアンフィルタの簡易実装（GPU/CPU両対応）"""
        # xpの確認
        self._ensure_xp()

        # カーネルサイズ
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # ガウシアンカーネル生成
        x = self.xp.arange(kernel_size) - kernel_size // 2
        kernel = self.xp.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / self.xp.sum(kernel)

        # 畳み込み（簡易版）
        n = len(data)
        smoothed = self.xp.zeros_like(data)
        pad = kernel_size // 2

        # パディング
        padded_data = self.xp.pad(data, pad, mode="edge")

        # 畳み込み演算
        for i in range(n):
            smoothed[i] = self.xp.sum(padded_data[i : i + kernel_size] * kernel)

        return smoothed

    def _analyze_periodicity_gpu(
        self, signal: NDArray, min_period: int, max_period: int, weight: float = 1.0
    ) -> tuple[NDArray, list]:
        """GPU上でFFT解析を実行"""
        # xpの確認
        self._ensure_xp()

        n = len(signal)
        periodic_scores = self.xp.zeros(n)
        detected_periods = []

        # DC成分除去
        signal_centered = signal - self.xp.mean(signal)

        if self.xp.std(signal_centered) < 1e-10:
            return periodic_scores, detected_periods

        # FFT実行
        yf = self.xp.fft.rfft(signal_centered)
        xf = self.xp.fft.rfftfreq(n, 1)
        power = self.xp.abs(yf) ** 2

        # 周波数範囲
        freq_max = 1.0 / min_period
        freq_min = 1.0 / max_period
        valid_mask = (xf > freq_min) & (xf < freq_max) & (xf > 0)

        if not self.xp.any(valid_mask):
            return periodic_scores, detected_periods

        valid_power = power[valid_mask]
        valid_freq = xf[valid_mask]

        # ピーク検出準備
        power_median = self.xp.median(valid_power)
        power_mad = self.xp.median(self.xp.abs(valid_power - power_median))
        power_threshold = power_median + 3 * power_mad

        # CPUに転送してscipy.signal.find_peaksを使用
        if self.is_gpu:
            valid_power_cpu = self.xp.asnumpy(valid_power)
        else:
            valid_power_cpu = valid_power

        # ピーク検出
        peaks, properties = find_peaks(
            valid_power_cpu,
            height=float(
                self.xp.asnumpy(power_threshold) if self.is_gpu else power_threshold
            ),
            distance=5,
            prominence=float(self.xp.asnumpy(power_mad) if self.is_gpu else power_mad),
        )

        # GPU上でピーク処理
        for peak_idx in peaks:
            freq = float(valid_freq[peak_idx])
            period = 1.0 / freq
            amplitude = float(self.xp.sqrt(valid_power[peak_idx]))

            detected_periods.append(
                {
                    "period": period,
                    "frequency": freq,
                    "amplitude": amplitude,
                    "power": float(valid_power[peak_idx]),
                    "snr": float(valid_power[peak_idx] / power_median),
                }
            )

            # 周期的位置にスコア加算
            phase = self.xp.arange(n) * freq * 2 * self.xp.pi
            periodic_contribution = amplitude * self.xp.abs(self.xp.sin(phase))
            periodic_scores += weight * periodic_contribution

        return periodic_scores, detected_periods

    def _extract_trend_gpu(self, signal: NDArray, window: int) -> NDArray:
        """GPUでトレンド抽出"""
        # xpの確認
        self._ensure_xp()

        sigma = window / 3
        return self._gaussian_smooth_gpu(signal, sigma=sigma)

    def _detect_sustained_gradient_gpu(self, gradient: NDArray, window: int) -> NDArray:
        """持続的な勾配の検出"""
        # xpの確認
        self._ensure_xp()

        # 勾配の絶対値を平滑化
        sustained = self._gaussian_smooth_gpu(self.xp.abs(gradient), sigma=window / 6)
        return sustained

    def _compute_sigma_gradient_gpu(self, sigma_s: NDArray) -> NDArray:
        """σsの勾配計算"""
        # xpの確認
        self._ensure_xp()

        # 長期的な変化を捉える
        smoothed = self._gaussian_smooth_gpu(sigma_s, sigma=1000 / 3)
        gradient = self.xp.abs(self.xp.gradient(smoothed))

        if self.xp.std(gradient) > 1e-10:
            normalized = (gradient - self.xp.mean(gradient)) / self.xp.std(gradient)
            return normalized
        else:
            return self.xp.zeros_like(gradient)

    def _compute_drift_scores_gpu(
        self, rho_t: NDArray, q_cumulative: NDArray, ref_window: int, n_frames: int
    ) -> NDArray:
        """ドリフトスコア計算（GPU最適化）"""
        # xpの確認
        self._ensure_xp()

        drift_scores = self.xp.zeros(n_frames)

        # 参照値
        ref_rho_t = self.xp.mean(rho_t[:ref_window])
        ref_q = self.xp.mean(q_cumulative[: min(ref_window, len(q_cumulative))])

        # GPU並列でドリフト計算
        if self.is_gpu and HAS_CUDA:
            threads = 256
            blocks = (n_frames + threads - 1) // threads

            self._drift_kernel[blocks, threads](
                rho_t,
                q_cumulative,
                drift_scores,
                ref_rho_t,
                ref_q,
                ref_window,
                n_frames,
                len(q_cumulative),
            )
        else:
            # CPU版フォールバック
            for idx in range(n_frames):
                start = max(0, idx - ref_window // 2)
                end = min(n_frames, idx + ref_window // 2)

                local_rho_t = self.xp.mean(rho_t[start:end])

                if idx < len(q_cumulative):
                    local_q = q_cumulative[idx]
                else:
                    local_q = q_cumulative[-1]

                rho_t_drift = abs(local_rho_t - ref_rho_t) / (ref_rho_t + 1e-10)
                q_drift = abs(local_q - ref_q) / (abs(ref_q) + 1e-10)

                drift_scores[idx] = rho_t_drift + 0.5 * q_drift

        return drift_scores

    def _compute_local_change_rate_gpu(
        self, rg: NDArray, gradient: NDArray, window: int
    ) -> NDArray:
        """局所変化率の計算"""
        # xpの確認
        self._ensure_xp()

        n = len(rg)
        change_rate = self.xp.zeros(n)

        # GPU並列処理
        if self.is_gpu and HAS_CUDA:
            threads = 256
            blocks = (n + threads - 1) // threads

            self._change_rate_kernel[blocks, threads](
                rg, gradient, change_rate, window, n
            )
        else:
            # CPU版フォールバック
            for idx in range(n):
                start = max(0, idx - window // 2)
                end = min(n, idx + window // 2)

                local_mean = self.xp.mean(rg[start:end])

                if local_mean > 0:
                    change_rate[idx] = abs(gradient[idx]) / local_mean

        return change_rate

    def _phase_space_analysis_gpu(
        self, rho_t: NDArray, embedding_dim: int, delay: int, embed_length: int
    ) -> NDArray:
        """位相空間での異常検出"""
        # xpの確認
        self._ensure_xp()

        n_frames = len(rho_t)
        anomaly_scores = self.xp.zeros(n_frames)

        # 位相空間の構築
        phase_space = self.xp.zeros((embed_length, embedding_dim))
        for i in range(embedding_dim):
            phase_space[:, i] = rho_t[i * delay : i * delay + embed_length]

        # k近傍密度による異常検出
        k = min(20, embed_length - 1)

        # GPU並列で各点の異常度計算
        if self.is_gpu and HAS_CUDA:
            threads = 256
            blocks = (embed_length + threads - 1) // threads

            self._knn_anomaly_kernel[blocks, threads](
                phase_space, anomaly_scores, k, embed_length, embedding_dim, delay
            )
        else:
            # CPU版フォールバック（簡易版）
            for idx in range(embed_length):
                point = phase_space[idx]

                # 全点との距離を計算
                distances = []
                for j in range(embed_length):
                    if j != idx:
                        dist = self.xp.linalg.norm(phase_space[j] - point)
                        distances.append(dist)

                # k近傍の平均距離
                distances = sorted(distances)[:k]
                knn_avg = sum(distances) / k

                # 異常スコア
                score_idx = idx + (embedding_dim - 1) * delay // 2
                if score_idx < n_frames:
                    anomaly_scores[score_idx] = knn_avg

        # 正規化
        if self.xp.std(anomaly_scores) > 1e-10:
            anomaly_scores = (
                anomaly_scores - self.xp.mean(anomaly_scores)
            ) / self.xp.std(anomaly_scores)

        return anomaly_scores

    def _finalize_periodic_scores_gpu(self, scores: NDArray) -> NDArray:
        """周期スコアの最終調整"""
        # xpの確認
        self._ensure_xp()

        if self.xp.max(scores) > 0:
            # 0-1範囲に正規化
            scores = scores / self.xp.max(scores)

            # 外れ値を強調（シグモイド変換）
            mean_score = self.xp.mean(scores)
            std_score = self.xp.std(scores)
            if std_score > 0:
                z_scores = (scores - mean_score) / std_score
                scores = 1 / (1 + self.xp.exp(-z_scores))

        return scores

    # === CUDAカーネル（GPUモードのみ） ===
    # 注：これらのカーネルは is_gpu=True かつ HAS_CUDA=True の時のみ使用される

    if HAS_CUDA:

        @cuda.jit
        def _drift_kernel(
            rho_t,
            q_cumulative,
            drift_scores,
            ref_rho_t,
            ref_q,
            reference_window,
            n_frames,
            q_len,
        ):
            """ドリフト計算カーネル"""
            idx = cuda.grid(1)

            if idx < n_frames:
                # ローカルウィンドウ
                start = max(0, idx - reference_window // 2)
                end = min(n_frames, idx + reference_window // 2)

                # ρTのローカル平均
                local_sum = 0.0
                count = 0
                for i in range(start, end):
                    local_sum += rho_t[i]
                    count += 1

                local_rho_t = local_sum / count if count > 0 else 0

                # Q値
                if idx < q_len:
                    local_q = q_cumulative[idx]
                else:
                    local_q = q_cumulative[q_len - 1]

                # ドリフト計算
                rho_t_drift = abs(local_rho_t - ref_rho_t) / (ref_rho_t + 1e-10)
                q_drift = abs(local_q - ref_q) / (abs(ref_q) + 1e-10)

                drift_scores[idx] = rho_t_drift + 0.5 * q_drift

        @cuda.jit
        def _change_rate_kernel(rg, gradient, change_rate, window, n):
            """変化率計算カーネル"""
            idx = cuda.grid(1)

            if idx < n:
                start = max(0, idx - window // 2)
                end = min(n, idx + window // 2)

                # ローカル平均
                local_sum = 0.0
                count = 0
                for i in range(start, end):
                    local_sum += rg[i]
                    count += 1

                local_mean = local_sum / count if count > 0 else 1.0

                if local_mean > 0:
                    change_rate[idx] = abs(gradient[idx]) / local_mean

        @cuda.jit
        def _knn_anomaly_kernel(
            phase_space, anomaly_scores, k, embed_length, embedding_dim, delay
        ):
            """k近傍異常検出カーネル"""
            idx = cuda.grid(1)

            if idx < embed_length:
                # 現在の点
                point = phase_space[idx]

                # 全点との距離を計算（簡易版）
                distances = cuda.local.array(100, dtype=cuda.float32)  # 最大100点
                n_dists = min(100, embed_length)

                for j in range(n_dists):
                    if j != idx:
                        dist = 0.0
                        for d in range(embedding_dim):
                            diff = phase_space[j, d] - point[d]
                            dist += diff * diff
                        distances[j] = cuda.sqrt(dist)
                    else:
                        distances[j] = 1e10  # 自分自身は除外

                # k近傍の平均距離（簡易ソート）
                knn_sum = 0.0
                for _ in range(k):
                    min_idx = 0
                    min_dist = distances[0]
                    for j in range(1, n_dists):
                        if distances[j] < min_dist:
                            min_dist = distances[j]
                            min_idx = j
                    knn_sum += min_dist
                    distances[min_idx] = 1e10  # 使用済み

                knn_avg = knn_sum / k

                # 異常スコア（距離が大きいほど異常）
                score_idx = idx + (embedding_dim - 1) * delay // 2
                if score_idx < len(anomaly_scores):
                    anomaly_scores[score_idx] = knn_avg
