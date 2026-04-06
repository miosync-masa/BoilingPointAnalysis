"""
Network Analyzer Core — Λ³ Native Engine
==========================================
Built by Masamichi & Tamaki

Λ³ネイティブなネットワーク解析エンジン。
偏相関ベースの旧エンジンを完全に置き換え、
ΔΛCイベント伝播確率によるsync/causal検出を行う。

Core mechanism:
  Sync  : ΔΛC co-firing rate (lag=0) + ρT correlation
  Causal: P(ΔΛC_j(t+k) | ΔΛC_i(t))  lagged propagation probability
  Filter: conditional propagation probability (common ancestor removal)
  Test  : permutation test for statistical significance

理論的根拠:
  旧エンジン (偏相関) は生の値の統計的関係を見ており、Λ³と無関係だった。
  新エンジンは ΔΛC 構造変化イベント同士の伝播を直接検出するため、
  Λ³理論とネイティブに接続されている。
"""

import logging
from dataclasses import dataclass, field

import numpy as np

# GPU 4-axis verification (optional)
try:
    from ..core.gpu_inverse import network_verify_all

    _HAS_GPU_VERIFY = True
except ImportError:
    _HAS_GPU_VERIFY = False

logger = logging.getLogger("getter_one.analysis.network_analyzer_core")


# ============================================
# Data Classes (既存インターフェース維持)
# ============================================


@dataclass
class DimensionLink:
    """A network link between two dimensions."""

    from_dim: int
    to_dim: int
    from_name: str
    to_name: str
    link_type: str  # 'sync' or 'causal'
    strength: float  # primary metric (co-firing rate or propagation prob)
    correlation: float  # signed correlation or directional indicator
    lag: int = 0  # lag in frames for causal links
    p_value: float = 1.0  # permutation test p-value
    rho_t_corr: float = 0.0  # ρT correlation (supplementary)
    event_quality: float = 0.0  # GPU 4-axis event genuineness (0〜1)


@dataclass
class NetworkResult:
    """Container for full network analysis results."""

    sync_network: list[DimensionLink] = field(default_factory=list)
    causal_network: list[DimensionLink] = field(default_factory=list)

    # Raw matrices
    sync_matrix: np.ndarray | None = None  # (n_dims, n_dims) co-firing rate
    causal_matrix: np.ndarray | None = None  # (n_dims, n_dims) max propagation prob
    causal_lag_matrix: np.ndarray | None = None  # (n_dims, n_dims) optimal lag
    rho_t_corr_matrix: np.ndarray | None = None  # (n_dims, n_dims) ρT correlation

    # Network-level characteristics
    pattern: str = "unknown"
    hub_dimensions: list[int] = field(default_factory=list)
    hub_names: list[str] = field(default_factory=list)

    # Causal structure summary
    causal_drivers: list[int] = field(default_factory=list)
    causal_followers: list[int] = field(default_factory=list)
    driver_names: list[str] = field(default_factory=list)
    follower_names: list[str] = field(default_factory=list)

    # Metadata
    n_dims: int = 0
    n_sync_links: int = 0
    n_causal_links: int = 0
    dimension_names: list[str] = field(default_factory=list)

    # Λ³ event statistics
    event_counts: dict = field(default_factory=dict)

    # GPU 4-axis event quality (per-dimension genuineness)
    event_quality_scores: dict = field(default_factory=dict)

    # Adaptive parameters
    adaptive_params: dict | None = None


@dataclass
class CooperativeEventNetwork:
    """Local network structure around a cooperative event."""

    event_frame: int
    event_timestamp: str | None = None
    delta_lambda_c: float = 0.0

    network: NetworkResult | None = None

    initiator_dims: list[int] = field(default_factory=list)
    initiator_names: list[str] = field(default_factory=list)
    propagation_order: list[int] = field(default_factory=list)


# ============================================
# 1D Λ³ Feature Functions
# ============================================


def _calculate_local_std_1d(data: np.ndarray, window: int) -> np.ndarray:
    """局所標準偏差（対称窓）— 無次元化の分母"""
    n = len(data)
    local_std = np.zeros(n)
    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        subset = data[start:end]
        if len(subset) > 0:
            local_std[i] = np.std(subset)
    return local_std


def _calculate_rho_t_1d(data: np.ndarray, window: int) -> np.ndarray:
    """1次元テンション密度 (ρT) — 過去窓の局所標準偏差"""
    n = len(data)
    rho_t = np.zeros(n)
    for i in range(n):
        start = max(0, i - window)
        end = i + 1
        subset = data[start:end]
        if len(subset) > 1:
            rho_t[i] = np.std(subset)
    return rho_t


# ============================================
# NetworkAnalyzerCore — Λ³ Native
# ============================================


class NetworkAnalyzerCore:
    """
    Λ³ネイティブ ネットワーク解析エンジン

    ΔΛCイベント伝播確率に基づくsync/causal検出を行う。
    偏相関は一切使用しない。

    Parameters
    ----------
    sync_threshold : float
        ΔΛC同時発火率の閾値（これ以上でsyncリンク候補）
    causal_threshold : float
        ΔΛC伝播確率の閾値（これ以上でcausalリンク候補）
    max_lag : int
        因果推定の最大ラグ
    delta_percentile : float
        ΔΛCイベント検出のパーセンタイル閾値
    local_std_window : int
        無次元化用の局所標準偏差ウィンドウ
    rho_t_window : int
        テンション密度計算のウィンドウ
    n_permutations : int
        有意性検定のpermutation回数
    p_value_threshold : float
        有意性の閾値
    rho_t_weight : float
        sync判定におけるρT相関の重み (0〜1)
    """

    def __init__(
        self,
        sync_threshold: float = 0.05,
        causal_threshold: float = 0.08,
        max_lag: int = 10,
        delta_percentile: float = 94.0,
        local_std_window: int = 20,
        rho_t_window: int = 30,
        n_permutations: int = 200,
        p_value_threshold: float = 0.05,
        rho_t_weight: float = 0.3,
        adaptive: bool = True,
        enable_gpu_verification: bool = False,
    ):
        # ユーザー指定値（adaptiveモードではヒントとして使用）
        self.sync_threshold_hint = sync_threshold
        self.causal_threshold_hint = causal_threshold
        self.max_lag_hint = max_lag
        self.delta_percentile_hint = delta_percentile
        self.local_std_window_hint = local_std_window
        self.rho_t_window_hint = rho_t_window

        # ランタイム値（adaptive計算後に上書き）
        self.sync_threshold = sync_threshold
        self.causal_threshold = causal_threshold
        self.max_lag = max_lag
        self.delta_percentile = delta_percentile
        self.local_std_window = local_std_window
        self.rho_t_window = rho_t_window

        self.n_permutations = n_permutations
        self.p_value_threshold = p_value_threshold
        self.rho_t_weight = rho_t_weight
        self.adaptive = adaptive
        self.enable_gpu_verification = enable_gpu_verification and _HAS_GPU_VERIFY

        # Adaptive計算結果の保存先
        self.adaptive_params: dict | None = None

        gpu_status = "enabled" if self.enable_gpu_verification else "disabled"
        adaptive_status = "ON" if adaptive else "OFF"
        logger.info(
            f"✅ NetworkAnalyzerCore [Λ³ Native] initialized "
            f"(sync>{sync_threshold}, causal>{causal_threshold}, "
            f"max_lag={max_lag}, adaptive={adaptive_status}, "
            f"n_perm={n_permutations}, p<{p_value_threshold}, "
            f"gpu_verify={gpu_status})"
        )

    # ================================================================
    # Main Entry Point
    # ================================================================

    def analyze(
        self,
        state_vectors: np.ndarray,
        dimension_names: list[str] | None = None,
        window: int | None = None,
    ) -> NetworkResult:
        """
        Λ³ネイティブなネットワーク解析を実行。

        Parameters
        ----------
        state_vectors : np.ndarray of shape (n_frames, n_dims)
            入力時系列データ
        dimension_names : list[str], optional
            各次元の名前
        window : int, optional
            解析ウィンドウ長。Noneの場合は全フレーム使用。

        Returns
        -------
        NetworkResult
        """
        n_frames, n_dims = state_vectors.shape

        if dimension_names is None:
            dimension_names = [f"dim_{i}" for i in range(n_dims)]

        if window is not None:
            state_vectors = state_vectors[:window]
            n_frames = len(state_vectors)

        logger.info(f"🔍 Λ³ Native Network Analysis ({n_dims} dims, {n_frames} frames)")

        # 1. Λ³特徴量抽出（各次元独立）
        events_pos, events_neg, rho_t, local_lambda_f, local_std = (
            self._extract_lambda3_events(state_vectors, n_frames, n_dims)
        )

        event_counts = {
            d: int(np.sum(events_pos[:, d]) + np.sum(events_neg[:, d]))
            for d in range(n_dims)
        }
        logger.info(
            "   ΔΛC events: "
            + ", ".join(
                f"{dimension_names[d]}={event_counts[d]}" for d in range(min(n_dims, 5))
            )
            + ("..." if n_dims > 5 else "")
        )

        # 1.5 Adaptive parameter tuning
        if self.adaptive:
            self.adaptive_params = self._compute_adaptive_parameters(
                state_vectors, events_pos, events_neg, rho_t, n_frames, n_dims
            )
            self.sync_threshold = self.adaptive_params["sync_threshold"]
            self.causal_threshold = self.adaptive_params["causal_threshold"]
            self.max_lag = self.adaptive_params["max_lag"]
            self.local_std_window = self.adaptive_params["local_std_window"]
            self.rho_t_window = self.adaptive_params["rho_t_window"]
            self.rho_t_weight = self.adaptive_params["rho_t_weight"]

        # 2. Sync行列（ΔΛC同時発火率 + ρT相関）
        sync_matrix, rho_t_corr_matrix = self._compute_sync_matrix(
            events_pos, events_neg, rho_t, n_dims
        )

        # 3. Causal行列（ΔΛCラグ付き伝播確率）
        causal_matrix, causal_lag_matrix = self._compute_causal_matrix(
            events_pos, events_neg, n_dims
        )

        # 4. Permutation test で有意性判定
        sync_pvalues = self._permutation_test_sync(
            events_pos, events_neg, rho_t, sync_matrix, n_dims
        )
        causal_pvalues = self._permutation_test_causal(
            events_pos, events_neg, causal_matrix, causal_lag_matrix, n_dims
        )

        # 5. ネットワーク構築
        sync_links, causal_links = self._build_networks(
            sync_matrix,
            causal_matrix,
            causal_lag_matrix,
            rho_t_corr_matrix,
            sync_pvalues,
            causal_pvalues,
            dimension_names,
            n_dims,
        )

        # 6. 共通祖先フィルタ（Λ³条件付き伝播確率）
        causal_links = self._filter_common_ancestor(
            causal_links, events_pos, events_neg, n_dims
        )

        # 6.5 GPU 4軸イベント品質検証（オプション）
        event_quality_scores = {}
        if self.enable_gpu_verification:
            event_quality_scores = self._gpu_event_quality(
                state_vectors,
                local_lambda_f,
                rho_t,
                local_std,
                events_pos,
                events_neg,
                sync_links,
                causal_links,
                n_dims,
            )

        # 7. パターン・ハブ・因果構造
        pattern = self._identify_pattern(sync_links, causal_links)
        hub_dims = self._detect_hubs(sync_links, causal_links, n_dims)
        drivers, followers = self._identify_causal_structure(causal_links, n_dims)

        result = NetworkResult(
            sync_network=sync_links,
            causal_network=causal_links,
            sync_matrix=sync_matrix,
            causal_matrix=causal_matrix,
            causal_lag_matrix=causal_lag_matrix,
            rho_t_corr_matrix=rho_t_corr_matrix,
            pattern=pattern,
            hub_dimensions=hub_dims,
            hub_names=[dimension_names[d] for d in hub_dims],
            causal_drivers=drivers,
            causal_followers=followers,
            driver_names=[dimension_names[d] for d in drivers],
            follower_names=[dimension_names[d] for d in followers],
            n_dims=n_dims,
            n_sync_links=len(sync_links),
            n_causal_links=len(causal_links),
            dimension_names=dimension_names,
            event_counts=event_counts,
            event_quality_scores=event_quality_scores,
            adaptive_params=self.adaptive_params,
        )

        self._print_summary(result)
        return result

    def analyze_event_network(
        self,
        state_vectors: np.ndarray,
        event_frame: int,
        window_before: int = 24,
        window_after: int = 6,
        dimension_names: list[str] | None = None,
    ) -> CooperativeEventNetwork:
        """ΔΛCイベント周辺の局所ネットワーク解析"""
        n_frames, n_dims = state_vectors.shape

        if dimension_names is None:
            dimension_names = [f"dim_{i}" for i in range(n_dims)]

        start = max(0, event_frame - window_before)
        end = min(n_frames, event_frame + window_after)
        local_data = state_vectors[start:end]

        network = self.analyze(local_data, dimension_names)

        initiators = self._identify_initiators(
            state_vectors, event_frame, window_before, n_dims
        )
        propagation = self._estimate_propagation_order(
            state_vectors, event_frame, window_before, n_dims
        )

        return CooperativeEventNetwork(
            event_frame=event_frame,
            network=network,
            initiator_dims=initiators,
            initiator_names=[dimension_names[d] for d in initiators],
            propagation_order=propagation,
        )

    # ================================================================
    # Step 1.5: Adaptive Parameter Computation
    # ================================================================

    def _compute_adaptive_parameters(
        self,
        state_vectors: np.ndarray,
        events_pos: np.ndarray,
        events_neg: np.ndarray,
        rho_t: np.ndarray,
        n_frames: int,
        n_dims: int,
    ) -> dict:
        """
        データ特性とΔΛCイベント統計に基づくパラメータ自動調整。

        調整対象:
          sync_threshold, causal_threshold, max_lag,
          local_std_window, rho_t_window, rho_t_weight
        """
        # --- 1. イベント密度 ---
        events_all = np.minimum(events_pos + events_neg, 1.0)
        event_density = np.mean(events_all)  # per dim per frame

        # --- 2. 次元間イベント同時発火率 ---
        cofiring_rates = []
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                cofiring = np.mean(events_all[:, i] * events_all[:, j])
                cofiring_rates.append(cofiring)
        mean_cofiring = float(np.mean(cofiring_rates)) if cofiring_rates else 0.0

        # --- 3. ρT variability ---
        rho_t_means = np.mean(rho_t, axis=0)
        rho_t_overall_mean = np.mean(rho_t_means)
        rho_t_cv = (
            float(np.std(rho_t_means) / (rho_t_overall_mean + 1e-10))
            if rho_t_overall_mean > 1e-10
            else 0.0
        )

        # --- 4. Temporal volatility ---
        temporal_changes = np.diff(state_vectors, axis=0)
        temporal_vol = float(np.mean(np.std(temporal_changes, axis=0)))
        global_std = float(np.std(state_vectors))
        vol_ratio = temporal_vol / (global_std + 1e-10)

        # === Adaptive sync_threshold ===
        sync_scale = 1.0
        # イベント密度が高い → 閾値を上げる
        if event_density > 0.12:
            sync_scale *= 1.3
        elif event_density < 0.04:
            sync_scale *= 0.8
        # 同時発火率が高い → 全部syncっぽい → 閾値上げる
        if mean_cofiring > 0.02:
            sync_scale *= 1.2
        # 次元数多い → 伝播確率が希薄化 → 閾値を下げる
        if n_dims > 10:
            sync_scale *= 0.85
        elif n_dims > 5:
            sync_scale *= 0.9

        sync_threshold = float(
            np.clip(self.sync_threshold_hint * sync_scale, 0.01, 0.30)
        )

        # === Adaptive causal_threshold ===
        causal_scale = 1.0
        if event_density > 0.12:
            causal_scale *= 1.4
        elif event_density < 0.04:
            causal_scale *= 0.7
        # 次元数多い → ペアあたりのイベント共起は減る → 閾値を下げる
        if n_dims > 10:
            causal_scale *= 0.8
        elif n_dims > 5:
            causal_scale *= 0.85
        # ρTの変動が大きい → 構造変化が多い → 少し上げる
        if rho_t_cv > 1.0:
            causal_scale *= 1.1

        causal_threshold = float(
            np.clip(self.causal_threshold_hint * causal_scale, 0.02, 0.40)
        )

        # === Adaptive max_lag ===
        # データ長に応じたスケーリング
        lag_scale = 1.0
        if n_frames < 200:
            lag_scale *= 0.7
        elif n_frames > 1000:
            lag_scale *= 1.3
        # 高volatility → 短いラグで伝播 → ラグ短縮
        if vol_ratio > 1.5:
            lag_scale *= 0.8
        elif vol_ratio < 0.3:
            lag_scale *= 1.2

        max_lag = int(np.clip(self.max_lag_hint * lag_scale, 3, max(5, n_frames // 10)))

        # === Adaptive windows ===
        window_scale = 1.0
        if n_frames < 200:
            window_scale *= 0.7
        elif n_frames > 1000:
            window_scale *= 1.3
        if vol_ratio > 1.5:
            window_scale *= 0.8

        local_std_window = int(
            np.clip(self.local_std_window_hint * window_scale, 5, n_frames // 5)
        )
        rho_t_window = int(
            np.clip(self.rho_t_window_hint * window_scale, 5, n_frames // 5)
        )

        # === Adaptive rho_t_weight ===
        # ρTの変動が大きい → ρTが情報を持ってる → weight上げる
        rho_t_weight = float(
            np.clip(self.rho_t_weight + (rho_t_cv - 0.5) * 0.2, 0.1, 0.5)
        )

        params = {
            "sync_threshold": sync_threshold,
            "causal_threshold": causal_threshold,
            "max_lag": max_lag,
            "local_std_window": local_std_window,
            "rho_t_window": rho_t_window,
            "rho_t_weight": rho_t_weight,
            "diagnostics": {
                "event_density": float(event_density),
                "mean_cofiring": mean_cofiring,
                "rho_t_cv": rho_t_cv,
                "vol_ratio": float(vol_ratio),
                "n_frames": n_frames,
                "n_dims": n_dims,
            },
        }

        logger.info(
            f"   🔧 Adaptive: sync={sync_threshold:.3f} "
            f"(hint={self.sync_threshold_hint}), "
            f"causal={causal_threshold:.3f} "
            f"(hint={self.causal_threshold_hint}), "
            f"max_lag={max_lag} (hint={self.max_lag_hint})"
        )
        logger.info(
            f"      event_density={event_density:.3f}, "
            f"cofiring={mean_cofiring:.4f}, "
            f"ρT_cv={rho_t_cv:.3f}, "
            f"vol_ratio={vol_ratio:.3f}"
        )

        return params

    # ================================================================
    # Step 1: Λ³ Event Extraction
    # ================================================================

    def _extract_lambda3_events(
        self,
        state_vectors: np.ndarray,
        n_frames: int,
        n_dims: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        各次元からΔΛCイベント（pos/neg）とρTを抽出。

        DualCoreの _compute_local と同じロジック:
          diff → local_std で無次元化 → percentile閾値 → バイナリイベント

        Returns
        -------
        events_pos    : (n_frames-1, n_dims)  正方向ΔΛC発火
        events_neg    : (n_frames-1, n_dims)  負方向ΔΛC発火
        rho_t         : (n_frames, n_dims)    テンション密度
        local_lambda_f: (n_frames-1, n_dims)  ジャンプ検出済みdiff（GPU検証用）
        local_std     : (n_frames, n_dims)    局所標準偏差（GPU検証用）
        """
        n_diff = n_frames - 1
        events_pos = np.zeros((n_diff, n_dims))
        events_neg = np.zeros((n_diff, n_dims))
        rho_t = np.zeros((n_frames, n_dims))
        local_lambda_f = np.zeros((n_diff, n_dims))
        local_std_all = np.zeros((n_frames, n_dims))

        for d in range(n_dims):
            series = state_vectors[:, d]

            # diff → 無次元化 → 閾値 → バイナリ
            diff = np.diff(series)
            lstd = _calculate_local_std_1d(series, self.local_std_window)
            lstd_diff = lstd[1:]
            score = np.abs(diff) / (lstd_diff + 1e-10)
            threshold = np.percentile(score, self.delta_percentile)

            jump_mask = score > threshold
            events_pos[:, d] = ((diff > 0) & jump_mask).astype(float)
            events_neg[:, d] = ((diff < 0) & jump_mask).astype(float)

            # DualCore互換: diff * mask
            local_lambda_f[:, d] = diff * jump_mask

            # ρT
            rho_t[:, d] = _calculate_rho_t_1d(series, self.rho_t_window)

            # local_std
            local_std_all[:, d] = lstd

        return events_pos, events_neg, rho_t, local_lambda_f, local_std_all

    # ================================================================
    # Step 2: Sync Matrix (ΔΛC co-firing + ρT correlation)
    # ================================================================

    def _compute_sync_matrix(
        self,
        events_pos: np.ndarray,
        events_neg: np.ndarray,
        rho_t: np.ndarray,
        n_dims: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        ΔΛC同時発火率 + ρT相関のブレンドによるsync行列

        sync_score = (1 - w) * cofiring_rate + w * |ρT_corr|
        """
        # 全イベント（pos + neg 合算）
        events_all = np.minimum(events_pos + events_neg, 1.0)

        cofiring_matrix = np.zeros((n_dims, n_dims))
        rho_t_corr_matrix = np.zeros((n_dims, n_dims))
        sync_matrix = np.zeros((n_dims, n_dims))

        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                # ΔΛC co-firing rate
                cofiring = np.mean(events_all[:, i] * events_all[:, j])
                cofiring_matrix[i, j] = cofiring
                cofiring_matrix[j, i] = cofiring

                # ρT correlation
                rt_i = rho_t[:, i]
                rt_j = rho_t[:, j]
                if np.std(rt_i) > 1e-10 and np.std(rt_j) > 1e-10:
                    corr = np.corrcoef(rt_i, rt_j)[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                else:
                    corr = 0.0
                rho_t_corr_matrix[i, j] = corr
                rho_t_corr_matrix[j, i] = corr

                # Blended sync score
                w = self.rho_t_weight
                score = (1 - w) * cofiring + w * abs(corr)
                sync_matrix[i, j] = score
                sync_matrix[j, i] = score

        return sync_matrix, rho_t_corr_matrix

    # ================================================================
    # Step 3: Causal Matrix (ΔΛC lagged propagation probability)
    # ================================================================

    def _compute_causal_matrix(
        self,
        events_pos: np.ndarray,
        events_neg: np.ndarray,
        n_dims: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        ΔΛCラグ付き伝播確率行列。

        P(ΔΛC_j(t+k) | ΔΛC_i(t)) を全ペア・全ラグで計算し、
        最大伝播確率とその最適ラグを記録する。

        pos→pos, pos→neg, neg→pos, neg→neg の4パターンを
        統合して最大を取る。
        """
        causal_matrix = np.zeros((n_dims, n_dims))
        lag_matrix = np.zeros((n_dims, n_dims), dtype=int)

        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue

                best_prob = 0.0
                best_lag = 0

                # 4パターンの伝播確率を計算
                for cause_ev, effect_ev in [
                    (events_pos[:, i], events_pos[:, j]),
                    (events_pos[:, i], events_neg[:, j]),
                    (events_neg[:, i], events_pos[:, j]),
                    (events_neg[:, i], events_neg[:, j]),
                ]:
                    for lag in range(1, self.max_lag + 1):
                        prob = self._propagation_probability(cause_ev, effect_ev, lag)
                        if prob > best_prob:
                            best_prob = prob
                            best_lag = lag

                causal_matrix[i, j] = best_prob
                lag_matrix[i, j] = best_lag

        return causal_matrix, lag_matrix

    @staticmethod
    def _propagation_probability(
        cause: np.ndarray,
        effect: np.ndarray,
        lag: int,
    ) -> float:
        """
        P(effect(t+lag) = 1 | cause(t) = 1)

        条件付き確率 = joint / marginal
        """
        if lag >= len(cause):
            return 0.0

        cause_past = cause[:-lag]
        effect_future = effect[lag:]

        n_cause = np.sum(cause_past)
        if n_cause < 1:
            return 0.0

        joint = np.sum(cause_past * effect_future)
        return joint / n_cause

    # ================================================================
    # Step 4: Permutation Tests
    # ================================================================

    def _permutation_test_sync(
        self,
        events_pos: np.ndarray,
        events_neg: np.ndarray,
        rho_t: np.ndarray,
        observed_sync: np.ndarray,
        n_dims: int,
    ) -> np.ndarray:
        """sync行列に対するpermutation test → p-value行列"""
        pvalues = np.ones((n_dims, n_dims))
        events_all = np.minimum(events_pos + events_neg, 1.0)

        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                obs = observed_sync[i, j]
                if obs < 1e-10:
                    continue

                count_ge = 0
                for _ in range(self.n_permutations):
                    # 片方のイベント系列をシャッフル
                    shuffled = events_all[:, j].copy()
                    np.random.shuffle(shuffled)

                    cofiring = np.mean(events_all[:, i] * shuffled)

                    # ρTもシャッフル
                    shuffled_rt = rho_t[:, j].copy()
                    np.random.shuffle(shuffled_rt)
                    rt_i = rho_t[:, i]
                    if np.std(rt_i) > 1e-10 and np.std(shuffled_rt) > 1e-10:
                        corr = np.corrcoef(rt_i, shuffled_rt)[0, 1]
                        if np.isnan(corr):
                            corr = 0.0
                    else:
                        corr = 0.0

                    w = self.rho_t_weight
                    null_score = (1 - w) * cofiring + w * abs(corr)

                    if null_score >= obs:
                        count_ge += 1

                p = (count_ge + 1) / (self.n_permutations + 1)
                pvalues[i, j] = p
                pvalues[j, i] = p

        return pvalues

    def _permutation_test_causal(
        self,
        events_pos: np.ndarray,
        events_neg: np.ndarray,
        observed_causal: np.ndarray,
        observed_lag: np.ndarray,
        n_dims: int,
    ) -> np.ndarray:
        """causal行列に対するpermutation test → p-value行列"""
        pvalues = np.ones((n_dims, n_dims))

        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue

                obs = observed_causal[i, j]
                if obs < 1e-10:
                    continue

                count_ge = 0
                for _ in range(self.n_permutations):
                    # effect側のイベント系列をシャッフル
                    best_null = 0.0
                    for cause_ev, effect_ev_orig in [
                        (events_pos[:, i], events_pos[:, j]),
                        (events_pos[:, i], events_neg[:, j]),
                        (events_neg[:, i], events_pos[:, j]),
                        (events_neg[:, i], events_neg[:, j]),
                    ]:
                        shuffled = effect_ev_orig.copy()
                        np.random.shuffle(shuffled)
                        for lag in range(1, self.max_lag + 1):
                            prob = self._propagation_probability(
                                cause_ev, shuffled, lag
                            )
                            if prob > best_null:
                                best_null = prob

                    if best_null >= obs:
                        count_ge += 1

                pvalues[i, j] = (count_ge + 1) / (self.n_permutations + 1)

        return pvalues

    # ================================================================
    # Step 5: Network Construction
    # ================================================================

    def _build_networks(
        self,
        sync_matrix: np.ndarray,
        causal_matrix: np.ndarray,
        causal_lag_matrix: np.ndarray,
        rho_t_corr_matrix: np.ndarray,
        sync_pvalues: np.ndarray,
        causal_pvalues: np.ndarray,
        dimension_names: list[str],
        n_dims: int,
    ) -> tuple[list[DimensionLink], list[DimensionLink]]:
        """閾値 + p-value でフィルタしてリンク構築"""
        sync_links = []
        causal_links = []

        # Sync links
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                score = sync_matrix[i, j]
                p = sync_pvalues[i, j]

                if score > self.sync_threshold and p < self.p_value_threshold:
                    sync_links.append(
                        DimensionLink(
                            from_dim=i,
                            to_dim=j,
                            from_name=dimension_names[i],
                            to_name=dimension_names[j],
                            link_type="sync",
                            strength=score,
                            correlation=rho_t_corr_matrix[i, j],
                            p_value=p,
                            rho_t_corr=rho_t_corr_matrix[i, j],
                        )
                    )

        # Causal links
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue

                prob = causal_matrix[i, j]
                lag = causal_lag_matrix[i, j]
                p = causal_pvalues[i, j]

                if prob > self.causal_threshold and p < self.p_value_threshold:
                    # causal > sync * 1.1 で方向性を確認
                    sync_score = sync_matrix[min(i, j), max(i, j)]
                    if prob > sync_score * 1.1 or sync_score < self.sync_threshold:
                        causal_links.append(
                            DimensionLink(
                                from_dim=i,
                                to_dim=j,
                                from_name=dimension_names[i],
                                to_name=dimension_names[j],
                                link_type="causal",
                                strength=prob,
                                correlation=1.0,  # 方向: i→j
                                lag=int(lag),
                                p_value=p,
                                rho_t_corr=rho_t_corr_matrix[min(i, j), max(i, j)],
                            )
                        )

        return sync_links, causal_links

    # ================================================================
    # Step 6: Common Ancestor Filter (Λ³ Conditional Propagation)
    # ================================================================

    def _filter_common_ancestor(
        self,
        causal_links: list[DimensionLink],
        events_pos: np.ndarray,
        events_neg: np.ndarray,
        n_dims: int,
    ) -> list[DimensionLink]:
        """
        スプリアスエッジ除去 — 共通祖先 + 媒介者の2パターン

        Pattern 1: Common Ancestor (既存)
          Z→A, Z→B が存在 → A→B はスプリアス
          判定: P(B|A, Z未発火) が大幅低下 → Z経由の偽因果

        Pattern 2: Mediator (新規)
          A→M, M→C が存在 → A→C は媒介された偽因果
          判定: P(C|A, M未発火) が大幅低下 → Mなしでは伝わらない
        """
        if len(causal_links) < 2 or n_dims < 3:
            return causal_links

        # 全イベント合算
        events_all = np.minimum(events_pos + events_neg, 1.0)

        # リンクマップ構築
        link_map: dict[tuple[int, int], DimensionLink] = {}
        for link in causal_links:
            key = (link.from_dim, link.to_dim)
            if key not in link_map or link.strength > link_map[key].strength:
                link_map[key] = link

        filtered = []
        n_ancestor = 0
        n_mediator = 0

        for link in causal_links:
            a, b = link.from_dim, link.to_dim
            lag_ab = link.lag
            removal_reason = None

            # --- Pattern 1: Common Ancestor ---
            # Z→A かつ Z→B が存在する次元を探す
            for z in range(n_dims):
                if z in (a, b):
                    continue

                z_to_a = link_map.get((z, a))
                z_to_b = link_map.get((z, b))

                if z_to_a is None or z_to_b is None:
                    continue

                # ラグ整合性チェック:
                # Z→A(lag_za) + A→B(lag_ab) ≈ Z→B(lag_zb) なら推移的偽因果
                lag_za = z_to_a.lag
                lag_zb = z_to_b.lag
                lag_consistent = abs((lag_za + lag_ab) - lag_zb) <= max(1, lag_zb // 3)

                # 条件付き確率チェック
                prob_with_z, prob_without_z = self._conditional_propagation(
                    events_all, a, b, z, lag_ab
                )

                # Z発火時のA発火数（検体数チェック）
                a_events_vec = (
                    events_all[:-lag_ab, a] if lag_ab > 0 else events_all[:, a]
                )
                z_active = np.zeros(len(a_events_vec))
                for t in range(len(a_events_vec)):
                    z_start = max(0, t - 2)
                    z_end = min(len(events_all), t + 3)
                    if np.any(events_all[z_start:z_end, z] > 0):
                        z_active[t] = 1.0
                n_a_without_z = np.sum((a_events_vec > 0) & (z_active == 0))
                insufficient_samples = n_a_without_z < 3

                # 判定: 条件付き確率で怪しい OR 検体不足
                prob_suspicious = (
                    prob_without_z < link.strength * 0.5
                    and prob_with_z > prob_without_z * 1.5
                )

                # 両方の証拠が揃ったとき削除:
                # (確率的に怪しい OR 検体不足) AND ラグ整合
                if (prob_suspicious or insufficient_samples) and lag_consistent:
                    removal_reason = "ancestor"
                    logger.debug(
                        f"   🔍 Spurious (ancestor+lag): "
                        f"{link.from_name}→{link.to_name} "
                        f"(Z={z_to_a.from_name}, "
                        f"lag_za={lag_za}+lag_ab={lag_ab}≈lag_zb={lag_zb}, "
                        f"P_with={prob_with_z:.3f}, "
                        f"P_without={prob_without_z:.3f}, "
                        f"n_without_z={int(n_a_without_z)})"
                    )
                    break

            # --- Pattern 2: Mediator ---
            # A→M かつ M→B が存在する次元を探す
            if removal_reason is None:
                for m in range(n_dims):
                    if m in (a, b):
                        continue

                    a_to_m = link_map.get((a, m))
                    m_to_b = link_map.get((m, b))

                    if a_to_m is None or m_to_b is None:
                        continue

                    # ラグ整合性: A→M(lag1) + M→B(lag2) ≈ A→B(lag_ab)
                    # 媒介経路のラグ合計が直接リンクのラグと近い場合のみ
                    mediated_lag = a_to_m.lag + m_to_b.lag
                    if abs(mediated_lag - lag_ab) > max(2, lag_ab // 2):
                        continue

                    # M未発火時のA→B伝播確率
                    prob_with_m, prob_without_m = self._conditional_propagation(
                        events_all, a, b, m, lag_ab
                    )

                    # Mなしでは伝わらない → 媒介された偽因果
                    if (
                        prob_without_m < link.strength * 0.4
                        and prob_with_m > prob_without_m * 2.0
                    ):
                        removal_reason = "mediator"
                        logger.debug(
                            f"   🔍 Spurious (mediator): "
                            f"{link.from_name}→{link.to_name} "
                            f"(M={link_map[(a, m)].to_name}, "
                            f"path: {link.from_name}→{link_map[(a, m)].to_name}"
                            f"→{link.to_name}, "
                            f"P_with={prob_with_m:.3f}, "
                            f"P_without={prob_without_m:.3f})"
                        )
                        break

            if removal_reason == "ancestor":
                n_ancestor += 1
            elif removal_reason == "mediator":
                n_mediator += 1
            else:
                filtered.append(link)

        n_total = n_ancestor + n_mediator
        if n_total > 0:
            logger.info(
                f"   🔍 Spurious filter: {n_total} removed "
                f"(ancestor={n_ancestor}, mediator={n_mediator}), "
                f"{len(filtered)} retained"
            )

        return filtered

    @staticmethod
    def _conditional_propagation(
        events_all: np.ndarray,
        a: int,
        b: int,
        z: int,
        lag: int,
    ) -> tuple[float, float]:
        """
        Z発火条件付きのA→B伝播確率

        Returns
        -------
        prob_with_z : Z発火時のP(B(t+lag)|A(t))
        prob_without_z : Z未発火時のP(B(t+lag)|A(t))
        """
        if lag >= len(events_all):
            return 0.0, 0.0

        a_events = events_all[:-lag, a]
        b_events = events_all[lag:, b]

        # Zの発火状態（Aと同時点 ± 数フレーム内で発火していたか）
        z_active = np.zeros(len(a_events))
        for t in range(len(a_events)):
            z_start = max(0, t - 2)
            z_end = min(len(events_all), t + 3)
            if np.any(events_all[z_start:z_end, z] > 0):
                z_active[t] = 1.0

        # A発火 かつ Z発火 のケース
        mask_with = (a_events > 0) & (z_active > 0)
        n_with = np.sum(mask_with)
        if n_with > 0:
            prob_with = np.sum(mask_with * b_events) / n_with
        else:
            prob_with = 0.0

        # A発火 かつ Z未発火 のケース
        mask_without = (a_events > 0) & (z_active == 0)
        n_without = np.sum(mask_without)
        if n_without > 0:
            prob_without = np.sum(mask_without * b_events) / n_without
        else:
            prob_without = 0.0

        return float(prob_with), float(prob_without)

    # ================================================================
    # Step 6.5: GPU 4-Axis Event Quality Verification
    # ================================================================

    def _gpu_event_quality(
        self,
        state_vectors: np.ndarray,
        local_lambda_f: np.ndarray,
        rho_t: np.ndarray,
        local_std: np.ndarray,
        events_pos: np.ndarray,
        events_neg: np.ndarray,
        sync_links: list[DimensionLink],
        causal_links: list[DimensionLink],
        n_dims: int,
    ) -> dict:
        """
        GPU 4軸（surrogate, ρT整合, 持続性, 協調性）で
        各次元のΔΛCイベント品質を検証し、リンクに反映する。

        Returns
        -------
        dict
            per-dimension genuineness scores and raw results
        """
        # 全イベントフレームを収集
        events_all = np.minimum(events_pos + events_neg, 1.0)
        event_frames_list = []
        for d in range(n_dims):
            frames_d = np.where(events_all[:, d] > 0)[0]
            event_frames_list.append(frames_d)

        # 全次元のイベントを統合
        all_frames = np.unique(
            np.concatenate(event_frames_list) if event_frames_list else []
        )

        if len(all_frames) == 0:
            return {}

        logger.info(
            f"   🔺 GPU 4-axis verification: {len(all_frames)} unique event frames"
        )

        # GPU 4軸検証実行
        raw = network_verify_all(
            event_frames=all_frames,
            state_vectors=state_vectors,
            local_lambda_f=local_lambda_f,
            local_rho_t=rho_t,
            local_std=local_std,
        )

        genuineness = raw["genuineness"]

        # フレーム → genuineness のマッピング
        frame_to_quality = {int(f): float(g) for f, g in zip(all_frames, genuineness)}

        # 各次元のイベント品質平均
        dim_quality = {}
        for d in range(n_dims):
            frames_d = event_frames_list[d]
            if len(frames_d) > 0:
                qualities = [frame_to_quality.get(int(f), 0.0) for f in frames_d]
                dim_quality[d] = float(np.mean(qualities))
            else:
                dim_quality[d] = 0.0

        # リンクにevent_quality反映
        for link in sync_links:
            q_from = dim_quality.get(link.from_dim, 0.0)
            q_to = dim_quality.get(link.to_dim, 0.0)
            link.event_quality = (q_from + q_to) / 2.0

        for link in causal_links:
            q_from = dim_quality.get(link.from_dim, 0.0)
            q_to = dim_quality.get(link.to_dim, 0.0)
            link.event_quality = (q_from + q_to) / 2.0

        logger.info(
            f"   ✅ Event quality: mean={np.mean(genuineness):.3f}, "
            f"min={np.min(genuineness):.3f}, max={np.max(genuineness):.3f}"
        )

        return {
            "dim_quality": dim_quality,
            "raw_genuineness": genuineness,
            "event_frames": all_frames,
            "frame_to_quality": frame_to_quality,
        }

    # ================================================================
    # Pattern / Hub / Causal Structure
    # ================================================================

    def _identify_pattern(
        self,
        sync_links: list[DimensionLink],
        causal_links: list[DimensionLink],
    ) -> str:
        n_sync = len(sync_links)
        n_causal = len(causal_links)

        if n_sync == 0 and n_causal == 0:
            return "independent"
        elif n_sync > n_causal * 2:
            return "parallel"
        elif n_causal > n_sync * 2:
            return "cascade"
        else:
            return "mixed"

    def _detect_hubs(
        self,
        sync_links: list[DimensionLink],
        causal_links: list[DimensionLink],
        n_dims: int,
    ) -> list[int]:
        connectivity = np.zeros(n_dims)
        for link in sync_links + causal_links:
            connectivity[link.from_dim] += link.strength
            connectivity[link.to_dim] += link.strength

        if np.max(connectivity) == 0:
            return []

        threshold = np.mean(connectivity) + np.std(connectivity)
        hubs = np.where(connectivity > threshold)[0].tolist()
        return sorted(hubs, key=lambda d: connectivity[d], reverse=True)

    def _identify_causal_structure(
        self,
        causal_links: list[DimensionLink],
        n_dims: int,
    ) -> tuple[list[int], list[int]]:
        out_strength = np.zeros(n_dims)
        in_strength = np.zeros(n_dims)

        for link in causal_links:
            out_strength[link.from_dim] += link.strength
            in_strength[link.to_dim] += link.strength

        drivers = [
            d
            for d in range(n_dims)
            if out_strength[d] > 0 and out_strength[d] > in_strength[d] * 1.5
        ]
        followers = [
            d
            for d in range(n_dims)
            if in_strength[d] > 0 and in_strength[d] > out_strength[d] * 1.5
        ]

        return (
            sorted(drivers, key=lambda d: out_strength[d], reverse=True),
            sorted(followers, key=lambda d: in_strength[d], reverse=True),
        )

    # ================================================================
    # Event Analysis Helpers
    # ================================================================

    def _identify_initiators(
        self,
        state_vectors: np.ndarray,
        event_frame: int,
        lookback: int,
        n_dims: int,
    ) -> list[int]:
        """ΔΛCイベントの発火タイミングで発起者を推定"""
        start = max(0, event_frame - lookback)
        pre_event = state_vectors[start : event_frame + 1]

        if len(pre_event) < 3:
            return []

        # 各次元でΔΛCイベントを検出し、最も早く発火した次元を特定
        displacement = np.diff(pre_event, axis=0)
        scores = np.zeros(n_dims)

        for d in range(n_dims):
            abs_disp = np.abs(displacement[:, d])
            # 早期に大きく動いた次元ほど高スコア
            weights = np.linspace(2.0, 0.5, len(abs_disp))
            scores[d] = np.sum(abs_disp * weights)

        threshold = np.mean(scores) + np.std(scores)
        initiators = np.where(scores > threshold)[0]
        return sorted(initiators, key=lambda d: scores[d], reverse=True)

    def _estimate_propagation_order(
        self,
        state_vectors: np.ndarray,
        event_frame: int,
        lookback: int,
        n_dims: int,
    ) -> list[int]:
        """ΔΛC発火の伝播順序を推定"""
        start = max(0, event_frame - lookback)
        window = state_vectors[start : event_frame + 1]

        if len(window) < 3:
            return list(range(n_dims))

        displacement = np.abs(np.diff(window, axis=0))
        onset_frames = np.full(n_dims, len(displacement))

        for d in range(n_dims):
            series = displacement[:, d]
            threshold_d = np.mean(series) + 1.5 * np.std(series)
            exceeding = np.where(series > threshold_d)[0]
            if len(exceeding) > 0:
                onset_frames[d] = exceeding[0]

        return list(np.argsort(onset_frames))

    # ================================================================
    # Output
    # ================================================================

    def _print_summary(self, result: NetworkResult) -> None:
        logger.info("=" * 55)
        logger.info("Λ³ Native Network Analysis Summary")
        logger.info("=" * 55)
        logger.info(f"  Pattern: {result.pattern}")
        logger.info(f"  Sync links: {result.n_sync_links}")
        logger.info(f"  Causal links: {result.n_causal_links}")

        if result.hub_names:
            logger.info(f"  Hubs: {', '.join(result.hub_names)}")
        if result.driver_names:
            logger.info(f"  Drivers: {', '.join(result.driver_names)}")
        if result.follower_names:
            logger.info(f"  Followers: {', '.join(result.follower_names)}")

        if result.sync_network:
            logger.info("  Sync Network:")
            for link in sorted(
                result.sync_network, key=lambda lnk: lnk.strength, reverse=True
            ):
                eq_str = (
                    f", eq={link.event_quality:.3f}" if link.event_quality > 0 else ""
                )
                logger.info(
                    f"    {link.from_name} ↔ {link.to_name}: "
                    f"{link.strength:.3f} (p={link.p_value:.3f}, "
                    f"ρT_corr={link.rho_t_corr:.3f}{eq_str})"
                )

        if result.causal_network:
            logger.info("  Causal Network:")
            for link in sorted(
                result.causal_network, key=lambda lnk: lnk.strength, reverse=True
            ):
                eq_str = (
                    f", eq={link.event_quality:.3f}" if link.event_quality > 0 else ""
                )
                logger.info(
                    f"    {link.from_name} → {link.to_name}: "
                    f"{link.strength:.3f} (lag={link.lag}, "
                    f"p={link.p_value:.3f}{eq_str})"
                )
