"""
Network Analyzer Core - Domain-Agnostic
========================================
Built with 💕 by Masamichi & Tamaki
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("getter_one.analysis.network_analyzer_core")


# ============================================
# Data Classes
# ============================================


@dataclass
class DimensionLink:
    """次元間ネットワークリンク"""

    from_dim: int
    to_dim: int
    from_name: str
    to_name: str
    link_type: str  # 'sync' or 'causal'
    strength: float  # 相関の絶対値
    correlation: float  # 相関の符号付き値（正/負の相関を区別）
    lag: int = 0  # causalの場合のラグ（フレーム数）


@dataclass
class NetworkResult:
    """ネットワーク解析結果"""

    sync_network: list[DimensionLink] = field(default_factory=list)
    causal_network: list[DimensionLink] = field(default_factory=list)

    # 相関行列（生データ）
    sync_matrix: np.ndarray | None = None  # (n_dims, n_dims)
    causal_matrix: np.ndarray | None = None  # (n_dims, n_dims) 最大ラグ相関
    causal_lag_matrix: np.ndarray | None = None  # (n_dims, n_dims) 最適ラグ

    # ネットワーク特性
    pattern: str = "unknown"  # 'parallel', 'cascade', 'mixed'
    hub_dimensions: list[int] = field(default_factory=list)
    hub_names: list[str] = field(default_factory=list)

    # 因果構造
    causal_drivers: list[int] = field(default_factory=list)  # 駆動次元
    causal_followers: list[int] = field(default_factory=list)  # 従属次元
    driver_names: list[str] = field(default_factory=list)
    follower_names: list[str] = field(default_factory=list)

    # メタデータ
    n_dims: int = 0
    n_sync_links: int = 0
    n_causal_links: int = 0
    dimension_names: list[str] = field(default_factory=list)

    # adaptive パラメータ（使用時のみ）
    adaptive_params: dict | None = None


@dataclass
class CooperativeEventNetwork:
    """cooperative event発生時のネットワーク構造"""

    event_frame: int
    event_timestamp: str | None = None
    delta_lambda_c: float = 0.0

    # イベント時のネットワーク
    network: NetworkResult | None = None

    # イベント固有の情報
    initiator_dims: list[int] = field(default_factory=list)
    initiator_names: list[str] = field(default_factory=list)
    propagation_order: list[int] = field(default_factory=list)


# ============================================
# Network Analyzer Core
# ============================================


class NetworkAnalyzerCore:
    """
    汎用次元間ネットワーク解析

    N次元時系列データの各次元間の同期・因果関係を検出し、
    ネットワーク構造として可視化可能な形で出力する。

    Parameters
    ----------
    sync_threshold : float
        同期ネットワーク判定閾値（adaptive=Trueならヒント値）
    causal_threshold : float
        因果ネットワーク判定閾値（adaptive=Trueならヒント値）
    max_lag : int
        因果推定の最大ラグ（adaptive=Trueならヒント値）
    adaptive : bool
        データ駆動の適応的パラメータ調整を有効にする
    """

    def __init__(
        self,
        sync_threshold: float = 0.5,
        causal_threshold: float = 0.4,
        max_lag: int = 12,
        adaptive: bool = True,
    ):
        # ユーザー指定値（adaptive=True ならヒント値）
        self.sync_threshold_hint = sync_threshold
        self.causal_threshold_hint = causal_threshold
        self.max_lag_hint = max_lag
        self.adaptive = adaptive

        # 実行時に更新されるパラメータ
        self.sync_threshold = sync_threshold
        self.causal_threshold = causal_threshold
        self.max_lag = max_lag

        # 適応的パラメータの計算結果を保持
        self.adaptive_params: dict | None = None

        logger.info(
            f"✅ NetworkAnalyzerCore initialized "
            f"(sync>{sync_threshold}, causal>{causal_threshold}, "
            f"max_lag={max_lag}, adaptive={adaptive})"
        )

    # ================================================================
    # Adaptive Parameter Computation
    # ================================================================

    def _compute_adaptive_parameters(
        self,
        state_vectors: np.ndarray,
    ) -> dict:
        """
        データ駆動の適応的パラメータ計算

        CascadeTrackerのadaptive機構と同一の5指標から
        ネットワーク解析用の閾値・ラグを動的に算出する。

        指標:
          1. グローバルボラティリティ（全体の変動度）
          2. 時間的変動性（隣接フレーム間の変化率）
          3. 次元間相関構造の複雑度
          4. 局所的変動パターン（非定常性）
          5. スペクトル特性（低周波支配度）

        Returns
        -------
        dict
            sync_threshold, causal_threshold, max_lag, scale_factor,
            volatility_metrics
        """
        n_frames, n_dims = state_vectors.shape

        # ── 1. グローバルボラティリティ ──
        global_std = np.std(state_vectors)
        global_mean = np.mean(np.abs(state_vectors))
        volatility_ratio = global_std / (global_mean + 1e-10)

        # ── 2. 時間的変動性（隣接フレーム間の変化率）──
        temporal_changes = np.diff(state_vectors, axis=0)
        temporal_volatility = np.mean(np.std(temporal_changes, axis=0))

        # ── 3. 次元間の相関構造の複雑度 ──
        if n_dims > 1:
            corr_matrix = np.corrcoef(state_vectors.T)
            triu = corr_matrix[np.triu_indices(n_dims, k=1)]
            triu = triu[~np.isnan(triu)]
            correlation_complexity = (
                1.0 - np.mean(np.abs(triu)) if len(triu) > 0 else 0.5
            )
            mean_abs_corr = np.mean(np.abs(triu)) if len(triu) > 0 else 0.0
        else:
            correlation_complexity = 0.5
            mean_abs_corr = 0.0

        # ── 4. 局所的変動パターン ──
        base_window = max(10, n_frames // 10)
        local_volatilities = []
        for i in range(0, n_frames - base_window, max(1, base_window // 2)):
            window_data = state_vectors[i : i + base_window]
            local_volatilities.append(np.std(window_data))

        if len(local_volatilities) > 1:
            volatility_variation = np.std(local_volatilities) / (
                np.mean(local_volatilities) + 1e-10
            )
        else:
            volatility_variation = 0.5

        # ── 5. スペクトル解析（支配的周期の推定）──
        fft_mag = np.abs(np.fft.fft(state_vectors, axis=0))
        low_cutoff = max(1, n_frames // 10)
        high_cutoff = max(2, n_frames // 2)
        low_freq_ratio = np.sum(fft_mag[:low_cutoff]) / (
            np.sum(fft_mag[:high_cutoff]) + 1e-10
        )

        # ── sync_threshold の動的調整 ──
        # ボラティリティが高い → 閾値を上げる（ノイズに強く）
        # 相関が全体的に高い → 閾値を上げる（意味のある相関だけ拾う）
        # 相関が全体的に低い → 閾値を下げる（微弱な構造も拾う）
        sync_adj = 0.0
        if volatility_ratio > 2.0:
            sync_adj += 0.1
        elif volatility_ratio < 0.3:
            sync_adj -= 0.1

        if mean_abs_corr > 0.6:
            sync_adj += 0.1  # 相関が強い系 → 閾値を上げて選別
        elif mean_abs_corr < 0.2:
            sync_adj -= 0.1  # 相関が弱い系 → 閾値を下げて拾う

        if volatility_variation > 1.0:
            sync_adj += 0.05  # 非定常 → やや厳しめに

        sync_threshold = np.clip(self.sync_threshold_hint + sync_adj, 0.15, 0.85)

        # ── causal_threshold の動的調整 ──
        # 時間的変動が小さい → 閾値を下げる（微細な因果も拾う）
        # 次元が多い → 閾値を上げる（偽因果を抑制）
        causal_adj = 0.0
        if temporal_volatility < global_std * 0.3:
            causal_adj -= 0.1  # 静かな系 → 小さい因果も拾う
        elif temporal_volatility > global_std * 2.0:
            causal_adj += 0.1  # 激しい系 → 厳しめに

        if n_dims > 50:
            causal_adj += 0.1  # 多次元 → 偽因果リスクが高い
        elif n_dims <= 5:
            causal_adj -= 0.05  # 低次元 → 緩めでOK

        if correlation_complexity > 0.7:
            causal_adj -= 0.05  # 複雑な相関 → 因果を丁寧に拾う

        causal_threshold = np.clip(self.causal_threshold_hint + causal_adj, 0.15, 0.80)

        # ── max_lag の動的調整 ──
        # 低周波支配 → ラグを伸ばす（ゆっくり伝播する系）
        # 高周波支配 → ラグを縮める（速い伝播）
        # データ長に応じてクリップ
        scale_factor = 1.0
        if low_freq_ratio > 0.8:
            scale_factor *= 1.5
        elif low_freq_ratio < 0.3:
            scale_factor *= 0.7

        if correlation_complexity > 0.7:
            scale_factor *= 1.3  # 複雑な相関 → 伝播が遅い可能性

        if temporal_volatility > global_std * 2.0:
            scale_factor *= 0.8  # 変動が激しい → 短いラグで十分

        raw_lag = int(self.max_lag_hint * scale_factor)
        max_lag = int(np.clip(raw_lag, 3, max(5, n_frames // 5)))

        params = {
            "sync_threshold": float(sync_threshold),
            "causal_threshold": float(causal_threshold),
            "max_lag": max_lag,
            "scale_factor": float(scale_factor),
            "volatility_metrics": {
                "global_volatility": float(volatility_ratio),
                "temporal_volatility": float(temporal_volatility),
                "correlation_complexity": float(correlation_complexity),
                "local_variation": float(volatility_variation),
                "low_freq_ratio": float(low_freq_ratio),
                "mean_abs_corr": float(mean_abs_corr),
            },
        }

        logger.info(
            f"   Adaptive params: "
            f"sync_th={sync_threshold:.3f} "
            f"(hint={self.sync_threshold_hint}), "
            f"causal_th={causal_threshold:.3f} "
            f"(hint={self.causal_threshold_hint}), "
            f"max_lag={max_lag} (hint={self.max_lag_hint})"
        )
        logger.info(
            f"   Volatility: global={volatility_ratio:.3f}, "
            f"temporal={temporal_volatility:.3f}, "
            f"corr_complexity={correlation_complexity:.3f}, "
            f"local_var={volatility_variation:.3f}, "
            f"low_freq={low_freq_ratio:.3f}, "
            f"mean_corr={mean_abs_corr:.3f}"
        )

        return params

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
        全体ネットワーク解析

        Parameters
        ----------
        state_vectors : np.ndarray (n_frames, n_dims)
            N次元状態ベクトル時系列
        dimension_names : list[str], optional
            各次元の名前
        window : int, optional
            相関計算ウィンドウ（Noneなら全フレーム使用）

        Returns
        -------
        NetworkResult
            ネットワーク解析結果
        """
        n_frames, n_dims = state_vectors.shape

        if dimension_names is None:
            dimension_names = [f"dim_{i}" for i in range(n_dims)]

        if window is None:
            window = n_frames

        # adaptive パラメータ計算
        if self.adaptive:
            self.adaptive_params = self._compute_adaptive_parameters(state_vectors)
            self.sync_threshold = self.adaptive_params["sync_threshold"]
            self.causal_threshold = self.adaptive_params["causal_threshold"]
            self.max_lag = self.adaptive_params["max_lag"]

        logger.info(
            f"🔍 Analyzing {n_dims}-dimensional network "
            f"({n_frames} frames, window={window}, "
            f"sync>{self.sync_threshold:.3f}, "
            f"causal>{self.causal_threshold:.3f}, "
            f"max_lag={self.max_lag})"
        )

        # 1. 相関計算
        correlations = self._compute_correlations(state_vectors, window)

        # 2. ネットワーク構築
        sync_links, causal_links = self._build_networks(correlations, dimension_names)

        # 3. パターン識別
        pattern = self._identify_pattern(sync_links, causal_links)

        # 4. ハブ次元検出
        hub_dims = self._detect_hubs(sync_links, causal_links, n_dims)

        # 5. 因果構造（ドライバー/フォロワー）
        drivers, followers = self._identify_causal_structure(causal_links, n_dims)

        result = NetworkResult(
            sync_network=sync_links,
            causal_network=causal_links,
            sync_matrix=correlations["sync"],
            causal_matrix=correlations["max_lagged"],
            causal_lag_matrix=correlations["best_lag"],
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
        """
        cooperative event発生時の局所ネットワーク解析

        イベント前後のウィンドウで因果構造を分析し、
        どの次元がイベントを「発火」させたかを推定する。
        """
        n_frames, n_dims = state_vectors.shape

        if dimension_names is None:
            dimension_names = [f"dim_{i}" for i in range(n_dims)]

        start = max(0, event_frame - window_before)
        end = min(n_frames, event_frame + window_after)
        local_data = state_vectors[start:end]

        # 局所ネットワーク解析
        network = self.analyze(local_data, dimension_names, window=len(local_data))

        # イベント発火次元の推定
        initiators = self._identify_initiators(
            state_vectors, event_frame, window_before, n_dims
        )

        # 伝播順序の推定
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
    # 相関計算
    # ================================================================

    def _compute_correlations(self, state_vectors: np.ndarray, window: int) -> dict:
        """全次元ペアの相関計算（同期・因果）"""
        n_frames, n_dims = state_vectors.shape
        w = min(window, n_frames)

        sync_matrix = np.zeros((n_dims, n_dims))
        max_lagged_matrix = np.zeros((n_dims, n_dims))
        best_lag_matrix = np.zeros((n_dims, n_dims), dtype=int)

        # 変位ベクトル（1次差分）で相関を計算
        # 生データの相関だとトレンドに引きずられるため
        displacement = np.diff(state_vectors[:w], axis=0)

        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                ts_i = displacement[:, i]
                ts_j = displacement[:, j]

                # ゼロ分散チェック
                if np.std(ts_i) < 1e-10 or np.std(ts_j) < 1e-10:
                    continue

                # ── 同期相関 ──
                sync_corr = np.corrcoef(ts_i, ts_j)[0, 1]
                if np.isnan(sync_corr):
                    sync_corr = 0.0
                sync_matrix[i, j] = sync_corr
                sync_matrix[j, i] = sync_corr

                # ── ラグ付き相関（因果推定） ──
                best_corr = 0.0
                best_lag = 0

                for lag in range(1, min(self.max_lag + 1, len(ts_i) - 1)):
                    # i → j（iがlagフレーム先行）
                    corr_ij = np.corrcoef(ts_i[:-lag], ts_j[lag:])[0, 1]
                    if np.isnan(corr_ij):
                        corr_ij = 0.0

                    if abs(corr_ij) > abs(best_corr):
                        best_corr = corr_ij
                        best_lag = lag

                    # j → i（jがlagフレーム先行）
                    corr_ji = np.corrcoef(ts_j[:-lag], ts_i[lag:])[0, 1]
                    if np.isnan(corr_ji):
                        corr_ji = 0.0

                    if abs(corr_ji) > abs(best_corr):
                        best_corr = corr_ji
                        best_lag = -lag  # 負のラグ = jが先行

                max_lagged_matrix[i, j] = best_corr
                max_lagged_matrix[j, i] = best_corr
                best_lag_matrix[i, j] = best_lag
                best_lag_matrix[j, i] = -best_lag

        return {
            "sync": sync_matrix,
            "max_lagged": max_lagged_matrix,
            "best_lag": best_lag_matrix,
        }

    # ================================================================
    # ネットワーク構築
    # ================================================================

    def _build_networks(
        self,
        correlations: dict,
        dimension_names: list[str],
    ) -> tuple[list[DimensionLink], list[DimensionLink]]:
        """相関行列からネットワークリンクを構築"""
        n_dims = len(dimension_names)
        sync_links = []
        causal_links = []

        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                sync_corr = correlations["sync"][i, j]
                causal_corr = correlations["max_lagged"][i, j]
                lag = correlations["best_lag"][i, j]

                # 同期リンク
                if abs(sync_corr) > self.sync_threshold:
                    sync_links.append(
                        DimensionLink(
                            from_dim=i,
                            to_dim=j,
                            from_name=dimension_names[i],
                            to_name=dimension_names[j],
                            link_type="sync",
                            strength=abs(sync_corr),
                            correlation=sync_corr,
                        )
                    )

                # 因果リンク
                # 同期相関より有意にラグ相関が強い場合のみ因果と判定
                if (
                    abs(causal_corr) > self.causal_threshold
                    and abs(causal_corr) > abs(sync_corr) * 1.1
                ):
                    # ラグの符号で因果の方向を決定
                    if lag > 0:
                        from_d, to_d = i, j
                    else:
                        from_d, to_d = j, i
                        lag = abs(lag)

                    causal_links.append(
                        DimensionLink(
                            from_dim=from_d,
                            to_dim=to_d,
                            from_name=dimension_names[from_d],
                            to_name=dimension_names[to_d],
                            link_type="causal",
                            strength=abs(causal_corr),
                            correlation=causal_corr,
                            lag=lag,
                        )
                    )

        return sync_links, causal_links

    # ================================================================
    # パターン識別・ハブ検出・因果構造
    # ================================================================

    def _identify_pattern(
        self,
        sync_links: list[DimensionLink],
        causal_links: list[DimensionLink],
    ) -> str:
        """ネットワークパターンの識別"""
        n_sync = len(sync_links)
        n_causal = len(causal_links)

        if n_sync == 0 and n_causal == 0:
            return "independent"
        elif n_sync > n_causal * 2:
            return "parallel"  # 同期的協調（全次元が同時に動く）
        elif n_causal > n_sync * 2:
            return "cascade"  # カスケード伝播（次元間に時間差）
        else:
            return "mixed"

    def _detect_hubs(
        self,
        sync_links: list[DimensionLink],
        causal_links: list[DimensionLink],
        n_dims: int,
    ) -> list[int]:
        """ハブ次元の検出（多くの次元と強く結合している次元）"""
        connectivity = np.zeros(n_dims)

        for link in sync_links + causal_links:
            connectivity[link.from_dim] += link.strength
            connectivity[link.to_dim] += link.strength

        if np.max(connectivity) == 0:
            return []

        # 平均+1σ以上の接続強度を持つ次元をハブとする
        threshold = np.mean(connectivity) + np.std(connectivity)
        hubs = np.where(connectivity > threshold)[0].tolist()

        return sorted(hubs, key=lambda d: connectivity[d], reverse=True)

    def _identify_causal_structure(
        self,
        causal_links: list[DimensionLink],
        n_dims: int,
    ) -> tuple[list[int], list[int]]:
        """因果構造の特定（ドライバー/フォロワー）"""
        out_degree = np.zeros(n_dims)  # 駆動する側
        in_degree = np.zeros(n_dims)  # 駆動される側

        for link in causal_links:
            out_degree[link.from_dim] += link.strength
            in_degree[link.to_dim] += link.strength

        # ドライバー: out_degree >> in_degree
        drivers = []
        followers = []

        for d in range(n_dims):
            if out_degree[d] > 0 and out_degree[d] > in_degree[d] * 1.5:
                drivers.append(d)
            elif in_degree[d] > 0 and in_degree[d] > out_degree[d] * 1.5:
                followers.append(d)

        return (
            sorted(drivers, key=lambda d: out_degree[d], reverse=True),
            sorted(followers, key=lambda d: in_degree[d], reverse=True),
        )

    # ================================================================
    # イベント解析ヘルパー
    # ================================================================

    def _identify_initiators(
        self,
        state_vectors: np.ndarray,
        event_frame: int,
        lookback: int,
        n_dims: int,
    ) -> list[int]:
        """
        cooperative eventの発火次元を推定

        イベント直前のウィンドウで最も早く・大きく動き始めた次元を特定。
        """
        start = max(0, event_frame - lookback)
        pre_event = state_vectors[start : event_frame + 1]

        if len(pre_event) < 3:
            return []

        displacement = np.diff(pre_event, axis=0)

        # 各次元の「動き出しの早さ × 大きさ」をスコア化
        scores = np.zeros(n_dims)
        for d in range(n_dims):
            abs_disp = np.abs(displacement[:, d])
            # 後半ほど重みが大きい（イベント直前の動きを重視）
            weights = np.linspace(0.5, 2.0, len(abs_disp))
            scores[d] = np.sum(abs_disp * weights)

        # 上位次元を返す
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
        """
        イベントの伝播順序を推定

        各次元が閾値を超えた最初のフレームで順序付け。
        """
        start = max(0, event_frame - lookback)
        window = state_vectors[start : event_frame + 1]

        if len(window) < 3:
            return list(range(n_dims))

        displacement = np.abs(np.diff(window, axis=0))

        # 各次元の「爆発タイミング」を検出
        # 移動平均を超えた最初のフレーム
        onset_frames = np.full(n_dims, len(displacement))

        for d in range(n_dims):
            series = displacement[:, d]
            threshold = np.mean(series) + 1.5 * np.std(series)

            exceeding = np.where(series > threshold)[0]
            if len(exceeding) > 0:
                onset_frames[d] = exceeding[0]

        # onset_frameの早い順にソート
        return list(np.argsort(onset_frames))

    # ================================================================
    # 出力
    # ================================================================

    def _print_summary(self, result: NetworkResult) -> None:
        """結果サマリーの表示"""
        logger.info("=" * 50)
        logger.info("Network Analysis Summary")
        logger.info("=" * 50)
        logger.info(f"  Pattern: {result.pattern}")
        logger.info(f"  Sync links: {result.n_sync_links}")
        logger.info(f"  Causal links: {result.n_causal_links}")

        if result.hub_names:
            logger.info(f"  Hub dimensions: {', '.join(result.hub_names)}")

        if result.driver_names:
            logger.info(f"  Causal drivers: {', '.join(result.driver_names)}")

        if result.follower_names:
            logger.info(f"  Causal followers: {', '.join(result.follower_names)}")

        if result.sync_network:
            logger.info("  Sync Network:")
            for link in sorted(
                result.sync_network,
                key=lambda lnk: lnk.strength,
                reverse=True,
            ):
                sign = "+" if link.correlation > 0 else "−"
                logger.info(
                    f"    {link.from_name} ↔ {link.to_name}: {sign}{link.strength:.3f}"
                )

        if result.causal_network:
            logger.info("  Causal Network:")
            for link in sorted(
                result.causal_network,
                key=lambda lnk: lnk.strength,
                reverse=True,
            ):
                logger.info(
                    f"    {link.from_name} → {link.to_name}: "
                    f"{link.strength:.3f} (lag={link.lag})"
                )
