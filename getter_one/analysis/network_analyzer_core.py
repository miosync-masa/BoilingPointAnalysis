"""
Network Analyzer Core - Domain-Agnostic
========================================

BANKAI-MDのthird_impact_analytics.pyの汎用版。
N次元時系列データの次元間ネットワーク構造を解析する。

MD版との対応:
  atom          → dimension（次元/チャネル）
  residue       → （廃止：物理実体データ非依存）
  sync_network  → 同期ネットワーク（同時相関）
  causal_network→ 因果ネットワーク（ラグ付き相関）
  async_network → （廃止：距離概念なし）
  residue_bridge→ （廃止）

天気データでの解釈例:
  sync:   「湿度と露点が常に同時に動く」
  causal: 「気温が動いた3時間後に気圧が動く」
  pattern: parallel（全次元同時変化）/ cascade（伝播的変化）

Built with 💕 by Masamichi & Tamaki
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("bankai.analysis.network_analyzer_core")


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
    link_type: str          # 'sync' or 'causal'
    strength: float         # 相関の絶対値
    correlation: float      # 相関の符号付き値（正/負の相関を区別）
    lag: int = 0            # causalの場合のラグ（フレーム数）


@dataclass
class NetworkResult:
    """ネットワーク解析結果"""
    sync_network: list[DimensionLink] = field(default_factory=list)
    causal_network: list[DimensionLink] = field(default_factory=list)

    # 相関行列（生データ）
    sync_matrix: np.ndarray | None = None      # (n_dims, n_dims)
    causal_matrix: np.ndarray | None = None     # (n_dims, n_dims) 最大ラグ相関
    causal_lag_matrix: np.ndarray | None = None  # (n_dims, n_dims) 最適ラグ

    # ネットワーク特性
    pattern: str = "unknown"         # 'parallel', 'cascade', 'mixed'
    hub_dimensions: list[int] = field(default_factory=list)
    hub_names: list[str] = field(default_factory=list)

    # 因果構造
    causal_drivers: list[int] = field(default_factory=list)   # 駆動次元
    causal_followers: list[int] = field(default_factory=list)  # 従属次元
    driver_names: list[str] = field(default_factory=list)
    follower_names: list[str] = field(default_factory=list)

    # メタデータ
    n_dims: int = 0
    n_sync_links: int = 0
    n_causal_links: int = 0
    dimension_names: list[str] = field(default_factory=list)


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
        同期ネットワーク判定閾値（相関の絶対値）
    causal_threshold : float
        因果ネットワーク判定閾値（ラグ付き相関の絶対値）
    max_lag : int
        因果推定の最大ラグ（フレーム数）
    """

    def __init__(
        self,
        sync_threshold: float = 0.5,
        causal_threshold: float = 0.4,
        max_lag: int = 12,
    ):
        self.sync_threshold = sync_threshold
        self.causal_threshold = causal_threshold
        self.max_lag = max_lag

        logger.info(
            f"✅ NetworkAnalyzerCore initialized "
            f"(sync>{sync_threshold}, causal>{causal_threshold}, max_lag={max_lag})"
        )

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

        logger.info(
            f"🔍 Analyzing {n_dims}-dimensional network "
            f"({n_frames} frames, window={window})"
        )

        # 1. 相関計算
        correlations = self._compute_correlations(state_vectors, window)

        # 2. ネットワーク構築
        sync_links, causal_links = self._build_networks(
            correlations, dimension_names
        )

        # 3. パターン識別
        pattern = self._identify_pattern(sync_links, causal_links)

        # 4. ハブ次元検出
        hub_dims = self._detect_hubs(sync_links, causal_links, n_dims)

        # 5. 因果構造（ドライバー/フォロワー）
        drivers, followers = self._identify_causal_structure(
            causal_links, n_dims
        )

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

        Parameters
        ----------
        state_vectors : np.ndarray (n_frames, n_dims)
        event_frame : int
            イベント発生フレーム
        window_before : int
            イベント前の解析ウィンドウ
        window_after : int
            イベント後の解析ウィンドウ
        dimension_names : list[str], optional
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

    def _compute_correlations(
        self, state_vectors: np.ndarray, window: int
    ) -> dict:
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
                    sync_links.append(DimensionLink(
                        from_dim=i,
                        to_dim=j,
                        from_name=dimension_names[i],
                        to_name=dimension_names[j],
                        link_type="sync",
                        strength=abs(sync_corr),
                        correlation=sync_corr,
                    ))

                # 因果リンク
                # 同期相関より有意にラグ相関が強い場合のみ因果と判定
                if (abs(causal_corr) > self.causal_threshold
                        and abs(causal_corr) > abs(sync_corr) * 1.1):

                    # ラグの符号で因果の方向を決定
                    if lag > 0:
                        from_d, to_d = i, j
                    else:
                        from_d, to_d = j, i
                        lag = abs(lag)

                    causal_links.append(DimensionLink(
                        from_dim=from_d,
                        to_dim=to_d,
                        from_name=dimension_names[from_d],
                        to_name=dimension_names[to_d],
                        link_type="causal",
                        strength=abs(causal_corr),
                        correlation=causal_corr,
                        lag=lag,
                    ))

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
            return "parallel"     # 同期的協調（全次元が同時に動く）
        elif n_causal > n_sync * 2:
            return "cascade"      # カスケード伝播（次元間に時間差）
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
        in_degree = np.zeros(n_dims)   # 駆動される側

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
        pre_event = state_vectors[start:event_frame + 1]

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
        window = state_vectors[start:event_frame + 1]

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

    def _print_summary(self, result: NetworkResult):
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
            for link in sorted(result.sync_network,
                              key=lambda lnk: lnk.strength, reverse=True):
                sign = "+" if link.correlation > 0 else "−"
                logger.info(
                    f"    {link.from_name} ↔ {link.to_name}: "
                    f"{sign}{link.strength:.3f}"
                )

        if result.causal_network:
            logger.info("  Causal Network:")
            for link in sorted(result.causal_network,
                              key=lambda lnk: lnk.strength, reverse=True):
                logger.info(
                    f"    {link.from_name} → {link.to_name}: "
                    f"{link.strength:.3f} (lag={link.lag})"
                )
