"""
Cascade Tracker - Domain-Agnostic Cascade Chain Reconstruction
================================================================

BANKAI-MD Third Impact Analytics の汎用版。
閉じた系（タンパク質残基）→ 開いた系（N次元時系列）への拡張。

Third Impact では残基の壁が系の境界だった。
GETTER One では因果が途切れるところが動的な系の境界になる。

Core Concept:
    複数の ΔΛC イベントを時系列で検出し、
    イベント間の因果チェーン（A→B→C→...）を再構成する。

    event_i.affected_dims ∩ event_j.genesis_dims ≠ ∅
    かつ event_j が event_i の直後に発生
    → event_i が event_j を引き起こした（因果リンク）

MD版との対応:
    genesis_atom     → genesis_dims（最初に動いた次元）
    first_wave_atom  → wave_dims（伝播先の次元）
    residue_bridge   → cross_group_bridge（次元グループ間の橋渡し）
    residue          → dimension_group（オプション）
    drug_target      → critical_dims（介入対象の次元）

Built with 💕 by Masamichi & Tamaki
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("getter_one.analysis.cascade_tracker")


# ============================================================
# Data Classes
# ============================================================

@dataclass
class CascadeEvent:
    """
    単一の ΔΛC イベント（Third Impact の EventOrigin 汎用版）

    MD版の genesis_atoms + first_wave_atoms に相当。
    """
    event_id: int
    frame: int                                    # イベント発生フレーム
    delta_lambda_c: float                         # ΔΛC 強度

    # 起源と伝播
    genesis_dims: list[int] = field(default_factory=list)    # 最初に動いた次元
    affected_dims: list[int] = field(default_factory=list)   # 影響を受けた全次元
    wave_dims: list[int] = field(default_factory=list)       # genesis以外で反応した次元

    # 名前付き
    genesis_names: list[str] = field(default_factory=list)
    affected_names: list[str] = field(default_factory=list)

    # 統計
    mean_displacement: float = 0.0
    std_displacement: float = 0.0
    threshold_used: float = 0.0

    # ネットワーク特性
    hub_dims: list[int] = field(default_factory=list)        # ハブ次元
    propagation_order: list[int] = field(default_factory=list)  # 伝播順序


@dataclass
class CascadeLink:
    """
    イベント間の因果リンク

    MD版の residue_bridge に相当するが、
    次元間の因果伝播を時間軸でつなぐ。
    """
    from_event_id: int
    to_event_id: int
    from_frame: int
    to_frame: int
    frame_gap: int                                # フレーム間隔

    # 因果の証拠
    shared_dims: list[int] = field(default_factory=list)  # 共有次元
    shared_names: list[str] = field(default_factory=list)
    causality_score: float = 0.0                  # 因果スコア

    # リンクタイプ
    link_type: str = "sequential"                 # sequential / branching / merging


@dataclass
class CascadeChain:
    """
    A→B→C→... のカスケードチェーン

    MD版では閉じた系で1本のchainだったが、
    汎用版では分岐・合流も許容する。
    """
    chain_id: int
    event_ids: list[int] = field(default_factory=list)
    links: list[CascadeLink] = field(default_factory=list)

    # チェーン特性
    length: int = 0                               # チェーン長
    total_frames: int = 0                         # 全体のフレーム幅
    origin_dims: list[int] = field(default_factory=list)     # 最初のgenesisの次元
    terminal_dims: list[int] = field(default_factory=list)   # 最後のaffectedの次元
    origin_names: list[str] = field(default_factory=list)
    terminal_names: list[str] = field(default_factory=list)

    # 強度
    mean_causality: float = 0.0
    total_delta_lambda_c: float = 0.0


@dataclass
class CascadeResult:
    """
    カスケード解析の全体結果

    MD版の ThirdImpactResult の汎用版。
    """
    # イベント
    events: list[CascadeEvent] = field(default_factory=list)
    n_events: int = 0

    # 因果リンク
    links: list[CascadeLink] = field(default_factory=list)
    n_links: int = 0

    # カスケードチェーン
    chains: list[CascadeChain] = field(default_factory=list)
    n_chains: int = 0
    longest_chain: int = 0

    # 因果DAG（隣接リスト表現）
    dag: dict[int, list[int]] = field(default_factory=dict)

    # クリティカル次元（drug_target_atoms の汎用版）
    critical_dims: list[int] = field(default_factory=list)
    critical_names: list[str] = field(default_factory=list)

    # グローバル統計
    cascade_coverage: float = 0.0       # 全フレームに対するカスケードの時間的カバー率
    mean_chain_length: float = 0.0
    branching_ratio: float = 0.0        # 分岐率（1イベントから複数への発火）

    # 次元名
    dimension_names: list[str] = field(default_factory=list)


# ============================================================
# Cascade Tracker
# ============================================================

class CascadeTracker:
    """
    カスケード因果チェーン追跡

    Third Impact (BANKAI-MD) の cascade 検出をドメイン非依存に汎用化。

    閉じた系（タンパク質）→ 残基の壁が境界
    開いた系（汎用時系列）→ 因果が途切れるところが動的境界

    Parameters
    ----------
    sigma_threshold : float
        genesis 検出の z-score 閾値（adaptive=True の場合はヒント値）
    max_gap : int
        因果リンク判定の最大フレーム間隔（adaptive=True の場合はデータから調整）
    min_shared_dims : int
        因果リンク判定に必要な最小共有次元数
    min_delta_lambda_c : float
        ΔΛC イベント検出の最小閾値（adaptive=True の場合はデータから調整）
    lookback : int
        イベント発火次元推定のルックバック幅（adaptive=True の場合はデータから調整）
    adaptive : bool
        適応的パラメータ調整を有効にするか。
        True の場合、上記パラメータはデフォルトヒントとして扱われ、
        データのボラティリティ・相関構造・スペクトル特性から動的に調整される。
    max_chains : int
        出力するカスケードチェーンの最大数（上位N件）。
        DAGのパス列挙で組み合わせ爆発を防ぐ。
    min_causality : float
        チェーンの平均因果スコアの最小値。これ未満のチェーンは除外される。
    """

    def __init__(
        self,
        sigma_threshold: float = 2.0,
        max_gap: int = 24,
        min_shared_dims: int = 1,
        min_delta_lambda_c: float = 2.0,
        lookback: int = 12,
        adaptive: bool = True,
        max_chains: int = 30,
        min_causality: float = 0.1,
    ):
        # ユーザー指定値（adaptive=True ならヒント値）
        self.sigma_threshold_hint = sigma_threshold
        self.max_gap_hint = max_gap
        self.min_shared_dims = min_shared_dims
        self.min_delta_lambda_c_hint = min_delta_lambda_c
        self.lookback_hint = lookback
        self.adaptive = adaptive
        self.max_chains = max_chains
        self.min_causality = min_causality

        # 実行時に更新されるパラメータ
        self.sigma_threshold = sigma_threshold
        self.max_gap = max_gap
        self.min_delta_lambda_c = min_delta_lambda_c
        self.lookback = lookback

        # 適応的パラメータの計算結果を保持
        self.adaptive_params: dict | None = None

        logger.info(
            f"✅ CascadeTracker initialized "
            f"(σ={sigma_threshold}, max_gap={max_gap}, "
            f"min_shared={min_shared_dims}, adaptive={adaptive})"
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

        BANKAI-MD の compute_adaptive_window_size を汎用化。
        データのボラティリティ・相関構造・スペクトル特性から
        閾値・ウィンドウサイズ・検出感度を動的に算出する。

        Returns
        -------
        dict
            adaptive_window: int     - ΔΛC検出のローカルウィンドウ
            lookback: int            - genesis検出のルックバック
            sigma_threshold: float   - genesis z-score 閾値
            delta_lc_sigma: float    - ΔΛCイベント検出の σ 倍率
            max_gap: int             - 因果リンクの最大フレーム間隔
            volatility_metrics: dict - 診断用メトリクス
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
            # NaN除去
            triu = triu[~np.isnan(triu)]
            correlation_complexity = 1.0 - np.mean(np.abs(triu)) if len(triu) > 0 else 0.5
        else:
            correlation_complexity = 0.5

        # ── 4. 局所的変動パターン ──
        base_window = max(10, n_frames // 10)
        local_volatilities = []
        for i in range(0, n_frames - base_window, max(1, base_window // 2)):
            window_data = state_vectors[i:i + base_window]
            local_volatilities.append(np.std(window_data))

        if len(local_volatilities) > 1:
            volatility_variation = (
                np.std(local_volatilities)
                / (np.mean(local_volatilities) + 1e-10)
            )
        else:
            volatility_variation = 0.5

        # ── 5. スペクトル解析（支配的周期の推定）──
        fft_mag = np.abs(np.fft.fft(state_vectors, axis=0))
        low_cutoff = max(1, n_frames // 10)
        high_cutoff = max(2, n_frames // 2)
        low_freq_ratio = (
            np.sum(fft_mag[:low_cutoff])
            / (np.sum(fft_mag[:high_cutoff]) + 1e-10)
        )

        # ── scale_factor の計算 ──
        scale_factor = 1.0

        # ボラティリティが高い → 小さいウィンドウ・低い閾値
        if volatility_ratio > 2.0:
            scale_factor *= 0.8
        elif volatility_ratio < 0.3:
            scale_factor *= 1.5

        # 時間的変動が大きい → 小さいウィンドウ
        if temporal_volatility > global_std * 2.0:
            scale_factor *= 0.9
        elif temporal_volatility < global_std * 0.3:
            scale_factor *= 1.4

        # 相関構造が複雑 → 大きいウィンドウ
        if correlation_complexity > 0.7:
            scale_factor *= 1.2
        elif correlation_complexity < 0.3:
            scale_factor *= 0.9

        # 局所変動が大きい → やや小さめ
        if volatility_variation > 1.0:
            scale_factor *= 0.85

        # 低周波支配 → 大きいウィンドウ
        if low_freq_ratio > 0.8:
            scale_factor *= 1.4
        elif low_freq_ratio < 0.3:
            scale_factor *= 0.8

        # ── パラメータの算出 ──

        # adaptive_window: ΔΛCローカル閾値のウィンドウ
        raw_window = int(base_window * scale_factor)
        adaptive_window = np.clip(raw_window, 10, max(20, n_frames // 3))

        # lookback: genesis検出のルックバック
        raw_lookback = int(self.lookback_hint * scale_factor)
        lookback = np.clip(raw_lookback, 3, max(5, n_frames // 10))

        # sigma_threshold: genesis z-score 閾値
        # ボラティリティが高い → 閾値を上げる（ノイズが多い）
        # ボラティリティが低い → 閾値を下げる（小さな異変も拾う）
        if volatility_ratio > 2.0:
            sigma_threshold = self.sigma_threshold_hint + 0.5
        elif volatility_ratio < 0.5:
            sigma_threshold = max(1.0, self.sigma_threshold_hint - 0.5)
        else:
            sigma_threshold = self.sigma_threshold_hint

        # delta_lc_sigma: ΔΛCイベント検出の σ 倍率
        # ボラティリティが高い → より厳しく
        if volatility_ratio > 1.5:
            delta_lc_sigma = self.min_delta_lambda_c_hint + 0.3
        elif volatility_ratio < 0.5:
            delta_lc_sigma = max(1.0, self.min_delta_lambda_c_hint - 0.3)
        else:
            delta_lc_sigma = self.min_delta_lambda_c_hint

        # max_gap: 因果リンクの最大フレーム間隔
        # データ長に応じてスケール
        raw_gap = int(self.max_gap_hint * scale_factor)
        max_gap = np.clip(raw_gap, 3, max(5, n_frames // 5))

        params = {
            "adaptive_window": int(adaptive_window),
            "lookback": int(lookback),
            "sigma_threshold": float(sigma_threshold),
            "delta_lc_sigma": float(delta_lc_sigma),
            "max_gap": int(max_gap),
            "scale_factor": float(scale_factor),
            "volatility_metrics": {
                "global_volatility": float(volatility_ratio),
                "temporal_volatility": float(temporal_volatility),
                "correlation_complexity": float(correlation_complexity),
                "local_variation": float(volatility_variation),
                "low_freq_ratio": float(low_freq_ratio),
            },
        }

        logger.info(
            f"   Adaptive params: scale={scale_factor:.2f}, "
            f"window={params['adaptive_window']}, "
            f"lookback={params['lookback']}, "
            f"σ_genesis={params['sigma_threshold']:.1f}, "
            f"σ_ΔΛC={params['delta_lc_sigma']:.1f}, "
            f"max_gap={params['max_gap']}"
        )
        logger.info(
            f"   Volatility: global={volatility_ratio:.3f}, "
            f"temporal={temporal_volatility:.3f}, "
            f"corr_complexity={correlation_complexity:.3f}, "
            f"local_var={volatility_variation:.3f}, "
            f"low_freq={low_freq_ratio:.3f}"
        )

        return params

    # ================================================================
    # Main Entry Point
    # ================================================================

    def track(
        self,
        state_vectors: np.ndarray,
        lambda_structures: dict[str, np.ndarray],
        dimension_names: list[str] | None = None,
        dimension_groups: dict[str, list[int]] | None = None,
    ) -> CascadeResult:
        """
        カスケード因果チェーンを追跡

        Parameters
        ----------
        state_vectors : np.ndarray (n_frames, n_dims)
            N次元状態ベクトル時系列
        lambda_structures : dict[str, np.ndarray]
            LambdaStructuresCore の出力
        dimension_names : list[str], optional
            各次元の名前
        dimension_groups : dict[str, list[int]], optional
            次元グループ定義（残基の代替）
            例: {"weather": [0,1,2], "ocean": [3,4,5]}

        Returns
        -------
        CascadeResult
        """
        n_frames, n_dims = state_vectors.shape

        if dimension_names is None:
            dimension_names = [f"dim_{i}" for i in range(n_dims)]

        logger.info(
            f"🔺 CascadeTracker: {n_dims} dims × {n_frames} frames"
        )

        # ── Step 0: 適応的パラメータ計算 ──
        if self.adaptive:
            self.adaptive_params = self._compute_adaptive_parameters(
                state_vectors
            )
            # 実行時パラメータを更新
            self.sigma_threshold = self.adaptive_params["sigma_threshold"]
            self.max_gap = self.adaptive_params["max_gap"]
            self.min_delta_lambda_c = self.adaptive_params["delta_lc_sigma"]
            self.lookback = self.adaptive_params["lookback"]
        else:
            # ヒント値をそのまま使用
            self.sigma_threshold = self.sigma_threshold_hint
            self.max_gap = self.max_gap_hint
            self.min_delta_lambda_c = self.min_delta_lambda_c_hint
            self.lookback = self.lookback_hint
            self.adaptive_params = None

        # ── Step 1: ΔΛC イベント検出 ──
        events = self._detect_events(
            state_vectors, lambda_structures, dimension_names
        )
        logger.info(f"   Events detected: {len(events)}")

        if len(events) < 2:
            logger.info("   < 2 events → no cascade possible")
            return CascadeResult(
                events=events,
                n_events=len(events),
                dimension_names=dimension_names,
            )

        # ── Step 2: イベント間因果リンク判定 ──
        links = self._build_causal_links(events, dimension_names)
        logger.info(f"   Causal links: {len(links)}")

        # ── Step 3: 因果DAG構築 ──
        dag = self._build_dag(events, links)

        # ── Step 4: カスケードチェーン抽出 ──
        chains = self._extract_chains(events, links, dag, dimension_names)
        logger.info(f"   Cascade chains: {len(chains)}")

        # ── Step 5: クリティカル次元特定 ──
        critical_dims = self._identify_critical_dims(
            events, links, chains, n_dims
        )

        # ── 結果構築 ──
        result = CascadeResult(
            events=events,
            n_events=len(events),
            links=links,
            n_links=len(links),
            chains=chains,
            n_chains=len(chains),
            longest_chain=max((c.length for c in chains), default=0),
            dag=dag,
            critical_dims=critical_dims,
            critical_names=[dimension_names[d] for d in critical_dims],
            dimension_names=dimension_names,
        )

        # 統計計算
        self._compute_statistics(result, n_frames)

        self._print_summary(result)
        return result

    # ================================================================
    # Step 1: ΔΛC イベント検出
    # ================================================================

    def _detect_events(
        self,
        state_vectors: np.ndarray,
        lambda_structures: dict[str, np.ndarray],
        dimension_names: list[str],
    ) -> list[CascadeEvent]:
        """
        ΔΛCイベントの検出（適応的ウィンドウ版）

        rho_T × sigma_s × |lambda_F| が **ローカル閾値** を超えるフレームを検出。

        グローバル閾値だと巨大イベント（リーマンショック本体等）が
        統計を支配し、前兆的な小さな異変が埋もれる。
        適応的ウィンドウにより「その時期における異常」を検出する。
        """
        events = []

        # ΔΛC の計算
        delta_lambda_c = self._compute_delta_lambda_c(lambda_structures)

        if delta_lambda_c is None or len(delta_lambda_c) == 0:
            return events

        n = len(delta_lambda_c)

        # 適応的ウィンドウサイズ
        if self.adaptive_params is not None:
            adaptive_window = self.adaptive_params["adaptive_window"]
        else:
            adaptive_window = max(20, self.lookback * 4)

        # ── ローカル閾値の計算 ──
        local_threshold = np.zeros(n)
        for i in range(n):
            # ウィンドウ範囲（前方のみ参照 = 未来の情報を使わない）
            w_start = max(0, i - adaptive_window)
            w_end = i + 1  # 現在フレームまで

            local_slice = delta_lambda_c[w_start:w_end]
            local_mean = np.mean(local_slice)
            local_std = np.std(local_slice)

            local_threshold[i] = local_mean + self.min_delta_lambda_c * local_std

        # グローバル閾値も参考に記録
        global_mean = np.mean(delta_lambda_c)
        global_std = np.std(delta_lambda_c)
        global_threshold = global_mean + self.min_delta_lambda_c * global_std

        # ── ローカル閾値超過のピーク検出 ──
        # 各フレームでローカル閾値を超えているかチェック
        above_local = delta_lambda_c > local_threshold
        above_indices = np.where(above_local)[0]

        # ピーク検出（局所最大のみ、最小距離フィルタ付き）
        event_frames = self._detect_peaks_adaptive(
            delta_lambda_c, above_indices, min_distance=3,
        )

        logger.info(
            f"   ΔΛC adaptive window: {adaptive_window} frames"
        )
        logger.info(
            f"   ΔΛC global ref: threshold={global_threshold:.4f} "
            f"(μ={global_mean:.4f}, σ={global_std:.4f})"
        )
        logger.info(
            f"   Adaptive detection: {len(event_frames)} events "
            f"(vs {len(self._detect_peaks(delta_lambda_c, global_threshold))} "
            f"with global threshold)"
        )

        for event_id, frame in enumerate(event_frames):
            event = self._analyze_single_event(
                event_id=event_id,
                frame=frame,
                state_vectors=state_vectors,
                lambda_structures=lambda_structures,
                delta_lambda_c_value=float(delta_lambda_c[frame]),
                dimension_names=dimension_names,
            )
            events.append(event)

        return events

    def _compute_delta_lambda_c(
        self,
        lambda_structures: dict[str, np.ndarray],
    ) -> np.ndarray | None:
        """
        ΔΛC = ρT · σₛ · |ΛF| を計算

        lambda_structures のキーに応じて柔軟に対応。
        """
        rho_t = lambda_structures.get("rho_T")
        sigma_s = lambda_structures.get("sigma_s")
        lambda_f_mag = lambda_structures.get("lambda_F_mag")

        if rho_t is None or sigma_s is None or lambda_f_mag is None:
            logger.warning("Missing lambda structure components for ΔΛC")
            return None

        # 長さを揃える（各成分のフレーム数が異なることがある）
        min_len = min(len(rho_t), len(sigma_s), len(lambda_f_mag))
        rho_t = rho_t[:min_len]
        sigma_s = sigma_s[:min_len]
        lambda_f_mag = lambda_f_mag[:min_len]

        # ΔΛC = ρT · σₛ · |ΛF|
        delta_lambda_c = rho_t * sigma_s * lambda_f_mag

        return delta_lambda_c

    def _detect_peaks(
        self,
        signal: np.ndarray,
        threshold: float,
        min_distance: int = 3,
    ) -> list[int]:
        """
        ピーク検出（閾値超過かつ局所最大）

        Parameters
        ----------
        signal : np.ndarray
            1D信号
        threshold : float
            最小ピーク高さ
        min_distance : int
            ピーク間の最小距離
        """
        # 閾値超過フレーム
        above = np.where(signal > threshold)[0]

        if len(above) == 0:
            return []

        # 局所最大のみ抽出
        peaks = []
        for idx in above:
            # 境界チェック
            if idx == 0 or idx >= len(signal) - 1:
                continue

            if signal[idx] >= signal[idx - 1] and signal[idx] >= signal[idx + 1]:
                peaks.append(idx)

        # 最小距離フィルタ
        if len(peaks) <= 1:
            return peaks

        filtered = [peaks[0]]
        for peak in peaks[1:]:
            if peak - filtered[-1] >= min_distance:
                filtered.append(peak)
            elif signal[peak] > signal[filtered[-1]]:
                # より強いピークで置換
                filtered[-1] = peak

        return filtered

    def _detect_peaks_adaptive(
        self,
        signal: np.ndarray,
        above_indices: np.ndarray,
        min_distance: int = 3,
    ) -> list[int]:
        """
        適応的閾値でのピーク検出

        Parameters
        ----------
        signal : np.ndarray
            1D信号
        above_indices : np.ndarray
            ローカル閾値を超過したフレームのインデックス
        min_distance : int
            ピーク間の最小距離
        """
        if len(above_indices) == 0:
            return []

        # 局所最大のみ抽出
        peaks = []
        for idx in above_indices:
            if idx == 0 or idx >= len(signal) - 1:
                continue
            if signal[idx] >= signal[idx - 1] and signal[idx] >= signal[idx + 1]:
                peaks.append(idx)

        if len(peaks) <= 1:
            return peaks

        # 最小距離フィルタ
        filtered = [peaks[0]]
        for peak in peaks[1:]:
            if peak - filtered[-1] >= min_distance:
                filtered.append(peak)
            elif signal[peak] > signal[filtered[-1]]:
                filtered[-1] = peak

        return filtered

    def _analyze_single_event(
        self,
        event_id: int,
        frame: int,
        state_vectors: np.ndarray,
        lambda_structures: dict[str, np.ndarray],
        delta_lambda_c_value: float,
        dimension_names: list[str],
    ) -> CascadeEvent:
        """
        単一イベントの解析

        Third Impact の _analyze_instantaneous_event + 伝播追跡の汎用版。
        各次元の変位 z-score から genesis/affected を特定する。

        ΔΛC検出フレームと実際の変位フレームにはオフセットが生じうる
        （lambda_F_mag が diff 演算ベースのため）。
        そのためイベントフレーム周辺 (±scan_radius) を探索して
        最大変位のフレームを真のイベントフレームとする。
        """
        n_frames, n_dims = state_vectors.shape
        scan_radius = 2

        event = CascadeEvent(
            event_id=event_id,
            frame=frame,
            delta_lambda_c=delta_lambda_c_value,
        )

        if frame < 1:
            return event

        # ── ベースライン統計: ルックバックウィンドウの「通常の変動」──
        baseline_start = max(0, frame - self.lookback - scan_radius)
        baseline_end = max(baseline_start + 2, frame - scan_radius)
        baseline_window = state_vectors[baseline_start:baseline_end]

        if len(baseline_window) < 3:
            # フォールバック: 全データから統計
            all_disp = np.abs(np.diff(state_vectors, axis=0))
            baseline_mean = np.mean(all_disp, axis=0)
            baseline_std = np.std(all_disp, axis=0)
        else:
            baseline_disp = np.abs(np.diff(baseline_window, axis=0))
            baseline_mean = np.mean(baseline_disp, axis=0)  # (n_dims,)
            baseline_std = np.std(baseline_disp, axis=0)    # (n_dims,)

        # ── Genesis 検出: フレーム周辺を探索 ──
        scan_start = max(1, frame - scan_radius)
        scan_end = min(n_frames - 1, frame + scan_radius + 1)

        best_z_scores = np.zeros(n_dims)

        for f in range(scan_start, scan_end):
            frame_disp = np.abs(state_vectors[f] - state_vectors[f - 1])
            z_scores = (frame_disp - baseline_mean) / (baseline_std + 1e-10)

            # 各次元で最大 z-score を保持
            best_z_scores = np.maximum(best_z_scores, z_scores)

        # 全次元の変位統計（レポート用）
        event.mean_displacement = float(np.mean(baseline_mean))
        event.std_displacement = float(np.mean(baseline_std))
        event.threshold_used = float(
            event.mean_displacement
            + self.sigma_threshold * event.std_displacement
        )

        # genesis dims: 周辺探索での最大 z-score が閾値超過
        for d in range(n_dims):
            if best_z_scores[d] > self.sigma_threshold:
                event.genesis_dims.append(d)

        # ── 伝播追跡: 後続フレームの波動 ──
        max_propagation = min(3, n_frames - frame - 1)
        all_affected = set(event.genesis_dims)

        for delta in range(1, max_propagation + 1):
            future_frame = frame + scan_radius + delta
            if future_frame >= n_frames:
                break

            future_disp = np.abs(
                state_vectors[future_frame] - state_vectors[future_frame - 1]
            )
            future_z = (future_disp - baseline_mean) / (baseline_std + 1e-10)

            for d in range(n_dims):
                if future_z[d] > self.sigma_threshold and d not in all_affected:
                    event.wave_dims.append(d)
                    all_affected.add(d)

        event.affected_dims = sorted(all_affected)

        # ── 名前付け ──
        event.genesis_names = [dimension_names[d] for d in event.genesis_dims]
        event.affected_names = [dimension_names[d] for d in event.affected_dims]

        # ── 伝播順序の推定 ──
        event.propagation_order = self._estimate_propagation_order(
            state_vectors, frame, n_dims
        )

        # ── ハブ次元の検出 ──
        event.hub_dims = self._detect_event_hubs(
            state_vectors, frame, event.affected_dims
        )

        return event

    def _estimate_propagation_order(
        self,
        state_vectors: np.ndarray,
        event_frame: int,
        n_dims: int,
    ) -> list[int]:
        """
        伝播順序の推定

        NetworkAnalyzerCore._estimate_propagation_order と同等ロジック。
        各次元が閾値を超えた最初のフレームで順序付け。
        """
        start = max(0, event_frame - self.lookback)
        window = state_vectors[start:event_frame + 1]

        if len(window) < 3:
            return list(range(n_dims))

        displacement = np.abs(np.diff(window, axis=0))

        onset_frames = np.full(n_dims, len(displacement))
        for d in range(n_dims):
            series = displacement[:, d]
            dim_threshold = np.mean(series) + 1.5 * np.std(series)

            exceeding = np.where(series > dim_threshold)[0]
            if len(exceeding) > 0:
                onset_frames[d] = exceeding[0]

        return list(np.argsort(onset_frames))

    def _detect_event_hubs(
        self,
        state_vectors: np.ndarray,
        event_frame: int,
        affected_dims: list[int],
    ) -> list[int]:
        """
        イベント時のハブ次元検出

        affected_dims の中で、他の次元との相関が高い次元をハブとする。
        """
        if len(affected_dims) < 3:
            return list(affected_dims)

        start = max(0, event_frame - self.lookback)
        window = state_vectors[start:event_frame + 1, :]

        if len(window) < 3:
            return []

        # affected_dims 内のペアワイズ相関
        connectivity = np.zeros(len(affected_dims))
        for i, d_i in enumerate(affected_dims):
            for j, d_j in enumerate(affected_dims):
                if i >= j:
                    continue
                corr = abs(np.corrcoef(window[:, d_i], window[:, d_j])[0, 1])
                if not np.isnan(corr):
                    connectivity[i] += corr
                    connectivity[j] += corr

        if np.max(connectivity) == 0:
            return []

        # 上位をハブとする
        hub_threshold = np.mean(connectivity) + np.std(connectivity)
        hubs = [
            affected_dims[i]
            for i in range(len(affected_dims))
            if connectivity[i] > hub_threshold
        ]

        return sorted(hubs, key=lambda d: connectivity[
            affected_dims.index(d)
        ], reverse=True)

    # ================================================================
    # Step 2: イベント間因果リンク判定
    # ================================================================

    def _build_causal_links(
        self,
        events: list[CascadeEvent],
        dimension_names: list[str],
    ) -> list[CascadeLink]:
        """
        イベント間の因果リンクを判定

        event_i.affected_dims ∩ event_j.genesis_dims ≠ ∅
        かつ event_j が event_i の直後（max_gap以内）に発生
        → event_i → event_j の因果リンク
        """
        links = []

        # 時系列順にソート（安全のため）
        sorted_events = sorted(events, key=lambda e: e.frame)

        for i, event_i in enumerate(sorted_events):
            for j in range(i + 1, len(sorted_events)):
                event_j = sorted_events[j]

                # フレーム間隔チェック
                gap = event_j.frame - event_i.frame
                if gap > self.max_gap:
                    break  # 以降のイベントはさらに遠いので打ち切り

                if gap <= 0:
                    continue

                # 因果判定: affected ∩ genesis
                shared = set(event_i.affected_dims) & set(event_j.genesis_dims)

                if len(shared) < self.min_shared_dims:
                    continue

                shared_list = sorted(shared)

                # 因果スコア
                # = 共有次元数 / max(affected, genesis) × 時間近接度
                coverage = len(shared) / max(
                    len(event_i.affected_dims),
                    len(event_j.genesis_dims),
                    1,
                )
                proximity = 1.0 / (1.0 + gap / self.max_gap)
                causality_score = coverage * proximity

                # リンクタイプ判定
                n_outgoing_i = sum(
                    1 for k in range(i + 1, len(sorted_events))
                    if sorted_events[k].frame - event_i.frame <= self.max_gap
                    and len(set(event_i.affected_dims)
                            & set(sorted_events[k].genesis_dims))
                        >= self.min_shared_dims
                )

                n_incoming_j = sum(
                    1 for k in range(j)
                    if event_j.frame - sorted_events[k].frame <= self.max_gap
                    and len(set(sorted_events[k].affected_dims)
                            & set(event_j.genesis_dims))
                        >= self.min_shared_dims
                )

                if n_outgoing_i > 1:
                    link_type = "branching"
                elif n_incoming_j > 1:
                    link_type = "merging"
                else:
                    link_type = "sequential"

                link = CascadeLink(
                    from_event_id=event_i.event_id,
                    to_event_id=event_j.event_id,
                    from_frame=event_i.frame,
                    to_frame=event_j.frame,
                    frame_gap=gap,
                    shared_dims=shared_list,
                    shared_names=[dimension_names[d] for d in shared_list],
                    causality_score=causality_score,
                    link_type=link_type,
                )
                links.append(link)

        return links

    # ================================================================
    # Step 3: 因果DAG構築
    # ================================================================

    def _build_dag(
        self,
        events: list[CascadeEvent],
        links: list[CascadeLink],
    ) -> dict[int, list[int]]:
        """
        因果DAGを隣接リスト形式で構築

        キー: event_id、値: 後続event_idのリスト
        """
        dag: dict[int, list[int]] = {e.event_id: [] for e in events}

        for link in links:
            dag[link.from_event_id].append(link.to_event_id)

        return dag

    # ================================================================
    # Step 4: カスケードチェーン抽出
    # ================================================================

    def _extract_chains(
        self,
        events: list[CascadeEvent],
        links: list[CascadeLink],
        dag: dict[int, list[int]],
        dimension_names: list[str],
    ) -> list[CascadeChain]:
        """
        DAGから全てのカスケードチェーンを抽出

        根ノード（入次数 0）から DFS で全パスを列挙。
        max_paths で組み合わせ爆発を防止。
        """
        # event_id → event のマッピング
        event_map = {e.event_id: e for e in events}

        # link のマッピング (from, to) → link
        link_map = {
            (lnk.from_event_id, lnk.to_event_id): lnk
            for lnk in links
        }

        # ── DAG枝刈り: 各ノードから上位 max_children 本のみ ──
        max_children = 3
        pruned_dag: dict[int, list[int]] = {}
        for node, children in dag.items():
            if len(children) <= max_children:
                pruned_dag[node] = children
            else:
                # causality_score の高い順に上位のみ保持
                scored = []
                for child in children:
                    key = (node, child)
                    score = link_map[key].causality_score if key in link_map else 0
                    scored.append((child, score))
                scored.sort(key=lambda x: x[1], reverse=True)
                pruned_dag[node] = [c for c, _ in scored[:max_children]]

        # 入次数の計算（枝刈り後のDAGで）
        in_degree: dict[int, int] = {e.event_id: 0 for e in events}
        for node, children in pruned_dag.items():
            for child in children:
                in_degree[child] = in_degree.get(child, 0) + 1

        # 根ノード（因果の起点）
        roots = [eid for eid, deg in in_degree.items() if deg == 0]

        # DFS で全パスを列挙（max_paths で制限）
        max_paths = self.max_chains * 10  # 十分な候補を生成
        chains = []
        chain_id = 0

        for root in roots:
            paths = self._dfs_all_paths(
                root, pruned_dag,
                max_depth=15,
                max_paths=max_paths - len(chains),
            )

            if len(chains) >= max_paths:
                break

            for path in paths:
                if len(path) < 2:
                    continue

                # チェーン構築
                chain_links = []
                for k in range(len(path) - 1):
                    key = (path[k], path[k + 1])
                    if key in link_map:
                        chain_links.append(link_map[key])

                origin_event = event_map[path[0]]
                terminal_event = event_map[path[-1]]

                chain = CascadeChain(
                    chain_id=chain_id,
                    event_ids=path,
                    links=chain_links,
                    length=len(path),
                    total_frames=(
                        terminal_event.frame - origin_event.frame
                    ),
                    origin_dims=origin_event.genesis_dims,
                    terminal_dims=terminal_event.affected_dims,
                    origin_names=[
                        dimension_names[d]
                        for d in origin_event.genesis_dims
                    ],
                    terminal_names=[
                        dimension_names[d]
                        for d in terminal_event.affected_dims
                    ],
                    mean_causality=(
                        np.mean([lnk.causality_score for lnk in chain_links])
                        if chain_links else 0.0
                    ),
                    total_delta_lambda_c=sum(
                        event_map[eid].delta_lambda_c for eid in path
                    ),
                )
                chains.append(chain)
                chain_id += 1

                if len(chains) >= max_paths:
                    break

        # 長さ × 因果スコアでソート
        chains.sort(
            key=lambda c: c.length * c.mean_causality,
            reverse=True,
        )

        # min_causality フィルタ
        chains = [c for c in chains if c.mean_causality >= self.min_causality]

        # max_chains で上位のみ保持
        chains = chains[:self.max_chains]

        return chains

    def _dfs_all_paths(
        self,
        start: int,
        dag: dict[int, list[int]],
        max_depth: int = 15,
        max_paths: int = 200,
    ) -> list[list[int]]:
        """
        DAG上のDFSで全パスを列挙

        max_depth + max_paths で組み合わせ爆発を防止
        """
        all_paths: list[list[int]] = []
        stack: list[tuple[int, list[int]]] = [(start, [start])]

        while stack:
            if len(all_paths) >= max_paths:
                break

            node, path = stack.pop()

            if len(path) > max_depth:
                all_paths.append(path)
                continue

            children = dag.get(node, [])

            if not children:
                all_paths.append(path)
            else:
                for child in children:
                    if child not in path:
                        stack.append((child, path + [child]))

        return all_paths

    # ================================================================
    # Step 5: クリティカル次元特定
    # ================================================================

    def _identify_critical_dims(
        self,
        events: list[CascadeEvent],
        links: list[CascadeLink],
        chains: list[CascadeChain],
        n_dims: int,
    ) -> list[int]:
        """
        クリティカル次元の特定（drug_target_atoms の汎用版）

        Third Impact の考え方を踏襲:
          1. 複数チェーンに genesis として登場する次元 → 起源的に重要
          2. 因果リンクの共有次元に頻出する次元 → 伝達の要
          3. ハブ次元 → ネットワーク的に重要
        """
        dim_scores = np.zeros(n_dims)

        # 1. genesis 頻度
        for event in events:
            for d in event.genesis_dims:
                dim_scores[d] += 2.0

        # 2. 因果リンク共有次元の頻度
        for link in links:
            for d in link.shared_dims:
                dim_scores[d] += link.causality_score * 3.0

        # 3. ハブ次元
        for event in events:
            for d in event.hub_dims:
                dim_scores[d] += 1.5

        # 4. チェーン起源・終端
        for chain in chains:
            for d in chain.origin_dims:
                dim_scores[d] += chain.length * 1.0
            for d in chain.terminal_dims:
                dim_scores[d] += 0.5

        # 上位を返す
        if np.max(dim_scores) == 0:
            return []

        threshold = np.mean(dim_scores) + np.std(dim_scores)
        critical = np.where(dim_scores > threshold)[0]

        return sorted(critical, key=lambda d: dim_scores[d], reverse=True)

    # ================================================================
    # Statistics
    # ================================================================

    def _compute_statistics(
        self,
        result: CascadeResult,
        n_frames: int,
    ):
        """グローバル統計を計算"""
        if result.n_chains > 0:
            result.mean_chain_length = np.mean(
                [c.length for c in result.chains]
            )

            # カスケードの時間的カバー率
            covered_frames = set()
            for event in result.events:
                covered_frames.add(event.frame)
            result.cascade_coverage = len(covered_frames) / max(n_frames, 1)

            # 分岐率
            branching_links = sum(
                1 for lnk in result.links if lnk.link_type == "branching"
            )
            result.branching_ratio = (
                branching_links / max(result.n_links, 1)
            )

    # ================================================================
    # Output
    # ================================================================

    def _print_summary(self, result: CascadeResult):
        """結果サマリーの表示"""
        logger.info("=" * 50)
        logger.info("🔺 Cascade Tracker Summary")
        logger.info("=" * 50)
        logger.info(f"  Events: {result.n_events}")
        logger.info(f"  Causal links: {result.n_links}")
        logger.info(f"  Cascade chains: {result.n_chains}")
        logger.info(f"  Longest chain: {result.longest_chain}")
        logger.info(f"  Mean chain length: {result.mean_chain_length:.1f}")
        logger.info(f"  Branching ratio: {result.branching_ratio:.2f}")
        logger.info(f"  Coverage: {result.cascade_coverage:.1%}")

        if result.critical_names:
            logger.info(
                f"  Critical dims: {', '.join(result.critical_names[:5])}"
            )

        # トップチェーンの詳細
        for chain in result.chains[:3]:
            logger.info(
                f"\n  Chain #{chain.chain_id}: "
                f"{' → '.join(str(eid) for eid in chain.event_ids)}"
            )
            logger.info(
                f"    Length: {chain.length}, "
                f"Frames: {chain.total_frames}, "
                f"Causality: {chain.mean_causality:.3f}"
            )
            logger.info(
                f"    Origin: {', '.join(chain.origin_names[:3])}"
            )
            logger.info(
                f"    Terminal: {', '.join(chain.terminal_names[:3])}"
            )
