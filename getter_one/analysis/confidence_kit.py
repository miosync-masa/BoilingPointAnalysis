"""
GETTER One - Confidence Kit
=============================

Detection結果とNetwork結果に対して統計的信頼度を付与する。

3本柱:
  ① Permutation Test → p値（偶然性の否定）
  ② Bootstrap CI     → 信頼区間（値の安定性）
  ③ Effect Size      → 効果量（実質的な意味の大きさ）

MCMCなし、PyMCなし、arvizなし。numpy/scipy完結。

入力:
  - state_vectors: 元データ（シャッフル用）
  - lambda_structures: Λ³構造計算結果
  - structural_boundaries: 境界検出結果
  - topological_breaks: トポロジカル破れ結果
  - anomaly_scores: 異常スコア結果
  - network_result: ネットワーク解析結果

Built with 💕 by Masamichi & Tamaki
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("getter_one.analysis.confidence_kit")


# ============================================================
# Result Data Classes
# ============================================================

@dataclass
class EventConfidence:
    """ΔΛCイベントの信頼度"""
    frame: int
    delta_lambda_c: float
    p_value: float
    ci_lower: float
    ci_upper: float
    effect_size: float          # Cohen's d
    is_significant: bool        # p < alpha


@dataclass
class BoundaryConfidence:
    """構造境界の信頼度"""
    frame: int
    effect_size: float          # 境界前後のΛ構造差
    p_value: float              # パーミュテーションp値
    ci_lower: float
    ci_upper: float
    is_significant: bool


@dataclass
class CausalLinkConfidence:
    """因果リンクの信頼度"""
    from_dim: int
    to_dim: int
    from_name: str
    to_name: str
    observed_strength: float
    p_value: float
    ci_lower: float
    ci_upper: float
    effect_size: float
    lag: int
    is_significant: bool


@dataclass
class SyncConfidence:
    """同期リンクの信頼度"""
    dim_i: int
    dim_j: int
    name_i: str
    name_j: str
    observed_correlation: float
    p_value: float
    ci_lower: float
    ci_upper: float
    is_significant: bool


@dataclass
class ConfidenceReport:
    """信頼度判定の全結果"""
    # イベント信頼度
    events: list[EventConfidence] = field(default_factory=list)
    n_significant_events: int = 0

    # 境界信頼度
    boundaries: list[BoundaryConfidence] = field(default_factory=list)
    n_significant_boundaries: int = 0

    # 因果リンク信頼度
    causal_links: list[CausalLinkConfidence] = field(default_factory=list)
    n_significant_causal: int = 0

    # 同期リンク信頼度
    sync_links: list[SyncConfidence] = field(default_factory=list)
    n_significant_sync: int = 0

    # 全体統計
    alpha: float = 0.05
    n_permutations: int = 1000
    n_bootstrap: int = 2000

    def summary(self) -> str:
        """サマリー文字列"""
        lines = [
            "=" * 50,
            "  GETTER One Confidence Report",
            "=" * 50,
            f"  Alpha: {self.alpha}  |  Permutations: {self.n_permutations}  |  Bootstrap: {self.n_bootstrap}",
            "",
            f"  Events:     {self.n_significant_events}/{len(self.events)} significant",
            f"  Boundaries: {self.n_significant_boundaries}/{len(self.boundaries)} significant",
            f"  Causal:     {self.n_significant_causal}/{len(self.causal_links)} significant",
            f"  Sync:       {self.n_significant_sync}/{len(self.sync_links)} significant",
            "=" * 50,
        ]
        return "\n".join(lines)


# ============================================================
# Main API
# ============================================================

def assess_confidence(
    state_vectors: np.ndarray,
    lambda_structures: dict[str, np.ndarray],
    structural_boundaries: dict | None = None,
    anomaly_scores: dict | None = None,
    network_result=None,
    dimension_names: list[str] | None = None,
    alpha: float = 0.05,
    n_permutations: int = 1000,
    n_bootstrap: int = 2000,
    window_steps: int = 24,
    seed: int = 42,
) -> ConfidenceReport:
    """
    検出結果に統計的信頼度を付与する。

    Parameters
    ----------
    state_vectors : np.ndarray (n_frames, n_dims)
        元の状態ベクトルデータ
    lambda_structures : dict
        LambdaStructuresCore の出力
    structural_boundaries : dict, optional
        BoundaryDetector の出力
    anomaly_scores : dict, optional
        AnomalyDetector の出力
    network_result : NetworkResult, optional
        NetworkAnalyzerCore の出力
    dimension_names : list[str], optional
    alpha : float
        有意水準
    n_permutations : int
        パーミュテーション回数
    n_bootstrap : int
        ブートストラップ回数
    seed : int
        乱数シード
    """
    rng = np.random.default_rng(seed)
    n_frames, n_dims = state_vectors.shape

    if dimension_names is None:
        dimension_names = [f"dim_{i}" for i in range(n_dims)]

    logger.info(
        f"🔍 Assessing confidence (α={alpha}, "
        f"perm={n_permutations}, boot={n_bootstrap})"
    )

    report = ConfidenceReport(
        alpha=alpha,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
    )

    # ① ΔΛCイベントの信頼度
    report.events = _assess_event_confidence(
        state_vectors, lambda_structures, alpha, n_permutations, n_bootstrap, rng,
        window_steps=window_steps,
    )
    report.n_significant_events = sum(1 for e in report.events if e.is_significant)

    # ② 構造境界の信頼度
    if structural_boundaries:
        report.boundaries = _assess_boundary_confidence(
            lambda_structures, structural_boundaries, alpha, n_permutations, n_bootstrap, rng
        )
        report.n_significant_boundaries = sum(
            1 for b in report.boundaries if b.is_significant
        )

    # ③④ ネットワークリンクの信頼度
    if network_result is not None:
        report.causal_links = _assess_causal_confidence(
            state_vectors, network_result, dimension_names,
            alpha, n_permutations, n_bootstrap, rng
        )
        report.n_significant_causal = sum(
            1 for c in report.causal_links if c.is_significant
        )

        report.sync_links = _assess_sync_confidence(
            state_vectors, network_result, dimension_names,
            alpha, n_permutations, n_bootstrap, rng
        )
        report.n_significant_sync = sum(
            1 for s in report.sync_links if s.is_significant
        )

    logger.info(f"\n{report.summary()}")
    return report


# ============================================================
# ① ΔΛCイベント信頼度
# ============================================================

def _assess_event_confidence(
    state_vectors: np.ndarray,
    lambda_structures: dict[str, np.ndarray],
    alpha: float,
    n_perm: int,
    n_boot: int,
    rng: np.random.Generator,
    window_steps: int = 24,
) -> list[EventConfidence]:
    """ΔΛCイベントがノイズでないことを検定

    帰無仮説: 次元間の同期構造が存在しない場合、
    観測されたΔΛCの大きさは偶然でも生じうるか？

    シャッフルしたデータからΛ³全体（ΛF, ρT, σₛ）を再計算して
    帰無分布を生成する。
    """
    lf_mag = lambda_structures.get("lambda_F_mag", np.array([]))
    rho_t = lambda_structures.get("rho_T", np.array([]))
    sigma_s = lambda_structures.get("sigma_s", np.array([]))

    if len(lf_mag) == 0:
        return []

    # ΔΛC = ρT × σₛ × |ΛF| を再計算
    n = min(len(lf_mag), len(rho_t) - 1, len(sigma_s) - 1)
    dlc = rho_t[1:n + 1] * sigma_s[1:n + 1] * lf_mag[:n]

    # イベント検出（2σ超え）
    threshold = np.mean(dlc) + 2 * np.std(dlc)
    event_frames = np.where(dlc > threshold)[0]

    if len(event_frames) == 0:
        return []

    # パーミュテーション帰無分布の生成
    # シャッフルしたデータからΛ³全体を再計算する
    null_dlc_distributions = []
    for p in range(n_perm):
        # 各次元を独立にシャッフル → 構造的同期を破壊
        shuffled = state_vectors.copy()
        for d in range(state_vectors.shape[1]):
            shuffled[:, d] = rng.permutation(shuffled[:, d])

        # ΛF再計算
        s_disp = np.diff(shuffled, axis=0)
        s_lf_mag = np.linalg.norm(s_disp, axis=1)

        # ρT再計算（rolling std）
        s_n_frames = len(shuffled)
        s_rho_t = np.zeros(s_n_frames)
        half_w = window_steps // 2
        for t in range(s_n_frames):
            start = max(0, t - half_w)
            end = min(len(s_lf_mag), t + half_w + 1)
            if end > start:
                s_rho_t[t] = np.std(s_lf_mag[start:end])

        # σₛ再計算（cross-dim correlation）
        s_sigma_s = np.zeros(s_n_frames)
        n_dims = shuffled.shape[1]
        for t in range(s_n_frames):
            start = max(0, t - half_w)
            end = min(len(s_disp), t + half_w + 1)
            if end - start >= 3 and n_dims >= 2:
                window_disp = s_disp[start:end]
                try:
                    corr = np.corrcoef(window_disp.T)
                    mask = ~np.eye(n_dims, dtype=bool)
                    s_sigma_s[t] = np.mean(np.abs(corr[mask]))
                except (ValueError, FloatingPointError):
                    s_sigma_s[t] = 0.0

        # ΔΛC再計算
        s_m = min(len(s_lf_mag), s_n_frames - 1)
        s_dlc = s_rho_t[1:s_m + 1] * s_sigma_s[1:s_m + 1] * s_lf_mag[:s_m]
        null_dlc_distributions.append(s_dlc)

    # 帰無分布: 全シャッフルの全フレームのΔΛCを統合
    null_all = np.concatenate(null_dlc_distributions)

    events = []
    for frame in event_frames:
        observed = dlc[frame]

        # p値: 帰無分布全体で観測値以上が出る確率
        p_value = float(np.mean(null_all >= observed))
        # 最小p値の補正
        p_value = max(p_value, 1.0 / (len(null_all) + 1))

        # ブートストラップCI
        ci_lo, ci_hi = _bootstrap_ci_scalar(dlc, frame, n_boot, rng)

        # 効果量（Cohen's d）
        effect = (observed - np.mean(null_all)) / max(np.std(null_all), 1e-10)

        events.append(EventConfidence(
            frame=frame,
            delta_lambda_c=float(observed),
            p_value=float(p_value),
            ci_lower=float(ci_lo),
            ci_upper=float(ci_hi),
            effect_size=float(effect),
            is_significant=p_value < alpha,
        ))

    return events


# ============================================================
# ② 構造境界の信頼度
# ============================================================

def _assess_boundary_confidence(
    lambda_structures: dict[str, np.ndarray],
    boundaries: dict,
    alpha: float,
    n_perm: int,
    n_boot: int,
    rng: np.random.Generator,
) -> list[BoundaryConfidence]:
    """構造境界が本物のレジーム変化であることを検定"""
    boundary_locs = boundaries.get("boundary_locations", [])
    if len(boundary_locs) == 0:
        return []

    rho_t = lambda_structures.get("rho_T", np.array([]))

    results = []
    for loc in boundary_locs:
        # 境界前後のΛ構造の差を効果量として計算
        window = min(50, loc, len(rho_t) - loc - 1)
        if window < 5:
            continue

        before = rho_t[loc - window:loc]
        after = rho_t[loc:loc + window]

        # 効果量: 前後の平均差 / プールド標準偏差
        effect = _cohens_d(before, after)

        # パーミュテーション検定: ρTを分割してランダムに割り当て
        combined = np.concatenate([before, after])
        null_effects = np.zeros(n_perm)
        for p in range(n_perm):
            perm = rng.permutation(combined)
            null_before = perm[:window]
            null_after = perm[window:]
            null_effects[p] = abs(_cohens_d(null_before, null_after))

        p_value = (1 + np.sum(null_effects >= abs(effect))) / (n_perm + 1)

        # ブートストラップCI for effect size
        boot_effects = np.zeros(n_boot)
        for b in range(n_boot):
            b_before = rng.choice(before, len(before), replace=True)
            b_after = rng.choice(after, len(after), replace=True)
            boot_effects[b] = _cohens_d(b_before, b_after)

        ci_lo = float(np.percentile(boot_effects, 2.5))
        ci_hi = float(np.percentile(boot_effects, 97.5))

        results.append(BoundaryConfidence(
            frame=int(loc),
            effect_size=float(effect),
            p_value=float(p_value),
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            is_significant=p_value < alpha,
        ))

    return results


# ============================================================
# ③ 因果リンクの信頼度
# ============================================================

def _assess_causal_confidence(
    state_vectors: np.ndarray,
    network_result,
    dimension_names: list[str],
    alpha: float,
    n_perm: int,
    n_boot: int,
    rng: np.random.Generator,
) -> list[CausalLinkConfidence]:
    """因果リンクが偶然でないことを検定"""
    causal_links = getattr(network_result, "causal_network", [])
    if not causal_links:
        return []

    displacement = np.diff(state_vectors, axis=0)
    results = []

    for link in causal_links:
        i, j = link.from_dim, link.to_dim
        lag = link.lag
        observed = link.strength

        ts_i = displacement[:, i]
        ts_j = displacement[:, j]

        # パーミュテーション: ソース系列をシャッフルしてラグ相関を再計算
        null_strengths = np.zeros(n_perm)
        for p in range(n_perm):
            perm_i = rng.permutation(ts_i)
            if lag > 0 and lag < len(perm_i):
                corr = np.corrcoef(perm_i[:-lag], ts_j[lag:])[0, 1]
                null_strengths[p] = abs(corr) if not np.isnan(corr) else 0.0

        p_value = (1 + np.sum(null_strengths >= observed)) / (n_perm + 1)

        # ブートストラップCI
        boot_strengths = np.zeros(n_boot)
        n_samples = len(ts_i) - lag
        for b in range(n_boot):
            idx = rng.choice(n_samples, n_samples, replace=True)
            corr = np.corrcoef(ts_i[idx], ts_j[idx + lag])[0, 1]
            boot_strengths[b] = abs(corr) if not np.isnan(corr) else 0.0

        ci_lo = float(np.percentile(boot_strengths, 2.5))
        ci_hi = float(np.percentile(boot_strengths, 97.5))

        # 効果量
        effect = (observed - np.mean(null_strengths)) / max(np.std(null_strengths), 1e-10)

        results.append(CausalLinkConfidence(
            from_dim=i,
            to_dim=j,
            from_name=dimension_names[i],
            to_name=dimension_names[j],
            observed_strength=float(observed),
            p_value=float(p_value),
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            effect_size=float(effect),
            lag=lag,
            is_significant=p_value < alpha,
        ))

    return results


# ============================================================
# ④ 同期リンクの信頼度
# ============================================================

def _assess_sync_confidence(
    state_vectors: np.ndarray,
    network_result,
    dimension_names: list[str],
    alpha: float,
    n_perm: int,
    n_boot: int,
    rng: np.random.Generator,
) -> list[SyncConfidence]:
    """同期リンクが偶然でないことを検定"""
    sync_links = getattr(network_result, "sync_network", [])
    if not sync_links:
        return []

    displacement = np.diff(state_vectors, axis=0)
    results = []

    for link in sync_links:
        i, j = link.from_dim, link.to_dim
        observed = link.correlation

        ts_i = displacement[:, i]
        ts_j = displacement[:, j]

        # パーミュテーション
        null_corrs = np.zeros(n_perm)
        for p in range(n_perm):
            perm_i = rng.permutation(ts_i)
            corr = np.corrcoef(perm_i, ts_j)[0, 1]
            null_corrs[p] = corr if not np.isnan(corr) else 0.0

        p_value = (1 + np.sum(np.abs(null_corrs) >= abs(observed))) / (n_perm + 1)

        # ブートストラップCI
        boot_corrs = np.zeros(n_boot)
        n_samples = len(ts_i)
        for b in range(n_boot):
            idx = rng.choice(n_samples, n_samples, replace=True)
            corr = np.corrcoef(ts_i[idx], ts_j[idx])[0, 1]
            boot_corrs[b] = corr if not np.isnan(corr) else 0.0

        ci_lo = float(np.percentile(boot_corrs, 2.5))
        ci_hi = float(np.percentile(boot_corrs, 97.5))

        results.append(SyncConfidence(
            dim_i=i,
            dim_j=j,
            name_i=dimension_names[i],
            name_j=dimension_names[j],
            observed_correlation=float(observed),
            p_value=float(p_value),
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            is_significant=p_value < alpha,
        ))

    return results


# ============================================================
# Utilities
# ============================================================

def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d 効果量"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def _bootstrap_ci_scalar(
    series: np.ndarray,
    index: int,
    n_bootstrap: int,
    rng: np.random.Generator,
    ci: float = 0.95,
) -> tuple[float, float]:
    """スカラー値のブートストラップ信頼区間"""
    window = min(50, index, len(series) - index - 1)
    if window < 3:
        return float(series[index]), float(series[index])

    local = series[max(0, index - window):index + window + 1]
    boot_vals = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(local, len(local), replace=True)
        boot_vals[b] = np.max(sample)

    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100
    return float(np.percentile(boot_vals, lo_pct)), float(np.percentile(boot_vals, hi_pct))
