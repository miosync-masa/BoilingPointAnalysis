"""
GETTER One - Report Generator
===============================

パイプライン全結果を人間可読なレポートに変換する。

出力フォーマット:
  - テキスト（ターミナル表示）
  - Markdown（保存・共有用）

Built with 💕 by Masamichi & Tamaki
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


import numpy as np

logger = logging.getLogger("getter_one.analysis.report_generator")


def generate_report(
    lambda_structures: dict[str, np.ndarray],
    structural_boundaries: dict | None = None,
    topological_breaks: dict | None = None,
    anomaly_scores: dict | None = None,
    network_result=None,
    confidence_report=None,
    dataset=None,
    output_path: str | None = None,
    title: str = "GETTER One Analysis Report",
) -> str:
    """
    パイプライン結果からレポートを生成する。

    Parameters
    ----------
    lambda_structures : dict
        LambdaStructuresCore の出力
    structural_boundaries : dict, optional
    topological_breaks : dict, optional
    anomaly_scores : dict, optional
    network_result : NetworkResult, optional
    confidence_report : ConfidenceReport, optional
    dataset : GetterDataset, optional
        元データ情報
    output_path : str, optional
        Markdownファイルの保存先
    title : str
        レポートタイトル

    Returns
    -------
    str
        レポート文字列（Markdown形式）
    """
    lines = []

    # ── ヘッダー ──
    lines.append(f"# {title}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # ── データ概要 ──
    lines.append("## 1. Data Summary")
    if dataset is not None:
        lines.append(f"- **Frames**: {dataset.n_frames}")
        lines.append(f"- **Dimensions**: {dataset.n_dims}")
        lines.append(f"- **Columns**: {', '.join(dataset.dimension_names)}")
        if dataset.target_name:
            lines.append(f"- **Target**: {dataset.target_name}")
        if dataset.timestamps is not None:
            lines.append(f"- **Time range**: {dataset.timestamps[0]} → {dataset.timestamps[-1]}")
        lines.append(f"- **Source**: {dataset.metadata.get('source', 'unknown')}")
    else:
        n_frames = len(lambda_structures.get("rho_T", []))
        n_dims = lambda_structures.get("lambda_F", np.zeros((1, 1))).shape[-1]
        lines.append(f"- **Frames**: {n_frames}")
        lines.append(f"- **Dimensions**: {n_dims}")
    lines.append("")

    # ── Λ³構造統計 ──
    lines.append("## 2. Lambda³ Structure Statistics")
    lines.append("")
    lines.append("| Metric | Min | Max | Mean | Std |")
    lines.append("|--------|-----|-----|------|-----|")

    for key in ["lambda_F_mag", "lambda_FF_mag", "rho_T",
                 "Q_cumulative", "sigma_s", "structural_coherence"]:
        if key in lambda_structures and len(lambda_structures[key]) > 0:
            data = lambda_structures[key]
            lines.append(
                f"| {key} | {np.min(data):.4e} | {np.max(data):.4e} | "
                f"{np.mean(data):.4e} | {np.std(data):.4e} |"
            )
    lines.append("")

    # ── 構造境界 ──
    if structural_boundaries:
        boundary_locs = structural_boundaries.get("boundary_locations", [])
        lines.append("## 3. Structural Boundaries")
        lines.append(f"**Detected: {len(boundary_locs)} boundaries**")
        lines.append("")

        if boundary_locs:
            lines.append("| # | Frame | Confidence |")
            lines.append("|---|-------|------------|")
            conf_map = {}
            if confidence_report:
                for b in confidence_report.boundaries:
                    conf_map[b.frame] = b

            for i, loc in enumerate(boundary_locs):
                conf = conf_map.get(int(loc))
                if conf:
                    sig = "✅" if conf.is_significant else "❌"
                    lines.append(
                        f"| {i+1} | {loc} | {sig} p={conf.p_value:.4f}, "
                        f"d={conf.effect_size:.3f} |"
                    )
                else:
                    lines.append(f"| {i+1} | {loc} | - |")
        lines.append("")

    # ── トポロジカル破れ ──
    if topological_breaks:
        lines.append("## 4. Topological Breaks")
        for key, val in topological_breaks.items():
            if isinstance(val, np.ndarray):
                lines.append(f"- **{key}**: mean={np.mean(val):.4e}, max={np.max(val):.4e}")
        lines.append("")

    # ── 異常スコア ──
    if anomaly_scores and "combined" in anomaly_scores:
        combined = anomaly_scores["combined"]
        threshold = np.mean(combined) + 2 * np.std(combined)
        n_critical = int(np.sum(combined > threshold))
        lines.append("## 5. Anomaly Detection")
        lines.append(f"- **Combined score**: mean={np.mean(combined):.4f}, max={np.max(combined):.4f}")
        lines.append(f"- **Threshold (2σ)**: {threshold:.4f}")
        lines.append(f"- **Critical frames**: {n_critical}")
        lines.append("")

    # ── ネットワーク解析 ──
    if network_result is not None:
        lines.append("## 6. Causal Network Analysis")
        lines.append(f"- **Pattern**: {network_result.pattern}")
        lines.append(f"- **Sync links**: {network_result.n_sync_links}")
        lines.append(f"- **Causal links**: {network_result.n_causal_links}")

        if network_result.hub_names:
            lines.append(f"- **Hub dimensions**: {', '.join(network_result.hub_names)}")
        if network_result.driver_names:
            lines.append(f"- **Causal drivers**: {', '.join(network_result.driver_names)}")
        if network_result.follower_names:
            lines.append(f"- **Causal followers**: {', '.join(network_result.follower_names)}")
        lines.append("")

        # 同期ネットワーク
        if network_result.sync_network:
            lines.append("### Sync Network")
            lines.append("| Dim A | Dim B | Correlation | Confidence |")
            lines.append("|-------|-------|-------------|------------|")

            conf_sync = {}
            if confidence_report:
                for s in confidence_report.sync_links:
                    conf_sync[(s.dim_i, s.dim_j)] = s

            for link in sorted(network_result.sync_network,
                              key=lambda lnk: lnk.strength, reverse=True):
                sign = "+" if link.correlation > 0 else "−"
                conf = conf_sync.get((link.from_dim, link.to_dim))
                if conf:
                    sig = "✅" if conf.is_significant else "❌"
                    conf_str = f"{sig} p={conf.p_value:.4f} CI=[{conf.ci_lower:.3f},{conf.ci_upper:.3f}]"
                else:
                    conf_str = "-"
                lines.append(
                    f"| {link.from_name} | {link.to_name} | "
                    f"{sign}{link.strength:.3f} | {conf_str} |"
                )
            lines.append("")

        # 因果ネットワーク
        if network_result.causal_network:
            lines.append("### Causal Network")
            lines.append("| Source | Target | Strength | Lag | Confidence |")
            lines.append("|--------|--------|----------|-----|------------|")

            conf_causal = {}
            if confidence_report:
                for c in confidence_report.causal_links:
                    conf_causal[(c.from_dim, c.to_dim)] = c

            for link in sorted(network_result.causal_network,
                              key=lambda lnk: lnk.strength, reverse=True):
                conf = conf_causal.get((link.from_dim, link.to_dim))
                if conf:
                    sig = "✅" if conf.is_significant else "❌"
                    conf_str = f"{sig} p={conf.p_value:.4f} CI=[{conf.ci_lower:.3f},{conf.ci_upper:.3f}]"
                else:
                    conf_str = "-"
                lines.append(
                    f"| {link.from_name} | {link.to_name} | "
                    f"{link.strength:.3f} | {link.lag} | {conf_str} |"
                )
            lines.append("")

    # ── 信頼度サマリー ──
    if confidence_report:
        lines.append("## 7. Confidence Summary")
        lines.append(f"- **Alpha**: {confidence_report.alpha}")
        lines.append(f"- **Permutations**: {confidence_report.n_permutations}")
        lines.append(f"- **Bootstrap samples**: {confidence_report.n_bootstrap}")
        lines.append(f"- **Significant events**: {confidence_report.n_significant_events}/{len(confidence_report.events)}")
        lines.append(f"- **Significant boundaries**: {confidence_report.n_significant_boundaries}/{len(confidence_report.boundaries)}")
        lines.append(f"- **Significant causal links**: {confidence_report.n_significant_causal}/{len(confidence_report.causal_links)}")
        lines.append(f"- **Significant sync links**: {confidence_report.n_significant_sync}/{len(confidence_report.sync_links)}")
        lines.append("")

    # ── フッター ──
    lines.append("---")
    lines.append("*Generated by GETTER One (Geometric Event-driven Tensor-based "
                 "Time-series Extraction & Recognition)*")
    lines.append(f"*Version 0.1.0 | {datetime.now().strftime('%Y-%m-%d')}*")

    report = "\n".join(lines)

    # 保存
    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")
        logger.info(f"📄 Report saved: {output_path}")

    return report
