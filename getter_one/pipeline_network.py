"""
GETTER One - Pipeline Network
===============================
Network系統専用パイプライン。

  DualCore (Λ³特徴量) → NetworkAnalyzerCore → Confidence → Report

ΔΛCイベント伝播確率に基づく構造推定のみを行う。
Detection / Cascade / InverseChecker はここには入らない。

Usage:
    from getter_one.pipeline_network import run_network

    result = run_network("data.csv")
    print(result.network.pattern)
    print(result.network.driver_names)

Built with 💕 by Masamichi & Tamaki
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .analysis.network_analyzer_core import NetworkAnalyzerCore, NetworkResult
from .data.loader import GetterDataset, load

logger = logging.getLogger("getter_one.pipeline_network")


# ============================================================
# Config
# ============================================================


@dataclass
class NetworkPipelineConfig:
    """Network Pipeline 設定"""

    # Λ³構造 (DualCore)
    window_steps: int | None = None  # None = アダプティブ
    adaptive_window: bool = True
    delta_percentile: float = 94.0

    # NetworkAnalyzerCore
    sync_threshold: float = 0.05
    causal_threshold: float = 0.08
    max_lag: int = 10
    local_std_window: int = 20
    rho_t_window: int = 30
    n_permutations: int = 200
    p_value_threshold: float = 0.05
    rho_t_weight: float = 0.3
    enable_gpu_verification: bool = False

    # Confidence (network-related only)
    enable_confidence: bool = True
    alpha: float = 0.05
    n_bootstrap: int = 2000

    # Report
    enable_report: bool = True
    report_path: str | None = None

    # General
    seed: int = 42
    verbose: bool = True


# ============================================================
# Result
# ============================================================


@dataclass
class NetworkPipelineResult:
    """Network Pipeline 結果"""

    config: NetworkPipelineConfig = field(default_factory=NetworkPipelineConfig)
    dataset: GetterDataset | None = None

    # Lambda (DualCore)
    lambda_structures: dict = field(default_factory=dict)
    local_lambda: dict = field(default_factory=dict)

    # Network
    network: NetworkResult | None = None

    # Confidence (network分のみ)
    confidence: object | None = None

    # Report
    report: str | None = None

    # Timing
    computation_time: float = 0.0


# ============================================================
# Pipeline
# ============================================================


def run_network(
    source,
    config: NetworkPipelineConfig | None = None,
    **kwargs,
) -> NetworkPipelineResult:
    """
    GETTER One Network Pipeline

    ΔΛCイベント伝播確率に基づく構造推定パイプライン。

    Parameters
    ----------
    source : str, Path, GetterDataset, or np.ndarray
        入力データ。以下のいずれか:
        - ファイルパス（csv/json/parquet/xlsx/npy/npz）
        - GetterDataset
        - numpy配列 (n_frames, n_dims)
    config : NetworkPipelineConfig, optional
        パイプライン設定
    **kwargs
        load() に渡す追加引数（target, time_column, normalize等）

    Returns
    -------
    NetworkPipelineResult
    """
    t_start = time.time()

    if config is None:
        config = NetworkPipelineConfig()

    result = NetworkPipelineResult(config=config)

    # ── 1. データ読み込み ──
    if isinstance(source, GetterDataset):
        dataset = source
    elif isinstance(source, np.ndarray):
        from .data.loader import from_numpy

        dataset = from_numpy(source, normalize=kwargs.pop("normalize", "range"))
    else:
        dataset = load(source, **kwargs)

    result.dataset = dataset

    if config.verbose:
        print(f"\n{'=' * 60}")
        print("  GETTER One — Network Pipeline")
        print(f"{'=' * 60}")
        print(f"  Data: {dataset.n_frames} frames × {dataset.n_dims} dims")
        print(f"  Dimensions: {', '.join(dataset.dimension_names)}")

    # ── 2. Lambda (DualCore) → Local + Global ──
    if config.verbose:
        print("\n  [1] Lambda DualCore...")

    from .structures.lambda_structures_dual_core import (
        DualCoreConfig,
        LambdaStructuresDualCore,
    )

    dual_config = DualCoreConfig(
        verbose=config.verbose,
        adaptive_window=config.adaptive_window,
        delta_percentile=config.delta_percentile,
    )
    core = LambdaStructuresDualCore(dual_config)
    all_structures = core.compute(
        state_vectors=dataset.state_vectors,
        window_steps=config.window_steps,
        dimension_names=dataset.dimension_names,
    )

    # Local と Global を分離
    result.local_lambda = {
        k: all_structures[k]
        for k in ["local_lambda_F", "local_rho_T", "local_Q_lambda", "local_std"]
        if k in all_structures
    }

    result.lambda_structures = {
        k: all_structures[k]
        for k in all_structures
        if not k.startswith("local_") and not k.startswith("_")
    }

    if config.verbose:
        n_jumps = np.count_nonzero(result.local_lambda.get("local_lambda_F", []))
        print(f"         Local: {n_jumps} jumps detected")

    # ── 3. Network Analysis (Λ³ Native) ──
    if config.verbose:
        print("\n  [2] Network Analysis (Λ³ Native)...")

    analyzer = NetworkAnalyzerCore(
        sync_threshold=config.sync_threshold,
        causal_threshold=config.causal_threshold,
        max_lag=config.max_lag,
        delta_percentile=config.delta_percentile,
        local_std_window=config.local_std_window,
        rho_t_window=config.rho_t_window,
        n_permutations=config.n_permutations,
        p_value_threshold=config.p_value_threshold,
        rho_t_weight=config.rho_t_weight,
        enable_gpu_verification=config.enable_gpu_verification,
    )

    result.network = analyzer.analyze(
        dataset.state_vectors,
        dimension_names=dataset.dimension_names,
    )

    if config.verbose:
        print(f"         Pattern: {result.network.pattern}")
        print(
            f"         Sync: {result.network.n_sync_links}, "
            f"Causal: {result.network.n_causal_links}"
        )
        if result.network.driver_names:
            print(f"         Drivers: {', '.join(result.network.driver_names)}")
        if result.network.follower_names:
            print(f"         Followers: {', '.join(result.network.follower_names)}")

    # ── 4. Confidence (Network分のみ) ──
    if config.enable_confidence:
        if config.verbose:
            print(f"\n  [3] Confidence (perm={config.n_permutations})...")

        try:
            from .analysis.confidence_kit import assess_confidence

            result.confidence = assess_confidence(
                state_vectors=dataset.state_vectors,
                lambda_structures=result.lambda_structures,
                structural_boundaries=None,
                anomaly_scores=None,
                network_result=result.network,
                dimension_names=dataset.dimension_names,
                alpha=config.alpha,
                n_permutations=config.n_permutations,
                n_bootstrap=config.n_bootstrap,
                window_steps=config.window_steps or 24,
                seed=config.seed,
            )

            if config.verbose:
                print(f"         Sig causal: {result.confidence.n_significant_causal}")
                print(f"         Sig sync: {result.confidence.n_significant_sync}")
        except Exception as e:
            logger.warning(f"Confidence assessment failed: {e}")

    # ── 5. Report ──
    if config.enable_report:
        if config.verbose:
            print("\n  [4] Report...")

        try:
            from .analysis.report_generator import generate_report

            result.report = generate_report(
                lambda_structures=result.lambda_structures,
                structural_boundaries=None,
                topological_breaks=None,
                anomaly_scores=None,
                network_result=result.network,
                confidence_report=result.confidence,
                dataset=dataset,
                output_path=config.report_path,
            )
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")

    # ── 完了 ──
    result.computation_time = time.time() - t_start

    if config.verbose:
        print(f"\n  ✅ Network Pipeline complete ({result.computation_time:.2f}s)")
        print(f"{'=' * 60}")

    return result
