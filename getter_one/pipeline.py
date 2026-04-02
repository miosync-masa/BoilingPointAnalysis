"""
GETTER One - Main Pipeline
============================

データ読み込み → Λ³構造計算 → Detection → Network → Confidence → Report
を一括で実行するエントリーポイント。

Usage:
    from getter_one.pipeline import run

    results = run("prepared.csv", window_steps=24)
    print(results.report)

CLI:
    $ python -m getter_one.pipeline run prepared.csv --window 24 --report output.md

Built with 💕 by Masamichi & Tamaki
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .analysis.confidence_kit import ConfidenceReport, assess_confidence
from .analysis.network_analyzer_core import NetworkAnalyzerCore, NetworkResult
from .data.loader import GetterDataset, load
from .structures.lambda_structures_core import LambdaStructuresCore

logger = logging.getLogger("getter_one.pipeline")


# ============================================================
# Pipeline Config
# ============================================================

@dataclass
class PipelineConfig:
    """パイプライン設定"""
    # Λ³構造
    window_steps: int = 24

    # Detection
    enable_boundary: bool = True
    enable_topology: bool = True
    enable_anomaly: bool = True

    # Network
    enable_network: bool = True
    sync_threshold: float = 0.3
    causal_threshold: float = 0.25
    max_lag: int = 12

    # Confidence
    enable_confidence: bool = True
    alpha: float = 0.05
    n_permutations: int = 1000
    n_bootstrap: int = 2000

    # Report
    enable_report: bool = True
    report_path: str | None = None

    # General
    seed: int = 42
    verbose: bool = True


# ============================================================
# Pipeline Result
# ============================================================

@dataclass
class PipelineResult:
    """パイプライン全結果"""
    # 入力データ
    dataset: GetterDataset | None = None

    # Λ³構造
    lambda_structures: dict[str, np.ndarray] = field(default_factory=dict)

    # Detection
    structural_boundaries: dict = field(default_factory=dict)
    topological_breaks: dict = field(default_factory=dict)
    anomaly_scores: dict = field(default_factory=dict)

    # Network
    network: NetworkResult | None = None

    # Confidence
    confidence: ConfidenceReport | None = None

    # Report
    report: str = ""

    # Metadata
    computation_time: float = 0.0
    config: PipelineConfig | None = None


# ============================================================
# Main Pipeline
# ============================================================

def run(
    source,
    config: PipelineConfig | None = None,
    **kwargs,
) -> PipelineResult:
    """
    GETTER One フルパイプライン実行。

    Parameters
    ----------
    source : str, Path, GetterDataset, or np.ndarray
        入力データ。以下のいずれか:
        - ファイルパス（csv/json/parquet/xlsx/npy/npz）
        - GetterDataset
        - numpy配列 (n_frames, n_dims)
    config : PipelineConfig, optional
        パイプライン設定
    **kwargs
        load() に渡す追加引数（target, time_column, normalize等）

    Returns
    -------
    PipelineResult
    """
    t_start = time.time()

    if config is None:
        config = PipelineConfig()

    result = PipelineResult(config=config)

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
        print("  GETTER One Pipeline")
        print(f"{'=' * 60}")
        print(f"  Data: {dataset.n_frames} frames × {dataset.n_dims} dims")
        print(f"  Dimensions: {', '.join(dataset.dimension_names)}")

    # ── 2. Λ³構造計算 ──
    if config.verbose:
        print(f"\n  [1/5] Computing Lambda³ structures (window={config.window_steps})...")

    core = LambdaStructuresCore()
    result.lambda_structures = core.compute_lambda_structures(
        state_vectors=dataset.state_vectors,
        window_steps=config.window_steps,
        dimension_names=dataset.dimension_names,
    )

    # ── 3. Detection ──
    if config.verbose:
        print("  [2/5] Running detection modules...")

    if config.enable_boundary:
        try:
            from .detection.boundary_detection_gpu import BoundaryDetectorGPU
            detector = BoundaryDetectorGPU(force_cpu=True)
            boundary_window = max(10, config.window_steps // 3)
            result.structural_boundaries = detector.detect_structural_boundaries(
                result.lambda_structures, boundary_window
            )
            n_bounds = len(result.structural_boundaries.get("boundary_locations", []))
            if config.verbose:
                print(f"         Boundaries: {n_bounds} detected")
        except Exception as e:
            logger.warning(f"Boundary detection failed: {e}")

    if config.enable_topology:
        try:
            from .detection.topology_breaks_gpu import TopologyBreaksDetectorGPU
            detector = TopologyBreaksDetectorGPU(force_cpu=True)
            fast_window = max(10, config.window_steps // 2)
            result.topological_breaks = detector.detect_topological_breaks(
                result.lambda_structures, fast_window
            )
            if config.verbose:
                print("         Topology breaks: computed")
        except Exception as e:
            logger.warning(f"Topology detection failed: {e}")

    if config.enable_anomaly:
        try:
            from .detection.anomaly_detection_gpu import AnomalyDetectorGPU
            detector = AnomalyDetectorGPU(force_cpu=True)

            # MDConfigの代わりに最小限の設定オブジェクトを作成
            class _MinimalConfig:
                w_lambda_f = 0.3
                w_lambda_ff = 0.2
                w_rho_t = 0.2
                w_topology = 0.3
                w_phase_coherence = 0.7
                w_singularities = 0.6
                global_weight = 0.6
                local_weight = 0.4

            result.anomaly_scores = detector.compute_multiscale_anomalies(
                result.lambda_structures,
                result.structural_boundaries,
                result.topological_breaks,
                {"com_positions": dataset.state_vectors},
                _MinimalConfig(),
            )
            if config.verbose and "combined" in result.anomaly_scores:
                combined = result.anomaly_scores["combined"]
                threshold = np.mean(combined) + 2 * np.std(combined)
                n_crit = int(np.sum(combined > threshold))
                print(f"         Anomaly: {n_crit} critical frames")
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")

    # ── 4. Network ──
    if config.enable_network:
        if config.verbose:
            print("  [3/5] Analyzing causal network...")

        analyzer = NetworkAnalyzerCore(
            sync_threshold=config.sync_threshold,
            causal_threshold=config.causal_threshold,
            max_lag=config.max_lag,
        )
        result.network = analyzer.analyze(
            dataset.state_vectors,
            dimension_names=dataset.dimension_names,
        )

        if config.verbose:
            print(f"         Pattern: {result.network.pattern}")
            print(f"         Sync: {result.network.n_sync_links}, "
                  f"Causal: {result.network.n_causal_links}")

    # ── 5. Confidence ──
    if config.enable_confidence:
        if config.verbose:
            print(f"  [4/5] Assessing confidence (perm={config.n_permutations})...")

        result.confidence = assess_confidence(
            state_vectors=dataset.state_vectors,
            lambda_structures=result.lambda_structures,
            structural_boundaries=result.structural_boundaries or None,
            anomaly_scores=result.anomaly_scores or None,
            network_result=result.network,
            dimension_names=dataset.dimension_names,
            alpha=config.alpha,
            n_permutations=config.n_permutations,
            n_bootstrap=config.n_bootstrap,
            seed=config.seed,
        )

        if config.verbose:
            print(f"         Sig boundaries: {result.confidence.n_significant_boundaries}")
            print(f"         Sig causal: {result.confidence.n_significant_causal}")
            print(f"         Sig sync: {result.confidence.n_significant_sync}")

    # ── 6. Report ──
    if config.enable_report:
        if config.verbose:
            print("  [5/5] Generating report...")

        from .analysis.report_generator import generate_report

        result.report = generate_report(
            lambda_structures=result.lambda_structures,
            structural_boundaries=result.structural_boundaries or None,
            topological_breaks=result.topological_breaks or None,
            anomaly_scores=result.anomaly_scores or None,
            network_result=result.network,
            confidence_report=result.confidence,
            dataset=dataset,
            output_path=config.report_path,
        )

    # ── 完了 ──
    result.computation_time = time.time() - t_start

    if config.verbose:
        print(f"\n  ✅ Pipeline complete ({result.computation_time:.2f}s)")
        print(f"{'=' * 60}")

    return result


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="getter-one",
        description="GETTER One - Structural event detection pipeline",
    )
    parser.add_argument("source", help="Input file path (csv/json/parquet/npy)")
    parser.add_argument("--target", help="Target column name")
    parser.add_argument("--time", help="Time column name")
    parser.add_argument("--normalize", default="range", choices=["range", "zscore", "none"])
    parser.add_argument("--window", type=int, default=24, help="Window steps for Λ³")
    parser.add_argument("--max-lag", type=int, default=12, help="Max lag for causal detection")
    parser.add_argument("--n-perm", type=int, default=1000, help="Permutation count")
    parser.add_argument("--report", help="Output report path (.md)")
    parser.add_argument("--no-confidence", action="store_true", help="Skip confidence assessment")
    parser.add_argument("--no-network", action="store_true", help="Skip network analysis")

    args = parser.parse_args()

    config = PipelineConfig(
        window_steps=args.window,
        max_lag=args.max_lag,
        n_permutations=args.n_perm,
        report_path=args.report,
        enable_confidence=not args.no_confidence,
        enable_network=not args.no_network,
    )

    result = run(
        args.source,
        config=config,
        target=args.target,
        time_column=args.time,
        normalize=args.normalize,
    )

    # レポートを標準出力に表示
    if result.report:
        print(result.report)


if __name__ == "__main__":
    main()
