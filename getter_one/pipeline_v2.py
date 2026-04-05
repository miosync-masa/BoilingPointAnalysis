"""
GETTER One - Pipeline V2 (DualCore Architecture)
=================================================

Lambda (DualCore) → Detection (event_mask) → Network → Cascade → InverseChecker

  Lambda:         Local(方式A) + Global(方式B)
  Detection:      Local+Global双方をOR統合 → event_mask
  Network:        local_std無次元化 + event_maskフィルタ
  Cascade:        Local ΛFでgenesis追跡 → 連鎖
  InverseChecker: 各イベントにNORMAL/CRITICALラベル（フィルタではなくラベル）

Usage:
    from getter_one.pipeline_v2 import run_v2

    results = run_v2("prepared.csv")

Built with 💕 by Masamichi & Tamaki
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .analysis.network_analyzer_core import NetworkAnalyzerCore, NetworkResult
from .data.loader import GetterDataset, load

logger = logging.getLogger("getter_one.pipeline_v2")


# ============================================================
# Config
# ============================================================


@dataclass
class PipelineV2Config:
    """Pipeline V2 設定"""

    # Λ³構造 (DualCore)
    window_steps: int | None = None  # None = アダプティブ
    adaptive_window: bool = True
    delta_percentile: float = 94.0

    # Detection
    enable_boundary: bool = True
    enable_topology: bool = True
    enable_anomaly: bool = True
    enable_extended: bool = True
    enable_phase_space: bool = True

    # Network
    enable_network: bool = True
    sync_threshold: float = 0.3
    causal_threshold: float = 0.25
    max_lag: int = 12
    adaptive_network: bool = True
    use_local_std_for_network: bool = True  # Local std で無次元化
    use_event_mask_for_network: bool = True  # Detection結果でフィルタ

    # Cascade
    enable_cascade: bool = True

    # InverseChecker
    enable_inverse_checker: bool = True

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
# Result
# ============================================================


@dataclass
class PipelineV2Result:
    """Pipeline V2 結果"""

    config: PipelineV2Config = field(default_factory=PipelineV2Config)
    dataset: GetterDataset | None = None

    # Lambda (DualCore)
    lambda_structures: dict = field(default_factory=dict)
    local_lambda: dict = field(default_factory=dict)  # Local (方式A)

    # Detection
    event_mask: np.ndarray | None = None  # (n_frames,) bool
    structural_boundaries: dict | None = None
    topological_breaks: dict | None = None
    anomaly_scores: dict = field(default_factory=dict)
    extended_detection: dict = field(default_factory=dict)
    phase_space_analysis: dict | None = None

    # Network
    network: NetworkResult | None = None

    # Cascade
    cascade: object | None = None  # CascadeResult

    # InverseChecker
    inverse_results: list = field(default_factory=list)  # per-event labels

    # Confidence
    confidence: object | None = None

    # Report
    report: str | None = None

    # Timing
    computation_time: float = 0.0


# ============================================================
# Pipeline V2
# ============================================================


def run_v2(
    source,
    config: PipelineV2Config | None = None,
    **kwargs,
) -> PipelineV2Result:
    """
    GETTER One Pipeline V2 (DualCore Architecture)

    Parameters
    ----------
    source : str, Path, GetterDataset, or np.ndarray
    config : PipelineV2Config, optional
    **kwargs : load() への追加引数
    """
    t_start = time.time()

    if config is None:
        config = PipelineV2Config()

    result = PipelineV2Result(config=config)

    # ── 1. データ読み込み ──
    if isinstance(source, GetterDataset):
        dataset = source
    elif isinstance(source, np.ndarray):
        from .data.loader import from_numpy

        dataset = from_numpy(source, normalize=kwargs.pop("normalize", "range"))
    else:
        dataset = load(source, **kwargs)

    result.dataset = dataset
    n_frames = dataset.n_frames

    if config.verbose:
        print(f"\n{'=' * 60}")
        print("  GETTER One Pipeline V2 (DualCore)")
        print(f"{'=' * 60}")
        print(f"  Data: {n_frames} frames × {dataset.n_dims} dims")
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

    # Global = 既存Detection互換フィールド
    result.lambda_structures = {
        k: all_structures[k]
        for k in all_structures
        if not k.startswith("local_") and not k.startswith("_")
    }

    if config.verbose:
        n_jumps = np.count_nonzero(result.local_lambda.get("local_lambda_F", []))
        print(f"         Local: {n_jumps} jumps detected")
        rho = result.lambda_structures.get("rho_T")
        if rho is not None:
            print(f"         Global rho_T: mean={np.mean(rho):.3e}")

    # ── 3. Detection → event_mask (OR統合) ──
    if config.verbose:
        print("  [2] Detection (event_mask)...")

    event_mask = np.zeros(n_frames, dtype=bool)

    # Local判定: local_lambda_F が非ゼロ
    local_lf = result.local_lambda.get("local_lambda_F")
    if local_lf is not None:
        # local_lambda_F は (n_frames-1, n_dims)
        local_events = np.any(local_lf != 0, axis=1)
        event_mask[1:] |= local_events
        if config.verbose:
            print(f"         Local events: {int(np.sum(local_events))}")

    # Global判定: 既存Detection系モジュール
    if config.enable_boundary:
        try:
            from .detection.boundary_detection_gpu import BoundaryDetectorGPU

            detector = BoundaryDetectorGPU(force_cpu=True)
            boundary_window = max(10, (config.window_steps or 24) // 3)
            result.structural_boundaries = detector.detect_structural_boundaries(
                result.lambda_structures, boundary_window
            )
            locs = result.structural_boundaries.get("boundary_locations", [])
            for loc in locs:
                if 0 <= loc < n_frames:
                    event_mask[loc] = True
            if config.verbose:
                print(f"         Boundaries: {len(locs)} detected")
        except Exception as e:
            logger.warning(f"Boundary detection failed: {e}")

    if config.enable_topology:
        try:
            from .detection.topology_breaks_gpu import TopologyBreaksDetectorGPU

            detector = TopologyBreaksDetectorGPU(force_cpu=True)
            fast_window = max(10, (config.window_steps or 24) // 2)
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

            if "combined" in result.anomaly_scores:
                combined = result.anomaly_scores["combined"]
                threshold = np.mean(combined) + 2 * np.std(combined)
                anomaly_mask = combined > threshold
                event_mask |= anomaly_mask
                if config.verbose:
                    print(f"         Anomaly: {int(np.sum(anomaly_mask))} critical")
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")

    # Global rho_T 急変
    rho = result.lambda_structures.get("rho_T")
    if rho is not None:
        rho_threshold = np.mean(rho) + 2 * np.std(rho)
        rho_events = rho > rho_threshold
        event_mask |= rho_events
        if config.verbose:
            print(f"         Global rho_T events: {int(np.sum(rho_events))}")

    result.event_mask = event_mask
    n_events = int(np.sum(event_mask))

    if config.verbose:
        print(
            f"         → event_mask: {n_events}/{n_frames} "
            f"({100 * n_events / n_frames:.1f}%)"
        )

    # ── 4. Network ──
    if config.enable_network:
        if config.verbose:
            modes = []
            if config.use_local_std_for_network:
                modes.append("dimensionless")
            if config.use_event_mask_for_network:
                modes.append("event-filtered")
            mode_str = " + ".join(modes) if modes else "raw"
            print(f"  [3] Network ({mode_str})...")

        analyzer = NetworkAnalyzerCore(
            sync_threshold=config.sync_threshold,
            causal_threshold=config.causal_threshold,
            max_lag=config.max_lag,
            adaptive=config.adaptive_network,
        )

        # フラグに応じてNetworkへの渡し方を切り替え
        net_local_lambda = (
            result.local_lambda if config.use_local_std_for_network else None
        )
        net_event_mask = (
            result.event_mask if config.use_event_mask_for_network else None
        )

        result.network = analyzer.analyze(
            dataset.state_vectors,
            dimension_names=dataset.dimension_names,
            local_lambda=net_local_lambda,
            event_mask=net_event_mask,
        )

        if config.verbose:
            print(f"         Pattern: {result.network.pattern}")
            print(
                f"         Sync: {result.network.n_sync_links}, "
                f"Causal: {result.network.n_causal_links}"
            )
            if result.network.driver_names:
                print(f"         Drivers: {result.network.driver_names}")

    # ── 5. Cascade (Local ΛF genesis追跡) ──
    if config.enable_cascade:
        if config.verbose:
            print("  [4] Cascade tracking (Local ΛF genesis)...")

        try:
            from .analysis.cascade_tracker import CascadeTracker

            cascade_tracker = CascadeTracker(adaptive=True)
            result.cascade = cascade_tracker.track(
                state_vectors=dataset.state_vectors,
                lambda_structures=result.lambda_structures,
                dimension_names=dataset.dimension_names,
                local_lambda=result.local_lambda,
            )

            if config.verbose:
                print(f"         Events: {result.cascade.n_events}")
                print(f"         Chains: {result.cascade.n_chains}")
                if result.cascade.critical_names:
                    print(f"         Critical: {result.cascade.critical_names}")
        except Exception as e:
            logger.warning(f"Cascade tracking failed: {e}")

    # ── 6. InverseChecker (NORMAL/CRITICALラベル) ──
    if config.enable_inverse_checker and result.cascade is not None:
        if config.verbose:
            print("  [5] InverseChecker (NORMAL/CRITICAL labeling)...")

        try:
            from .analysis.inverse_checker import InverseChecker

            checker = InverseChecker()
            inverse_results = []

            for event in result.cascade.events:
                frame = event.frame
                # イベント前後でInverseChecker実行
                window_before = min(frame, 30)
                window_after = min(n_frames - frame - 1, 10)

                if window_before >= 5:
                    verdict = checker.check_event(
                        state_vectors=dataset.state_vectors,
                        event_frame=frame,
                        window_before=window_before,
                        window_after=window_after,
                    )
                    inverse_results.append(
                        {
                            "event_id": event.event_id,
                            "frame": frame,
                            "verdict": verdict,
                            "genesis_dims": event.genesis_dims,
                            "genesis_names": event.genesis_names,
                        }
                    )

            result.inverse_results = inverse_results

            if config.verbose and inverse_results:
                n_normal = sum(
                    1
                    for r in inverse_results
                    if hasattr(r["verdict"], "is_normal") and r["verdict"].is_normal
                )
                n_critical = len(inverse_results) - n_normal
                print(f"         NORMAL: {n_normal}, CRITICAL: {n_critical}")
        except Exception as e:
            logger.warning(f"InverseChecker failed: {e}")

    # ── 7. Confidence ──
    if config.enable_confidence:
        if config.verbose:
            print(f"  [6] Confidence (perm={config.n_permutations})...")

        try:
            from .analysis.confidence_kit import assess_confidence

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
                window_steps=config.window_steps or 24,
                seed=config.seed,
            )

            if config.verbose:
                print(
                    f"         Sig boundaries: "
                    f"{result.confidence.n_significant_boundaries}"
                )
                print(f"         Sig causal: {result.confidence.n_significant_causal}")
        except Exception as e:
            logger.warning(f"Confidence assessment failed: {e}")

    # ── 8. Report ──
    if config.enable_report:
        if config.verbose:
            print("  [7] Report...")

        try:
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
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")

    # ── 完了 ──
    result.computation_time = time.time() - t_start

    if config.verbose:
        print(f"\n  ✅ Pipeline V2 complete ({result.computation_time:.2f}s)")
        print(f"{'=' * 60}")

    return result
