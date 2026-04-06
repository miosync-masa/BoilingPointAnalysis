"""
GETTER One - Pipeline Detection
=================================
Detection系統専用パイプライン。

  DualCore (Λ³特徴量)
    → Detection 5種 (OR統合 → event_mask)
    → InverseChecker (逆問題 3軸 → GENUINE/SPURIOUS)
    → CascadeTracker (GENUINEイベントのみ)
    → Confidence
    → Report

Network（構造推定）はここには入らない。
構造推定が必要な場合は pipeline_network.py を使うこと。

Usage:
    from getter_one.pipeline_detection import run_detection

    result = run_detection("data.csv")
    print(result.cascade.n_chains)
    print(result.inverse_result.genuine_ratio)

Built with 💕 by Masamichi & Tamaki
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .data.loader import GetterDataset, load

logger = logging.getLogger("getter_one.pipeline_detection")


# ============================================================
# Config
# ============================================================


@dataclass
class DetectionPipelineConfig:
    """Detection Pipeline 設定"""

    # Λ³構造 (DualCore)
    window_steps: int | None = None  # None = アダプティブ
    adaptive_window: bool = True
    delta_percentile: float = 94.0

    # Detection modules (OR統合)
    enable_boundary: bool = True
    enable_topology: bool = True
    enable_anomaly: bool = True
    enable_extended: bool = True
    enable_phase_space: bool = True

    # InverseChecker
    enable_inverse_checker: bool = True
    w_recon: float = 0.4
    w_topo: float = 0.3
    w_jump: float = 0.3
    jump_sigma: float = 2.5
    verdict_threshold: float = 0.0

    # CascadeTracker
    enable_cascade: bool = True

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
class DetectionPipelineResult:
    """Detection Pipeline 結果"""

    config: DetectionPipelineConfig = field(default_factory=DetectionPipelineConfig)
    dataset: GetterDataset | None = None

    # Lambda (DualCore)
    lambda_structures: dict = field(default_factory=dict)
    local_lambda: dict = field(default_factory=dict)

    # Detection (OR統合)
    event_mask: np.ndarray | None = None  # (n_frames,) 全候補
    structural_boundaries: dict | None = None
    topological_breaks: dict | None = None
    anomaly_scores: dict = field(default_factory=dict)
    extended_detection: dict = field(default_factory=dict)
    phase_space_analysis: dict | None = None
    detector_labels: dict = field(default_factory=dict)  # frame→[detector名]

    # InverseChecker
    inverse_result: object | None = None  # VerificationResult
    genuine_mask: np.ndarray | None = None  # (n_frames,) 検証済み

    # Cascade
    cascade: object | None = None  # CascadeResult

    # Confidence
    confidence: object | None = None

    # Report
    report: str | None = None

    # Timing
    computation_time: float = 0.0


# ============================================================
# Pipeline
# ============================================================


def run_detection(
    source,
    config: DetectionPipelineConfig | None = None,
    **kwargs,
) -> DetectionPipelineResult:
    """
    GETTER One Detection Pipeline

    構造変化イベントの検出 → 逆問題精査 → カスケード追跡。

    Parameters
    ----------
    source : str, Path, GetterDataset, or np.ndarray
        入力データ
    config : DetectionPipelineConfig, optional
        パイプライン設定
    **kwargs
        load() に渡す追加引数

    Returns
    -------
    DetectionPipelineResult
    """
    t_start = time.time()

    if config is None:
        config = DetectionPipelineConfig()

    result = DetectionPipelineResult(config=config)

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
        print("  GETTER One — Detection Pipeline")
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
        for k in [
            "local_lambda_F",
            "local_rho_T",
            "local_Q_lambda",
            "local_std",
        ]
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
        rho = result.lambda_structures.get("rho_T")
        if rho is not None:
            print(f"         Global rho_T: mean={np.mean(rho):.3e}")

    # ── 3. Detection → event_mask (OR統合) ──
    if config.verbose:
        print("\n  [2] Detection (OR union → event_mask)...")

    event_mask = np.zeros(n_frames, dtype=bool)
    detector_labels: dict[int, list[str]] = {}

    def _register_events(frames, detector_name: str) -> None:
        """イベントフレームをevent_maskとdetector_labelsに登録"""
        for f in frames:
            if 0 <= f < n_frames:
                event_mask[f] = True
                if f not in detector_labels:
                    detector_labels[f] = []
                detector_labels[f].append(detector_name)

    # (a) Local ΛF: local_lambda_F が非ゼロ
    local_lf = result.local_lambda.get("local_lambda_F")
    if local_lf is not None:
        local_events = np.where(np.any(local_lf != 0, axis=1))[0]
        # local_lambda_F は (n_frames-1, n_dims) なのでオフセット+1
        _register_events(local_events + 1, "local_lambda_F")
        if config.verbose:
            print(f"         Local ΛF events: {len(local_events)}")

    # (b) Boundary Detection
    if config.enable_boundary:
        try:
            from .detection.boundary_detection_gpu import (
                BoundaryDetectorGPU,
            )

            detector = BoundaryDetectorGPU(force_cpu=True)
            boundary_window = max(10, (config.window_steps or 24) // 3)
            result.structural_boundaries = detector.detect_structural_boundaries(
                result.lambda_structures, boundary_window
            )
            locs = result.structural_boundaries.get("boundary_locations", [])
            _register_events(locs, "boundary")
            if config.verbose:
                print(f"         Boundaries: {len(locs)} detected")
        except Exception as e:
            logger.warning(f"Boundary detection failed: {e}")

    # (c) Topology Breaks
    if config.enable_topology:
        try:
            from .detection.topology_breaks_gpu import (
                TopologyBreaksDetectorGPU,
            )

            detector = TopologyBreaksDetectorGPU(force_cpu=True)
            fast_window = max(10, (config.window_steps or 24) // 2)
            result.topological_breaks = detector.detect_topological_breaks(
                result.lambda_structures, fast_window
            )
            # トポロジー破壊もスコアベースでevent_maskに統合
            topo_scores = result.topological_breaks.get("combined_breaks", None)
            if topo_scores is not None and hasattr(topo_scores, "__len__"):
                topo_thresh = np.mean(topo_scores) + 2 * np.std(topo_scores)
                topo_frames = np.where(topo_scores > topo_thresh)[0]
                _register_events(topo_frames, "topology")
                if config.verbose:
                    print(f"         Topology: {len(topo_frames)} breaks")
            elif config.verbose:
                print("         Topology: computed (no frame-level events)")
        except Exception as e:
            logger.warning(f"Topology detection failed: {e}")

    # (d) Anomaly Detection
    if config.enable_anomaly:
        try:
            from .detection.anomaly_detection_gpu import (
                AnomalyDetectorGPU,
            )

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
                anomaly_frames = np.where(combined > threshold)[0]
                _register_events(anomaly_frames, "anomaly")
                if config.verbose:
                    print(f"         Anomaly: {len(anomaly_frames)} critical")
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")

    # (e) Extended Detection
    if config.enable_extended:
        try:
            from .detection.extended_detection_gpu import (
                ExtendedDetectorGPU,
            )

            ext_detector = ExtendedDetectorGPU(force_cpu=True)

            # 周期検出
            min_period = max(10, n_frames // 100)
            max_period = max(100, n_frames // 2)
            periodic = ext_detector.detect_periodic_transitions(
                result.lambda_structures,
                min_period=min_period,
                max_period=max_period,
            )
            result.extended_detection["periodic"] = periodic

            # 緩慢遷移検出
            gradual = ext_detector.detect_gradual_transitions(
                result.lambda_structures,
            )
            result.extended_detection["gradual"] = gradual

            # 構造ドリフト検出
            drift = ext_detector.detect_structural_drift(
                result.lambda_structures,
            )
            result.extended_detection["drift"] = drift

            # gradualスコアをevent_maskに統合
            gradual_scores = gradual.get("gradual_scores", None)
            if gradual_scores is not None and hasattr(gradual_scores, "__len__"):
                g_thresh = np.mean(gradual_scores) + 2 * np.std(gradual_scores)
                gradual_frames = np.where(gradual_scores > g_thresh)[0]
                _register_events(gradual_frames, "extended_gradual")

            if config.verbose:
                n_periodic = len(periodic.get("detected_periods", []))
                print(f"         Extended: {n_periodic} periods")
        except Exception as e:
            logger.warning(f"Extended detection failed: {e}")

    # (f) Phase Space Analysis
    if config.enable_phase_space:
        try:
            from .detection.phase_space_gpu import (
                PhaseSpaceAnalyzerGPU,
            )

            ps_analyzer = PhaseSpaceAnalyzerGPU(force_cpu=True)
            result.phase_space_analysis = ps_analyzer.analyze_phase_space(
                result.lambda_structures,
            )

            ps_score = result.phase_space_analysis.get("integrated_anomaly_score", None)
            if ps_score is not None and hasattr(ps_score, "__len__"):
                ps_thresh = np.mean(ps_score) + 2 * np.std(ps_score)
                ps_frames = np.where(ps_score > ps_thresh)[0]
                _register_events(ps_frames, "phase_space")
                if config.verbose:
                    print(f"         Phase space: {len(ps_frames)} anomalies")
            elif config.verbose:
                print("         Phase space: computed")
        except Exception as e:
            logger.warning(f"Phase space analysis failed: {e}")

    # (g) Global ρT 急変
    rho = result.lambda_structures.get("rho_T")
    if rho is not None:
        rho_threshold = np.mean(rho) + 2 * np.std(rho)
        rho_frames = np.where(rho > rho_threshold)[0]
        _register_events(rho_frames, "global_rho_T")
        if config.verbose:
            print(f"         Global ρT events: {len(rho_frames)}")

    result.event_mask = event_mask
    result.detector_labels = detector_labels
    n_candidates = int(np.sum(event_mask))

    if config.verbose:
        print(
            f"         → event_mask (OR): {n_candidates}/{n_frames} "
            f"({100 * n_candidates / n_frames:.1f}%)"
        )

    # ── 4. InverseChecker (逆問題精査 → GENUINE/SPURIOUS) ──
    if config.enable_inverse_checker and n_candidates > 0:
        if config.verbose:
            print("\n  [3] InverseChecker (GENUINE/SPURIOUS)...")

        try:
            from .analysis.inverse_checker import InverseChecker

            checker = InverseChecker(
                w_recon=config.w_recon,
                w_topo=config.w_topo,
                w_jump=config.w_jump,
                jump_sigma=config.jump_sigma,
                verdict_threshold=config.verdict_threshold,
            )

            result.inverse_result = checker.verify(
                event_mask=event_mask,
                state_vectors=dataset.state_vectors,
                local_lambda=result.local_lambda,
                dimension_names=dataset.dimension_names,
            )

            # GENUINEイベントのみのマスクを構築
            result.genuine_mask = result.inverse_result.genuine_mask(n_frames)

            if config.verbose:
                ir = result.inverse_result
                print(f"         GENUINE: {ir.n_genuine} ({ir.genuine_ratio:.1%})")
                print(f"         SPURIOUS: {ir.n_spurious}")
        except ImportError:
            logger.warning(
                "InverseChecker requires GPU (CuPy). "
                "Skipping — using raw event_mask for Cascade."
            )
            result.genuine_mask = event_mask
        except Exception as e:
            logger.warning(f"InverseChecker failed: {e}")
            result.genuine_mask = event_mask
    else:
        # InverseChecker無効時はevent_maskをそのまま使用
        result.genuine_mask = event_mask

    # ── 5. CascadeTracker (GENUINEイベントのみ) ──
    if config.enable_cascade and result.genuine_mask is not None:
        n_genuine = int(np.sum(result.genuine_mask))
        if n_genuine > 0:
            if config.verbose:
                print(f"\n  [4] Cascade ({n_genuine} genuine events)...")

            try:
                from .analysis.cascade_tracker import CascadeTracker

                cascade_tracker = CascadeTracker(adaptive=True)
                result.cascade = cascade_tracker.track(
                    state_vectors=dataset.state_vectors,
                    lambda_structures=result.lambda_structures,
                    dimension_names=dataset.dimension_names,
                    local_lambda=result.local_lambda,
                    event_mask=result.genuine_mask,
                )

                if config.verbose:
                    print(f"         Events: {result.cascade.n_events}")
                    print(f"         Chains: {result.cascade.n_chains}")
                    if result.cascade.critical_names:
                        print(f"         Critical: {result.cascade.critical_names}")
            except Exception as e:
                logger.warning(f"Cascade tracking failed: {e}")
        elif config.verbose:
            print("\n  [4] Cascade: skipped (0 genuine events)")

    # ── 6. Confidence ──
    if config.enable_confidence:
        if config.verbose:
            print(f"\n  [5] Confidence (perm={config.n_permutations})...")

        try:
            from .analysis.confidence_kit import assess_confidence

            result.confidence = assess_confidence(
                state_vectors=dataset.state_vectors,
                lambda_structures=result.lambda_structures,
                structural_boundaries=result.structural_boundaries or None,
                anomaly_scores=result.anomaly_scores or None,
                network_result=None,  # Detection側なのでNetworkなし
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
        except Exception as e:
            logger.warning(f"Confidence assessment failed: {e}")

    # ── 7. Report ──
    if config.enable_report:
        if config.verbose:
            print("\n  [6] Report...")

        try:
            from .analysis.report_generator import generate_report

            result.report = generate_report(
                lambda_structures=result.lambda_structures,
                structural_boundaries=result.structural_boundaries or None,
                topological_breaks=result.topological_breaks or None,
                anomaly_scores=result.anomaly_scores or None,
                network_result=None,  # Detection側なのでNetworkなし
                confidence_report=result.confidence,
                dataset=dataset,
                output_path=config.report_path,
            )
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")

    # ── 完了 ──
    result.computation_time = time.time() - t_start

    if config.verbose:
        print(f"\n  ✅ Detection Pipeline complete ({result.computation_time:.2f}s)")
        print(f"{'=' * 60}")

    return result
