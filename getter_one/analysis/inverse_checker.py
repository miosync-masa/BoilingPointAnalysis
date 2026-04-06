"""
Inverse Checker — Structural Event Verification Gate
=====================================================
⚠️ GPU ONLY — pip install getter-one[gpu]
Built with 💕 by Masamichi & Tamaki

Detection パイプラインの品質保証ゲート。
5種のDetectorが検出した ΔΛC イベント候補を精査し、
GENUINE（本物の構造変化）か SPURIOUS（ノイズ）かを判定する。

位置づけ:
  Detection (OR union) → ★ InverseChecker ★ → Cascade (GENUINEのみ)

検証4軸:
  1. Surrogate Test   — シャッフルしても再検出されるか（データ依存性）
  2. ρT Consistency   — テンション蓄積後の発火か（物理的整合性）
  3. Persistence      — 発火後に構造が変化したまま戻らないか（不可逆性）
  4. Coordination     — 複数次元で協調的に発火しているか（構造的協調）

4軸はGPU並列で全イベント同時計算され、重み付き統合スコアで判定する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

# GPU ONLY
from ..core.gpu_inverse import inverse_verify_all

logger = logging.getLogger("getter_one.analysis.inverse_checker")


# ============================================
# Data Classes
# ============================================


@dataclass
class DetectionCandidate:
    """Detection OR統合後のイベント候補"""

    frame: int
    detected_by: list[str] = field(default_factory=list)  # どのdetectorが検出したか
    n_detectors: int = 0  # 検出したdetector数（投票数）

    # Λ³特徴量（抽出済み）
    delta_lambda_c: float = 0.0  # ΔΛC強度
    rho_t_at_event: float = 0.0  # イベント時点のρT
    rho_t_before: float = 0.0  # イベント直前のρT平均
    n_dims_active: int = 0  # 同時発火次元数


@dataclass
class EventVerdict:
    """単一イベントの検証結果"""

    event_id: int
    frame: int
    delta_lambda_c: float

    # 4軸スコア（各 0.0〜1.0, 高い = GENUINE寄り）
    surrogate_score: float = 0.0  # シャッフル再検出率
    rho_t_score: float = 0.0  # テンション整合性
    persistence_score: float = 0.0  # 構造変化の持続性
    coordination_score: float = 0.0  # 多次元協調度

    # 統合
    genuineness: float = 0.0  # 重み付き統合スコア
    is_genuine: bool = True  # True = 本物, False = ノイズ

    # メタデータ
    detected_by: list[str] = field(default_factory=list)
    n_detectors: int = 0
    genesis_dims: list[int] = field(default_factory=list)
    genesis_names: list[str] = field(default_factory=list)

    @property
    def is_spurious(self) -> bool:
        return not self.is_genuine

    @property
    def verdict_label(self) -> str:
        return "GENUINE" if self.is_genuine else "SPURIOUS"

    @property
    def confidence_label(self) -> str:
        g = self.genuineness
        if g > 0.8:
            return "very_high"
        if g > 0.6:
            return "high"
        if g > 0.4:
            return "marginal"
        if g > 0.2:
            return "low"
        return "very_low"


@dataclass
class VerificationResult:
    """逆問題検証の全体結果"""

    verdicts: list[EventVerdict] = field(default_factory=list)

    n_total: int = 0
    n_genuine: int = 0
    n_spurious: int = 0
    genuine_ratio: float = 0.0

    # 生スコア（npzで保存用）
    raw_surrogate_scores: np.ndarray | None = None
    raw_rho_t_scores: np.ndarray | None = None
    raw_persistence_scores: np.ndarray | None = None
    raw_coordination_scores: np.ndarray | None = None
    raw_genuineness: np.ndarray | None = None

    def genuine_events(self) -> list[EventVerdict]:
        """GENUINEイベント（本物の構造変化）"""
        return [v for v in self.verdicts if v.is_genuine]

    def spurious_events(self) -> list[EventVerdict]:
        """SPURIOUSイベント（ノイズ）"""
        return [v for v in self.verdicts if v.is_spurious]

    def genuine_frames(self) -> np.ndarray:
        """GENUINEイベントのフレーム配列（CascadeTracker入力用）"""
        return np.array([v.frame for v in self.verdicts if v.is_genuine])

    def genuine_mask(self, n_frames: int) -> np.ndarray:
        """GENUINEイベントのboolマスク（CascadeTracker入力用）"""
        mask = np.zeros(n_frames, dtype=bool)
        for v in self.verdicts:
            if v.is_genuine and v.frame < n_frames:
                mask[v.frame] = True
        return mask

    def get_verdict(self, event_id: int) -> EventVerdict | None:
        for v in self.verdicts:
            if v.event_id == event_id:
                return v
        return None

    def save_raw(self, path: str) -> None:
        np.savez_compressed(
            path,
            surrogate_scores=self.raw_surrogate_scores,
            rho_t_scores=self.raw_rho_t_scores,
            persistence_scores=self.raw_persistence_scores,
            coordination_scores=self.raw_coordination_scores,
            genuineness=self.raw_genuineness,
        )
        logger.info(f"💾 Raw scores saved to {path}")

    # === 後方互換エイリアス ===
    @property
    def n_structural(self) -> int:
        return self.n_genuine

    @property
    def n_noise(self) -> int:
        return self.n_spurious

    @property
    def structural_ratio(self) -> float:
        return self.genuine_ratio

    @property
    def n_normal(self) -> int:
        return self.n_genuine

    @property
    def n_critical(self) -> int:
        return self.n_spurious

    @property
    def normal_ratio(self) -> float:
        return self.genuine_ratio

    def structural_events(self) -> list[EventVerdict]:
        return self.genuine_events()

    def noise_events(self) -> list[EventVerdict]:
        return self.spurious_events()

    def normal_events(self) -> list[EventVerdict]:
        return self.genuine_events()

    def critical_events(self) -> list[EventVerdict]:
        return self.spurious_events()


# ============================================
# InverseChecker
# ============================================


class InverseChecker:
    """
    逆問題による構造的イベント検証ゲート

    Detectionの後、Cascadeの前に挟まり、
    検出されたΔΛCイベント候補が本物かノイズかを精査する。

    Parameters
    ----------
    w_surrogate : float
        Surrogate Test の重み
    w_rho_t : float
        ρT整合性の重み
    w_persistence : float
        持続性の重み
    w_coordination : float
        協調性の重み
    genuineness_threshold : float
        GENUINE判定の閾値（これ以上でGENUINE）
    surrogate_n_trials : int
        Surrogate Testのシャッフル回数
    persistence_window : int
        持続性判定の前後ウィンドウ
    coordination_window : int
        協調性判定のフレーム範囲
    delta_percentile : float
        ΔΛCイベント検出のパーセンタイル閾値
    """

    def __init__(
        self,
        *,
        w_surrogate: float = 0.3,
        w_rho_t: float = 0.25,
        w_persistence: float = 0.25,
        w_coordination: float = 0.2,
        genuineness_threshold: float = 0.4,
        surrogate_n_trials: int = 100,
        persistence_window: int = 20,
        coordination_window: int = 3,
        delta_percentile: float = 94.0,
    ) -> None:
        self.w_surrogate = w_surrogate
        self.w_rho_t = w_rho_t
        self.w_persistence = w_persistence
        self.w_coordination = w_coordination
        self.genuineness_threshold = genuineness_threshold
        self.surrogate_n_trials = surrogate_n_trials
        self.persistence_window = persistence_window
        self.coordination_window = coordination_window
        self.delta_percentile = delta_percentile

        logger.info(
            f"✅ InverseChecker initialized "
            f"(w_surr={w_surrogate}, w_rhoT={w_rho_t}, "
            f"w_persist={w_persistence}, w_coord={w_coordination}, "
            f"threshold={genuineness_threshold})"
        )

    # ================================================================
    # Main Entry Point
    # ================================================================

    def verify(
        self,
        event_mask: np.ndarray,
        state_vectors: np.ndarray,
        local_lambda: dict[str, np.ndarray],
        *,
        detector_labels: dict[int, list[str]] | None = None,
        dimension_names: list[str] | None = None,
    ) -> VerificationResult:
        """
        Detectionの結果を逆問題で検証

        Parameters
        ----------
        event_mask : np.ndarray of shape (n_frames,)
            Detection OR統合後のboolマスク（True = イベント候補）
        state_vectors : np.ndarray of shape (n_frames, n_dims)
            元の状態ベクトル
        local_lambda : dict[str, np.ndarray]
            DualCore._compute_local() の出力:
              local_lambda_F : (n_frames-1, n_dims)
              local_rho_T    : (n_frames, n_dims)
              local_std      : (n_frames, n_dims)
        detector_labels : dict[int, list[str]], optional
            各フレームを検出したdetector名のリスト
        dimension_names : list[str], optional
            次元名

        Returns
        -------
        VerificationResult
            全イベント候補の検証結果。
            genuine_mask() で CascadeTracker に渡すマスクが取れる。
        """
        n_frames, n_dims = state_vectors.shape

        if dimension_names is None:
            dimension_names = [f"dim_{i}" for i in range(n_dims)]

        # イベント候補フレームの抽出
        event_frames = np.where(event_mask)[0]

        if len(event_frames) == 0:
            logger.warning("⚠️ No event candidates to verify")
            return VerificationResult()

        logger.info(f"🔺 Verifying {len(event_frames)} event candidates...")

        # Λ³ Local特徴量の取得
        local_lambda_f = local_lambda.get("local_lambda_F")
        local_rho_t = local_lambda.get("local_rho_T")
        local_std = local_lambda.get("local_std")

        if local_lambda_f is None or local_rho_t is None:
            raise ValueError(
                "local_lambda must contain 'local_lambda_F' and 'local_rho_T'"
            )

        # 候補イベントの特徴量行列を構築
        event_features = self._build_event_features(
            event_frames,
            state_vectors,
            local_lambda_f,
            local_rho_t,
            local_std,
            n_frames,
            n_dims,
        )

        # GPU並列で4軸検証
        raw = inverse_verify_all(
            event_frames=event_frames,
            state_vectors=state_vectors,
            local_lambda_f=local_lambda_f,
            local_rho_t=local_rho_t,
            local_std=local_std,
            event_features=event_features,
            w_surrogate=self.w_surrogate,
            w_rho_t=self.w_rho_t,
            w_persistence=self.w_persistence,
            w_coordination=self.w_coordination,
            surrogate_n_trials=self.surrogate_n_trials,
            persistence_window=self.persistence_window,
            coordination_window=self.coordination_window,
            delta_percentile=self.delta_percentile,
        )

        # 結果の構築
        result = self._build_result(
            event_frames,
            raw,
            local_lambda_f,
            local_rho_t,
            detector_labels,
            dimension_names,
            n_dims,
        )

        self._print_summary(result)

        return result

    # ================================================================
    # Event Feature Extraction
    # ================================================================

    def _build_event_features(
        self,
        event_frames: np.ndarray,
        state_vectors: np.ndarray,
        local_lambda_f: np.ndarray,
        local_rho_t: np.ndarray,
        local_std: np.ndarray | None,
        n_frames: int,
        n_dims: int,
    ) -> np.ndarray:
        """
        イベント候補の特徴量行列を構築

        各イベントフレームについて:
          - state_vector
          - local_rho_T (全次元)
          - local_lambda_F (全次元、フレーム調整)
          - local_std (全次元)
        を結合する。
        """
        n_events = len(event_frames)

        # 特徴量: state + rho_T + |lambda_F| + local_std
        n_features = n_dims * 4
        features = np.zeros((n_events, n_features), dtype=np.float32)

        n_diff = local_lambda_f.shape[0]

        for i, frame in enumerate(event_frames):
            f = min(frame, n_frames - 1)
            f_diff = min(frame, n_diff - 1)

            features[i, :n_dims] = state_vectors[f]
            features[i, n_dims : n_dims * 2] = local_rho_t[f]
            features[i, n_dims * 2 : n_dims * 3] = np.abs(
                local_lambda_f[f_diff]
            )
            if local_std is not None:
                features[i, n_dims * 3 :] = local_std[f]

        return features

    # ================================================================
    # Result Construction
    # ================================================================

    def _build_result(
        self,
        event_frames: np.ndarray,
        raw: dict[str, np.ndarray],
        local_lambda_f: np.ndarray,
        local_rho_t: np.ndarray,
        detector_labels: dict[int, list[str]] | None,
        dimension_names: list[str],
        n_dims: int,
    ) -> VerificationResult:
        """GPU結果からVerificationResultを構築"""

        verdicts = []
        n_diff = local_lambda_f.shape[0]

        for i, frame in enumerate(event_frames):
            f_diff = min(frame, n_diff - 1)

            # ΔΛC強度: local_lambda_Fの全次元の絶対値合計
            delta_lc = float(np.sum(np.abs(local_lambda_f[f_diff])))

            # 発火次元の特定
            active_dims = np.where(np.abs(local_lambda_f[f_diff]) > 0)[0]
            genesis_dims = active_dims.tolist()
            genesis_names = [
                dimension_names[d]
                for d in genesis_dims
                if d < len(dimension_names)
            ]

            # detector情報
            det_by = []
            n_det = 0
            if detector_labels and frame in detector_labels:
                det_by = detector_labels[frame]
                n_det = len(det_by)

            genuineness = float(raw["genuineness"][i])

            verdict = EventVerdict(
                event_id=i,
                frame=int(frame),
                delta_lambda_c=delta_lc,
                surrogate_score=float(raw["surrogate_scores"][i]),
                rho_t_score=float(raw["rho_t_scores"][i]),
                persistence_score=float(raw["persistence_scores"][i]),
                coordination_score=float(raw["coordination_scores"][i]),
                genuineness=genuineness,
                is_genuine=genuineness > self.genuineness_threshold,
                detected_by=det_by,
                n_detectors=n_det,
                genesis_dims=genesis_dims,
                genesis_names=genesis_names,
            )
            verdicts.append(verdict)

        n_genuine = sum(1 for v in verdicts if v.is_genuine)
        n_total = len(verdicts)

        return VerificationResult(
            verdicts=verdicts,
            n_total=n_total,
            n_genuine=n_genuine,
            n_spurious=n_total - n_genuine,
            genuine_ratio=n_genuine / n_total if n_total > 0 else 0.0,
            raw_surrogate_scores=raw["surrogate_scores"],
            raw_rho_t_scores=raw["rho_t_scores"],
            raw_persistence_scores=raw["persistence_scores"],
            raw_coordination_scores=raw["coordination_scores"],
            raw_genuineness=raw["genuineness"],
        )

    # ================================================================
    # Output
    # ================================================================

    def _print_summary(self, result: VerificationResult) -> None:
        logger.info("=" * 55)
        logger.info("🔺 Inverse Verification Summary")
        logger.info("=" * 55)
        logger.info(f"  Total candidates: {result.n_total}")
        logger.info(
            f"  GENUINE:  {result.n_genuine} ({result.genuine_ratio:.1%})"
        )
        logger.info(f"  SPURIOUS: {result.n_spurious}")

        if result.verdicts:
            sorted_v = sorted(
                result.verdicts,
                key=lambda v: v.genuineness,
                reverse=True,
            )

            logger.info("")
            logger.info("  Top GENUINE events (real structural changes):")
            for v in sorted_v[:5]:
                if v.is_genuine:
                    dims_str = (
                        ", ".join(v.genesis_names[:3])
                        if v.genesis_names
                        else "N/A"
                    )
                    logger.info(
                        f"    [GENUINE] E{v.event_id:>3} "
                        f"(frame={v.frame}) "
                        f"genuineness={v.genuineness:.3f} "
                        f"[{v.confidence_label}] "
                        f"dims=[{dims_str}]"
                    )

            spurious = [v for v in sorted_v if v.is_spurious]
            if spurious:
                logger.info("")
                logger.info("  SPURIOUS events (noise, rejected):")
                for v in spurious[-3:]:
                    logger.info(
                        f"    [SPURIOUS] E{v.event_id:>3} "
                        f"(frame={v.frame}) "
                        f"genuineness={v.genuineness:.3f} "
                        f"[{v.confidence_label}]"
                    )

        # 4軸スコアの統計
        if result.raw_genuineness is not None and len(result.raw_genuineness) > 0:
            logger.info("")
            logger.info("  Axis statistics (mean ± std):")
            for name, scores in [
                ("Surrogate", result.raw_surrogate_scores),
                ("ρT Consistency", result.raw_rho_t_scores),
                ("Persistence", result.raw_persistence_scores),
                ("Coordination", result.raw_coordination_scores),
            ]:
                if scores is not None:
                    logger.info(
                        f"    {name:<16}: "
                        f"{np.mean(scores):.3f} ± {np.std(scores):.3f}"
                    )
