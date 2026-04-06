"""
Inverse Checker — Structural Event Verification Gate
=====================================================
⚠️ GPU ONLY — pip install getter-one[gpu]
Built with 💕 by Masamichi & Tamaki

Detection パイプラインの品質保証ゲート。
5種のDetectorが検出した ΔΛC イベント候補を逆問題で精査し、
GENUINE（本物の構造変化）か SPURIOUS（ノイズ）かを判定する。

位置づけ:
  Detection (OR union) → ★ InverseChecker ★ → Cascade (GENUINEのみ)

逆問題 3軸:
  1. Reconstruction Error — Local Λ³構造でデータを再構成できるか
  2. Topo Preservation   — 位相構造の連続性が保たれているか
  3. Jump Integrity      — 変化が構造的に有意か

3軸を重み付き統合し、hybrid_score > 0 なら GENUINE、≤ 0 なら SPURIOUS。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..core.gpu_inverse import inverse_verify_all

logger = logging.getLogger("getter_one.analysis.inverse_checker")


# ============================================
# Data Classes
# ============================================


@dataclass
class EventVerdict:
    """単一イベントの検証結果"""

    event_id: int
    frame: int
    delta_lambda_c: float

    # 3軸スコア（逆問題）
    reconstruction_error: float = 0.0  # 低い = 構造で説明可能
    topo_score: float = 0.0  # 低い = 位相整合的
    jump_score: float = 0.0  # 高い = 構造的に有意なジャンプ

    # 統合
    hybrid_score: float = 0.0  # 正 = GENUINE, 負 = SPURIOUS
    is_genuine: bool = True

    # メタデータ
    genesis_dims: list[int] = field(default_factory=list)
    genesis_names: list[str] = field(default_factory=list)

    @property
    def is_spurious(self) -> bool:
        return not self.is_genuine

    @property
    def is_normal(self) -> bool:
        """後方互換"""
        return self.is_genuine

    @property
    def is_critical(self) -> bool:
        """後方互換"""
        return self.is_spurious

    @property
    def verdict_label(self) -> str:
        return "GENUINE" if self.is_genuine else "SPURIOUS"

    @property
    def confidence_label(self) -> str:
        h = self.hybrid_score
        if h > 1.5:
            return "very_high"
        if h > 0.5:
            return "high"
        if h > 0.0:
            return "marginal"
        if h > -0.5:
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

    # 生スコア配列
    raw_reconstruction_errors: np.ndarray | None = None
    raw_topo_scores: np.ndarray | None = None
    raw_jump_scores: np.ndarray | None = None
    raw_hybrid_scores: np.ndarray | None = None

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
            reconstruction_errors=self.raw_reconstruction_errors,
            topo_scores=self.raw_topo_scores,
            jump_scores=self.raw_jump_scores,
            hybrid_scores=self.raw_hybrid_scores,
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
    検出されたΔΛCイベント候補が本物かノイズかを
    Λ³逆問題（3軸）で精査する。

    Parameters
    ----------
    w_recon : float
        再構成誤差の重み
    w_topo : float
        トポロジー保存の重み
    w_jump : float
        ジャンプ整合性の重み
    jump_sigma : float
        ジャンプ判定のσ倍率
    verdict_threshold : float
        hybrid_score の判定閾値（これ以上でGENUINE）
    """

    def __init__(
        self,
        *,
        w_recon: float = 0.4,
        w_topo: float = 0.3,
        w_jump: float = 0.3,
        jump_sigma: float = 2.5,
        verdict_threshold: float = 0.0,
    ) -> None:
        self.w_recon = w_recon
        self.w_topo = w_topo
        self.w_jump = w_jump
        self.jump_sigma = jump_sigma
        self.verdict_threshold = verdict_threshold

        logger.info(
            f"✅ InverseChecker initialized "
            f"(w_recon={w_recon}, w_topo={w_topo}, w_jump={w_jump}, "
            f"jump_σ={jump_sigma}, threshold={verdict_threshold})"
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
        dimension_names: list[str] | None = None,
    ) -> VerificationResult:
        """
        DetectionのOR統合結果を逆問題で検証

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
        dimension_names : list[str], optional
            次元名

        Returns
        -------
        VerificationResult
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

        # イベント特徴量の抽出
        event_features = self._extract_event_features(
            event_frames, state_vectors, local_lambda
        )

        # Λ³パス行列の構築（Local ΛF から）
        lambda_matrix = self._build_lambda_matrix(
            event_frames, local_lambda
        )

        # GPU逆問題検証（3軸並列）
        raw = inverse_verify_all(
            lambda_matrix=lambda_matrix,
            events=event_features,
            w_recon=self.w_recon,
            w_topo=self.w_topo,
            w_jump=self.w_jump,
            jump_sigma=self.jump_sigma,
        )

        # 結果の構築
        result = self._build_result(
            event_frames, raw, local_lambda, dimension_names
        )

        self._print_summary(result)
        return result

    # ================================================================
    # Feature Extraction
    # ================================================================

    def _extract_event_features(
        self,
        event_frames: np.ndarray,
        state_vectors: np.ndarray,
        local_lambda: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        イベント特徴量を抽出

        state_vector + local_rho_T + local_Q_lambda(あれば)
        """
        n_events = len(event_frames)
        n_frames, n_dims = state_vectors.shape

        local_rho_t = local_lambda.get(
            "local_rho_T", np.zeros((n_frames, n_dims))
        )
        local_q = local_lambda.get(
            "local_Q_lambda", np.zeros((n_frames, n_dims))
        )

        # 特徴量: state + rho_T + Q_lambda
        n_features = n_dims * 3
        features = np.zeros((n_events, n_features), dtype=np.float32)

        for i, frame in enumerate(event_frames):
            f = min(frame, n_frames - 1)
            features[i, :n_dims] = state_vectors[f]
            features[i, n_dims : n_dims * 2] = local_rho_t[f]
            features[i, n_dims * 2 :] = local_q[f]

        return features

    # ================================================================
    # Lambda Matrix Construction (Local ΛF)
    # ================================================================

    def _build_lambda_matrix(
        self,
        event_frames: np.ndarray,
        local_lambda: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        イベントフレームのΛ³パス行列を構築（Local ΛF版）

        Local ΛFは (n_diff, n_dims) なので、
        n_paths = n_dims として
        (n_dims, n_events) のパス行列を返す。

        旧版との違い:
          旧: lambda_structures["lambda_F"] (Global, 多次元ベクトル)
          新: local_lambda["local_lambda_F"] (Local, 各次元独立ジャンプ検出済み)
        """
        local_lambda_f = local_lambda.get("local_lambda_F")
        if local_lambda_f is None:
            raise ValueError(
                "local_lambda must contain 'local_lambda_F'"
            )

        n_diff, n_dims = local_lambda_f.shape
        n_events = len(event_frames)

        # (n_dims, n_events) — 各パス(=次元)でのイベント時点の値
        matrix = np.zeros((n_dims, n_events), dtype=np.float32)

        for i, frame in enumerate(event_frames):
            f = min(frame, n_diff - 1)
            matrix[:, i] = local_lambda_f[f]

        return matrix

    # ================================================================
    # Result Construction
    # ================================================================

    def _build_result(
        self,
        event_frames: np.ndarray,
        raw: dict[str, np.ndarray],
        local_lambda: dict[str, np.ndarray],
        dimension_names: list[str],
    ) -> VerificationResult:
        """GPU結果からVerificationResultを構築"""

        local_lambda_f = local_lambda.get("local_lambda_F")
        n_diff = local_lambda_f.shape[0] if local_lambda_f is not None else 0

        verdicts = []
        for i, frame in enumerate(event_frames):
            f_diff = min(frame, n_diff - 1) if n_diff > 0 else 0

            # ΔΛC強度
            if local_lambda_f is not None:
                delta_lc = float(
                    np.sum(np.abs(local_lambda_f[f_diff]))
                )
                # 発火次元の特定
                active = np.where(np.abs(local_lambda_f[f_diff]) > 0)[0]
                genesis_dims = active.tolist()
                genesis_names = [
                    dimension_names[d]
                    for d in genesis_dims
                    if d < len(dimension_names)
                ]
            else:
                delta_lc = 0.0
                genesis_dims = []
                genesis_names = []

            hybrid = float(raw["hybrid_scores"][i])

            verdict = EventVerdict(
                event_id=i,
                frame=int(frame),
                delta_lambda_c=delta_lc,
                reconstruction_error=float(raw["reconstruction_errors"][i]),
                topo_score=float(raw["topo_scores"][i]),
                jump_score=float(raw["jump_scores"][i]),
                hybrid_score=hybrid,
                is_genuine=hybrid > self.verdict_threshold,
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
            raw_reconstruction_errors=raw["reconstruction_errors"],
            raw_topo_scores=raw["topo_scores"],
            raw_jump_scores=raw["jump_scores"],
            raw_hybrid_scores=raw["hybrid_scores"],
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
                key=lambda v: v.hybrid_score,
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
                        f"hybrid={v.hybrid_score:+.3f} "
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
                        f"hybrid={v.hybrid_score:+.3f} "
                        f"[{v.confidence_label}]"
                    )
