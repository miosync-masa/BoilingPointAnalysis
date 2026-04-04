"""
Inverse Checker — Structural Verification for Cascade Events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CascadeTrackerが検出したイベントを逆問題で構造的に検証するよ！🔥

検出（System 1）と検証（System 2）の分離により：
- 閾値ガン下げ → Recall ≈ 100%
- inverse_checker → Precision ↑
- Pareto frontの外側へ 🏆

⚠️ GPU ONLY — pip install getter-one[gpu]

by 環ちゃん

Built with 💕 by Masamichi & Tamaki
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..analysis.cascade_tracker import CascadeEvent, CascadeResult

# GPU ONLY import — raises ImportError if no CUDA
from ..core.gpu_inverse import inverse_verify_all

logger = logging.getLogger("getter_one.analysis.inverse_checker")


# ===============================
# Result Data Classes
# ===============================


@dataclass
class EventVerdict:
    """単一イベントの検証結果"""

    event_id: int
    frame: int
    delta_lambda_c: float

    # 3軸スコア
    reconstruction_error: float = 0.0  # 低い = 構造的に説明可能
    topo_score: float = 0.0  # 低い = 位相整合的
    jump_score: float = 0.0  # 高い = 構造的に有意なジャンプ

    # 統合
    hybrid_score: float = 0.0  # 高い = 構造的イベント
    is_structural: bool = False  # True = 本物, False = ノイズ

    # 名前付き情報（cascade_trackerから引き継ぎ）
    genesis_names: list[str] = field(default_factory=list)

    @property
    def confidence_label(self) -> str:
        """信頼度ラベル"""
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

    # イベント別結果
    verdicts: list[EventVerdict] = field(default_factory=list)

    # 統計
    n_total: int = 0
    n_structural: int = 0
    n_noise: int = 0
    structural_ratio: float = 0.0

    # 生スコア配列（npyで保存用）
    raw_reconstruction_errors: np.ndarray | None = None
    raw_topo_scores: np.ndarray | None = None
    raw_jump_scores: np.ndarray | None = None
    raw_hybrid_scores: np.ndarray | None = None

    def structural_events(self) -> list[EventVerdict]:
        """構造的イベントのみ返す"""
        return [v for v in self.verdicts if v.is_structural]

    def noise_events(self) -> list[EventVerdict]:
        """ノイズ判定イベントのみ返す"""
        return [v for v in self.verdicts if not v.is_structural]

    def get_verdict(self, event_id: int) -> EventVerdict | None:
        """event_idで検索"""
        for v in self.verdicts:
            if v.event_id == event_id:
                return v
        return None

    def save_raw(self, path: str) -> None:
        """生スコアをnpzで保存"""
        np.savez_compressed(
            path,
            reconstruction_errors=self.raw_reconstruction_errors,
            topo_scores=self.raw_topo_scores,
            jump_scores=self.raw_jump_scores,
            hybrid_scores=self.raw_hybrid_scores,
        )
        logger.info(f"💾 Raw scores saved to {path}")


# ===============================
# Inverse Checker
# ===============================


class InverseChecker:
    """
    逆問題による構造的イベント検証

    CascadeTrackerの検出結果を受け取り、各イベントが
    「Λ³構造で説明可能か」をGPU並列で検証する。

    Parameters
    ----------
    w_recon : float
        再構成誤差の重み (default: 0.4)
    w_topo : float
        トポロジー保存の重み (default: 0.3)
    w_jump : float
        ジャンプ整合性の重み (default: 0.3)
    jump_sigma : float
        ジャンプ判定のσ倍率 (default: 2.5)
    verdict_threshold : float
        hybrid_score の判定閾値 (default: 0.0)
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

    def verify(
        self,
        cascade_result: CascadeResult,
        state_vectors: np.ndarray,
        lambda_structures: dict[str, np.ndarray],
        *,
        event_ids: list[int] | None = None,
    ) -> VerificationResult:
        """
        CascadeTrackerの結果を逆問題で検証

        Parameters
        ----------
        cascade_result : CascadeResult
            CascadeTracker.track() の出力
        state_vectors : np.ndarray
            (n_frames, n_dims) 元の状態ベクトル
        lambda_structures : dict[str, np.ndarray]
            LambdaStructuresCore.compute_lambda_structures() の出力
        event_ids : list[int], optional
            検証対象イベントのID群。Noneなら全イベント検証

        Returns
        -------
        VerificationResult
            全イベントの構造的検証結果
        """
        events = cascade_result.events
        if not events:
            logger.warning("⚠️ No events to verify")
            return VerificationResult()

        # 対象イベントのフィルタリング
        if event_ids is not None:
            target_ids = set(event_ids)
            events = [e for e in events if e.event_id in target_ids]
            if not events:
                logger.warning("⚠️ No matching events found")
                return VerificationResult()

        logger.info(f"🔺 Verifying {len(events)} events...")

        # イベント特徴量の抽出
        event_features = self._extract_event_features(
            events, state_vectors, lambda_structures
        )

        # Λ³パス行列の構築
        lambda_matrix = self._build_lambda_matrix(events, lambda_structures)

        # GPU逆問題検証（全イベント並列）
        raw = inverse_verify_all(
            lambda_matrix=lambda_matrix,
            events=event_features,
            w_recon=self.w_recon,
            w_topo=self.w_topo,
            w_jump=self.w_jump,
            jump_sigma=self.jump_sigma,
        )

        # 結果の構築
        result = self._build_result(events, raw)

        self._print_summary(result)

        return result

    def verify_genesis_only(
        self,
        cascade_result: CascadeResult,
        state_vectors: np.ndarray,
        lambda_structures: dict[str, np.ndarray],
    ) -> VerificationResult:
        """
        カスケードの起点（genesis）イベントのみ検証

        カスケードの根っこが本物かどうかを判定する。
        能登地震の教訓: 小さな変化こそ検証が重要。
        """
        # チェーンの起点イベントIDを収集
        genesis_ids: set[int] = set()
        for chain in cascade_result.chains:
            if chain.event_ids:
                genesis_ids.add(chain.event_ids[0])

        if not genesis_ids:
            logger.warning("⚠️ No genesis events found in chains")
            return VerificationResult()

        logger.info(f"🌱 Verifying {len(genesis_ids)} genesis events (chain origins)")

        return self.verify(
            cascade_result,
            state_vectors,
            lambda_structures,
            event_ids=list(genesis_ids),
        )

    def filter_cascade(
        self,
        cascade_result: CascadeResult,
        verification: VerificationResult,
    ) -> CascadeResult:
        """
        検証結果に基づいてカスケードをフィルタリング

        信頼度の低いgenesisから始まるチェーンを棄却し、
        構造的に検証済みのカスケードマップを返す。

        Parameters
        ----------
        cascade_result : CascadeResult
            元のCascadeResult
        verification : VerificationResult
            verify()の結果

        Returns
        -------
        CascadeResult
            フィルタ済みの結果
        """
        # 構造的イベントのIDセット
        structural_ids = {v.event_id for v in verification.verdicts if v.is_structural}

        # チェーンのフィルタ: 起点が構造的なもののみ
        filtered_chains = []
        for chain in cascade_result.chains:
            if not chain.event_ids:
                continue
            origin_id = chain.event_ids[0]
            if origin_id in structural_ids:
                filtered_chains.append(chain)

        n_kept = len(filtered_chains)
        n_dropped = len(cascade_result.chains) - n_kept

        logger.info(
            f"🔗 Cascade filter: {n_kept} chains kept, {n_dropped} chains dropped"
        )

        # 新しいCascadeResultを構築
        filtered = CascadeResult(
            events=cascade_result.events,
            n_events=cascade_result.n_events,
            links=cascade_result.links,
            n_links=cascade_result.n_links,
            chains=filtered_chains,
            n_chains=n_kept,
            longest_chain=(max((c.length for c in filtered_chains), default=0)),
            dag=cascade_result.dag,
            critical_dims=cascade_result.critical_dims,
            critical_names=cascade_result.critical_names,
            cascade_coverage=cascade_result.cascade_coverage,
            mean_chain_length=(
                float(np.mean([c.length for c in filtered_chains]))
                if filtered_chains
                else 0.0
            ),
            branching_ratio=cascade_result.branching_ratio,
            dimension_names=cascade_result.dimension_names,
        )

        return filtered

    # ===============================
    # Internal Methods
    # ===============================

    def _extract_event_features(
        self,
        events: list[CascadeEvent],
        state_vectors: np.ndarray,
        lambda_structures: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        イベントフレームの特徴量を抽出

        state_vectors + Λ³構造の値を結合して
        イベント特徴量ベクトルを構築する。
        """
        n_events = len(events)
        _, n_dims = state_vectors.shape

        rho_t = lambda_structures.get("rho_T", np.zeros(len(state_vectors)))
        sigma_s = lambda_structures.get("sigma_s", np.zeros(len(state_vectors)))
        coherence = lambda_structures.get(
            "structural_coherence", np.zeros(len(state_vectors))
        )

        # 特徴量: state_vector + rho_T + sigma_s + coherence
        n_features = n_dims + 3
        features = np.zeros((n_events, n_features), dtype=np.float32)

        for i, event in enumerate(events):
            f = min(event.frame, len(state_vectors) - 1)
            features[i, :n_dims] = state_vectors[f]
            features[i, n_dims] = rho_t[f] if f < len(rho_t) else 0.0
            features[i, n_dims + 1] = sigma_s[f] if f < len(sigma_s) else 0.0
            features[i, n_dims + 2] = coherence[f] if f < len(coherence) else 0.0

        return features

    def _build_lambda_matrix(
        self,
        events: list[CascadeEvent],
        lambda_structures: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        イベントフレームのΛ³パス行列を構築

        lambda_F の各次元を「パス」として、イベントフレームでの
        値を抽出してパス行列を構成する。
        """
        lambda_f = lambda_structures.get("lambda_F")
        if lambda_f is None:
            raise ValueError("lambda_structures must contain 'lambda_F'")

        n_events = len(events)
        frames = [min(e.frame, len(lambda_f) - 1) for e in events]

        if lambda_f.ndim == 2:
            # (n_frames, n_dims) → n_paths = n_dims
            n_paths = lambda_f.shape[1]
            matrix = np.zeros((n_paths, n_events), dtype=np.float32)
            for i, f in enumerate(frames):
                matrix[:, i] = lambda_f[f]
        elif lambda_f.ndim == 3:
            # (n_frames, n_residues, 3) → n_paths = n_residues * 3
            n_frames, n_res, n_comp = lambda_f.shape
            n_paths = n_res * n_comp
            matrix = np.zeros((n_paths, n_events), dtype=np.float32)
            for i, f in enumerate(frames):
                matrix[:, i] = lambda_f[f].ravel()
        else:
            raise ValueError(f"Unexpected lambda_F shape: {lambda_f.shape}")

        return matrix

    def _build_result(
        self,
        events: list[CascadeEvent],
        raw: dict[str, np.ndarray],
    ) -> VerificationResult:
        """GPU結果からVerificationResultを構築"""
        verdicts = []
        for i, event in enumerate(events):
            hybrid = float(raw["hybrid_scores"][i])
            verdict = EventVerdict(
                event_id=event.event_id,
                frame=event.frame,
                delta_lambda_c=event.delta_lambda_c,
                reconstruction_error=float(raw["reconstruction_errors"][i]),
                topo_score=float(raw["topo_scores"][i]),
                jump_score=float(raw["jump_scores"][i]),
                hybrid_score=hybrid,
                is_structural=hybrid > self.verdict_threshold,
                genesis_names=event.genesis_names,
            )
            verdicts.append(verdict)

        n_structural = sum(1 for v in verdicts if v.is_structural)
        n_total = len(verdicts)

        return VerificationResult(
            verdicts=verdicts,
            n_total=n_total,
            n_structural=n_structural,
            n_noise=n_total - n_structural,
            structural_ratio=(n_structural / n_total if n_total > 0 else 0.0),
            raw_reconstruction_errors=raw["reconstruction_errors"],
            raw_topo_scores=raw["topo_scores"],
            raw_jump_scores=raw["jump_scores"],
            raw_hybrid_scores=raw["hybrid_scores"],
        )

    def _print_summary(self, result: VerificationResult) -> None:
        """検証サマリーを出力"""
        logger.info("=" * 50)
        logger.info("🔺 Inverse Verification Summary")
        logger.info("=" * 50)
        logger.info(f"  Total events: {result.n_total}")
        logger.info(
            f"  Structural:   {result.n_structural} ({result.structural_ratio:.1%})"
        )
        logger.info(f"  Noise:        {result.n_noise}")

        # 上位イベント
        if result.verdicts:
            sorted_v = sorted(
                result.verdicts, key=lambda v: v.hybrid_score, reverse=True
            )
            logger.info("")
            logger.info("  Top structural events:")
            for v in sorted_v[:5]:
                status = "✅" if v.is_structural else "❌"
                logger.info(
                    f"    {status} E{v.event_id:>3} "
                    f"(frame={v.frame}) "
                    f"hybrid={v.hybrid_score:+.3f} "
                    f"[{v.confidence_label}]"
                )

            logger.info("")
            logger.info("  Bottom events (noise candidates):")
            for v in sorted_v[-3:]:
                status = "✅" if v.is_structural else "❌"
                logger.info(
                    f"    {status} E{v.event_id:>3} "
                    f"(frame={v.frame}) "
                    f"hybrid={v.hybrid_score:+.3f} "
                    f"[{v.confidence_label}]"
                )
