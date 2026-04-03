"""
Lambda³ Structure Computation - Core (Domain-Agnostic)
=====================================================
Built with 💕 by Masamichi & Tamaki
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

# GPU imports
try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None

# CUDA kernel imports（3Dパス用）
try:
    from ..core.gpu_kernels import (
        tension_field_kernel,
    )

    HAS_KERNELS = True
except ImportError:
    HAS_KERNELS = False

logger = logging.getLogger("bankai.structures.lambda_structures_core")


@dataclass
class LambdaCoreConfig:
    """Lambda³ Core 計算設定"""
    verbose: bool = True
    force_cpu: bool = False      # True ならGPUを無効化


class LambdaStructuresCore:
    """
    汎用Lambda³構造計算

    任意のN次元時系列データに対してLambda³構造特徴量を計算する。

    GPU加速:
      - CuPy が利用可能なら自動的にGPU上で計算
      - n_dims==3 の場合は CUDA カーネル（gpu_kernels.py）を直接使用
      - force_cpu=True で強制的にCPUモード

    Parameters
    ----------
    config : LambdaCoreConfig, optional
        計算設定
    """

    def __init__(self, config: LambdaCoreConfig | None = None):
        self.config = config or LambdaCoreConfig()

        # GPU判定
        self.use_gpu = HAS_GPU and not self.config.force_cpu
        self.use_kernels = HAS_KERNELS and self.use_gpu

        # 演算バックエンド
        self.xp = cp if self.use_gpu else np

        if self.config.verbose:
            if self.use_kernels:
                mode = "GPU + CUDA Kernels"
            elif self.use_gpu:
                mode = "GPU (CuPy)"
            else:
                mode = "CPU"
            logger.info(f"✅ LambdaStructuresCore initialized ({mode})")

    def compute_lambda_structures(
        self,
        state_vectors: np.ndarray,
        window_steps: int,
        dimension_names: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Lambda³構造を計算

        Parameters
        ----------
        state_vectors : np.ndarray
            N次元状態ベクトル時系列 (n_frames, n_dims)
        window_steps : int
            スライディングウィンドウサイズ
        dimension_names : list[str], optional
            各次元の名前（ログ出力用）

        Returns
        -------
        dict[str, np.ndarray]
            Lambda構造辞書（lambda_structures_gpu.pyと同一フォーマット）
            全出力は numpy 配列（GPU計算後も自動変換）
        """
        if state_vectors.ndim != 2:
            raise ValueError(
                f"state_vectors must be 2D (n_frames, n_dims), "
                f"got shape {state_vectors.shape}"
            )

        n_frames, n_dims = state_vectors.shape

        if self.config.verbose:
            dim_str = (
                ", ".join(dimension_names)
                if dimension_names
                else f"{n_dims}D"
            )
            logger.info(
                f"🚀 Computing Lambda³ structures "
                f"(frames={n_frames}, dims={dim_str}, window={window_steps})"
            )

        # GPU転送（使用時のみ）
        if self.use_gpu:
            sv_gpu = cp.asarray(state_vectors, dtype=cp.float32)
        else:
            sv_gpu = state_vectors

        xp = self.xp

        # 3D + CUDA kernels 利用可能？
        # rho_T: Bessel補正済みカーネル → ✅ 使用（diff=6e-8）
        # Q_lambda: float32 I/O で cross_z≈0 の符号が不安定
        #   → カーネルの入力を double* に変更するまでは CPU パス
        can_use_kernels_rho = self.use_kernels and n_dims == 3

        # 1. ΛF - 構造フロー
        lambda_F, lambda_F_mag = self._compute_lambda_F(sv_gpu, xp)

        # 2. ΛFF - 二次構造フロー
        lambda_FF, lambda_FF_mag = self._compute_lambda_FF(lambda_F, xp)

        # 3. ρT - テンション場
        if can_use_kernels_rho:
            rho_T = self._compute_rho_T_kernel(state_vectors, window_steps)
        elif self.use_gpu:
            rho_T = self._compute_rho_T_gpu(sv_gpu, window_steps)
        else:
            rho_T = self._compute_rho_T_cpu(state_vectors, window_steps)

        # 4. Q_Λ - トポロジカルチャージ（CPU/CuPyパス）
        # float32 I/O カーネルでは cross_z≈0 で符号不安定のため
        Q_lambda, Q_cumulative = self._compute_Q_lambda(
            lambda_F, lambda_F_mag, xp
        )

        # 5. σₛ - 構造同期率
        sigma_s = self._compute_sigma_s(sv_gpu, lambda_F, window_steps, xp)

        # 6. 構造的コヒーレンス
        coherence = self._compute_coherence(lambda_F, window_steps, xp)

        # GPU → CPU 変換
        results = {
            "lambda_F": self._to_numpy(lambda_F),
            "lambda_F_mag": self._to_numpy(lambda_F_mag),
            "lambda_FF": self._to_numpy(lambda_FF),
            "lambda_FF_mag": self._to_numpy(lambda_FF_mag),
            "rho_T": self._to_numpy(rho_T),
            "Q_lambda": self._to_numpy(Q_lambda),
            "Q_cumulative": self._to_numpy(Q_cumulative),
            "sigma_s": self._to_numpy(sigma_s),
            "structural_coherence": self._to_numpy(coherence),
        }

        self._print_statistics(results)

        return results

    # ================================================================
    # GPU/CPU共通: xp版（numpy or cupy）
    # ================================================================

    def _compute_lambda_F(
        self, positions, xp,
    ) -> tuple:
        """ΛF - 構造フロー計算（次元数フリー）"""
        lambda_F = xp.diff(positions, axis=0)
        lambda_F_mag = xp.linalg.norm(lambda_F, axis=1)
        return lambda_F, lambda_F_mag

    def _compute_lambda_FF(
        self, lambda_F, xp,
    ) -> tuple:
        """ΛFF - 二次構造フロー計算"""
        lambda_FF = xp.diff(lambda_F, axis=0)
        lambda_FF_mag = xp.linalg.norm(lambda_FF, axis=1)
        return lambda_FF, lambda_FF_mag

    def _compute_Q_lambda(
        self, lambda_F, lambda_F_mag, xp,
    ) -> tuple:
        """Q_Λ - トポロジカルチャージ計算（N次元汎用）"""
        n_steps = len(lambda_F_mag)

        # CuPyの場合はnumpy配列に一旦戻してループ
        # （ここはカーネル化候補だが、n_stepsが小さいので影響小）
        lf = self._to_numpy(lambda_F)
        lfm = self._to_numpy(lambda_F_mag)

        q_np = np.zeros(n_steps)
        for step in range(1, n_steps):
            if lfm[step] > 1e-10 and lfm[step - 1] > 1e-10:
                v1 = lf[step - 1] / lfm[step - 1]
                v2 = lf[step] / lfm[step]

                cos_angle = np.clip(np.dot(v1, v2), -1, 1)
                angle = np.arccos(cos_angle)

                if len(v1) >= 2:
                    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
                    signed_angle = angle if cross_z >= 0 else -angle
                else:
                    signed_angle = angle

                q_np[step] = signed_angle / (2 * np.pi)

        Q_cumulative_np = np.cumsum(q_np)

        if self.use_gpu:
            return cp.asarray(q_np), cp.asarray(Q_cumulative_np)
        return q_np, Q_cumulative_np

    def _compute_coherence(
        self, lambda_F, window: int, xp,
    ) -> np.ndarray:
        """構造的コヒーレンス計算"""
        # ループベースなのでCPUで計算
        lf = self._to_numpy(lambda_F)
        n_frames = len(lf)
        coherence = np.zeros(n_frames)

        for i in range(window, n_frames - window):
            local_F = lf[i - window: i + window]
            mean_dir = np.mean(local_F, axis=0)
            mean_norm = np.linalg.norm(mean_dir)

            if mean_norm > 1e-10:
                mean_dir /= mean_norm
                norms = np.linalg.norm(local_F, axis=1)
                valid_mask = norms > 1e-10

                if np.any(valid_mask):
                    normalized_F = (
                        local_F[valid_mask] / norms[valid_mask, np.newaxis]
                    )
                    coherences = np.dot(normalized_F, mean_dir)
                    coherence[i] = np.mean(coherences)

        return coherence

    def _compute_sigma_s(
        self, state_vectors, lambda_F, window_steps: int, xp,
    ) -> np.ndarray:
        """
        σₛ - 構造同期率の汎用計算

        GPU版: CuPyのcorrcoefを使用（大きなウィンドウで効果大）
        CPU版: numpy のcorrcoef
        """
        # ループベースなのでnumpy側で実行
        lf = self._to_numpy(lambda_F)
        n_frames = len(lf)
        n_dims = lf.shape[1]
        sigma_s_arr = np.zeros(n_frames + 1)

        if n_dims < 2:
            sv_np = self._to_numpy(state_vectors)
            return sigma_s_arr[:len(sv_np)]

        half_w = window_steps

        for t in range(n_frames):
            start = max(0, t - half_w)
            end = min(n_frames, t + half_w + 1)
            window = lf[start:end]

            if len(window) < 3:
                continue

            stds = window.std(axis=0)
            active_dims = np.where(stds > 1e-10)[0]

            if len(active_dims) < 2:
                sigma_s_arr[t] = 0.0
                continue

            active_window = window[:, active_dims]

            try:
                corr_matrix = np.corrcoef(active_window.T)

                if np.any(np.isnan(corr_matrix)):
                    sigma_s_arr[t] = 0.0
                    continue

                n_active = len(active_dims)
                mask = ~np.eye(n_active, dtype=bool)
                sigma_s_arr[t] = np.abs(corr_matrix[mask]).mean()
            except Exception:
                sigma_s_arr[t] = 0.0

        if n_frames > 0:
            sigma_s_arr[n_frames] = sigma_s_arr[n_frames - 1]

        sv_np = self._to_numpy(state_vectors)
        return sigma_s_arr[:len(sv_np)]

    # ================================================================
    # CUDA Kernel パス（n_dims == 3 専用）
    # ================================================================

    def _compute_rho_T_kernel(
        self, positions: np.ndarray, window_steps: int,
    ) -> np.ndarray:
        """ρT - CUDAカーネル版（3D専用、Bessel補正済み）"""
        logger.debug("   ρT: CUDA kernel path (3D)")
        rho_gpu = tension_field_kernel(positions, window_steps)
        return cp.asnumpy(rho_gpu)

    # ================================================================
    # CuPy GPU パス（N次元汎用）
    # ================================================================

    def _compute_rho_T_gpu(
        self, positions_gpu, window_steps: int,
    ) -> np.ndarray:
        """
        ρT - CuPy GPU版（N次元汎用）

        CUDAカーネルの3D制約を回避しつつGPU高速化。
        共分散トレースをベクトル化操作で計算。
        """
        logger.debug("   ρT: CuPy GPU path (N-dim)")
        n_frames = positions_gpu.shape[0]
        rho_T = cp.zeros(n_frames, dtype=cp.float32)

        for step in range(n_frames):
            start = max(0, step - window_steps)
            end = min(n_frames, step + window_steps + 1)
            local = positions_gpu[start:end]

            if len(local) > 1:
                centered = local - cp.mean(local, axis=0, keepdims=True)
                # 共分散トレース = 各次元の分散の和
                # GPU上のベクトル化: (centered ** 2).mean(axis=0).sum()
                rho_T[step] = cp.mean(centered ** 2, axis=0).sum() * len(local) / (len(local) - 1)

        return cp.asnumpy(rho_T)

    # ================================================================
    # CPU フォールバック
    # ================================================================

    def _compute_rho_T_cpu(
        self, positions: np.ndarray, window_steps: int,
    ) -> np.ndarray:
        """ρT - CPU版（numpy、元の実装）"""
        n_frames = len(positions)
        rho_T = np.zeros(n_frames)

        for step in range(n_frames):
            start = max(0, step - window_steps)
            end = min(n_frames, step + window_steps + 1)
            local_positions = positions[start:end]

            if len(local_positions) > 1:
                centered = local_positions - np.mean(
                    local_positions, axis=0, keepdims=True
                )
                cov = np.cov(centered.T)
                if cov.ndim == 0:
                    rho_T[step] = float(cov)
                else:
                    rho_T[step] = np.trace(cov)

        return rho_T

    # ================================================================
    # ユーティリティ
    # ================================================================

    def _to_numpy(self, arr) -> np.ndarray:
        """GPU配列をnumpy配列に安全に変換"""
        if self.use_gpu and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def _print_statistics(self, results: dict[str, np.ndarray]):
        """統計情報を出力"""
        if not self.config.verbose:
            return

        logger.info("📊 Lambda³ Structure Statistics:")
        for key in [
            "lambda_F_mag", "lambda_FF_mag", "rho_T",
            "Q_cumulative", "sigma_s", "structural_coherence",
        ]:
            if key in results and len(results[key]) > 0:
                data = results[key]
                logger.info(
                    f"   {key}: min={np.min(data):.3e}, "
                    f"max={np.max(data):.3e}, "
                    f"mean={np.mean(data):.3e}, std={np.std(data):.3e}"
                )
