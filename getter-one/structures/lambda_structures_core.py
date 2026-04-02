"""
Lambda³ Structure Computation - Core (Domain-Agnostic)
=====================================================

BANKAI-MDのlambda_structures_gpu.pyの汎用版。
GPU依存・MD依存を完全に排除し、任意のN次元時系列データに対して
Lambda³構造を計算する。

核心的変更点:
  - σₛ (構造同期率) をRMSD/Rg依存から、状態ベクトルの
    次元間相関に変更。これにより任意のN次元データに適用可能。
  - 入力を md_features dict から state_vectors (np.ndarray) に簡素化。
  - 出力フォーマットはlambda_structures_gpu.pyと完全互換。
    → 下流の detection/ モジュール群がそのまま使える。

Compatibility:
  - 出力 dict のキーは lambda_structures_gpu.py と同一
  - BoundaryDetectorGPU, TopologyBreaksDetectorGPU,
    AnomalyDetectorGPU にそのまま接続可能

Built with 💕 by Masamichi & Tamaki
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("bankai.structures.lambda_structures_core")


@dataclass
class LambdaCoreConfig:
    """Lambda³ Core 計算設定"""
    verbose: bool = True


class LambdaStructuresCore:
    """
    汎用Lambda³構造計算

    任意のN次元時系列データに対してLambda³構造特徴量を計算する。
    GPU非依存・MD非依存。numpy/scipyのみで動作。

    Parameters
    ----------
    config : LambdaCoreConfig, optional
        計算設定
    """

    def __init__(self, config: Optional[LambdaCoreConfig] = None):
        self.config = config or LambdaCoreConfig()
        if self.config.verbose:
            logger.info("✅ LambdaStructuresCore initialized (CPU, domain-agnostic)")

    def compute_lambda_structures(
        self,
        state_vectors: np.ndarray,
        window_steps: int,
        dimension_names: Optional[list[str]] = None,
    ) -> dict[str, np.ndarray]:
        """
        Lambda³構造を計算

        Parameters
        ----------
        state_vectors : np.ndarray
            N次元状態ベクトル時系列 (n_frames, n_dims)
            天気なら (748, 6)、MDなら com_positions (n_frames, 3) に相当
        window_steps : int
            スライディングウィンドウサイズ
        dimension_names : list[str], optional
            各次元の名前（ログ出力用）

        Returns
        -------
        dict[str, np.ndarray]
            Lambda構造辞書（lambda_structures_gpu.pyと同一フォーマット）
            - lambda_F: 構造フロー (n_frames-1, n_dims)
            - lambda_F_mag: フロー大きさ (n_frames-1,)
            - lambda_FF: 二次フロー (n_frames-2, n_dims)
            - lambda_FF_mag: 二次フロー大きさ (n_frames-2,)
            - rho_T: テンション場 (n_frames,)
            - Q_lambda: トポロジカルチャージ (n_frames-1,)
            - Q_cumulative: 累積チャージ (n_frames-1,)
            - sigma_s: 構造同期率 (n_frames,)
            - structural_coherence: コヒーレンス (n_frames-1,)
        """
        if state_vectors.ndim != 2:
            raise ValueError(
                f"state_vectors must be 2D (n_frames, n_dims), got shape {state_vectors.shape}"
            )

        n_frames, n_dims = state_vectors.shape

        if self.config.verbose:
            dim_str = ", ".join(dimension_names) if dimension_names else f"{n_dims}D"
            logger.info(
                f"🚀 Computing Lambda³ structures "
                f"(frames={n_frames}, dims={dim_str}, window={window_steps})"
            )

        # 1. ΛF - 構造フロー
        lambda_F, lambda_F_mag = self._compute_lambda_F(state_vectors)

        # 2. ΛFF - 二次構造フロー
        lambda_FF, lambda_FF_mag = self._compute_lambda_FF(lambda_F)

        # 3. ρT - テンション場
        rho_T = self._compute_rho_T(state_vectors, window_steps)

        # 4. Q_Λ - トポロジカルチャージ
        Q_lambda, Q_cumulative = self._compute_Q_lambda(lambda_F, lambda_F_mag)

        # 5. σₛ - 構造同期率（★汎用版の核心！）
        sigma_s = self._compute_sigma_s(state_vectors, lambda_F, window_steps)

        # 6. 構造的コヒーレンス
        coherence = self._compute_coherence(lambda_F, window_steps)

        results = {
            "lambda_F": lambda_F,
            "lambda_F_mag": lambda_F_mag,
            "lambda_FF": lambda_FF,
            "lambda_FF_mag": lambda_FF_mag,
            "rho_T": rho_T,
            "Q_lambda": Q_lambda,
            "Q_cumulative": Q_cumulative,
            "sigma_s": sigma_s,
            "structural_coherence": coherence,
        }

        self._print_statistics(results)

        return results

    # ================================================================
    # 以下のメソッドは lambda_structures_gpu.py と同一ロジック（numpy版）
    # ================================================================

    def _compute_lambda_F(
        self, positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """ΛF - 構造フロー計算（次元数フリー）"""
        lambda_F = np.diff(positions, axis=0)
        lambda_F_mag = np.linalg.norm(lambda_F, axis=1)
        return lambda_F, lambda_F_mag

    def _compute_lambda_FF(
        self, lambda_F: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """ΛFF - 二次構造フロー計算"""
        lambda_FF = np.diff(lambda_F, axis=0)
        lambda_FF_mag = np.linalg.norm(lambda_FF, axis=1)
        return lambda_FF, lambda_FF_mag

    def _compute_rho_T(
        self, positions: np.ndarray, window_steps: int
    ) -> np.ndarray:
        """ρT - テンション場計算（共分散トレース）"""
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

    def _compute_Q_lambda(
        self, lambda_F: np.ndarray, lambda_F_mag: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Q_Λ - トポロジカルチャージ計算"""
        n_steps = len(lambda_F_mag)
        Q_lambda = np.zeros(n_steps)

        for step in range(1, n_steps):
            if lambda_F_mag[step] > 1e-10 and lambda_F_mag[step - 1] > 1e-10:
                v1 = lambda_F[step - 1] / lambda_F_mag[step - 1]
                v2 = lambda_F[step] / lambda_F_mag[step]

                cos_angle = np.clip(np.dot(v1, v2), -1, 1)
                angle = np.arccos(cos_angle)

                # 2D以上: 最初の2成分で回転方向を決定
                if len(v1) >= 2:
                    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
                    signed_angle = angle if cross_z >= 0 else -angle
                else:
                    signed_angle = angle

                Q_lambda[step] = signed_angle / (2 * np.pi)

        Q_cumulative = np.cumsum(Q_lambda)
        return Q_lambda, Q_cumulative

    def _compute_coherence(
        self, lambda_F: np.ndarray, window: int
    ) -> np.ndarray:
        """構造的コヒーレンス計算"""
        n_frames = len(lambda_F)
        coherence = np.zeros(n_frames)

        for i in range(window, n_frames - window):
            local_F = lambda_F[i - window : i + window]

            mean_dir = np.mean(local_F, axis=0)
            mean_norm = np.linalg.norm(mean_dir)

            if mean_norm > 1e-10:
                mean_dir /= mean_norm

                norms = np.linalg.norm(local_F, axis=1)
                valid_mask = norms > 1e-10

                if np.any(valid_mask):
                    normalized_F = local_F[valid_mask] / norms[valid_mask, np.newaxis]
                    coherences = np.dot(normalized_F, mean_dir)
                    coherence[i] = np.mean(coherences)

        return coherence

    # ================================================================
    # σₛ - 構造同期率（★汎用版の核心的変更）
    # ================================================================

    def _compute_sigma_s(
        self,
        state_vectors: np.ndarray,
        lambda_F: np.ndarray,
        window_steps: int,
    ) -> np.ndarray:
        """
        σₛ - 構造同期率の汎用計算

        MD版 (lambda_structures_gpu.py):
          RMSD と Rg の相関 → 2つのスカラー量の協調度

        汎用版 (本モジュール):
          状態ベクトルの全次元間の変位相関 → N次元の協調度

        「全次元が同時に同方向に動く」= 高同期
        「各次元がバラバラに動く」= 低同期

        これがBANKAI汎用化の核心:
          MDではRMSD/Rgという「物理的意味のある2指標」に依存していたが、
          汎用版では「状態空間のN次元全てが協調的に変化しているか」を
          直接測定する。意味はドメインに依存するが、数学は共通。

        Parameters
        ----------
        state_vectors : np.ndarray (n_frames, n_dims)
        lambda_F : np.ndarray (n_frames-1, n_dims)
            変位ベクトル
        window_steps : int
        """
        n_frames = len(lambda_F)
        n_dims = lambda_F.shape[1]
        sigma_s_arr = np.zeros(n_frames + 1)  # state_vectorsのフレーム数に合わせる

        if n_dims < 2:
            # 1次元データでは同期率は定義不能
            return sigma_s_arr

        half_w = window_steps

        for t in range(n_frames):
            start = max(0, t - half_w)
            end = min(n_frames, t + half_w + 1)
            window = lambda_F[start:end]  # (w, n_dims)

            if len(window) < 3:
                continue

            # 各次元のstd確認: ゼロ分散の次元を除外
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

                # 対角を除いた相関の平均 = 全次元の協調度
                n_active = len(active_dims)
                mask = ~np.eye(n_active, dtype=bool)
                sigma_s_arr[t] = np.abs(corr_matrix[mask]).mean()
            except Exception:
                sigma_s_arr[t] = 0.0

        # n_frames+1 のうち最後の要素は前方からコピー
        if n_frames > 0:
            sigma_s_arr[n_frames] = sigma_s_arr[n_frames - 1]

        # state_vectorsのフレーム数に合わせて返す
        return sigma_s_arr[:len(state_vectors)]

    # ================================================================
    # ユーティリティ
    # ================================================================

    def _print_statistics(self, results: dict[str, np.ndarray]):
        """統計情報を出力"""
        if not self.config.verbose:
            return

        logger.info("📊 Lambda³ Structure Statistics:")
        for key in ["lambda_F_mag", "lambda_FF_mag", "rho_T",
                     "Q_cumulative", "sigma_s", "structural_coherence"]:
            if key in results and len(results[key]) > 0:
                data = results[key]
                logger.info(
                    f"   {key}: min={np.min(data):.3e}, max={np.max(data):.3e}, "
                    f"mean={np.mean(data):.3e}, std={np.std(data):.3e}"
                )
