"""
Lambda³ Structure Computation — DualCore Architecture
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

logger = logging.getLogger("getter_one.structures.lambda_structures_dual_core")


# ================================================================
# 設定
# ================================================================


@dataclass
class DualCoreConfig:
    """DualCore 計算設定"""

    verbose: bool = True
    force_cpu: bool = False

    # アダプティブウィンドウ
    adaptive_window: bool = True
    base_window: int = 30
    min_window: int = 10

    # ジャンプ検出（Local方式A用）
    delta_percentile: float = 94.0


# ================================================================
# 1D コア計算関数 (方式A用、numpy版、将来CUDA化候補)
# ================================================================


def _calculate_rho_t_1d(data: np.ndarray, window: int) -> np.ndarray:
    """1次元テンション密度 (ρT) — 過去窓の局所標準偏差"""
    n = len(data)
    rho_t = np.zeros(n)
    for i in range(n):
        start = max(0, i - window)
        end = i + 1
        subset = data[start:end]
        if len(subset) > 1:
            mean = np.mean(subset)
            variance = np.sum((subset - mean) ** 2) / len(subset)
            rho_t[i] = np.sqrt(variance)
    return rho_t


def _calculate_local_std_1d(data: np.ndarray, window: int) -> np.ndarray:
    """1次元の局所標準偏差（前後window の対称窓）— 無次元化の分母"""
    n = len(data)
    local_std = np.zeros(n)
    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        subset = data[start:end]
        mean = np.mean(subset)
        variance = np.sum((subset - mean) ** 2) / len(subset)
        local_std[i] = np.sqrt(variance)
    return local_std


def _compute_Q_lambda_1d(path: np.ndarray) -> np.ndarray:
    """1次元のトポロジカルチャージ (Q_Λ) — arctan2ベース"""
    n = len(path)
    if n < 3:
        return np.zeros(n)
    closed = np.concatenate([path, [path[0]]])
    theta = np.arctan2(closed[1:], closed[:-1])
    q = np.zeros(n)
    for i in range(n - 1):
        d = theta[i + 1] - theta[i]
        if d > np.pi:
            d -= 2 * np.pi
        elif d < -np.pi:
            d += 2 * np.pi
        q[i + 1] = d / (2 * np.pi)
    return q


# ================================================================
# アダプティブウィンドウ
# ================================================================


def compute_adaptive_window(
    state_vectors: np.ndarray,
    base_window: int = 30,
    min_window: int = 10,
) -> dict[str, int]:
    """データ特性に基づく動的ウィンドウサイズ"""
    n_frames, n_dims = state_vectors.shape
    max_window = max(100, min(n_frames // 10, 2000))

    if n_frames > 300:
        adjusted_base = base_window
    elif n_frames > 100:
        adjusted_base = int(base_window * 0.8)
    else:
        adjusted_base = int(base_window * 0.6)
    adjusted_base = max(adjusted_base, n_frames // 20)

    global_std = np.std(state_vectors)
    global_mean = np.mean(np.abs(state_vectors))
    volatility = global_std / (global_mean + 1e-10)

    temporal_changes = np.diff(state_vectors, axis=0)
    temporal_vol = np.mean(np.std(temporal_changes, axis=0))

    scale = 1.0
    if volatility > 2.0:
        scale *= 0.8
    elif volatility < 0.3:
        scale *= 1.5
    if temporal_vol > global_std * 2.0:
        scale *= 0.9
    elif temporal_vol < global_std * 0.3:
        scale *= 1.4

    local_w = int(np.clip(adjusted_base * scale, min_window, max_window))
    jump_w = int(np.clip(local_w * 0.5, max(min_window // 2, 3), max_window // 3))
    tension_w = int(np.clip(local_w * 1.5, min_window, max_window))

    return {
        "local": local_w,
        "jump": jump_w,
        "tension": tension_w,
        "volatility": float(volatility),
        "scale_factor": float(scale),
    }


# ================================================================
# DualCore 本体
# ================================================================


class LambdaStructuresDualCore:
    """
    Local/Global Dual-Path Lambda³ Structure Computation

    Local (方式A) — 新規:
      local_lambda_F  : (n_frames-1, n_dims)  ジャンプ検出済み。正負方向あり。
      local_rho_T     : (n_frames, n_dims)    各次元のρT
      local_Q_lambda  : (n_frames, n_dims)    各次元のQ_Λ
      local_std       : (n_frames, n_dims)    無次元化基準

    Global (方式B) — 元のLambdaStructuresCore継承:
      lambda_F, lambda_F_mag, lambda_FF, lambda_FF_mag
      rho_T, Q_lambda, Q_cumulative
      sigma_s, structural_coherence
    """

    def __init__(self, config: DualCoreConfig | None = None):
        self.config = config or DualCoreConfig()

        self.use_gpu = HAS_GPU and not self.config.force_cpu
        self.use_kernels = HAS_KERNELS and self.use_gpu
        self.xp = cp if self.use_gpu else np

        if self.config.verbose:
            if self.use_kernels:
                mode = "GPU + CUDA Kernels"
            elif self.use_gpu:
                mode = "GPU (CuPy)"
            else:
                mode = "CPU"
            logger.info(f"✅ LambdaStructuresDualCore initialized ({mode})")

    def compute(
        self,
        state_vectors: np.ndarray,
        window_steps: int | None = None,
        dimension_names: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Lambda³構造を計算（Local + Global）"""
        if state_vectors.ndim != 2:
            raise ValueError(
                f"state_vectors must be 2D, got shape {state_vectors.shape}"
            )

        n_frames, n_dims = state_vectors.shape

        # アダプティブウィンドウ
        if self.config.adaptive_window:
            adaptive = compute_adaptive_window(
                state_vectors,
                self.config.base_window,
                self.config.min_window,
            )
            tension_w = adaptive["tension"]
            jump_w = adaptive["jump"]
            local_w = adaptive["local"]
            if self.config.verbose:
                logger.info(
                    f"🔧 Adaptive: tension={tension_w}, "
                    f"jump={jump_w}, local={local_w}, "
                    f"vol={adaptive['volatility']:.3f}"
                )
        else:
            w = window_steps or self.config.base_window
            tension_w = w
            jump_w = max(w // 2, 3)
            local_w = w

        window_for_global = window_steps or local_w

        if self.config.verbose:
            dim_str = ", ".join(dimension_names) if dimension_names else f"{n_dims}D"
            logger.info(f"🚀 DualCore (frames={n_frames}, dims={dim_str})")

        # ── Local Lambda (方式A) ──
        local = self._compute_local(
            state_vectors,
            n_frames,
            n_dims,
            tension_w,
            jump_w,
            local_w,
        )

        # ── Global Lambda (方式B) ──
        glob = self._compute_global(
            state_vectors,
            n_frames,
            n_dims,
            window_for_global,
        )

        # 統合
        results = {}
        results.update(local)
        results.update(glob)
        results["_adaptive_windows"] = {
            "tension": tension_w,
            "jump": jump_w,
            "local": local_w,
        }

        if self.config.verbose:
            self._print_statistics(results, n_dims, dimension_names)

        return results

    # 後方互換エイリアス
    def compute_lambda_structures(
        self,
        state_vectors,
        window_steps=None,
        dimension_names=None,
    ):
        return self.compute(state_vectors, window_steps, dimension_names)

    # ================================================================
    # Local Lambda (方式A) — 各次元ごと独立
    # ================================================================

    def _compute_local(self, sv, n_frames, n_dims, tension_w, jump_w, local_w):
        n_diff = n_frames - 1

        local_lambda_F = np.zeros((n_diff, n_dims))
        local_rho_T = np.zeros((n_frames, n_dims))
        local_Q_lambda = np.zeros((n_frames, n_dims))
        local_std = np.zeros((n_frames, n_dims))

        for d in range(n_dims):
            series = sv[:, d]

            # diff → 無次元化 → ジャンプ検出 → マスク → ΛF
            diff = np.diff(series)
            lstd = _calculate_local_std_1d(series, local_w)
            lstd_diff = lstd[1:]
            score = np.abs(diff) / (lstd_diff + 1e-10)
            threshold = np.percentile(score, self.config.delta_percentile)
            jump_mask = score > threshold

            local_lambda_F[:, d] = diff * jump_mask

            # ρT
            local_rho_T[:, d] = _calculate_rho_t_1d(series, tension_w)

            # Q_Λ
            local_Q_lambda[:, d] = _compute_Q_lambda_1d(series)

            # local_std
            local_std[:, d] = lstd

        return {
            "local_lambda_F": local_lambda_F,
            "local_rho_T": local_rho_T,
            "local_Q_lambda": local_Q_lambda,
            "local_std": local_std,
        }

    # ================================================================
    # Global Lambda (方式B) — 元の LambdaStructuresCore そのまま
    # ================================================================

    def _compute_global(self, state_vectors, n_frames, n_dims, window_steps):
        xp = self.xp

        if self.use_gpu:
            sv_gpu = cp.asarray(state_vectors, dtype=cp.float32)
        else:
            sv_gpu = state_vectors

        # ΛF
        lambda_F = xp.diff(sv_gpu, axis=0)
        lambda_F_mag = xp.linalg.norm(lambda_F, axis=1)

        # ΛFF
        lambda_FF = xp.diff(lambda_F, axis=0)
        lambda_FF_mag = xp.linalg.norm(lambda_FF, axis=1)

        # ρT — Global（trace(cov)）
        can_use_kernels = self.use_kernels and n_dims == 3
        if can_use_kernels:
            rho_T = self._rho_T_kernel(state_vectors, window_steps)
        elif self.use_gpu:
            rho_T = self._rho_T_gpu(sv_gpu, window_steps)
        else:
            rho_T = self._rho_T_cpu(state_vectors, window_steps)

        # Q_Λ — N次元ベクトル間角度
        Q_lambda, Q_cumulative = self._Q_lambda_global(
            lambda_F,
            lambda_F_mag,
            xp,
        )

        # σₛ — 構造同期率
        sigma_s = self._sigma_s(sv_gpu, lambda_F, window_steps, xp)

        # コヒーレンス
        coherence = self._coherence(lambda_F, window_steps, xp)

        return {
            "lambda_F": self._np(lambda_F),
            "lambda_F_mag": self._np(lambda_F_mag),
            "lambda_FF": self._np(lambda_FF),
            "lambda_FF_mag": self._np(lambda_FF_mag),
            "rho_T": self._np(rho_T),
            "Q_lambda": self._np(Q_lambda),
            "Q_cumulative": self._np(Q_cumulative),
            "sigma_s": self._np(sigma_s),
            "structural_coherence": self._np(coherence),
        }

    # ================================================================
    # Global 内部メソッド (元の LambdaStructuresCore から継承)
    # ================================================================

    def _Q_lambda_global(self, lambda_F, lambda_F_mag, xp):
        n_steps = len(lambda_F_mag)
        lf = self._np(lambda_F)
        lfm = self._np(lambda_F_mag)
        q = np.zeros(n_steps)
        for step in range(1, n_steps):
            if lfm[step] > 1e-10 and lfm[step - 1] > 1e-10:
                v1 = lf[step - 1] / lfm[step - 1]
                v2 = lf[step] / lfm[step]
                cos_a = np.clip(np.dot(v1, v2), -1, 1)
                angle = np.arccos(cos_a)
                if len(v1) >= 2:
                    cross_z = v1[0] * v2[1] - v1[1] * v2[0]
                    angle = angle if cross_z >= 0 else -angle
                q[step] = angle / (2 * np.pi)
        q_cum = np.cumsum(q)
        if self.use_gpu:
            return cp.asarray(q), cp.asarray(q_cum)
        return q, q_cum

    def _sigma_s(self, state_vectors, lambda_F, window_steps, xp):
        lf = self._np(lambda_F)
        n_frames = len(lf)
        n_dims = lf.shape[1]
        sigma_s = np.zeros(n_frames + 1)
        if n_dims < 2:
            return sigma_s[: self._np(state_vectors).shape[0]]
        for t in range(n_frames):
            start = max(0, t - window_steps)
            end = min(n_frames, t + window_steps + 1)
            w = lf[start:end]
            if len(w) < 3:
                continue
            stds = w.std(axis=0)
            active = np.where(stds > 1e-10)[0]
            if len(active) < 2:
                continue
            try:
                corr = np.corrcoef(w[:, active].T)
                if np.any(np.isnan(corr)):
                    continue
                mask = ~np.eye(len(active), dtype=bool)
                sigma_s[t] = np.abs(corr[mask]).mean()
            except Exception:
                pass
        if n_frames > 0:
            sigma_s[n_frames] = sigma_s[n_frames - 1]
        return sigma_s[: self._np(state_vectors).shape[0]]

    def _coherence(self, lambda_F, window, xp):
        lf = self._np(lambda_F)
        n = len(lf)
        coh = np.zeros(n)
        for i in range(window, n - window):
            local_F = lf[i - window : i + window]
            mean_dir = np.mean(local_F, axis=0)
            mn = np.linalg.norm(mean_dir)
            if mn > 1e-10:
                mean_dir /= mn
                norms = np.linalg.norm(local_F, axis=1)
                valid = norms > 1e-10
                if np.any(valid):
                    normed = local_F[valid] / norms[valid, np.newaxis]
                    coh[i] = np.mean(np.dot(normed, mean_dir))
        return coh

    def _rho_T_kernel(self, positions, window_steps):
        logger.debug("   ρT Global: CUDA kernel (3D)")
        return cp.asnumpy(tension_field_kernel(positions, window_steps))

    def _rho_T_gpu(self, pos_gpu, window_steps):
        logger.debug("   ρT Global: CuPy (N-dim)")
        n = pos_gpu.shape[0]
        rho = cp.zeros(n, dtype=cp.float32)
        for i in range(n):
            s = max(0, i - window_steps)
            e = min(n, i + window_steps + 1)
            loc = pos_gpu[s:e]
            if len(loc) > 1:
                c = loc - cp.mean(loc, axis=0, keepdims=True)
                rho[i] = cp.mean(c**2, axis=0).sum() * len(loc) / (len(loc) - 1)
        return cp.asnumpy(rho)

    def _rho_T_cpu(self, positions, window_steps):
        n = len(positions)
        rho = np.zeros(n)
        for i in range(n):
            s = max(0, i - window_steps)
            e = min(n, i + window_steps + 1)
            loc = positions[s:e]
            if len(loc) > 1:
                c = loc - np.mean(loc, axis=0, keepdims=True)
                cov = np.cov(c.T)
                rho[i] = float(cov) if cov.ndim == 0 else np.trace(cov)
        return rho

    def _np(self, arr):
        if self.use_gpu and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    # ================================================================
    # 統計表示
    # ================================================================

    def _print_statistics(self, results, n_dims, dim_names):
        if not self.config.verbose:
            return

        logger.info("📊 DualCore Statistics:")

        # Local
        lr = results["local_rho_T"]
        logger.info(
            f"   [Local]  ρT: shape={lr.shape}, "
            f"min={np.min(lr):.3e}, max={np.max(lr):.3e}"
        )
        lf = results["local_lambda_F"]
        n_nz = np.count_nonzero(lf)
        logger.info(
            f"   [Local]  ΛF: {n_nz}/{lf.size} nonzero "
            f"({100 * n_nz / lf.size:.1f}% = detected jumps)"
        )

        # Global
        for key in ["rho_T", "sigma_s"]:
            d = results[key]
            logger.info(
                f"   [Global] {key}: min={np.min(d):.3e}, "
                f"max={np.max(d):.3e}, mean={np.mean(d):.3e}"
            )
