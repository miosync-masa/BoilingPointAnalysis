"""
CuPy find_peaks互換性パッチ（改良版）
引数の形式を自動調整
"""
from __future__ import annotations

import numpy as np

try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None


def find_peaks_wrapper(array, height=None, distance=None, prominence=None, **kwargs):
    """
    find_peaksのラッパー（引数を自動調整）
    """
    try:
        # オリジナルのfind_peaksを試す
        from cupyx.scipy.signal import find_peaks as _find_peaks_original

        return _find_peaks_original(
            array, height=height, distance=distance, prominence=prominence, **kwargs
        )
    except ValueError as e:
        if "array size" in str(e):
            # heightがスカラーの場合、配列に変換
            if height is not None and np.isscalar(height):
                # 全要素に同じ閾値を適用
                height = cp.full_like(array, height)

            # 再度試す
            try:
                return _find_peaks_original(
                    array,
                    height=height,
                    distance=distance,
                    prominence=prominence,
                    **kwargs,
                )
            except Exception:
                # それでもダメならシンプル実装
                return find_peaks_simple_gpu(array, threshold=height)
        else:
            raise


def find_peaks_simple_gpu(array, threshold=None, min_distance=1):
    """
    シンプルなピーク検出実装（GPU版）
    """
    if not isinstance(array, cp.ndarray):
        array = cp.asarray(array)

    n = len(array)
    peaks = []

    # 閾値処理
    if threshold is not None:
        if isinstance(threshold, cp.ndarray):
            threshold_val = cp.mean(threshold)
        else:
            threshold_val = threshold
    else:
        threshold_val = cp.mean(array)

    # ピーク検出
    for i in range(1, n - 1):
        if array[i] > array[i - 1] and array[i] > array[i + 1]:
            if array[i] >= threshold_val:
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)

    return cp.array(peaks, dtype=cp.int64), {}


# モンキーパッチ関数
def apply_find_peaks_patch():
    """CuPyのfind_peaksにパッチを適用"""
    try:
        import cupyx.scipy.signal as signal

        # オリジナルを保存
        if hasattr(signal, "find_peaks"):
            signal._find_peaks_original = signal.find_peaks

        # ラッパーで置き換え
        signal.find_peaks = find_peaks_wrapper
        print("✅ Applied find_peaks compatibility patch")

    except ImportError:
        print("⚠️ cupyx.scipy.signal not available")


# 環境変数チェック付きで自動適用
import os

if os.environ.get("LAMBDA3_GPU_PATCH_FIND_PEAKS", "1") == "1":
    if HAS_GPU:
        apply_find_peaks_patch()
