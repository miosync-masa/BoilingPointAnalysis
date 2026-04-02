"""
GPU Utilities and Base Classes - Refactored Version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPUとCPUを賢く切り替える基底クラスと便利なユーティリティ集だよ〜！💕
環ちゃんがリファクタリングして、メモリマネージャのバグも修正したよ！

by 環ちゃん
"""

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Union

import numpy as np

# ===============================
# GPU Availability Check
# ===============================

try:
    import cupy as cp
    from cupy import cuda

    HAS_GPU = True
    GPU_AVAILABLE = cp.cuda.is_available()
except ImportError:
    HAS_GPU = False
    GPU_AVAILABLE = False
    cp = None
    cuda = None

# ===============================
# Type Definitions
# ===============================

if TYPE_CHECKING:
    from numpy.typing import NDArray as NPArray

    if HAS_GPU:
        from cupy.typing import NDArray as CPArray

        NDArray = Union[NPArray, CPArray]
    else:
        NDArray = NPArray
else:
    NDArray = Union[np.ndarray, "cp.ndarray"] if HAS_GPU else np.ndarray

ArrayType = NDArray  # エイリアス

# ===============================
# Logger Setup
# ===============================

logger = logging.getLogger("bankai.core.utils")

# ===============================
# GPU Backend Base Class
# ===============================


class GPUBackend:
    """
    GPU/CPU自動切り替えの基底クラス
    全てのGPU対応クラスはこれを継承するよ〜！

    メモリマネージャーも自動で初期化されるから安心！💕
    """

    def __init__(
        self,
        device: str | int = "auto",
        force_cpu: bool = False,
        mixed_precision: bool = False,
        profile: bool = False,
        memory_manager_config: dict[str, Any] | None = None,
    ):
        """
        Parameters
        ----------
        device : str or int, default='auto'
            'auto': 自動選択
            'cpu': CPU強制
            'gpu': GPU強制
            0,1,2...: 特定のGPU番号
        force_cpu : bool, default=False
            True時はGPUがあってもCPU使用
        mixed_precision : bool, default=False
            FP16使用で高速化（精度は落ちる）
        profile : bool, default=False
            プロファイリングモード
        memory_manager_config : dict, optional
            メモリマネージャーの設定
        """
        # 基本設定
        self.force_cpu = force_cpu
        self.mixed_precision = mixed_precision
        self.profile = profile
        self._timers = {}

        # デバイス選択
        if force_cpu or not GPU_AVAILABLE:
            self.device = "cpu"
            self.device_id = -1
            self.xp = np
            self.is_gpu = False
            logger.info("Using CPU backend")
        else:
            self.device_id = self._select_device(device)
            self.device = f"gpu:{self.device_id}"
            self.xp = cp
            self.is_gpu = True

            # GPUデバイスをセット
            cp.cuda.Device(self.device_id).use()
            logger.info(f"Using GPU backend: Device {self.device_id}")

            # Mixed precision設定
            if mixed_precision:
                self._setup_mixed_precision()

        # メモリマネージャーの初期化（重要！）
        self._initialize_memory_manager(memory_manager_config)

        # デバイス情報をキャッシュ
        self._cache_device_info()

    def _initialize_memory_manager(self, config: dict[str, Any] | None = None):
        """メモリマネージャーを初期化"""
        # インポートは実行時に（循環参照回避）
        from .gpu_memory import GPUMemoryManager

        # デフォルト設定
        default_config = {
            "max_memory_gb": None,  # 自動
            "reserve_percent": 10.0,
            "enable_pooling": True,
        }

        if config:
            default_config.update(config)

        # メモリマネージャー作成
        self.memory_manager = GPUMemoryManager(**default_config)
        logger.debug(
            f"Memory manager initialized: {self.memory_manager.get_memory_info()}"
        )

    def _select_device(self, device: str | int) -> int:
        """デバイス選択ロジック"""
        if device == "auto":
            # 自動選択（メモリが一番空いてるGPU）
            return auto_select_device()
        elif device == "cpu":
            return -1
        elif device == "gpu":
            return 0  # デフォルトGPU
        elif isinstance(device, int):
            # 指定されたGPUが存在するかチェック
            n_devices = cp.cuda.runtime.getDeviceCount()
            if device >= n_devices:
                logger.warning(f"GPU {device} not found. Using GPU 0.")
                return 0
            return device
        else:
            raise ValueError(f"Invalid device: {device}")

    def _setup_mixed_precision(self):
        """Mixed precision設定"""
        if self.is_gpu and HAS_GPU:
            try:
                # TensorCoreを使う設定
                cp.cuda.cublas.setMathMode(cp.cuda.cublas.CUBLAS_TENSOR_OP_MATH)
                logger.info("Mixed precision (FP16) enabled")
            except Exception as e:
                logger.warning(f"Failed to enable mixed precision: {e}")

    def _cache_device_info(self):
        """デバイス情報をキャッシュ"""
        self.device_info = {}

        if self.is_gpu and HAS_GPU:
            try:
                props = cp.cuda.runtime.getDeviceProperties(self.device_id)
                self.device_info = {
                    "name": props["name"].decode(),
                    "compute_capability": f"{props['major']}.{props['minor']}",
                    "total_memory": props["totalGlobalMem"],
                    "multiprocessor_count": props["multiProcessorCount"],
                }
            except Exception as e:
                logger.warning(f"Failed to get device properties: {e}")

    # ===============================
    # Array Operations
    # ===============================

    def to_gpu(self, array: NDArray, dtype: np.dtype | None = None) -> NDArray:
        """配列をGPUに転送（またはそのまま）"""
        if self.is_gpu and HAS_GPU:
            if isinstance(array, cp.ndarray):
                return array.astype(dtype) if dtype else array
            return cp.asarray(array, dtype=dtype)
        else:
            return np.asarray(array, dtype=dtype) if dtype else np.asarray(array)

    def to_cpu(self, array: NDArray) -> np.ndarray:
        """配列をCPUに転送（またはそのまま）"""
        if self.is_gpu and HAS_GPU and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return np.asarray(array)

    def zeros(self, shape: tuple[int, ...], dtype: np.dtype = np.float32) -> NDArray:
        """ゼロ配列を作成"""
        return self.xp.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: np.dtype = np.float32) -> NDArray:
        """1配列を作成"""
        return self.xp.ones(shape, dtype=dtype)

    def empty(self, shape: tuple[int, ...], dtype: np.dtype = np.float32) -> NDArray:
        """空配列を作成（高速だけど初期化されない）"""
        return self.xp.empty(shape, dtype=dtype)

    def arange(self, *args, **kwargs) -> NDArray:
        """連番配列を作成"""
        return self.xp.arange(*args, **kwargs)

    # ===============================
    # Memory Management
    # ===============================

    def clear_memory(self):
        """メモリをクリア"""
        if hasattr(self, "memory_manager"):
            self.memory_manager.clear_cache()

        # 追加のクリーンアップ
        if self.is_gpu and HAS_GPU:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    @contextmanager
    def batch_context(self, estimated_memory: int):
        """バッチ処理用のメモリコンテキスト"""
        if hasattr(self, "memory_manager"):
            with self.memory_manager.temporary_allocation(estimated_memory, "batch"):
                yield
        else:
            yield

    # ===============================
    # Profiling
    # ===============================

    @contextmanager
    def timer(self, name: str):
        """タイマーコンテキスト"""
        if self.profile:
            start = time.perf_counter()

            # GPU同期（正確な計測のため）
            if self.is_gpu and HAS_GPU:
                cp.cuda.Stream.null.synchronize()

            yield

            # GPU同期
            if self.is_gpu and HAS_GPU:
                cp.cuda.Stream.null.synchronize()

            elapsed = time.perf_counter() - start

            if name not in self._timers:
                self._timers[name] = []
            self._timers[name].append(elapsed)

            logger.debug(f"{name}: {elapsed:.3f}s")
        else:
            yield

    def get_profile_summary(self) -> dict[str, dict[str, float]]:
        """プロファイル結果のサマリー"""
        summary = {}
        for name, times in self._timers.items():
            summary[name] = {
                "total": sum(times),
                "mean": np.mean(times),
                "std": np.std(times),
                "min": min(times),
                "max": max(times),
                "count": len(times),
            }
        return summary

    # ===============================
    # Device Info
    # ===============================

    def get_device_info(self) -> dict[str, Any]:
        """デバイス情報を取得"""
        info = {
            "backend": "GPU" if self.is_gpu else "CPU",
            "device": self.device,
            "mixed_precision": self.mixed_precision,
        }

        if self.device_info:
            info.update(self.device_info)

        # メモリ情報も追加
        if hasattr(self, "memory_manager"):
            mem_info = self.memory_manager.get_memory_info()
            info["memory"] = {
                "total_gb": mem_info.total_gb,
                "used_gb": mem_info.used_gb,
                "free_gb": mem_info.free_gb,
            }

        return info


# ===============================
# Utility Functions
# ===============================


def auto_select_device() -> int:
    """
    自動的に最適なGPUを選択
    メモリが一番空いてるやつを選ぶよ〜！
    """
    if not GPU_AVAILABLE:
        return 0

    n_devices = cp.cuda.runtime.getDeviceCount()
    if n_devices == 0:
        return 0

    best_device = 0
    max_free_memory = 0

    for i in range(n_devices):
        with cp.cuda.Device(i):
            free_mem, total_mem = cp.cuda.runtime.memGetInfo()
            if free_mem > max_free_memory:
                max_free_memory = free_mem
                best_device = i

    logger.info(
        f"Auto-selected GPU {best_device} with {max_free_memory / 1e9:.1f}GB free"
    )
    return best_device


def profile_gpu(func: Callable) -> Callable:
    """
    GPU関数のプロファイリングデコレータ
    GPUBackendのtimerメソッドを利用
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # self.timerを使う（GPUBackendのメソッド）
        if hasattr(self, "timer"):
            with self.timer(func.__name__):
                return func(self, *args, **kwargs)
        else:
            # timerがない場合は通常実行
            return func(self, *args, **kwargs)

    return wrapper


def handle_gpu_errors(func: Callable) -> Callable:
    """
    GPU関連のエラーをハンドリングするデコレータ
    メモリ不足時は自動でリトライ！
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            error_str = str(e)

            # GPUメモリ不足の場合
            if "out of memory" in error_str.lower():
                logger.warning(f"GPU out of memory in {func.__name__}: {e}")

                # メモリクリア試行
                if hasattr(self, "clear_memory"):
                    self.clear_memory()
                    logger.info("Cleared GPU memory, retrying...")

                    # リトライは1回のみ
                    try:
                        return func(self, *args, **kwargs)
                    except Exception as retry_error:
                        logger.error(f"Retry failed: {retry_error}")

                        # バッチサイズ調整の提案
                        if hasattr(self, "config") and hasattr(
                            self.config, "gpu_batch_size"
                        ):
                            suggested_size = self.config.gpu_batch_size // 2
                            logger.error(
                                f"Consider reducing batch size. "
                                f"Current: {self.config.gpu_batch_size}, "
                                f"Suggested: {suggested_size}"
                            )
                        raise

            # CuPy/CUDA特有のエラー
            elif any(
                keyword in error_str.lower() for keyword in ["cuda", "cupy", "gpu"]
            ):
                logger.error(f"GPU error in {func.__name__}: {e}")

                # デバイス情報を出力
                if hasattr(self, "get_device_info"):
                    logger.error(f"Device info: {self.get_device_info()}")

            raise

    return wrapper


# ===============================
# Global Utility Functions
# ===============================


def get_gpu_info() -> dict[str, Any]:
    """システムのGPU情報を取得"""
    info = {"gpu_available": GPU_AVAILABLE, "has_cupy": HAS_GPU, "devices": []}

    if GPU_AVAILABLE:
        n_devices = cp.cuda.runtime.getDeviceCount()
        info["device_count"] = n_devices

        for i in range(n_devices):
            try:
                with cp.cuda.Device(i):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    free_mem, total_mem = cp.cuda.runtime.memGetInfo()

                    device_info = {
                        "id": i,
                        "name": props["name"].decode(),
                        "compute_capability": f"{props['major']}.{props['minor']}",
                        "total_memory_gb": total_mem / 1e9,
                        "free_memory_gb": free_mem / 1e9,
                        "multiprocessor_count": props["multiProcessorCount"],
                    }
                    info["devices"].append(device_info)
            except Exception as e:
                logger.warning(f"Failed to get info for GPU {i}: {e}")

    return info


def set_gpu_device(device_id: int):
    """使用するGPUデバイスを設定"""
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")

    n_devices = cp.cuda.runtime.getDeviceCount()
    if device_id >= n_devices:
        raise ValueError(f"GPU {device_id} not found. Available: 0-{n_devices - 1}")

    cp.cuda.Device(device_id).use()
    logger.info(f"Set active GPU to device {device_id}")


def enable_gpu_logging(level: str = "INFO"):
    """GPU関連のログレベルを設定"""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Lambda3 GPUのロガー
    logging.getLogger("bankai").setLevel(log_level)

    # CuPyのロガー（存在する場合）
    if HAS_GPU:
        logging.getLogger("cupy").setLevel(log_level)


def benchmark_gpu() -> dict[str, float]:
    """簡単なGPUベンチマーク"""
    if not GPU_AVAILABLE:
        return {"error": "GPU not available"}

    results = {}
    sizes = [1000, 5000, 10000]

    for size in sizes:
        # 行列積のベンチマーク
        a = cp.random.randn(size, size, dtype=np.float32)
        b = cp.random.randn(size, size, dtype=np.float32)

        # ウォームアップ
        _ = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()

        # 計測
        start = time.perf_counter()
        for _ in range(5):
            c = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()
        elapsed = (time.perf_counter() - start) / 5

        results[f"matmul_{size}x{size}"] = elapsed

        # メモリクリア
        del a, b, c
        cp.get_default_memory_pool().free_all_blocks()

    return results


# ===============================
# Initialize on Import
# ===============================

# ログレベル設定（環境変数から）
import os

log_level = os.environ.get("LAMBDA3_LOG_LEVEL", "INFO")
enable_gpu_logging(log_level)

# GPU情報を表示（デバッグモード時）
if logger.isEnabledFor(logging.DEBUG):
    gpu_info = get_gpu_info()
    logger.debug(f"GPU Info: {gpu_info}")
