"""
GETTER One
==========

Geometric Event-driven Tensor-based
Time-series Extraction & Recognition

Omnidimensional Network Engine

Discrete geometric structural change detection
and causal network extraction for N-dimensional time series.

Usage:
    from getter_one.data import load
    from getter_one.structures import LambdaStructuresCore
    from getter_one.analysis import NetworkAnalyzerCore, assess_confidence

CLI:
    $ getter-one-loader load data.csv --target y -o prepared.csv
    $ getter-one-loader merge a.csv b.json --time date -o merged.csv

Author: Masamichi Iizumi (Miosync, Inc.)
License: MIT
"""

from __future__ import annotations

import logging
import os
from typing import Any

# ===============================
# Version
# ===============================

__version__ = "0.1.5"
__author__ = "Masamichi Iizumi"

# ===============================
# Logging
# ===============================

logger = logging.getLogger("getter_one")
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ===============================
# GPU Environment Detection
# ===============================


class _GPUEnvironment:
    """GPU環境の検出と情報管理（内部クラス）"""

    __slots__ = (
        "has_cupy",
        "gpu_available",
        "gpu_name",
        "gpu_memory",
        "cuda_version",
        "compute_capability",
    )

    def __init__(self):
        self.has_cupy: bool = False
        self.gpu_available: bool = False
        self.gpu_name: str = "Not Available"
        self.gpu_memory: float = 0.0
        self.cuda_version: str = "Not Available"
        self.compute_capability: str = "7.5"
        self._detect()

    def _detect(self):
        try:
            import cupy as cp

            self.has_cupy = True

            if cp.cuda.runtime.getDeviceCount() > 0:
                self.gpu_available = True
                self.gpu_name = self._device_name(cp)
                self.gpu_memory = self._device_memory(cp)
                self.cuda_version = self._cuda_ver(cp)
                self.compute_capability = self._compute_cap(cp)
                logger.info(
                    f"GPU detected: {self.gpu_name} "
                    f"({self.gpu_memory:.1f} GB, CUDA {self.cuda_version})"
                )
            else:
                logger.info("No GPU devices found - running in CPU mode")
        except ImportError:
            logger.info("CuPy not installed - running in CPU mode")
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")

    @staticmethod
    def _device_name(cp) -> str:
        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"]
            return name.decode("utf-8") if isinstance(name, bytes) else name
        except Exception:
            return "Unknown GPU"

    @staticmethod
    def _device_memory(cp) -> float:
        try:
            return cp.cuda.runtime.memGetInfo()[1] / (1024**3)
        except Exception:
            return 0.0

    @staticmethod
    def _compute_cap(cp) -> str:
        try:
            dev = cp.cuda.runtime.getDevice()
            major = cp.cuda.runtime.deviceGetAttribute(75, dev)
            minor = cp.cuda.runtime.deviceGetAttribute(76, dev)
            return f"{major}.{minor}"
        except Exception:
            return "7.5"

    @staticmethod
    def _cuda_ver(cp) -> str:
        try:
            v = cp.cuda.runtime.runtimeGetVersion()
            return f"{v // 1000}.{(v % 1000) // 10}"
        except Exception:
            return "Unknown"

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.gpu_available,
            "name": self.gpu_name,
            "memory_gb": self.gpu_memory,
            "cuda_version": self.cuda_version,
            "compute_capability": self.compute_capability,
            "has_cupy": self.has_cupy,
        }


# シングルトン初期化
_gpu_env = _GPUEnvironment()

# グローバル変数エクスポート
HAS_CUPY: bool = _gpu_env.has_cupy
GPU_AVAILABLE: bool = _gpu_env.gpu_available
GPU_NAME: str = _gpu_env.gpu_name
GPU_MEMORY: float = _gpu_env.gpu_memory
GPU_COMPUTE_CAPABILITY: str = _gpu_env.compute_capability
CUDA_VERSION_STR: str = _gpu_env.cuda_version

# ===============================
# Utility Functions
# ===============================


def get_gpu_info() -> dict[str, Any]:
    """GPU環境情報を辞書で返す"""
    return _gpu_env.as_dict()


def set_gpu_device(device_id: int) -> None:
    """使用するGPUデバイスを切り替え"""
    if GPU_AVAILABLE:
        import cupy as cp

        cp.cuda.Device(device_id).use()
        logger.info(f"GPU device set to: {device_id}")
    else:
        logger.warning("No GPU available")


def set_log_level(level: str = "INFO") -> None:
    """GETTER One パッケージのログレベルを設定"""
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {level}")
    logger.setLevel(numeric)


# ===============================
# GPU Memory Limit (env var)
# ===============================

if GPU_AVAILABLE and "GETTER_ONE_GPU_MEMORY_LIMIT" in os.environ:
    try:
        _limit_gb = float(os.environ["GETTER_ONE_GPU_MEMORY_LIMIT"])
        import cupy as _cp

        _cp.cuda.MemoryPool().set_limit(size=int(_limit_gb * 1024**3))
        logger.info(f"GPU memory limit: {_limit_gb} GB")
        del _cp, _limit_gb
    except (ValueError, Exception) as e:
        logger.warning(f"Failed to set GPU memory limit: {e}")

# Debug mode
if os.environ.get("GETTER_ONE_DEBUG", "").lower() in ("1", "true", "yes"):
    set_log_level("DEBUG")

# ===============================
# Public API
# ===============================

__all__ = [
    # Version
    "__version__",
    "__author__",
    # GPU info
    "GPU_AVAILABLE",
    "GPU_NAME",
    "GPU_MEMORY",
    "GPU_COMPUTE_CAPABILITY",
    "CUDA_VERSION_STR",
    "HAS_CUPY",
    # Utility
    "get_gpu_info",
    "set_gpu_device",
    "set_log_level",
]

# Lazy-importable names (via __getattr__):
#   Data:       load, merge, from_dataframe, from_numpy, GetterDataset
#   Structures: LambdaStructuresCore, LambdaCoreConfig
#   Analysis:   NetworkAnalyzerCore, NetworkResult, DimensionLink,
#               CooperativeEventNetwork, assess_confidence, ConfidenceReport,
#               EventConfidence, BoundaryConfidence, CausalLinkConfidence,
#               SyncConfidence

# ===============================
# Lazy Imports
# ===============================


def __getattr__(name: str):
    """遅延インポートで起動時間を最小化"""

    # --- Data ---
    _data_names = {"load", "merge", "from_dataframe", "from_numpy", "GetterDataset"}
    if name in _data_names:
        from getter_one.data import loader as _loader

        return getattr(_loader, name)

    # --- Structures ---
    if name == "LambdaStructuresCore":
        from getter_one.structures.lambda_structures_core import LambdaStructuresCore

        return LambdaStructuresCore

    if name == "LambdaCoreConfig":
        from getter_one.structures.lambda_structures_core import LambdaCoreConfig

        return LambdaCoreConfig

    # --- Analysis: Network ---
    _network_names = {
        "NetworkAnalyzerCore",
        "NetworkResult",
        "DimensionLink",
        "CooperativeEventNetwork",
    }
    if name in _network_names:
        from getter_one.analysis import network_analyzer_core as _net

        return getattr(_net, name)

    # --- Analysis: Confidence ---
    _confidence_names = {
        "assess_confidence",
        "ConfidenceReport",
        "EventConfidence",
        "BoundaryConfidence",
        "CausalLinkConfidence",
        "SyncConfidence",
    }
    if name in _confidence_names:
        from getter_one.analysis import confidence_kit as _conf

        return getattr(_conf, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
