"""
BANKAI - Bond-vector ANalysis of Kinetic Amino acid Initiator
==============================================================

GPU-accelerated sub-picosecond causal cascade detection
in GROMACS molecular dynamics trajectories.

Usage:
    import bankai
    from bankai import MDLambda3DetectorGPU, TwoStageAnalyzerGPU

CLI:
    $ bankai-run --help
    $ python -m bankai --help

Author: Masamichi Iizumi (Miosync, Inc.)
License: MIT
"""

import logging
import os
from typing import Any

# ===============================
# Version
# ===============================

__version__ = "1.1.6"
__author__ = "Masamichi Iizumi"

# ===============================
# Logging
# ===============================

logger = logging.getLogger("bankai")
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
                    f"GPU detected: {self.gpu_name} ({self.gpu_memory:.1f} GB, CUDA {self.cuda_version})"
                )
            else:
                logger.warning("No GPU devices found")
        except ImportError:
            logger.warning("CuPy not installed - GPU features disabled")
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
    """bankai パッケージのログレベルを設定"""
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise ValueError(f"Invalid log level: {level}")
    logger.setLevel(numeric)


# ===============================
# GPU Memory Limit (env var)
# ===============================

if GPU_AVAILABLE and "BANKAI_GPU_MEMORY_LIMIT" in os.environ:
    try:
        _limit_gb = float(os.environ["BANKAI_GPU_MEMORY_LIMIT"])
        import cupy as _cp

        _cp.cuda.MemoryPool().set_limit(size=int(_limit_gb * 1024**3))
        logger.info(f"GPU memory limit: {_limit_gb} GB")
        del _cp, _limit_gb
    except (ValueError, Exception) as e:
        logger.warning(f"Failed to set GPU memory limit: {e}")

# Debug mode
if os.environ.get("BANKAI_DEBUG", "").lower() in ("1", "true", "yes"):
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
    # Core classes (lazy)
    "MDConfig",
    "ResidueAnalysisConfig",
    "MDLambda3DetectorGPU",
    "TwoStageAnalyzerGPU",
    # Functions (lazy)
    "perform_two_stage_analysis_gpu",
    # Result types (lazy)
    "MDLambda3Result",
    "TwoStageLambda3Result",
    "ResidueLevelAnalysis",
    "ResidueEvent",
    # Visualization (lazy)
    "Lambda3VisualizerGPU",
    "CausalityVisualizerGPU",
    # Errors (lazy)
    "BankaiError",
    "GPUMemoryError",
    "GPUNotAvailableError",
]

# ===============================
# Lazy Imports
# ===============================


def __getattr__(name: str):
    """遅延インポートで起動時間を最小化"""

    # --- Config ---
    if name == "MDConfig":
        from bankai.analysis.md_lambda3_detector_gpu import MDConfig

        return MDConfig

    if name == "ResidueAnalysisConfig":
        from bankai.analysis.two_stage_analyzer_gpu import ResidueAnalysisConfig

        return ResidueAnalysisConfig

    # --- Core classes ---
    if name == "MDLambda3DetectorGPU":
        from bankai.analysis.md_lambda3_detector_gpu import MDLambda3DetectorGPU

        return MDLambda3DetectorGPU

    if name == "TwoStageAnalyzerGPU":
        from bankai.analysis.two_stage_analyzer_gpu import TwoStageAnalyzerGPU

        return TwoStageAnalyzerGPU

    # --- Functions ---
    if name == "perform_two_stage_analysis_gpu":
        from bankai.analysis.two_stage_analyzer_gpu import (
            perform_two_stage_analysis_gpu,
        )

        return perform_two_stage_analysis_gpu

    # --- Result types ---
    _result_types = {
        "MDLambda3Result",
        "TwoStageLambda3Result",
        "ResidueLevelAnalysis",
        "ResidueEvent",
    }
    if name in _result_types:
        from bankai import models as _models

        return getattr(_models, name)

    # --- Visualization ---
    if name == "Lambda3VisualizerGPU":
        from bankai.visualization import Lambda3VisualizerGPU

        return Lambda3VisualizerGPU

    if name == "CausalityVisualizerGPU":
        from bankai.visualization import CausalityVisualizerGPU

        return CausalityVisualizerGPU

    # --- Errors ---
    _error_types = {"BankaiError", "GPUMemoryError", "GPUNotAvailableError"}
    if name in _error_types:
        from bankai import errors as _errors

        return getattr(_errors, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
