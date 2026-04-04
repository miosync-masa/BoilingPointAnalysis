"""
Getter_one GPU Core Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Components:
    - GPUBackend: GPU/CPU自動切り替えの基底クラス
    - GPUMemoryManager: メモリ管理システム
    - CUDAKernels: 高速カスタムカーネル集
    - InverseKernels: 逆問題検証カーネル（GPU ONLY 🔥）
    - Decorators: プロファイリングとエラーハンドリング
"""

# gpu_utils.py から
# gpu_kernels.py から
from .gpu_kernels import (
    CUDAKernels,
    anomaly_detection_kernel,
    benchmark_kernels,
    compute_gradient_kernel,
    compute_local_fractal_dimension_kernel,
    create_elementwise_kernel,
    distance_matrix_kernel,
    get_kernel_manager,
    residue_com_kernel,
    tension_field_kernel,
    topological_charge_kernel,
)

# gpu_memory.py から
from .gpu_memory import (
    BatchProcessor,
    GPUMemoryManager,
    GPUMemoryPool,
    MemoryError,
    MemoryInfo,
    clear_gpu_cache,
    estimate_memory_usage,
    get_memory_summary,
)
from .gpu_utils import GPUBackend, auto_select_device, handle_gpu_errors, profile_gpu

# GPU ONLY: gpu_inverse は CuPy が無い環境ではスキップ
try:
    from .gpu_inverse import (
        InverseKernels,
        compute_gram_matrix,
        compute_path_statistics,
        inverse_verify_all,
    )

    _HAS_INVERSE_KERNELS = True
except ImportError:
    _HAS_INVERSE_KERNELS = False

__all__ = [
    # Utils
    "GPUBackend",
    "auto_select_device",
    "profile_gpu",
    "handle_gpu_errors",
    # Memory
    "MemoryInfo",
    "GPUMemoryManager",
    "GPUMemoryPool",
    "BatchProcessor",
    "estimate_memory_usage",
    "clear_gpu_cache",
    "get_memory_summary",
    "MemoryError",
    # Kernels
    "CUDAKernels",
    "residue_com_kernel",
    "tension_field_kernel",
    "anomaly_detection_kernel",
    "distance_matrix_kernel",
    "topological_charge_kernel",
    "compute_local_fractal_dimension_kernel",
    "compute_gradient_kernel",
    "create_elementwise_kernel",
    "benchmark_kernels",
    "get_kernel_manager",
    # Inverse Kernels (GPU ONLY)
    "InverseKernels",
    "compute_gram_matrix",
    "compute_path_statistics",
    "inverse_verify_all",
]
# 初期化メッセージ
import logging

logger = logging.getLogger("getter_one.core")
logger.debug("Getter_one GPU Core initialized")
