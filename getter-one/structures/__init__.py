"""
Lambda³ Structures Module
~~~~~~~~~~~~~~~~~~~~~~~~~

Lambda³構造計算モジュール 💕

Components:
    - LambdaStructuresCore: 汎用Lambda³構造計算（CPU, domain-agnostic）
    - LambdaStructuresGPU: GPU版Lambda構造計算（MD特化）
    - MDFeaturesGPU: MD特徴抽出
    - TensorOperationsGPU: テンソル演算
"""

# ── Core（汎用版・CPU・ドメイン非依存） ──
from .lambda_structures_core import (
    LambdaCoreConfig,
    LambdaStructuresCore,
)

from .tensor_operations_gpu import (
    TensorOperationsGPU,
    batch_tensor_operation,
    compute_correlation_gpu,
    compute_covariance_gpu,
    compute_gradient_gpu,
    sliding_window_operation_gpu,
)

__all__ = [
    # Lambda Structures Core (domain-agnostic)
    "LambdaStructuresCore",
    "LambdaCoreConfig",
    # Tensor Operations
    "TensorOperationsGPU",
    "compute_gradient_gpu",
    "compute_covariance_gpu",
    "compute_correlation_gpu",
    "sliding_window_operation_gpu",
    "batch_tensor_operation",
]

# 初期化メッセージ
import logging

logger = logging.getLogger("bankai.structures")
logger.debug("Lambda³ Structures module initialized (Core + GPU)")
