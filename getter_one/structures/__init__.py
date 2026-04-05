"""
GETTER One - Structures Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lambda³構造計算 💕

Components:
    - LambdaStructuresCore: 汎用Lambda³構造計算（CPU, domain-agnostic）
    - LambdaStructuresDualCore: Local/Global Dual-Path Architecture
"""

from .lambda_structures_core import (
    LambdaCoreConfig,
    LambdaStructuresCore,
)
from .lambda_structures_dual_core import (
    DualCoreConfig,
    LambdaStructuresDualCore,
)

__all__ = [
    "LambdaStructuresCore",
    "LambdaCoreConfig",
    "LambdaStructuresDualCore",
    "DualCoreConfig",
]
