"""
GETTER One - Structures Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lambda³構造計算 💕

Components:
    - LambdaStructuresCore: 汎用Lambda³構造計算（CPU, domain-agnostic）
"""

from .lambda_structures_core import (
    LambdaCoreConfig,
    LambdaStructuresCore,
)

__all__ = [
    "LambdaStructuresCore",
    "LambdaCoreConfig",
]
