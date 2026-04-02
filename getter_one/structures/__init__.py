"""
Getter-one Structures Module
~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# ── Core（汎用版・CPU・ドメイン非依存） ──
from .lambda_structures_core import (
    LambdaCoreConfig,
    LambdaStructuresCore,
)


__all__ = [
    # Lambda Structures Core (domain-agnostic)
    "LambdaStructuresCore",
    "LambdaCoreConfig",
]

# 初期化メッセージ
import logging

logger = logging.getLogger("bankai.structures")
logger.debug("Getter-one Structures module initialized (Core + GPU)")
