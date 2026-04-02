"""
getter-oneU版解析モジュール
"""

# 🌐 Network Analyzer Core (domain-agnostic)
from .network_analyzer_core import (
    CooperativeEventNetwork,
    DimensionLink,
    NetworkAnalyzerCore,
    NetworkResult,
)


__all__ = [
    # メイン検出器
    # 🌐 Network Analyzer Core (domain-agnostic)
    "NetworkAnalyzerCore",
    "NetworkResult",
    "DimensionLink",
    "CooperativeEventNetwork",
]

__version__ = "1.0.0"


# ========================================
# バージョン情報
# ========================================

def get_version_info():
    """
    Lambda³ GPU バージョン情報取得

    Returns
    -------
    dict
        バージョン情報
    """
    return {
        "version": __version__,
        "features": {
            "bankai_core": True,
            "gpu_acceleration": True,
        },
        "description": "Getter One",
    }
