"""
GETTER One - Analysis Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

構造ネットワーク解析・カスケード因果追跡・信頼度判定・レポート生成 💕

Components:
    - NetworkAnalyzerCore: 次元間因果ネットワーク解析
    - CascadeTracker: カスケード因果チェーン追跡（Third Impact汎用版）
    - ConfidenceKit: 統計的信頼度判定（p値/CI/効果量）
    - InverseChecker: 逆問題による構造的イベント検証（GPU ONLY 🔥）
    - ReportGenerator: 結果レポート生成
"""

from .cascade_tracker import (
    CascadeChain,
    CascadeEvent,
    CascadeLink,
    CascadeResult,
    CascadeTracker,
)
from .confidence_kit import (
    BoundaryConfidence,
    CausalLinkConfidence,
    ConfidenceReport,
    EventConfidence,
    SyncConfidence,
    assess_confidence,
)
from .network_analyzer_core import (
    CooperativeEventNetwork,
    DimensionLink,
    NetworkAnalyzerCore,
    NetworkResult,
)
from .report_generator import generate_report

# GPU ONLY: InverseChecker は CuPy が無い環境ではスキップ
try:
    from .inverse_checker import (
        EventVerdict,
        InverseChecker,
        VerificationResult,
    )

    _HAS_INVERSE = True
except ImportError:
    _HAS_INVERSE = False

__all__ = [
    # Cascade Tracker
    "CascadeChain",
    "CascadeEvent",
    "CascadeLink",
    "CascadeResult",
    "CascadeTracker",
    # Confidence Kit
    "BoundaryConfidence",
    "CausalLinkConfidence",
    "ConfidenceReport",
    "EventConfidence",
    "SyncConfidence",
    "assess_confidence",
    # Network Analyzer
    "CooperativeEventNetwork",
    "DimensionLink",
    "NetworkAnalyzerCore",
    "NetworkResult",
    # Inverse Checker (GPU ONLY)
    "EventVerdict",
    "InverseChecker",
    "VerificationResult",
    # Report
    "generate_report",
]
