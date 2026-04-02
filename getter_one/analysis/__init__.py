"""
GETTER One - Analysis Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

構造ネットワーク解析・信頼度判定・レポート生成 💕

Components:
    - NetworkAnalyzerCore: 次元間因果ネットワーク解析
    - ConfidenceKit: 統計的信頼度判定（p値/CI/効果量）
    - ReportGenerator: 結果レポート生成
"""

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

__all__ = [
    # Network Analyzer
    "NetworkAnalyzerCore",
    "NetworkResult",
    "DimensionLink",
    "CooperativeEventNetwork",
    # Confidence Kit
    "assess_confidence",
    "ConfidenceReport",
    "EventConfidence",
    "BoundaryConfidence",
    "CausalLinkConfidence",
    "SyncConfidence",
    # Report
    "generate_report",
]
