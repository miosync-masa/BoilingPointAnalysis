"""
Lambda³ GPU版検出モジュール
異常検出、境界検出、トポロジカル破れ検出などの高速GPU実装
"""

from .anomaly_detection_gpu import AnomalyDetectorGPU
from .boundary_detection_gpu import (
    BoundaryDetectorGPU,
    compute_structural_boundaries_batch_gpu,
)
from .extended_detection_gpu import ExtendedDetectorGPU
from .phase_space_gpu import PhaseSpaceAnalyzerGPU
from .topology_breaks_gpu import TopologyBreaksDetectorGPU

__all__ = [
    "AnomalyDetectorGPU",
    "BoundaryDetectorGPU",
    "TopologyBreaksDetectorGPU",
    "ExtendedDetectorGPU",
    "PhaseSpaceAnalyzerGPU",
    "compute_structural_boundaries_batch_gpu",
]

__version__ = "1.0.0"
