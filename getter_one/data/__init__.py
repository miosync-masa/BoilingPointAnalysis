
"""
GETTER One - Data Module
~~~~~~~~~~~~~~~~~~~~~~~~~

多フォーマット対応データローダー 💕

Supported: .csv, .tsv, .json, .parquet, .xlsx, .npy, .npz
"""

from data.loader import (
    GetterDataset,
    from_dataframe,
    from_numpy,
    load,
    merge,
)

__all__ = [
    "load",
    "merge",
    "from_dataframe",
    "from_numpy",
    "GetterDataset",
]
