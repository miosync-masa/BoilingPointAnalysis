"""
GETTER One - Data Loader
=========================

多次元時系列データの読み込み・マージ・前処理を行う汎用データローダー。

対応フォーマット:
  .csv, .tsv   - テーブルデータ（pandas経由）
  .json        - JSON配列/オブジェクト
  .parquet     - Apache Parquet（高速・圧縮）
  .xlsx, .xls  - Excelファイル
  .npy         - NumPy配列
  .npz         - NumPy複数配列

使い方:
  from getter_one.data import load, merge

  # 単一ファイル（拡張子で自動判別）
  dataset = load("weather.csv", target="precipitation")

  # 複数ソースのマージ
  dataset = merge(
      ["weather.csv", "air_quality.json"],
      time_column="date",
  )

# ファイル情報を見る
python -m getter_one.data.loader info weather.csv

# 単体読み込み＆正規化＆ターゲット分離
python -m getter_one.data.loader load weather.csv \
  --target precipitation \
  --normalize range \
  -o prepared.csv

# 複数ファイルをマージ
python -m getter_one.data.loader merge \
  weather.csv air_quality.json solar.parquet \
  --time date \
  --target precipitation \
  -o merged.csv

# 出力フォーマットは拡張子で自動判別
-o output.csv   → CSV
-o output.npy   → NumPy配列
-o output.npz   → NumPy複数配列（dimension_names付き）

Built with 💕 by Masamichi & Tamaki
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("getter_one.data.loader")


# ============================================================
# Data Container
# ============================================================

@dataclass
class GetterDataset:
    """
    GETTER One 統一データセット

    全ての入力フォーマットからこの形式に変換される。
    lambda_structures_core.py の compute_lambda_structures() に
    そのまま渡せる形式。
    """
    state_vectors: np.ndarray                  # (n_frames, n_dims)
    dimension_names: list[str]                 # 各次元の名前
    timestamps: Optional[np.ndarray] = None    # タイムスタンプ（あれば）
    target: Optional[np.ndarray] = None        # ターゲット変数（あれば）
    target_name: Optional[str] = None          # ターゲット変数名
    metadata: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return self.state_vectors.shape[0]

    @property
    def n_dims(self) -> int:
        return self.state_vectors.shape[1]

    def __repr__(self) -> str:
        target_str = f", target='{self.target_name}'" if self.target_name else ""
        ts_str = " with timestamps" if self.timestamps is not None else ""
        return (
            f"GetterDataset(frames={self.n_frames}, dims={self.n_dims}, "
            f"dimensions={self.dimension_names}{target_str}{ts_str})"
        )


# ============================================================
# Main API
# ============================================================

def load(
    path: Union[str, Path],
    target: Optional[str] = None,
    time_column: Optional[str] = None,
    columns: Optional[list[str]] = None,
    exclude_columns: Optional[list[str]] = None,
    normalize: str = "range",
    dropna: bool = True,
    dtype: type = np.float64,
    **kwargs,
) -> GetterDataset:
    """
    ファイルを読み込んでGetterDatasetに変換する。
    拡張子で自動判別。

    Parameters
    ----------
    path : str or Path
        入力ファイルパス
    target : str, optional
        ターゲット変数名（予測タスク用。入力から除外される）
    time_column : str, optional
        タイムスタンプ列名（自動検出も試みる）
    columns : list[str], optional
        使用する列（指定しなければ数値列を自動検出）
    exclude_columns : list[str], optional
        除外する列
    normalize : str
        正規化方法: "range", "zscore", "none"
    dropna : bool
        NaN行を削除するか
    dtype : type
        数値型（default: float64）
    **kwargs
        各フォーマット固有のオプション

    Returns
    -------
    GetterDataset
    """
    path = Path(path)
    suffix = path.suffix.lower()

    logger.info(f"📂 Loading {path.name} (format: {suffix})")

    # フォーマット別読み込み → DataFrame
    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, **kwargs)

    elif suffix == ".json":
        df = _load_json(path, **kwargs)

    elif suffix == ".parquet":
        df = pd.read_parquet(path, **kwargs)

    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, **kwargs)

    elif suffix == ".npy":
        return _load_npy(path, target=target, normalize=normalize, dtype=dtype)

    elif suffix == ".npz":
        return _load_npz(path, target=target, normalize=normalize, dtype=dtype)

    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    # DataFrame → GetterDataset
    return _dataframe_to_dataset(
        df,
        target=target,
        time_column=time_column,
        columns=columns,
        exclude_columns=exclude_columns,
        normalize=normalize,
        dropna=dropna,
        dtype=dtype,
        source=str(path),
    )


def from_dataframe(
    df: pd.DataFrame,
    target: Optional[str] = None,
    time_column: Optional[str] = None,
    columns: Optional[list[str]] = None,
    exclude_columns: Optional[list[str]] = None,
    normalize: str = "range",
    dropna: bool = True,
    dtype: type = np.float64,
) -> GetterDataset:
    """
    pandas DataFrameから直接GetterDatasetを作成する。

    Parameters
    ----------
    df : pd.DataFrame
    (他のパラメータはload()と同じ)
    """
    return _dataframe_to_dataset(
        df.copy(),
        target=target,
        time_column=time_column,
        columns=columns,
        exclude_columns=exclude_columns,
        normalize=normalize,
        dropna=dropna,
        dtype=dtype,
        source="DataFrame",
    )


def from_numpy(
    data: np.ndarray,
    dimension_names: Optional[list[str]] = None,
    timestamps: Optional[np.ndarray] = None,
    target: Optional[np.ndarray] = None,
    target_name: Optional[str] = None,
    normalize: str = "range",
) -> GetterDataset:
    """
    NumPy配列から直接GetterDatasetを作成する。

    Parameters
    ----------
    data : np.ndarray (n_frames, n_dims)
    dimension_names : list[str], optional
    timestamps : np.ndarray, optional
    target : np.ndarray, optional
    target_name : str, optional
    normalize : str
    """
    if data.ndim == 1:
        data = data[:, np.newaxis]

    if dimension_names is None:
        dimension_names = [f"dim_{i}" for i in range(data.shape[1])]

    data_norm = _normalize(data, normalize)

    return GetterDataset(
        state_vectors=data_norm,
        dimension_names=dimension_names,
        timestamps=timestamps,
        target=target,
        target_name=target_name,
        metadata={"source": "numpy", "normalize": normalize},
    )


def merge(
    paths: list[Union[str, Path]],
    time_column: str = "date",
    target: Optional[str] = None,
    normalize: str = "range",
    dropna: bool = True,
    how: str = "inner",
    **kwargs,
) -> GetterDataset:
    """
    複数ファイルをタイムスタンプでマージしてGetterDatasetを作成する。

    Parameters
    ----------
    paths : list[str or Path]
        入力ファイルパスのリスト
    time_column : str
        マージに使うタイムスタンプ列名
    target : str, optional
        ターゲット変数名
    normalize : str
        正規化方法
    dropna : bool
        NaN行を削除するか
    how : str
        マージ方法: "inner", "outer", "left", "right"
    """
    if not paths:
        raise ValueError("No paths provided")

    logger.info(f"📂 Merging {len(paths)} files on '{time_column}'")

    dfs = []
    for p in paths:
        p = Path(p)
        suffix = p.suffix.lower()

        if suffix in (".csv", ".tsv"):
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(p, sep=sep, **kwargs)
        elif suffix == ".json":
            df = _load_json(p, **kwargs)
        elif suffix == ".parquet":
            df = pd.read_parquet(p, **kwargs)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(p, **kwargs)
        else:
            raise ValueError(f"Unsupported format for merge: {suffix}")

        # 重複列名にソース名をプレフィックス
        overlap = set(df.columns) - {time_column}
        for existing_df in dfs:
            existing_cols = set(existing_df.columns) - {time_column}
            dupes = overlap & existing_cols
            if dupes:
                prefix = p.stem + "_"
                df = df.rename(columns={c: prefix + c for c in dupes})

        dfs.append(df)

    # タイムスタンプでマージ
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on=time_column, how=how)

    logger.info(f"   Merged shape: {merged.shape}")

    return _dataframe_to_dataset(
        merged,
        target=target,
        time_column=time_column,
        normalize=normalize,
        dropna=dropna,
        source=f"merged({len(paths)} files)",
    )


# ============================================================
# Internal: Format-specific loaders
# ============================================================

def _load_json(path: Path, **kwargs) -> pd.DataFrame:
    """JSON → DataFrame"""
    with open(path) as f:
        data = json.load(f)

    # JSON形式の自動判別
    if isinstance(data, list):
        # [{col: val, ...}, ...] 形式
        return pd.DataFrame(data)

    elif isinstance(data, dict):
        # {col: [val, ...], ...} 形式
        if all(isinstance(v, list) for v in data.values()):
            return pd.DataFrame(data)

        # {"data": [...], "metadata": {...}} 形式
        if "data" in data:
            df = pd.DataFrame(data["data"])
            return df

        # 単一レコード → 1行DataFrame
        return pd.DataFrame([data])

    raise ValueError(f"Unsupported JSON structure in {path}")


def _load_npy(
    path: Path,
    target: Optional[str] = None,
    normalize: str = "range",
    dtype: type = np.float64,
) -> GetterDataset:
    """NumPy .npy → GetterDataset"""
    data = np.load(path).astype(dtype)

    if data.ndim == 1:
        data = data[:, np.newaxis]

    n_dims = data.shape[1]
    dim_names = [f"dim_{i}" for i in range(n_dims)]

    target_arr = None
    target_name = None
    if target is not None and target.isdigit():
        idx = int(target)
        target_arr = data[:, idx]
        target_name = f"dim_{idx}"
        data = np.delete(data, idx, axis=1)
        dim_names = [f"dim_{i}" for i in range(data.shape[1])]

    data_norm = _normalize(data, normalize)

    return GetterDataset(
        state_vectors=data_norm,
        dimension_names=dim_names,
        target=target_arr,
        target_name=target_name,
        metadata={"source": str(path), "normalize": normalize},
    )


def _load_npz(
    path: Path,
    target: Optional[str] = None,
    normalize: str = "range",
    dtype: type = np.float64,
) -> GetterDataset:
    """NumPy .npz → GetterDataset"""
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.keys())

    # "data" or "state_vectors" キーを探す
    data_key = None
    for candidate in ["data", "state_vectors", "X", "x", keys[0]]:
        if candidate in keys:
            data_key = candidate
            break

    data = npz[data_key].astype(dtype)
    if data.ndim == 1:
        data = data[:, np.newaxis]

    dim_names = [f"dim_{i}" for i in range(data.shape[1])]

    # dimension_names が含まれてれば使う
    if "dimension_names" in keys:
        dim_names = list(npz["dimension_names"])

    # timestamps
    timestamps = None
    if "timestamps" in keys:
        timestamps = npz["timestamps"]

    # target
    target_arr = None
    target_name = None
    if target is not None and target in keys:
        target_arr = npz[target].astype(dtype)
        target_name = target

    data_norm = _normalize(data, normalize)

    return GetterDataset(
        state_vectors=data_norm,
        dimension_names=dim_names,
        timestamps=timestamps,
        target=target_arr,
        target_name=target_name,
        metadata={"source": str(path), "normalize": normalize, "npz_keys": keys},
    )


# ============================================================
# Internal: DataFrame → GetterDataset
# ============================================================

def _dataframe_to_dataset(
    df: pd.DataFrame,
    target: Optional[str] = None,
    time_column: Optional[str] = None,
    columns: Optional[list[str]] = None,
    exclude_columns: Optional[list[str]] = None,
    normalize: str = "range",
    dropna: bool = True,
    dtype: type = np.float64,
    source: str = "unknown",
) -> GetterDataset:
    """DataFrame → GetterDataset の共通変換ロジック"""

    # タイムスタンプ列の検出
    timestamps = None
    if time_column is None:
        time_column = _detect_time_column(df)

    if time_column and time_column in df.columns:
        try:
            timestamps = pd.to_datetime(df[time_column]).values
        except Exception:
            logger.warning(f"   Could not parse '{time_column}' as datetime")
            timestamps = None

    # ターゲット列の分離
    target_arr = None
    target_name = None
    if target and target in df.columns:
        target_arr = df[target].values.astype(dtype)
        target_name = target

    # 入力列の決定
    if columns is not None:
        input_cols = [c for c in columns if c in df.columns]
    else:
        # 数値列を自動検出（time, target列は除外）
        exclude = {time_column, target} if time_column else {target}
        exclude = {c for c in exclude if c is not None}
        if exclude_columns:
            exclude.update(exclude_columns)
        input_cols = [
            c for c in df.columns
            if c not in exclude and df[c].dtype in ("float64", "float32", "int64", "int32")
        ]

    if not input_cols:
        raise ValueError(
            f"No numeric input columns found. "
            f"Available columns: {list(df.columns)}"
        )

    # NaN処理
    if dropna:
        mask = df[input_cols].notna().all(axis=1)
        if target_arr is not None:
            mask &= pd.Series(target_arr).notna().values
        df = df[mask].reset_index(drop=True)
        if target_arr is not None:
            target_arr = target_arr[mask.values]
        if timestamps is not None:
            timestamps = timestamps[mask.values]

    # 数値変換
    data = df[input_cols].values.astype(dtype)

    # 正規化
    data_norm = _normalize(data, normalize)

    logger.info(
        f"   ✅ Loaded: {data_norm.shape[0]} frames × {data_norm.shape[1]} dims "
        f"({', '.join(input_cols[:3])}{'...' if len(input_cols) > 3 else ''})"
    )

    return GetterDataset(
        state_vectors=data_norm,
        dimension_names=input_cols,
        timestamps=timestamps,
        target=target_arr,
        target_name=target_name,
        metadata={
            "source": source,
            "normalize": normalize,
            "original_columns": list(df.columns),
            "n_original_rows": len(df),
        },
    )


# ============================================================
# Internal: Utilities
# ============================================================

def _normalize(data: np.ndarray, method: str) -> np.ndarray:
    """正規化"""
    if method == "none":
        return data.copy()

    elif method == "range":
        dmin = data.min(axis=0)
        dmax = data.max(axis=0)
        drange = dmax - dmin
        drange[drange == 0] = 1.0
        return (data - dmin) / drange

    elif method == "zscore":
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std == 0] = 1.0
        return (data - mean) / std

    else:
        raise ValueError(f"Unknown normalize method: {method}. Use 'range', 'zscore', or 'none'")


def _detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """タイムスタンプ列を自動検出"""
    candidates = ["date", "datetime", "timestamp", "time", "Date", "DateTime", "Timestamp"]
    for c in candidates:
        if c in df.columns:
            return c

    # object型の列でdatetime parseを試みる
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                pd.to_datetime(df[c].head(5))
                return c
            except Exception:
                continue

    return None


# ============================================================
# CLI: コマンドラインから単体実行
# ============================================================
# Usage:
#   python -m getter_one.data.loader load weather.csv --target precipitation -o prepared.csv
#   python -m getter_one.data.loader merge weather.csv air.json --time date -o merged.csv
#   python -m getter_one.data.loader info weather.csv
# ============================================================

def _cli_load(args):
    """CLI: load サブコマンド"""
    dataset = load(
        args.file,
        target=args.target,
        time_column=args.time,
        normalize=args.normalize,
    )
    _cli_print_info(dataset)
    if args.output:
        _cli_save(dataset, args.output)


def _cli_merge(args):
    """CLI: merge サブコマンド"""
    dataset = merge(
        args.files,
        time_column=args.time or "date",
        target=args.target,
        normalize=args.normalize,
    )
    _cli_print_info(dataset)
    if args.output:
        _cli_save(dataset, args.output)


def _cli_info(args):
    """CLI: info サブコマンド"""
    dataset = load(args.file, normalize="none")
    _cli_print_info(dataset)


def _cli_print_info(dataset: GetterDataset):
    """データセット情報を表示"""
    print(f"\n{'=' * 50}")
    print(f"  GETTER One Dataset Info")
    print(f"{'=' * 50}")
    print(f"  Frames:     {dataset.n_frames}")
    print(f"  Dimensions: {dataset.n_dims}")
    print(f"  Columns:    {dataset.dimension_names}")
    if dataset.target_name:
        print(f"  Target:     {dataset.target_name}")
    if dataset.timestamps is not None:
        print(f"  Time range: {dataset.timestamps[0]} → {dataset.timestamps[-1]}")
    print(f"  Normalize:  {dataset.metadata.get('normalize', 'unknown')}")

    # 統計情報
    print(f"\n  Statistics:")
    for i, name in enumerate(dataset.dimension_names):
        col = dataset.state_vectors[:, i]
        print(f"    {name:>30s}: mean={col.mean():.4f} std={col.std():.4f} "
              f"min={col.min():.4f} max={col.max():.4f}")

    if dataset.target is not None:
        t = dataset.target
        print(f"    {'[TARGET] ' + dataset.target_name:>30s}: mean={t.mean():.4f} "
              f"std={t.std():.4f} zero={100*(t==0).mean():.1f}%")


def _cli_save(dataset: GetterDataset, output_path: str):
    """データセットを保存"""
    path = Path(output_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.DataFrame(dataset.state_vectors, columns=dataset.dimension_names)
        if dataset.timestamps is not None:
            df.insert(0, "date", dataset.timestamps)
        if dataset.target is not None:
            df[dataset.target_name or "target"] = dataset.target
        df.to_csv(path, index=False)

    elif suffix == ".npy":
        np.save(path, dataset.state_vectors)

    elif suffix == ".npz":
        save_dict = {"state_vectors": dataset.state_vectors,
                     "dimension_names": np.array(dataset.dimension_names)}
        if dataset.timestamps is not None:
            save_dict["timestamps"] = dataset.timestamps
        if dataset.target is not None:
            save_dict[dataset.target_name or "target"] = dataset.target
        np.savez(path, **save_dict)

    else:
        raise ValueError(f"Unsupported output format: {suffix}")

    print(f"\n  ✅ Saved: {path} ({path.stat().st_size / 1024:.1f} KB)")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="getter-one-loader",
        description="GETTER One Data Loader - Prepare data for GETTER One pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # load
    p_load = subparsers.add_parser("load", help="Load and prepare a single file")
    p_load.add_argument("file", help="Input file path")
    p_load.add_argument("-o", "--output", help="Output file path (csv/npy/npz)")
    p_load.add_argument("--target", help="Target column name")
    p_load.add_argument("--time", help="Time column name")
    p_load.add_argument("--normalize", default="range", choices=["range", "zscore", "none"])

    # merge
    p_merge = subparsers.add_parser("merge", help="Merge multiple files")
    p_merge.add_argument("files", nargs="+", help="Input file paths")
    p_merge.add_argument("-o", "--output", help="Output file path")
    p_merge.add_argument("--target", help="Target column name")
    p_merge.add_argument("--time", help="Time column name for merge key")
    p_merge.add_argument("--normalize", default="range", choices=["range", "zscore", "none"])

    # info
    p_info = subparsers.add_parser("info", help="Show file info")
    p_info.add_argument("file", help="Input file path")

    args = parser.parse_args()

    if args.command == "load":
        _cli_load(args)
    elif args.command == "merge":
        _cli_merge(args)
    elif args.command == "info":
        _cli_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
