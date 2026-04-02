"""
GETTER One - Test Suite
========================

pytest で実行:
  $ pytest tests/ -v
  $ pytest tests/ -v --tb=short

GitHub Actions:
  $ pytest tests/ -v --junitxml=results.xml
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_data():
    """基本テスト用の合成時系列データ"""
    rng = np.random.default_rng(42)
    n_frames = 500
    n_dims = 4

    # 構造的な相関を持つデータ
    t = np.arange(n_frames, dtype=np.float64)
    data = np.zeros((n_frames, n_dims))
    data[:, 0] = np.sin(0.05 * t) + rng.normal(0, 0.1, n_frames)           # dim_0
    data[:, 1] = np.sin(0.05 * t + 0.5) + rng.normal(0, 0.1, n_frames)     # dim_1 (lagged)
    data[:, 2] = -0.8 * data[:, 0] + rng.normal(0, 0.15, n_frames)          # dim_2 (anti-corr)
    data[:, 3] = rng.normal(0, 0.3, n_frames)                               # dim_3 (noise)

    return data


@pytest.fixture
def sample_names():
    return ["alpha", "beta", "gamma", "noise"]


@pytest.fixture
def sample_csv(sample_data, sample_names, tmp_path):
    """テスト用CSV"""
    df = pd.DataFrame(sample_data, columns=sample_names)
    df.insert(0, "date", pd.date_range("2024-01-01", periods=len(df), freq="h"))
    df["target"] = np.random.default_rng(42).random(len(df))
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_json(sample_data, sample_names, tmp_path):
    """テスト用JSON"""
    records = []
    for i in range(len(sample_data)):
        row = {name: float(sample_data[i, j]) for j, name in enumerate(sample_names)}
        row["date"] = f"2024-01-{(i % 28) + 1:02d}T{i % 24:02d}:00:00"
        records.append(row)
    path = tmp_path / "test_data.json"
    path.write_text(json.dumps(records))
    return path


@pytest.fixture
def sample_npy(sample_data, tmp_path):
    """テスト用npy"""
    path = tmp_path / "test_data.npy"
    np.save(path, sample_data)
    return path


@pytest.fixture
def normalized_data(sample_data):
    """レンジ正規化済みデータ"""
    dmin = sample_data.min(axis=0)
    dmax = sample_data.max(axis=0)
    drange = dmax - dmin
    drange[drange == 0] = 1.0
    return (sample_data - dmin) / drange


# ============================================================
# Tests: Data Loader
# ============================================================

class TestDataLoader:
    """getter_one.data.loader のテスト"""

    def test_load_csv(self, sample_csv):
        from getter_one.data.loader import load
        dataset = load(sample_csv, target="target")
        assert dataset.n_frames > 0
        assert dataset.n_dims == 4
        assert dataset.target is not None
        assert dataset.target_name == "target"
        assert dataset.timestamps is not None

    def test_load_json(self, sample_json):
        from getter_one.data.loader import load
        dataset = load(sample_json)
        assert dataset.n_frames > 0
        assert dataset.n_dims >= 4

    def test_load_npy(self, sample_npy):
        from getter_one.data.loader import load
        dataset = load(sample_npy)
        assert dataset.n_frames == 500
        assert dataset.n_dims == 4

    def test_from_numpy(self, sample_data, sample_names):
        from getter_one.data.loader import from_numpy
        dataset = from_numpy(sample_data, dimension_names=sample_names)
        assert dataset.n_frames == 500
        assert dataset.n_dims == 4
        assert dataset.dimension_names == sample_names

    def test_from_dataframe(self, sample_data, sample_names):
        from getter_one.data.loader import from_dataframe
        df = pd.DataFrame(sample_data, columns=sample_names)
        dataset = from_dataframe(df)
        assert dataset.n_frames == 500
        assert dataset.n_dims == 4

    def test_normalize_range(self, sample_csv):
        from getter_one.data.loader import load
        dataset = load(sample_csv, target="target", normalize="range")
        assert dataset.state_vectors.min() >= 0.0
        assert dataset.state_vectors.max() <= 1.0

    def test_normalize_zscore(self, sample_csv):
        from getter_one.data.loader import load
        dataset = load(sample_csv, target="target", normalize="zscore")
        means = dataset.state_vectors.mean(axis=0)
        np.testing.assert_allclose(means, 0.0, atol=1e-10)

    def test_normalize_none(self, sample_csv):
        from getter_one.data.loader import load
        dataset = load(sample_csv, target="target", normalize="none")
        assert dataset.state_vectors.max() > 1.0  # 正規化されてない

    def test_target_separation(self, sample_csv):
        from getter_one.data.loader import load
        dataset = load(sample_csv, target="target")
        assert "target" not in dataset.dimension_names
        assert dataset.target is not None
        assert len(dataset.target) == dataset.n_frames

    def test_save_csv(self, sample_csv, tmp_path):
        from getter_one.data.loader import load, _cli_save
        dataset = load(sample_csv)
        out_path = tmp_path / "output.csv"
        _cli_save(dataset, str(out_path))
        assert out_path.exists()
        df = pd.read_csv(out_path)
        assert len(df) == dataset.n_frames

    def test_save_npy(self, sample_csv, tmp_path):
        from getter_one.data.loader import load, _cli_save
        dataset = load(sample_csv)
        out_path = tmp_path / "output.npy"
        _cli_save(dataset, str(out_path))
        assert out_path.exists()
        loaded = np.load(out_path)
        assert loaded.shape == dataset.state_vectors.shape

    def test_unsupported_format(self, tmp_path):
        from getter_one.data.loader import load
        fake = tmp_path / "data.xyz"
        fake.write_text("nope")
        with pytest.raises(ValueError, match="Unsupported"):
            load(fake)

    def test_merge(self, tmp_path):
        from getter_one.data.loader import merge
        df1 = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "a": [1.0, 2.0]})
        df2 = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "b": [3.0, 4.0]})
        p1 = tmp_path / "f1.csv"
        df1.to_csv(p1, index=False)
        p2 = tmp_path / "f2.csv"
        df2.to_csv(p2, index=False)
        dataset = merge([str(p1), str(p2)], time_column="date")
        assert dataset.n_dims == 2
        assert dataset.n_frames == 2


# ============================================================
# Tests: Lambda Structures Core
# ============================================================

class TestLambdaStructuresCore:
    """getter_one.structures.lambda_structures_core のテスト"""

    def test_basic_computation(self, normalized_data, sample_names):
        from getter_one.structures.lambda_structures_core import LambdaStructuresCore
        core = LambdaStructuresCore()
        results = core.compute_lambda_structures(
            normalized_data, window_steps=24, dimension_names=sample_names
        )
        assert "lambda_F" in results
        assert "lambda_F_mag" in results
        assert "lambda_FF" in results
        assert "lambda_FF_mag" in results
        assert "rho_T" in results
        assert "Q_lambda" in results
        assert "Q_cumulative" in results
        assert "sigma_s" in results
        assert "structural_coherence" in results

    def test_output_shapes(self, normalized_data):
        from getter_one.structures.lambda_structures_core import LambdaStructuresCore
        core = LambdaStructuresCore()
        n = len(normalized_data)
        results = core.compute_lambda_structures(normalized_data, window_steps=10)

        assert results["lambda_F"].shape == (n - 1, normalized_data.shape[1])
        assert results["lambda_F_mag"].shape == (n - 1,)
        assert results["lambda_FF"].shape == (n - 2, normalized_data.shape[1])
        assert results["lambda_FF_mag"].shape == (n - 2,)
        assert results["rho_T"].shape == (n,)
        assert results["sigma_s"].shape == (n,)

    def test_sigma_s_nonzero(self, normalized_data):
        """σₛが相関のあるデータで非ゼロであること"""
        from getter_one.structures.lambda_structures_core import LambdaStructuresCore
        core = LambdaStructuresCore()
        results = core.compute_lambda_structures(normalized_data, window_steps=24)
        assert np.max(results["sigma_s"]) > 0

    def test_q_lambda_cumulative(self, normalized_data):
        """Q_cumulativeがQ_lambdaの累積和であること"""
        from getter_one.structures.lambda_structures_core import LambdaStructuresCore
        core = LambdaStructuresCore()
        results = core.compute_lambda_structures(normalized_data, window_steps=10)
        np.testing.assert_allclose(
            results["Q_cumulative"],
            np.cumsum(results["Q_lambda"]),
            atol=1e-10,
        )

    def test_1d_input_rejection(self):
        """1Dデータが正しくエラーになること"""
        from getter_one.structures.lambda_structures_core import LambdaStructuresCore
        core = LambdaStructuresCore()
        with pytest.raises(ValueError, match="2D"):
            core.compute_lambda_structures(np.array([1, 2, 3]), window_steps=2)

    def test_different_window_sizes(self, normalized_data):
        """異なるwindow_sizeで動作すること"""
        from getter_one.structures.lambda_structures_core import LambdaStructuresCore
        core = LambdaStructuresCore()
        for w in [5, 10, 50, 100]:
            results = core.compute_lambda_structures(normalized_data, window_steps=w)
            assert results["rho_T"].shape[0] == len(normalized_data)


# ============================================================
# Tests: Network Analyzer Core
# ============================================================

class TestNetworkAnalyzerCore:
    """getter_one.analysis.network_analyzer_core のテスト"""

    def test_basic_analysis(self, normalized_data, sample_names):
        from getter_one.analysis.network_analyzer_core import NetworkAnalyzerCore
        analyzer = NetworkAnalyzerCore(sync_threshold=0.3, causal_threshold=0.25)
        result = analyzer.analyze(normalized_data, dimension_names=sample_names)

        assert result.n_dims == 4
        assert result.pattern in ("parallel", "cascade", "mixed", "independent")
        assert isinstance(result.sync_network, list)
        assert isinstance(result.causal_network, list)

    def test_sync_detection(self, normalized_data, sample_names):
        """相関データから同期リンクが検出されること"""
        from getter_one.analysis.network_analyzer_core import NetworkAnalyzerCore
        analyzer = NetworkAnalyzerCore(sync_threshold=0.2)
        result = analyzer.analyze(normalized_data, dimension_names=sample_names)
        assert result.n_sync_links > 0

    def test_hub_detection(self, normalized_data, sample_names):
        from getter_one.analysis.network_analyzer_core import NetworkAnalyzerCore
        analyzer = NetworkAnalyzerCore(sync_threshold=0.2, causal_threshold=0.15)
        result = analyzer.analyze(normalized_data, dimension_names=sample_names)
        assert isinstance(result.hub_dimensions, list)

    def test_event_network(self, normalized_data, sample_names):
        from getter_one.analysis.network_analyzer_core import NetworkAnalyzerCore
        analyzer = NetworkAnalyzerCore()
        event_net = analyzer.analyze_event_network(
            normalized_data, event_frame=250,
            window_before=24, window_after=6,
            dimension_names=sample_names,
        )
        assert event_net.event_frame == 250
        assert isinstance(event_net.propagation_order, list)
        assert len(event_net.propagation_order) == 4

    def test_sync_matrix_symmetric(self, normalized_data):
        from getter_one.analysis.network_analyzer_core import NetworkAnalyzerCore
        analyzer = NetworkAnalyzerCore()
        result = analyzer.analyze(normalized_data)
        np.testing.assert_allclose(
            result.sync_matrix, result.sync_matrix.T, atol=1e-10
        )

    def test_independent_data(self):
        """独立なデータではリンクが少ないこと"""
        from getter_one.analysis.network_analyzer_core import NetworkAnalyzerCore
        rng = np.random.default_rng(99)
        independent = rng.normal(0, 1, (500, 4))
        analyzer = NetworkAnalyzerCore(sync_threshold=0.5, causal_threshold=0.5)
        result = analyzer.analyze(independent)
        assert result.n_sync_links == 0
        assert result.n_causal_links == 0


# ============================================================
# Tests: Confidence Kit
# ============================================================

class TestConfidenceKit:
    """getter_one.analysis.confidence_kit のテスト"""

    def test_basic_confidence(self, normalized_data, sample_names):
        from getter_one.structures.lambda_structures_core import LambdaStructuresCore
        from getter_one.analysis.confidence_kit import assess_confidence

        core = LambdaStructuresCore()
        structures = core.compute_lambda_structures(normalized_data, window_steps=24)

        report = assess_confidence(
            state_vectors=normalized_data,
            lambda_structures=structures,
            dimension_names=sample_names,
            n_permutations=50,
            n_bootstrap=100,
        )
        assert report.alpha == 0.05
        assert report.n_permutations == 50

    def test_boundary_confidence(self, normalized_data):
        from getter_one.structures.lambda_structures_core import LambdaStructuresCore
        from getter_one.analysis.confidence_kit import assess_confidence

        core = LambdaStructuresCore()
        structures = core.compute_lambda_structures(normalized_data, window_steps=24)
        boundaries = {"boundary_locations": [100, 300]}

        report = assess_confidence(
            state_vectors=normalized_data,
            lambda_structures=structures,
            structural_boundaries=boundaries,
            n_permutations=50,
            n_bootstrap=100,
        )
        assert len(report.boundaries) > 0

    def test_network_confidence(self, normalized_data, sample_names):
        from getter_one.structures.lambda_structures_core import LambdaStructuresCore
        from getter_one.analysis.network_analyzer_core import NetworkAnalyzerCore
        from getter_one.analysis.confidence_kit import assess_confidence

        core = LambdaStructuresCore()
        structures = core.compute_lambda_structures(normalized_data, window_steps=24)

        analyzer = NetworkAnalyzerCore(sync_threshold=0.2)
        network = analyzer.analyze(normalized_data, dimension_names=sample_names)

        report = assess_confidence(
            state_vectors=normalized_data,
            lambda_structures=structures,
            network_result=network,
            dimension_names=sample_names,
            n_permutations=50,
            n_bootstrap=100,
        )
        assert isinstance(report.sync_links, list)
        assert isinstance(report.causal_links, list)

    def test_confidence_report_summary(self, normalized_data):
        from getter_one.structures.lambda_structures_core import LambdaStructuresCore
        from getter_one.analysis.confidence_kit import assess_confidence

        core = LambdaStructuresCore()
        structures = core.compute_lambda_structures(normalized_data, window_steps=24)

        report = assess_confidence(
            state_vectors=normalized_data,
            lambda_structures=structures,
            n_permutations=20,
            n_bootstrap=50,
        )
        summary = report.summary()
        assert "GETTER One Confidence Report" in summary


# ============================================================
# Tests: Report Generator
# ============================================================

class TestReportGenerator:
    """getter_one.analysis.report_generator のテスト"""

    def test_basic_report(self, normalized_data):
        from getter_one.structures.lambda_structures_core import LambdaStructuresCore
        from getter_one.analysis.report_generator import generate_report

        core = LambdaStructuresCore()
        structures = core.compute_lambda_structures(normalized_data, window_steps=24)

        report = generate_report(lambda_structures=structures)
        assert "Lambda³ Structure Statistics" in report
        assert "GETTER One" in report

    def test_report_with_all_components(self, normalized_data, sample_names, tmp_path):
        from getter_one.structures.lambda_structures_core import LambdaStructuresCore
        from getter_one.analysis.network_analyzer_core import NetworkAnalyzerCore
        from getter_one.analysis.confidence_kit import assess_confidence
        from getter_one.analysis.report_generator import generate_report

        core = LambdaStructuresCore()
        structures = core.compute_lambda_structures(normalized_data, window_steps=24)
        boundaries = {"boundary_locations": [100, 300]}

        analyzer = NetworkAnalyzerCore(sync_threshold=0.2)
        network = analyzer.analyze(normalized_data, dimension_names=sample_names)

        confidence = assess_confidence(
            state_vectors=normalized_data,
            lambda_structures=structures,
            structural_boundaries=boundaries,
            network_result=network,
            dimension_names=sample_names,
            n_permutations=20,
            n_bootstrap=50,
        )

        out_path = tmp_path / "report.md"
        report = generate_report(
            lambda_structures=structures,
            structural_boundaries=boundaries,
            network_result=network,
            confidence_report=confidence,
            output_path=str(out_path),
        )

        assert out_path.exists()
        assert "Causal Network" in report
        assert "Confidence Summary" in report


# ============================================================
# Tests: Integration (Pipeline-like)
# ============================================================

class TestIntegration:
    """パイプライン統合テスト"""

    def test_full_flow_numpy(self, normalized_data, sample_names):
        """numpy → Λ³ → network → confidence → report の全フロー"""
        from getter_one.structures.lambda_structures_core import LambdaStructuresCore
        from getter_one.analysis.network_analyzer_core import NetworkAnalyzerCore
        from getter_one.analysis.confidence_kit import assess_confidence
        from getter_one.analysis.report_generator import generate_report

        # Λ³
        core = LambdaStructuresCore()
        structures = core.compute_lambda_structures(
            normalized_data, window_steps=24, dimension_names=sample_names
        )
        assert len(structures) == 9

        # Network
        analyzer = NetworkAnalyzerCore(sync_threshold=0.2)
        network = analyzer.analyze(normalized_data, dimension_names=sample_names)
        assert network.n_dims == 4

        # Confidence
        confidence = assess_confidence(
            state_vectors=normalized_data,
            lambda_structures=structures,
            network_result=network,
            dimension_names=sample_names,
            n_permutations=20,
            n_bootstrap=50,
        )
        assert confidence.alpha == 0.05

        # Report
        report = generate_report(
            lambda_structures=structures,
            network_result=network,
            confidence_report=confidence,
        )
        assert len(report) > 100

    def test_full_flow_csv(self, sample_csv):
        """CSV → load → Λ³ → network の全フロー"""
        from getter_one.data.loader import load
        from getter_one.structures.lambda_structures_core import LambdaStructuresCore
        from getter_one.analysis.network_analyzer_core import NetworkAnalyzerCore

        dataset = load(sample_csv, target="target")
        core = LambdaStructuresCore()
        structures = core.compute_lambda_structures(
            dataset.state_vectors, window_steps=24,
            dimension_names=dataset.dimension_names,
        )
        assert len(structures) == 9

        analyzer = NetworkAnalyzerCore()
        network = analyzer.analyze(
            dataset.state_vectors,
            dimension_names=dataset.dimension_names,
        )
        assert network.n_dims == dataset.n_dims
