"""
GETTER One — Hell Mode Robustness Benchmark
=============================================

Non-stationary, non-linear, non-Gaussian data with:
  - Phase jumps, bifurcations, cascades
  - Resonance amplification, topological jumps
  - Multi-path interactions, structural decay

Question: "Which methods survive?"

Usage (Colab):
  !pip install getter-one statsmodels tigramite
  %run hell_mode_benchmark.py

Built with 💕 by Masamichi & Tamaki
"""

import time
import warnings

import numpy as np
import pandas as pd

# ============================================================
# 1. Hell Mode Data Generator (from BANKAI-MD anomaly zoo)
# ============================================================


class HellModeGenerator:
    """地獄モードデータ生成器 — MD異常検知用パターンを因果構造に注入"""

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def _pulse(self, series, intensity=3.0):
        s = series.copy()
        n = len(s)
        n_pulses = min(3, n)
        for idx in self.rng.choice(n, size=n_pulses, replace=False):
            s[idx] += self.rng.standard_normal() * intensity
            for offset in range(1, 4):
                decay = np.exp(-0.5 * offset)
                if idx + offset < n:
                    s[idx + offset] += s[idx] * decay * 0.3
        return s

    def _phase_jump(self, series, intensity=3.0):
        s = series.copy()
        n = len(s)
        jump = self.rng.integers(n // 3, 2 * n // 3)
        s[jump:] += intensity * self.rng.choice([-1, 1])
        s[jump] += self.rng.standard_normal() * intensity * 2
        return s

    def _bifurcation(self, series, intensity=2.0):
        s = series.copy()
        n = len(s)
        split = n // 2
        for i in range(split, n):
            t = (i - split) / (n - split)
            if i % 2 == 0:
                s[i] += intensity * np.sqrt(t) * self.rng.standard_normal()
            else:
                s[i] -= intensity * np.sqrt(t) * self.rng.standard_normal()
        return s

    def _cascade(self, series, intensity=2.0):
        s = series.copy()
        n = len(s)
        start = self.rng.integers(0, n // 2)
        s[start] += intensity * 3
        for i in range(start + 1, min(start + 15, n)):
            decay = np.exp(-0.3 * (i - start))
            s[i] += s[i - 1] * 0.5 * decay + self.rng.standard_normal() * 0.3
        return s

    def _resonance(self, series, intensity=2.0):
        s = series.copy()
        fft = np.fft.fft(s)
        max_freq = max(2, len(fft) // 4)
        res_freq = self.rng.integers(1, max_freq)
        fft[res_freq] *= intensity * 3
        if len(fft) - res_freq > 0:
            fft[-res_freq] *= intensity * 3
        s = np.real(np.fft.ifft(fft))
        return s

    def _structural_decay(self, series, intensity=2.0):
        s = series.copy()
        n = len(s)
        mid = n // 2
        for i in range(mid, n):
            decay = np.exp(-intensity * (i - mid) / (n - mid))
            s[i] *= decay
            s[i] += (1 - decay) * self.rng.standard_normal() * np.std(s[:mid])
        return s

    def generate_scenario(self, name, T=400, n_series=5):
        """地獄モードシナリオを生成（因果構造付き！）"""

        # まず因果構造のある基本データを作る
        # A → B (lag=2), A → C (lag=3) の構造
        noise_scale = 0.5
        A = np.zeros(T)
        B = np.zeros(T)
        C = np.zeros(T)

        for t in range(T):
            A[t] = 0.6 * (A[t - 1] if t > 0 else 0) + self.rng.standard_normal() * noise_scale
            if t >= 2:
                B[t] = 0.5 * (B[t - 1] if t > 0 else 0) + 0.7 * A[t - 2] + self.rng.standard_normal() * noise_scale
            if t >= 3:
                C[t] = 0.4 * (C[t - 1] if t > 0 else 0) + 0.6 * A[t - 3] + self.rng.standard_normal() * noise_scale

        # 追加の独立系列
        extras = []
        for _ in range(n_series - 3):
            s = np.zeros(T)
            for t in range(1, T):
                s[t] = 0.5 * s[t - 1] + self.rng.standard_normal() * noise_scale
            extras.append(s)

        all_series = [A, B, C] + extras

        # 地獄モード注入！
        hell_functions = {
            "H1_pulse": lambda s: self._pulse(s, 5.0),
            "H2_phase_jump": lambda s: self._phase_jump(s, 4.0),
            "H3_bifurcation": lambda s: self._bifurcation(s, 3.0),
            "H4_cascade": lambda s: self._cascade(s, 4.0),
            "H5_resonance": lambda s: self._resonance(s, 3.0),
            "H6_decay": lambda s: self._structural_decay(s, 2.0),
            "H7_multi_hell": self._multi_hell,
            "H8_progressive": self._progressive_hell,
        }

        if name not in hell_functions:
            raise ValueError(f"Unknown scenario: {name}")

        # 各系列に地獄を注入
        corrupted = []
        for s in all_series:
            result = hell_functions[name](s)
            # resonanceがcomplex返す可能性があるのでrealを取る
            corrupted.append(np.real(result).astype(np.float64))

        data = np.column_stack(corrupted).astype(np.float64)
        # NaN/Inf を除去
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        columns = [f"X{i}" for i in range(data.shape[1])]

        # Ground truth: A→B(lag=2), A→C(lag=3) は不変
        # 地獄モードの中でも因果構造が生き残るか？がテスト
        gt = {
            "edges": [("X0", "X1", 2), ("X0", "X2", 3)],
            "forbidden": [("X1", "X0"), ("X2", "X0"), ("X1", "X2"), ("X2", "X1")],
        }

        return pd.DataFrame(data=data, columns=columns, dtype=np.float64), gt

    def _multi_hell(self, series, intensity=3.0):
        """複数の地獄パターンを同時適用"""
        s = series.copy()
        s = self._pulse(s, intensity)
        s = self._phase_jump(s, intensity * 0.5)
        s = self._cascade(s, intensity * 0.7)
        return s

    def _progressive_hell(self, series, intensity=3.0):
        """時間とともに悪化する地獄"""
        s = series.copy()
        n = len(s)
        # 前半は平和、後半に向けて地獄度が増す
        for i in range(n):
            hell_factor = (i / n) ** 2  # 二次関数的に増加
            s[i] += self.rng.standard_normal() * intensity * hell_factor
            if self.rng.random() < hell_factor * 0.1:
                s[i] *= self.rng.choice([-1, 1]) * self.rng.uniform(2, 5)
        return s


# ============================================================
# 2. Method Adapters (return detected edges for scoring)
# ============================================================

# Edge format: list of (source, target, lag) tuples


def run_getter_one(df):
    """GETTER One"""
    from getter_one.analysis.network_analyzer_core import NetworkAnalyzerCore
    from getter_one.data.loader import from_dataframe
    from getter_one.structures.lambda_structures_core import LambdaStructuresCore

    dataset = from_dataframe(df, normalize="range")
    core = LambdaStructuresCore()
    structures = core.compute_lambda_structures(
        dataset.state_vectors, window_steps=24,
        dimension_names=list(df.columns),
    )
    analyzer = NetworkAnalyzerCore(
        sync_threshold=0.3, causal_threshold=0.25, max_lag=8,
    )
    network = analyzer.analyze(
        dataset.state_vectors, dimension_names=list(df.columns),
    )

    edges = []
    if network.causal_network:
        for link in network.causal_network:
            edges.append((link.from_name, link.to_name, link.lag))

    return {
        "status": "✅ OK",
        "edges": edges,
        "sync": network.n_sync_links,
        "causal": network.n_causal_links,
        "pattern": network.pattern,
        "hubs": network.hub_names,
        "rho_T_max": float(np.max(structures.get("rho_T", [0]))),
        "sigma_s_mean": float(np.mean(structures.get("sigma_s", [0]))),
    }


def run_var_granger(df):
    """VAR Granger Causality"""
    from statsmodels.tsa.api import VAR

    model = VAR(df)
    fitted = model.fit(maxlags=8, ic="aic")
    cols = list(df.columns)

    # ペアワイズのGranger因果検定
    edges = []
    for i, caused in enumerate(cols):
        for j, causing in enumerate(cols):
            if i == j:
                continue
            test = fitted.test_causality(caused, causing=[causing], kind="f")
            if test.pvalue < 0.05:
                # VARはlagを直接返さないのでAIC選択のラグを使用
                edges.append((causing, caused, fitted.k_ar))

    return {
        "status": "✅ OK",
        "edges": edges,
    }


def run_pcmci(df):
    """PCMCI+"""
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    cols = list(df.columns)
    dataframe = pp.DataFrame(df.values, var_names=cols)
    parcorr = ParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
    results = pcmci.run_pcmciplus(tau_min=0, tau_max=8, pc_alpha=0.05)

    edges = []
    n_vars = len(cols)
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            for tau in range(1, 9):
                if results["p_matrix"][i, j, tau] < 0.05:
                    edges.append((cols[i], cols[j], tau))

    return {
        "status": "✅ OK",
        "edges": edges,
    }


def run_transfer_entropy(df):
    """Transfer Entropy (simplified, multi-lag)"""
    from scipy.stats import entropy

    data = df.values
    cols = list(df.columns)
    n_dims = data.shape[1]
    edges = []

    for lag in range(1, 9):
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue
                n_bins = 10
                x = data[:-lag, i]
                y = data[lag:, j]
                if len(x) < 10:
                    continue

                hist_joint, _, _ = np.histogram2d(x, y, bins=n_bins)
                hist_x, _ = np.histogram(x, bins=n_bins)
                hist_y, _ = np.histogram(y, bins=n_bins)

                p_joint = hist_joint / hist_joint.sum()
                p_x = hist_x / hist_x.sum()
                p_y = hist_y / hist_y.sum()

                te = entropy(p_joint.ravel()) - entropy(p_x) - entropy(p_y)
                if abs(te) > 0.1:
                    edges.append((cols[i], cols[j], lag))

    return {
        "status": "✅ OK",
        "edges": edges,
    }


# ============================================================
# 3. Ground Truth Scoring
# ============================================================


def score_edges(detected_edges, ground_truth, lag_tolerance=1):
    """検出されたエッジをground truthと比較してスコアリング

    Parameters
    ----------
    detected_edges : list of (source, target, lag)
    ground_truth : dict with "edges" and "forbidden"
    lag_tolerance : int, lag一致の許容範囲

    Returns
    -------
    dict with precision, recall, f1, tp, fp, fn, lag_mae, spurious
    """
    gt_edges = ground_truth["edges"]  # [(src, tgt, lag), ...]
    forbidden = ground_truth.get("forbidden", [])  # [(src, tgt), ...]

    # True positives: 方向が合っていてlagが許容範囲内
    tp = 0
    matched_gt = set()
    lag_errors = []

    for d_src, d_tgt, d_lag in detected_edges:
        for gi, (g_src, g_tgt, g_lag) in enumerate(gt_edges):
            if d_src == g_src and d_tgt == g_tgt and gi not in matched_gt:
                if abs(d_lag - g_lag) <= lag_tolerance:
                    tp += 1
                    matched_gt.add(gi)
                    lag_errors.append(abs(d_lag - g_lag))
                    break

    fn = len(gt_edges) - tp

    # False positives: ground truthにないエッジ
    fp = len(detected_edges) - tp

    # Spurious: forbiddenエッジの検出
    n_spurious = 0
    for d_src, d_tgt, _d_lag in detected_edges:
        for f_src, f_tgt in forbidden:
            if d_src == f_src and d_tgt == f_tgt:
                n_spurious += 1
                break

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    lag_mae = float(np.mean(lag_errors)) if lag_errors else float("nan")
    spurious_rate = n_spurious / max(len(detected_edges), 1) if detected_edges else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "lag_mae": lag_mae,
        "n_detected": len(detected_edges),
        "n_spurious": n_spurious,
        "spurious_rate": spurious_rate,
    }


# ============================================================
# 3. Hell Mode Benchmark Runner
# ============================================================


HELL_SCENARIOS = [
    "H1_pulse",
    "H2_phase_jump",
    "H3_bifurcation",
    "H4_cascade",
    "H5_resonance",
    "H6_decay",
    "H7_multi_hell",
    "H8_progressive",
]

METHOD_RUNNERS = {
    "GETTER_One": run_getter_one,
    "VAR_Granger": run_var_granger,
    "PCMCI+": run_pcmci,
    "Transfer_Entropy": run_transfer_entropy,
}


def run_hell_benchmark(n_repeats=5, T=400, n_series=5):
    print("=" * 70)
    print("  🔥🔥🔥 HELL MODE ROBUSTNESS BENCHMARK 🔥🔥🔥")
    print("  Non-stationary, non-linear, non-Gaussian chaos")
    print(f"  Scenarios: {len(HELL_SCENARIOS)} | Methods: {len(METHOD_RUNNERS)}")
    print(f"  Repeats: {n_repeats} | T={T} | n_series={n_series}")
    print("  Question: Which methods survive?")
    print("=" * 70)

    results = []

    for scenario in HELL_SCENARIOS:
        print(f"\n{'─' * 60}")
        print(f"  🔥 {scenario}")
        print(f"{'─' * 60}")

        for repeat in range(n_repeats):
            gen = HellModeGenerator(seed=42 + repeat * 7)
            df, gt = gen.generate_scenario(scenario, T=T, n_series=n_series)

            # データの狂い具合を表示
            if repeat == 0:
                print(f"    Data stats: min={df.values.min():.1f} max={df.values.max():.1f} "
                      f"std={df.values.std():.2f} "
                      f"kurtosis={float(pd.DataFrame(df.values.ravel()).kurtosis().iloc[0]):.1f}")

            for method_name, runner in METHOD_RUNNERS.items():
                row = {
                    "scenario": scenario,
                    "repeat": repeat,
                    "method": method_name,
                }

                t0 = time.time()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        output = runner(df)
                    elapsed = time.time() - t0
                    row["status"] = "✅ survived"
                    row["time_s"] = elapsed
                    row["detail"] = str(output.get("status", ""))

                    # Ground truth scoring
                    detected = output.get("edges", [])
                    scores = score_edges(detected, gt, lag_tolerance=1)
                    row["f1"] = scores["f1"]
                    row["precision"] = scores["precision"]
                    row["recall"] = scores["recall"]
                    row["lag_mae"] = scores["lag_mae"]
                    row["n_detected"] = scores["n_detected"]
                    row["tp"] = scores["tp"]
                    row["fp"] = scores["fp"]
                    row["fn"] = scores["fn"]
                    row["n_spurious"] = scores["n_spurious"]
                    row["spurious_rate"] = scores["spurious_rate"]

                    # GETTER One特有の指標
                    if "rho_T_max" in output:
                        row["rho_T_max"] = output["rho_T_max"]
                        row["sigma_s_mean"] = output["sigma_s_mean"]

                except Exception as e:
                    elapsed = time.time() - t0
                    row["status"] = "💀 CRASHED"
                    row["time_s"] = elapsed
                    row["detail"] = f"{type(e).__name__}: {str(e)[:80]}"

                    if repeat == 0:
                        print(f"    💀 {method_name}: {type(e).__name__}: {str(e)[:60]}")

                results.append(row)

            if repeat == 0:
                for method_name in METHOD_RUNNERS:
                    r = [x for x in results if x["scenario"] == scenario
                         and x["repeat"] == 0 and x["method"] == method_name]
                    if r:
                        status = r[0]["status"]
                        t = r[0]["time_s"]
                        print(f"    {status} {method_name:20s} ({t:.3f}s)")

    # ============================================================
    # Summary
    # ============================================================

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("  SURVIVAL MATRIX")
    print("=" * 70)

    survival = results_df.groupby(["scenario", "method"])["status"].apply(
        lambda x: (x == "✅ survived").sum()
    ).unstack(fill_value=0)
    total = n_repeats
    survival_pct = (survival / total * 100).round(0).astype(int)
    print(survival_pct.to_string())

    print("\n" + "=" * 70)
    print("  OVERALL SURVIVAL RATE")
    print("=" * 70)

    overall = results_df.groupby("method")["status"].apply(
        lambda x: (x == "✅ survived").sum() / len(x) * 100
    ).round(1)
    for method in sorted(overall.index, key=lambda m: overall[m], reverse=True):
        rate = overall[method]
        bar = "█" * int(rate / 5) + "░" * (20 - int(rate / 5))
        emoji = "✅" if rate == 100 else ("⚠️" if rate > 50 else "💀")
        print(f"  {emoji} {method:20s}: {bar} {rate:.1f}%")

    # F1 scoring summary
    survived = results_df[results_df["status"] == "✅ survived"]
    if "f1" in survived.columns and len(survived) > 0:
        print("\n" + "=" * 70)
        print("  CAUSAL DETECTION IN HELL (Ground Truth: X0→X1 lag=2, X0→X2 lag=3)")
        print("=" * 70)

        # Per-method average
        print("\n  Per-Method Average:")
        print(f"  {'Method':20s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s} {'LagMAE':>8s} {'Detect':>8s} {'Spur':>8s}")
        print(f"  {'─' * 72}")
        for method in sorted(survived["method"].unique()):
            m = survived[survived["method"] == method]
            f1 = m["f1"].mean()
            prec = m["precision"].mean()
            rec = m["recall"].mean()
            lag = m["lag_mae"].mean()
            det = m["n_detected"].mean()
            spur = m["spurious_rate"].mean()
            print(f"  {method:20s} {f1:8.3f} {prec:8.3f} {rec:8.3f} {lag:8.3f} {det:8.1f} {spur:8.3f}")

        # Per-scenario × method F1
        print("\n  Per-Scenario F1:")
        pivot_f1 = survived.groupby(["scenario", "method"])["f1"].mean().unstack(fill_value=0)
        print(pivot_f1.to_string(float_format="%.3f"))

        # Mean ± Std
        print("\n  F1 Mean ± Std:")
        for method in sorted(survived["method"].unique()):
            m = survived[survived["method"] == method]
            f1_mean = m["f1"].mean()
            f1_std = m["f1"].std()
            print(f"    {method:20s}: {f1_mean:.3f} ± {f1_std:.3f}")

    # GETTER One detailed stats
    getter_rows = results_df[
        (results_df["method"] == "GETTER_One")
        & (results_df["status"] == "✅ survived")
    ]
    if len(getter_rows) > 0 and "rho_T_max" in getter_rows.columns:
        print("\n" + "=" * 70)
        print("  GETTER ONE — HELL MODE STRUCTURAL STATS")
        print("=" * 70)

        for scenario in HELL_SCENARIOS:
            s_rows = getter_rows[getter_rows["scenario"] == scenario]
            if len(s_rows) > 0:
                rho_max = s_rows["rho_T_max"].mean()
                sigma_mean = s_rows["sigma_s_mean"].mean()
                f1 = s_rows["f1"].mean()
                tp = s_rows["tp"].mean()
                fp = s_rows["fp"].mean()
                print(f"  {scenario:25s}: ρT={rho_max:.3f} σₛ={sigma_mean:.3f} "
                      f"F1={f1:.3f} TP={tp:.1f} FP={fp:.1f}")

    # Crash details
    crashes = results_df[results_df["status"] == "💀 CRASHED"]
    if len(crashes) > 0:
        print("\n" + "=" * 70)
        print("  💀 CRASH LOG")
        print("=" * 70)
        for _, row in crashes.drop_duplicates(subset=["method", "scenario", "detail"]).iterrows():
            print(f"  {row['method']:20s} | {row['scenario']:20s} | {row['detail']}")

    # Save
    csv_path = f"hell_mode_results_{n_repeats}repeats.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  📄 Results saved: {csv_path}")

    return results_df


# ============================================================
# 4. Main
# ============================================================

if __name__ == "__main__":
    import argparse
    import logging

    logging.disable(logging.CRITICAL)

    parser = argparse.ArgumentParser(description="Hell Mode Robustness Benchmark")
    parser.add_argument("-n", "--repeats", type=int, default=5,
                        help="Number of repeats (default: 5)")
    parser.add_argument("-T", type=int, default=400,
                        help="Time series length (default: 400)")
    parser.add_argument("--n-series", type=int, default=5,
                        help="Number of time series (default: 5)")
    args = parser.parse_args()

    run_hell_benchmark(
        n_repeats=args.repeats, T=args.T, n_series=args.n_series,
    )
