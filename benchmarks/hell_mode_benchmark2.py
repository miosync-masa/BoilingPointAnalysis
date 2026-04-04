"""
GETTER One — Hell Mode Robustness Benchmark (Emperor Edition)
==============================================================

Non-stationary, non-linear, non-Gaussian data with:
  - Phase jumps, bifurcations, cascades
  - Resonance amplification, topological jumps
  - Multi-path interactions, structural decay

Question: "Which methods survive?"

Built with 💕 by Masamichi & Tamaki
"""

import time
import warnings

import numpy as np
import pandas as pd

_HAS_STATSMODELS = False
try:
    from statsmodels.tsa.api import VAR  # noqa: F401

    _HAS_STATSMODELS = True
except ImportError:
    pass

_HAS_TIGRAMITE = False
try:
    import tigramite.data_processing as pp  # noqa: F401
    from tigramite.independence_tests.parcorr import ParCorr  # noqa: F401
    from tigramite.pcmci import PCMCI  # noqa: F401

    _HAS_TIGRAMITE = True
except ImportError:
    pass

from getter_one.data.loader import from_dataframe  # noqa: E402
from getter_one.pipeline import PipelineConfig  # noqa: E402
from getter_one.pipeline import run as getter_run  # noqa: E402


# ============================================================
# 1. Hell Mode Data Generator
# ============================================================


class HellModeGenerator:
    """地獄モードデータ生成器"""

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

    def _multi_hell(self, series, intensity=3.0):
        s = series.copy()
        s = self._pulse(s, intensity)
        s = self._phase_jump(s, intensity * 0.5)
        s = self._cascade(s, intensity * 0.7)
        return s

    def _progressive_hell(self, series, intensity=3.0):
        s = series.copy()
        n = len(s)
        for i in range(n):
            hell_factor = (i / n) ** 2
            s[i] += self.rng.standard_normal() * intensity * hell_factor
            if self.rng.random() < hell_factor * 0.1:
                s[i] *= self.rng.choice([-1, 1]) * self.rng.uniform(2, 5)
        return s

    def generate_scenario(self, name, T=400, n_series=5):
        noise_scale = 0.5
        A = np.zeros(T)
        B = np.zeros(T)
        C = np.zeros(T)

        for t in range(T):
            A[t] = (
                0.6 * (A[t - 1] if t > 0 else 0)
                + self.rng.standard_normal() * noise_scale
            )
            if t >= 2:
                B[t] = (
                    0.5 * (B[t - 1] if t > 0 else 0)
                    + 0.7 * A[t - 2]
                    + self.rng.standard_normal() * noise_scale
                )
            if t >= 3:
                C[t] = (
                    0.4 * (C[t - 1] if t > 0 else 0)
                    + 0.6 * A[t - 3]
                    + self.rng.standard_normal() * noise_scale
                )

        extras = []
        for _ in range(n_series - 3):
            s = np.zeros(T)
            for t in range(1, T):
                s[t] = 0.5 * s[t - 1] + self.rng.standard_normal() * noise_scale
            extras.append(s)

        all_series = [A, B, C] + extras

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

        corrupted = []
        for s in all_series:
            result = hell_functions[name](s)
            corrupted.append(np.real(result).astype(np.float64))

        data = np.column_stack(corrupted).astype(np.float64)
        data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        columns = [f"X{i}" for i in range(data.shape[1])]

        gt = {
            "edges": [("X0", "X1", 2), ("X0", "X2", 3)],
            "forbidden": [("X1", "X0"), ("X2", "X0"), ("X1", "X2"), ("X2", "X1")],
        }

        return pd.DataFrame(data=data, columns=columns, dtype=np.float64), gt


# ============================================================
# 2. Method Runners
# ============================================================


def run_getter_emperor(df):
    """GETTER One Emperor Mode (v0.1.9 全武装)"""
    config = PipelineConfig(
        window_steps=min(24, max(5, len(df) // 10)),
        enable_boundary=True,
        enable_topology=True,
        enable_anomaly=True,
        enable_extended=True,
        enable_phase_space=True,
        enable_network=True,
        sync_threshold=0.3,
        causal_threshold=0.25,
        max_lag=8,
        adaptive_network=True,
        enable_confidence=False,
        enable_report=False,
        verbose=False,
    )

    dataset = from_dataframe(df, normalize="range")
    result = getter_run(dataset, config=config)
    network = result.network
    structures = result.lambda_structures

    edges = []
    if network and network.causal_network:
        for link in network.causal_network:
            edges.append((link.from_name, link.to_name, link.lag))

    return {
        "status": "✅ OK",
        "edges": edges,
        "sync": network.n_sync_links if network else 0,
        "causal": network.n_causal_links if network else 0,
        "pattern": network.pattern if network else "none",
        "hubs": network.hub_names if network else [],
        "adaptive": network.adaptive_params if network else None,
        "rho_T_max": float(np.max(structures.get("rho_T", [0]))),
        "sigma_s_mean": float(np.mean(structures.get("sigma_s", [0]))),
    }


def run_var_granger(df):
    """VAR Granger Causality"""
    from statsmodels.tsa.api import VAR

    model = VAR(df)
    fitted = model.fit(maxlags=8, ic="aic")
    cols = list(df.columns)

    edges = []
    for i, caused in enumerate(cols):
        for j, causing in enumerate(cols):
            if i == j:
                continue
            test = fitted.test_causality(caused, causing=[causing], kind="f")
            if test.pvalue < 0.05:
                edges.append((causing, caused, fitted.k_ar))

    return {"status": "✅ OK", "edges": edges}


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

    return {"status": "✅ OK", "edges": edges}


def run_transfer_entropy(df):
    """Transfer Entropy"""
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

    return {"status": "✅ OK", "edges": edges}


# ============================================================
# 3. Ground Truth Scoring
# ============================================================


def score_edges(detected_edges, ground_truth, lag_tolerance=1):
    gt_edges = ground_truth["edges"]
    forbidden = ground_truth.get("forbidden", [])

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
    fp = len(detected_edges) - tp

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
# 4. Hell Mode Runner
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
    "GETTER_Emperor": run_getter_emperor,
    "VAR_Granger": run_var_granger,
    "PCMCI+": run_pcmci,
    "Transfer_Entropy": run_transfer_entropy,
}


def run_hell_benchmark(n_repeats=5, T=400, n_series=5):
    print("=" * 70)
    print("  🔥🔥🔥 HELL MODE ROBUSTNESS BENCHMARK 🔥🔥🔥")
    print("  ⚡ GETTER ONE: EMPEROR MODE (v0.1.9 全武装) ⚡")
    print(f"  Scenarios: {len(HELL_SCENARIOS)} | Methods: {len(METHOD_RUNNERS)}")
    print(f"  Repeats: {n_repeats} | T={T} | n_series={n_series}")
    print("  Question: Which methods survive the inferno?")
    print("=" * 70)

    results = []

    for scenario in HELL_SCENARIOS:
        print(f"\n{'─' * 60}")
        print(f"  🔥 {scenario}")
        print(f"{'─' * 60}")

        for repeat in range(n_repeats):
            gen = HellModeGenerator(seed=42 + repeat * 7)
            df, gt = gen.generate_scenario(scenario, T=T, n_series=n_series)

            if repeat == 0:
                print(
                    f"    Data stats: min={df.values.min():.1f} "
                    f"max={df.values.max():.1f} "
                    f"std={df.values.std():.2f} "
                    f"kurtosis={float(pd.DataFrame(df.values.ravel()).kurtosis().iloc[0]):.1f}"
                )

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

                    detected = output.get("edges", [])
                    scores = score_edges(detected, gt, lag_tolerance=1)
                    row.update(scores)

                    if "rho_T_max" in output:
                        row["rho_T_max"] = output["rho_T_max"]
                        row["sigma_s_mean"] = output["sigma_s_mean"]

                except Exception as e:
                    elapsed = time.time() - t0
                    row["status"] = "💀 CRASHED"
                    row["time_s"] = elapsed
                    row["detail"] = f"{type(e).__name__}: {str(e)[:80]}"
                    if repeat == 0:
                        print(
                            f"    💀 {method_name}: {type(e).__name__}: {str(e)[:60]}"
                        )

                results.append(row)

            if repeat == 0:
                for method_name in METHOD_RUNNERS:
                    r = [
                        x
                        for x in results
                        if x["scenario"] == scenario
                        and x["repeat"] == 0
                        and x["method"] == method_name
                    ]
                    if r:
                        status = r[0]["status"]
                        t = r[0]["time_s"]
                        f1 = r[0].get("f1", "N/A")
                        f1_str = f"F1={f1:.3f}" if isinstance(f1, float) else "CRASHED"
                        print(f"    {status} {method_name:20s} ({t:.3f}s) {f1_str}")

    # ── Summary ──
    results_df = pd.DataFrame(results)
    survived = results_df[results_df["status"] == "✅ survived"]

    print("\n" + "=" * 70)
    print("  SURVIVAL RATE")
    print("=" * 70)
    overall = (
        results_df.groupby("method")["status"]
        .apply(lambda x: (x == "✅ survived").sum() / len(x) * 100)
        .round(1)
    )
    for method in sorted(overall.index, key=lambda m: overall[m], reverse=True):
        rate = overall[method]
        bar = "█" * int(rate / 5) + "░" * (20 - int(rate / 5))
        emoji = "✅" if rate == 100 else ("⚠️" if rate > 50 else "💀")
        print(f"  {emoji} {method:20s}: {bar} {rate:.1f}%")

    if "f1" in survived.columns and len(survived) > 0:
        print("\n" + "=" * 70)
        print("  CAUSAL DETECTION IN HELL (GT: X0→X1 lag=2, X0→X2 lag=3)")
        print("=" * 70)

        print(
            f"\n  {'Method':20s} {'F1':>8s} {'Prec':>8s} {'Rec':>8s} "
            f"{'LagMAE':>8s} {'Detect':>8s} {'Spur':>8s} {'Time':>8s}"
        )
        print(f"  {'─' * 80}")
        for method in sorted(survived["method"].unique()):
            m = survived[survived["method"] == method]
            print(
                f"  {method:20s} {m['f1'].mean():8.3f} "
                f"{m['precision'].mean():8.3f} {m['recall'].mean():8.3f} "
                f"{m['lag_mae'].mean():8.3f} {m['n_detected'].mean():8.1f} "
                f"{m['spurious_rate'].mean():8.3f} {m['time_s'].mean():8.3f}"
            )

        # F1 Mean ± Std
        print("\n  F1 Mean ± Std:")
        for method in sorted(survived["method"].unique()):
            m = survived[survived["method"] == method]
            print(f"    {method:20s}: {m['f1'].mean():.3f} ± {m['f1'].std():.3f}")

        # Per-Scenario F1
        print("\n  Per-Scenario F1:")
        pivot = (
            survived.groupby(["scenario", "method"])["f1"].mean().unstack(fill_value=0)
        )
        print(pivot.to_string(float_format="%.3f"))

        # Composite
        print("\n" + "=" * 70)
        print("  🏆 HELL MODE COMPOSITE SCORE")
        print("=" * 70)
        composite = {}
        for method in survived["method"].unique():
            m = survived[survived["method"] == method]
            f1 = m["f1"].mean()
            prec = m["precision"].mean()
            lag = m["lag_mae"].mean()
            lag_s = 1 / (1 + lag) if not np.isnan(lag) else 0
            spur = m["spurious_rate"].mean()
            spur_s = 1 - spur
            surv = overall.get(method, 0) / 100
            c = (f1 * 2.0 + prec * 1.5 + lag_s * 1.0 + spur_s * 1.0 + surv * 0.5) / 6.0
            composite[method] = c
            print(
                f"  {method:20s}: {c:.3f}  "
                f"(F1={f1:.3f} Prec={prec:.3f} Lag={lag_s:.3f} "
                f"Spur={spur_s:.3f} Surv={surv:.1f})"
            )

        print("\n  🏆 Ranking:")
        for rank, (method, score) in enumerate(
            sorted(composite.items(), key=lambda x: x[1], reverse=True), 1
        ):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
            print(f"    {medal} {rank}. {method}: {score:.3f}")

    # GETTER structural stats
    getter_rows = survived[survived["method"] == "GETTER_Emperor"]
    if len(getter_rows) > 0 and "rho_T_max" in getter_rows.columns:
        print("\n" + "=" * 70)
        print("  GETTER EMPEROR — HELL STRUCTURAL STATS")
        print("=" * 70)
        for scenario in HELL_SCENARIOS:
            s_rows = getter_rows[getter_rows["scenario"] == scenario]
            if len(s_rows) > 0:
                print(
                    f"  {scenario:25s}: "
                    f"ρT={s_rows['rho_T_max'].mean():.3f} "
                    f"σₛ={s_rows['sigma_s_mean'].mean():.3f} "
                    f"F1={s_rows['f1'].mean():.3f} "
                    f"TP={s_rows['tp'].mean():.1f} FP={s_rows['fp'].mean():.1f}"
                )

    # Crash log
    crashes = results_df[results_df["status"] == "💀 CRASHED"]
    if len(crashes) > 0:
        print("\n" + "=" * 70)
        print("  💀 CRASH LOG")
        print("=" * 70)
        for _, row in crashes.drop_duplicates(
            subset=["method", "scenario", "detail"]
        ).iterrows():
            print(
                f"  {row['method']:20s} | {row['scenario']:20s} | {row.get('detail', 'unknown')}"
            )

    csv_path = f"hell_emperor_{n_repeats}repeats.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  📄 Results saved: {csv_path}")

    return results_df


if __name__ == "__main__":
    import argparse
    import logging

    logging.disable(logging.CRITICAL)

    parser = argparse.ArgumentParser(description="Hell Mode Emperor Benchmark")
    parser.add_argument("-n", "--repeats", type=int, default=5)
    parser.add_argument("-T", type=int, default=400)
    parser.add_argument("--n-series", type=int, default=5)
    args = parser.parse_args()

    run_hell_benchmark(n_repeats=args.repeats, T=args.T, n_series=args.n_series)
