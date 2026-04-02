"""
GETTER One × Financial Markets
================================
  USD/JPY, JPY/GBP, GBP/USD, Nikkei 225, Dow Jones

Usage (Colab):
  !pip install getter-one yfinance
  # then run this script
"""

import numpy as np
import yfinance as yf

from getter_one.analysis.confidence_kit import assess_confidence
from getter_one.analysis.network_analyzer_core import NetworkAnalyzerCore
from getter_one.analysis.report_generator import generate_report
from getter_one.data.loader import from_dataframe
from getter_one.structures.lambda_structures_core import LambdaStructuresCore

# ============================================================
# 1. Fetch Financial Data
# ============================================================

def fetch_financial_data(
    start_date="2024-01-01",
    end_date="2024-12-31",
):
    print(f"Fetching market data: {start_date} → {end_date}...")

    tickers = {
        "USD/JPY": "JPY=X",
        "GBP/USD": "GBPUSD=X",
        "GBP/JPY": "GBPJPY=X",
        "Nikkei 225": "^N225",
        "Dow Jones": "^DJI",
    }

    data_close = yf.download(
        list(tickers.values()),
        start=start_date,
        end=end_date,
    )["Close"]

    # JPY/GBP = 1 / GBP/JPY
    data_close["JPY/GBP"] = 1 / data_close["GBPJPY=X"]
    data_close = data_close.drop(columns=["GBPJPY=X"])

    reversed_tickers = {v: k for k, v in tickers.items()}
    final_data = data_close.rename(columns=reversed_tickers)

    desired_order = ["USD/JPY", "JPY/GBP", "GBP/USD", "Nikkei 225", "Dow Jones"]
    final_data = final_data[desired_order].dropna()

    print(f"  ✅ {len(final_data)} trading days, {len(desired_order)} dimensions")
    return final_data


# ============================================================
# 2. GETTER One Analysis
# ============================================================

def run_analysis(df, window_steps=20):
    print(f"\n{'=' * 60}")
    print("  GETTER One × Financial Market Analysis")
    print(f"  {len(df)} days × {len(df.columns)} dimensions")
    print(f"{'=' * 60}")

    names = list(df.columns)
    dataset = from_dataframe(df, normalize="range")

    # Λ³ structures
    print("\n  [1/4] Computing Lambda³ structures...")
    core = LambdaStructuresCore()
    structures = core.compute_lambda_structures(
        dataset.state_vectors,
        window_steps=window_steps,
        dimension_names=names,
    )

    # Print Λ³ stats
    for key in ["lambda_F_mag", "rho_T", "sigma_s"]:
        if key in structures:
            v = structures[key]
            print(f"    {key}: mean={np.mean(v):.4f} max={np.max(v):.4f}")

    # Network
    print("\n  [2/4] Extracting causal network...")
    analyzer = NetworkAnalyzerCore(
        sync_threshold=0.25,
        causal_threshold=0.2,
        max_lag=5,
    )
    network = analyzer.analyze(dataset.state_vectors, dimension_names=names)

    print(f"    Pattern: {network.pattern}")
    print(f"    Sync links: {network.n_sync_links}")
    print(f"    Causal links: {network.n_causal_links}")

    if network.sync_network:
        print("\n    Sync Network:")
        for link in sorted(network.sync_network, key=lambda l: l.strength, reverse=True):
            sign = "+" if link.correlation > 0 else "−"
            print(f"      {link.from_name} ↔ {link.to_name}: {sign}{link.strength:.3f}")

    if network.causal_network:
        print("\n    Causal Network:")
        for link in sorted(network.causal_network, key=lambda l: l.strength, reverse=True):
            print(f"      {link.from_name} → {link.to_name}: "
                  f"{link.strength:.3f} (lag={link.lag} days)")

    if network.hub_names:
        print(f"\n    Hubs: {', '.join(network.hub_names)}")
    if network.driver_names:
        print(f"    Drivers: {', '.join(network.driver_names)}")
    if network.follower_names:
        print(f"    Followers: {', '.join(network.follower_names)}")

    # Confidence
    print("\n  [3/4] Assessing confidence...")
    confidence = assess_confidence(
        state_vectors=dataset.state_vectors,
        lambda_structures=structures,
        network_result=network,
        dimension_names=names,
        n_permutations=500,
        n_bootstrap=1000,
    )

    if confidence.causal_links:
        print("\n    Causal Link Confidence:")
        for c in confidence.causal_links:
            sig = "✅" if c.is_significant else "❌"
            print(f"      {sig} {c.from_name} → {c.to_name} (lag={c.lag}): "
                  f"p={c.p_value:.4f} CI=[{c.ci_lower:.3f}, {c.ci_upper:.3f}]")

    if confidence.sync_links:
        print("\n    Sync Link Confidence:")
        for s in confidence.sync_links:
            sig = "✅" if s.is_significant else "❌"
            print(f"      {sig} {s.name_i} ↔ {s.name_j}: "
                  f"r={s.observed_correlation:.3f} p={s.p_value:.4f}")

    # Report
    print("\n  [4/4] Generating report...")
    report = generate_report(
        lambda_structures=structures,
        network_result=network,
        confidence_report=confidence,
        dataset=dataset,
        title="GETTER One × Financial Market Analysis",
        output_path="financial_report.md",
    )

    print("\n  ✅ Analysis complete!")
    print("  📄 Report saved: financial_report.md")

    return {
        "dataset": dataset,
        "structures": structures,
        "network": network,
        "confidence": confidence,
        "report": report,
    }


# ============================================================
# 3. Main
# ============================================================

if __name__ == "__main__":
    # Fetch data
    df = fetch_financial_data(
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    # Save raw data
    df.to_csv("financial_data_2024.csv")
    print("\n  Raw data saved: financial_data_2024.csv")

    # Run GETTER One
    results = run_analysis(df, window_steps=20)

    # Print full report
    print("\n" + results["report"])
