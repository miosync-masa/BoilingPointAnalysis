"""
GETTER One × Global Market Tensor
====================================
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
# 1. Global Market Data
# ============================================================

GLOBAL_TICKERS = {
    # 為替
    "USD/JPY": "JPY=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    # 株式指数
    "Nikkei": "^N225",
    "Dow": "^DJI",
    "S&P500": "^GSPC",
    # コモディティ
    "Gold": "GC=F",
    "Oil": "CL=F",
    "Silver": "SI=F",
    # 恐怖指数
    "VIX": "^VIX",
    # 暗号通貨
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}


def fetch_global_market(start="2024-01-01", end="2024-12-31", tickers=None):
    if tickers is None:
        tickers = GLOBAL_TICKERS

    print(f"Fetching {len(tickers)} markets: {start} → {end}...")

    data = yf.download(list(tickers.values()), start=start, end=end)["Close"]
    reversed_map = {v: k for k, v in tickers.items()}
    data = data.rename(columns=reversed_map)
    data = data[list(tickers.keys())].dropna()

    print(f"  ✅ {len(data)} trading days × {len(data.columns)} markets")

    for col in data.columns:
        vals = data[col]
        print(f"    {col:>10s}: {vals.min():.2f} ~ {vals.max():.2f}")

    return data


# ============================================================
# 2. Analysis
# ============================================================

def analyze_market(df, window_steps=20):
    names = list(df.columns)
    n_dims = len(names)

    print(f"\n{'=' * 70}")
    print("  GETTER One × Global Market Tensor")
    print(f"  {len(df)} days × {n_dims} dimensions")
    print(f"  Markets: {', '.join(names)}")
    print(f"{'=' * 70}")

    dataset = from_dataframe(df, normalize="range")

    # Λ³
    print("\n  [1/4] Lambda³ structures...")
    core = LambdaStructuresCore()
    structures = core.compute_lambda_structures(
        dataset.state_vectors, window_steps=window_steps,
        dimension_names=names,
    )

    # ΔΛC events
    lf = structures["lambda_F_mag"]
    rho = structures["rho_T"]
    sigma = structures["sigma_s"]
    n = min(len(lf), len(rho) - 1, len(sigma) - 1)
    dlc = rho[1:n + 1] * sigma[1:n + 1] * lf[:n]
    threshold = np.mean(dlc) + 2 * np.std(dlc)
    event_frames = np.where(dlc > threshold)[0]

    print(f"    ΔΛC events (2σ): {len(event_frames)} detected")
    if len(event_frames) > 0 and hasattr(df, 'index'):
        print("    Top 5 events:")
        top_idx = np.argsort(dlc[event_frames])[-5:][::-1]
        for idx in top_idx:
            frame = event_frames[idx]
            if frame < len(df.index):
                date = df.index[frame]
                print(f"      {date}: ΔΛC={dlc[frame]:.4f}")

    # Network
    print("\n  [2/4] Causal network...")
    analyzer = NetworkAnalyzerCore(
        sync_threshold=0.2, causal_threshold=0.15, max_lag=5,
    )
    network = analyzer.analyze(dataset.state_vectors, dimension_names=names)

    print(f"    Pattern: {network.pattern}")
    print(f"    Sync: {network.n_sync_links}, Causal: {network.n_causal_links}")

    if network.sync_network:
        print("\n    Sync Network:")
        for link in sorted(network.sync_network,
                          key=lambda lnk: lnk.strength, reverse=True)[:15]:
            sign = "+" if link.correlation > 0 else "−"
            print(f"      {link.from_name:>10s} ↔ {link.to_name:<10s}: "
                  f"{sign}{link.strength:.3f}")

    if network.causal_network:
        print("\n    Causal Network:")
        for link in sorted(network.causal_network,
                          key=lambda lnk: lnk.strength, reverse=True)[:15]:
            print(f"      {link.from_name:>10s} → {link.to_name:<10s}: "
                  f"{link.strength:.3f} (lag={link.lag}d)")

    if network.hub_names:
        print(f"\n    Hubs: {', '.join(network.hub_names)}")
    if network.driver_names:
        print(f"    Drivers: {', '.join(network.driver_names)}")
    if network.follower_names:
        print(f"    Followers: {', '.join(network.follower_names)}")

    # Confidence
    print("\n  [3/4] Confidence assessment...")
    confidence = assess_confidence(
        state_vectors=dataset.state_vectors,
        lambda_structures=structures,
        network_result=network,
        dimension_names=names,
        n_permutations=500,
        n_bootstrap=1000,
    )

    sig_sync = [s for s in confidence.sync_links if s.is_significant]
    sig_causal = [c for c in confidence.causal_links if c.is_significant]
    print(f"    Significant sync: {len(sig_sync)}/{len(confidence.sync_links)}")
    print(f"    Significant causal: {len(sig_causal)}/{len(confidence.causal_links)}")

    # Report
    print("\n  [4/4] Report generation...")
    report = generate_report(
        lambda_structures=structures,
        network_result=network,
        confidence_report=confidence,
        dataset=dataset,
        title="GETTER One × Global Market Tensor Analysis",
        output_path="global_market_report.md",
    )

    print("\n  ✅ Complete!")

    return {
        "dataset": dataset,
        "structures": structures,
        "network": network,
        "confidence": confidence,
        "report": report,
        "delta_lambda_c": dlc,
        "event_frames": event_frames,
    }


# ============================================================
# 3. Main
# ============================================================

if __name__ == "__main__":
    # Full global market
    df = fetch_global_market("2024-01-01", "2024-12-31")
    df.to_csv("global_market_2024.csv")

    results = analyze_market(df, window_steps=20)
    print("\n" + results["report"])
