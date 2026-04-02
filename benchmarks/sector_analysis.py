"""
GETTER One × Japanese Equity Sector Network
=============================================
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
# 1. Sector Representatives
# ============================================================

# 各セクターから代表銘柄を選定
SECTOR_TICKERS = {
    # 半導体/AI
    "Tokyo_Electron": "8035.T",
    "Advantest": "6857.T",
    "Renesas": "6723.T",
    # 自動車
    "Toyota": "7203.T",
    "Honda": "7267.T",
    "Nissan": "7201.T",
    # 銀行
    "MUFG": "8306.T",
    "SMFG": "8316.T",
    "Mizuho": "8411.T",
    # 不動産
    "Mitsui_Fudosan": "8801.T",
    "Mitsubishi_Estate": "8802.T",
    # 商社
    "Mitsubishi_Corp": "8058.T",
    "Itochu": "8001.T",
    "Mitsui_Bussan": "8031.T",
    # 医薬
    "Takeda": "4502.T",
    "Daiichi_Sankyo": "4568.T",
    # 通信
    "NTT": "9432.T",
    "SoftBank_G": "9984.T",
    "KDDI": "9433.T",
    # 小売/サービス
    "Fast_Retailing": "9983.T",
    "Seven_i": "3382.T",
    # 電力
    "TEPCO": "9501.T",
    "Kansai_Electric": "9503.T",
    # 鉄鋼/素材
    "Nippon_Steel": "5401.T",
    "Shin_Etsu": "4063.T",
}

# セクターラベル（解析後のグルーピング用）
SECTOR_MAP = {
    "Tokyo_Electron": "Semiconductor", "Advantest": "Semiconductor", "Renesas": "Semiconductor",
    "Toyota": "Auto", "Honda": "Auto", "Nissan": "Auto",
    "MUFG": "Bank", "SMFG": "Bank", "Mizuho": "Bank",
    "Mitsui_Fudosan": "RealEstate", "Mitsubishi_Estate": "RealEstate",
    "Mitsubishi_Corp": "Trading", "Itochu": "Trading", "Mitsui_Bussan": "Trading",
    "Takeda": "Pharma", "Daiichi_Sankyo": "Pharma",
    "NTT": "Telecom", "SoftBank_G": "Telecom", "KDDI": "Telecom",
    "Fast_Retailing": "Retail", "Seven_i": "Retail",
    "TEPCO": "Utility", "Kansai_Electric": "Utility",
    "Nippon_Steel": "Materials", "Shin_Etsu": "Materials",
}


def fetch_sector_data(start="2024-01-01", end="2024-12-31"):
    print(f"Fetching {len(SECTOR_TICKERS)} stocks: {start} → {end}...")

    data = yf.download(
        list(SECTOR_TICKERS.values()),
        start=start, end=end,
    )["Close"]

    reversed_map = {v: k for k, v in SECTOR_TICKERS.items()}
    data = data.rename(columns=reversed_map)
    data = data[list(SECTOR_TICKERS.keys())].dropna()

    print(f"  ✅ {len(data)} trading days × {len(data.columns)} stocks")
    return data


# ============================================================
# 2. Analysis
# ============================================================

def analyze_sectors(df, window_steps=20, max_lag=10):
    names = list(df.columns)
    n_dims = len(names)

    print(f"\n{'=' * 70}")
    print("  GETTER One × Japanese Equity Sector Network")
    print(f"  {len(df)} days × {n_dims} stocks across {len(set(SECTOR_MAP.values()))} sectors")
    print(f"{'=' * 70}")

    dataset = from_dataframe(df, normalize="range")

    # Λ³
    print(f"\n  [1/4] Lambda³ structures (window={window_steps})...")
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
    if len(event_frames) > 0:
        print("    Top events:")
        top_idx = np.argsort(dlc[event_frames])[-10:][::-1]
        for idx in top_idx:
            frame = event_frames[idx]
            if frame < len(df.index):
                print(f"      {df.index[frame]}: ΔΛC={dlc[frame]:.4f}")

    # Network
    print(f"\n  [2/4] Causal network (max_lag={max_lag})...")
    analyzer = NetworkAnalyzerCore(
        sync_threshold=0.25, causal_threshold=0.2, max_lag=max_lag,
    )
    network = analyzer.analyze(dataset.state_vectors, dimension_names=names)

    print(f"    Pattern: {network.pattern}")
    print(f"    Sync: {network.n_sync_links}, Causal: {network.n_causal_links}")

    # セクター間因果の集計
    if network.causal_network:
        print("\n    Cross-Sector Causal Links:")
        cross_sector = []
        intra_sector = []
        for link in sorted(network.causal_network,
                          key=lambda lnk: lnk.strength, reverse=True):
            src_sector = SECTOR_MAP.get(link.from_name, "?")
            dst_sector = SECTOR_MAP.get(link.to_name, "?")
            if src_sector != dst_sector:
                cross_sector.append(link)
                print(f"      [{src_sector}] {link.from_name} → "
                      f"[{dst_sector}] {link.to_name}: "
                      f"{link.strength:.3f} (lag={link.lag}d)")
            else:
                intra_sector.append(link)

        print(f"\n    Cross-sector: {len(cross_sector)} links")
        print(f"    Intra-sector: {len(intra_sector)} links")

        # セクター間因果行列
        sorted(set(SECTOR_MAP.values()))
        print("\n    Sector-level Causal Flow:")
        sector_flows = {}
        for link in cross_sector:
            src = SECTOR_MAP.get(link.from_name, "?")
            dst = SECTOR_MAP.get(link.to_name, "?")
            key = f"{src} → {dst}"
            if key not in sector_flows:
                sector_flows[key] = []
            sector_flows[key].append(link.strength)

        for flow, strengths in sorted(sector_flows.items(),
                                      key=lambda x: np.mean(x[1]),
                                      reverse=True):
            print(f"      {flow}: avg={np.mean(strengths):.3f} "
                  f"(n={len(strengths)})")

    # 同期クラスタ
    if network.sync_network:
        print("\n    Top Sync Links (intra-sector expected):")
        for link in sorted(network.sync_network,
                          key=lambda lnk: lnk.strength, reverse=True)[:15]:
            src_s = SECTOR_MAP.get(link.from_name, "?")
            dst_s = SECTOR_MAP.get(link.to_name, "?")
            tag = "INTRA" if src_s == dst_s else "CROSS"
            sign = "+" if link.correlation > 0 else "−"
            print(f"      [{tag}] {link.from_name} ↔ {link.to_name}: "
                  f"{sign}{link.strength:.3f} ({src_s}↔{dst_s})")

    if network.hub_names:
        print(f"\n    Hubs: {', '.join(network.hub_names)}")
    if network.driver_names:
        print(f"    Drivers: {', '.join(network.driver_names)}")
    if network.follower_names:
        print(f"    Followers: {', '.join(network.follower_names)}")

    # Confidence
    print("\n  [3/4] Confidence (perm=500)...")
    confidence = assess_confidence(
        state_vectors=dataset.state_vectors,
        lambda_structures=structures,
        network_result=network,
        dimension_names=names,
        n_permutations=500,
        n_bootstrap=1000,
    )
    sig_sync = sum(1 for s in confidence.sync_links if s.is_significant)
    sig_causal = sum(1 for c in confidence.causal_links if c.is_significant)
    print(f"    Significant sync: {sig_sync}/{len(confidence.sync_links)}")
    print(f"    Significant causal: {sig_causal}/{len(confidence.causal_links)}")

    # Report
    print("\n  [4/4] Report...")
    report = generate_report(
        lambda_structures=structures,
        network_result=network,
        confidence_report=confidence,
        dataset=dataset,
        title="GETTER One × Japanese Equity Sector Network",
        output_path="sector_network_report.md",
    )

    print("\n  ✅ Complete!")
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
    df = fetch_sector_data("2024-01-01", "2024-12-31")
    df.to_csv("sector_data_2024.csv")

    results = analyze_sectors(df, window_steps=20, max_lag=10)
    print("\n" + results["report"])
