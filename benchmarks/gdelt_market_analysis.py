"""
GETTER One × GDELT × Financial Markets
=========================================
Usage (Colab):
  !pip install getter-one yfinance gdeltdoc
"""

import numpy as np
import pandas as pd
import yfinance as yf
from gdeltdoc import Filters, GdeltDoc

from getter_one.analysis.confidence_kit import assess_confidence
from getter_one.analysis.network_analyzer_core import NetworkAnalyzerCore
from getter_one.data.loader import from_dataframe
from getter_one.structures.lambda_structures_core import LambdaStructuresCore

# ============================================================
# 1. GDELT News Tensor
# ============================================================

def fetch_gdelt_timelines(start_date, end_date, keywords=None):
    """GDELTからニューストーン＋ボリュームを取得"""
    if keywords is None:
        keywords = {
            "BOJ": "Bank of Japan",
            "Fed": "Federal Reserve",
            "Trade_War": "trade war tariff",
            "AI_Tech": "artificial intelligence technology",
            "Oil_Crisis": "oil price crude",
            "Crypto": "bitcoin cryptocurrency",
        }

    gd = GdeltDoc()
    all_data = {}

    print(f"Fetching GDELT timelines: {start_date} → {end_date}")
    print(f"  Keywords: {len(keywords)}")

    for label, kw in keywords.items():
        try:
            f = Filters(keyword=kw, start_date=start_date, end_date=end_date)

            # トーン（感情スコア）
            tone = gd.timeline_search("timelinetone", f)
            if tone is not None and len(tone) > 0:
                tone = tone.set_index("datetime")
                tone.index = pd.to_datetime(tone.index).date
                all_data[f"{label}_tone"] = tone["Average Tone"]
                print(f"    ✅ {label} tone: {len(tone)} days")

            # ボリューム（報道量）
            vol = gd.timeline_search("timelinevol", f)
            if vol is not None and len(vol) > 0:
                vol = vol.set_index("datetime")
                vol.index = pd.to_datetime(vol.index).date
                all_data[f"{label}_vol"] = vol["Volume Intensity"]
                print(f"    ✅ {label} vol: {len(vol)} days")

        except Exception as e:
            print(f"    ❌ {label}: {e}")

    if not all_data:
        print("  No GDELT data retrieved!")
        return None

    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    print(f"\n  GDELT tensor: {len(df)} days × {len(df.columns)} dimensions")
    return df


# ============================================================
# 2. Financial Market Tensor
# ============================================================

def fetch_market_data(start_date, end_date, tickers=None):
    """yfinanceから金融データを取得"""
    if tickers is None:
        tickers = {
            "USD/JPY": "JPY=X",
            "Nikkei": "^N225",
            "S&P500": "^GSPC",
            "Gold": "GC=F",
            "VIX": "^VIX",
            "BTC": "BTC-USD",
        }

    print(f"\nFetching market data: {start_date} → {end_date}")
    data = yf.download(list(tickers.values()), start=start_date, end=end_date)["Close"]

    reversed_map = {v: k for k, v in tickers.items()}
    data = data.rename(columns=reversed_map)
    data = data[list(tickers.keys())].dropna()
    data.index = pd.to_datetime(data.index).date
    data.index = pd.to_datetime(data.index)

    print(f"  Market tensor: {len(data)} days × {len(data.columns)} dimensions")
    return data


# ============================================================
# 3. Merge & Analyze
# ============================================================

def analyze_news_market(gdelt_df, market_df, window_steps=15, max_lag=5):
    """ニューステンソル × 市場テンソルの統合解析"""
    # 日付でマージ
    merged = pd.merge(
        gdelt_df, market_df,
        left_index=True, right_index=True,
        how="inner",
    ).dropna()

    names = list(merged.columns)
    n_news = len(gdelt_df.columns)
    n_market = len(market_df.columns)

    print(f"\n{'=' * 70}")
    print("  GETTER One × News-Market Tensor Analysis")
    print(f"  {len(merged)} days × {len(names)} dimensions")
    print(f"  News dims: {n_news} | Market dims: {n_market}")
    print(f"{'=' * 70}")

    dataset = from_dataframe(merged, normalize="range")

    # Λ³
    print(f"\n  [1/3] Lambda³ (window={window_steps})...")
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

    if len(event_frames) > 0:
        print(f"    ΔΛC events: {len(event_frames)}")
        top_idx = np.argsort(dlc[event_frames])[-5:][::-1]
        for idx in top_idx:
            frame = event_frames[idx]
            if frame < len(merged.index):
                print(f"      {merged.index[frame].date()}: ΔΛC={dlc[frame]:.4f}")

    # Network
    print(f"\n  [2/3] Causal network (max_lag={max_lag})...")
    analyzer = NetworkAnalyzerCore(
        sync_threshold=0.2, causal_threshold=0.15, max_lag=max_lag,
    )
    network = analyzer.analyze(dataset.state_vectors, dimension_names=names)

    # ニュース→市場の因果だけ抽出！
    news_cols = set(gdelt_df.columns)
    market_cols = set(market_df.columns)

    print("\n    === NEWS → MARKET Causal Links ===")
    news_to_market = []
    market_to_news = []
    for link in sorted(network.causal_network,
                      key=lambda lnk: lnk.strength, reverse=True):
        src_is_news = link.from_name in news_cols
        dst_is_market = link.to_name in market_cols
        src_is_market = link.from_name in market_cols
        dst_is_news = link.to_name in news_cols

        if src_is_news and dst_is_market:
            news_to_market.append(link)
            print(f"      📰→💰 {link.from_name} → {link.to_name}: "
                  f"{link.strength:.3f} (lag={link.lag}d)")
        elif src_is_market and dst_is_news:
            market_to_news.append(link)

    print("\n    === MARKET → NEWS Causal Links ===")
    for link in sorted(market_to_news,
                      key=lambda lnk: lnk.strength, reverse=True):
        print(f"      💰→📰 {link.from_name} → {link.to_name}: "
              f"{link.strength:.3f} (lag={link.lag}d)")

    print(f"\n    News → Market: {len(news_to_market)} links")
    print(f"    Market → News: {len(market_to_news)} links")

    if network.sync_network:
        print("\n    === Cross-Domain Sync ===")
        for link in sorted(network.sync_network,
                          key=lambda lnk: lnk.strength, reverse=True)[:10]:
            src_type = "📰" if link.from_name in news_cols else "💰"
            dst_type = "📰" if link.to_name in news_cols else "💰"
            if src_type != dst_type:
                sign = "+" if link.correlation > 0 else "−"
                print(f"      {src_type}{link.from_name} ↔ "
                      f"{dst_type}{link.to_name}: {sign}{link.strength:.3f}")

    # Confidence
    print("\n  [3/3] Confidence...")
    confidence = assess_confidence(
        state_vectors=dataset.state_vectors,
        lambda_structures=structures,
        network_result=network,
        dimension_names=names,
        n_permutations=300,
        n_bootstrap=500,
        window_steps=window_steps,
    )
    sig_causal = sum(1 for c in confidence.causal_links if c.is_significant)
    print(f"    Significant causal: {sig_causal}/{len(confidence.causal_links)}")

    return {
        "merged": merged,
        "structures": structures,
        "network": network,
        "confidence": confidence,
        "news_to_market": news_to_market,
        "market_to_news": market_to_news,
    }


# ============================================================
# 4. Main
# ============================================================

if __name__ == "__main__":
    # 2024年7-8月（日銀ショック期間）
    START = "2024-07-01"
    END = "2024-08-31"

    # GDELT
    gdelt_df = fetch_gdelt_timelines(START, END)

    # Market
    market_df = fetch_market_data(START, END)

    # Analyze!
    if gdelt_df is not None:
        results = analyze_news_market(gdelt_df, market_df,
                                      window_steps=10, max_lag=5)
