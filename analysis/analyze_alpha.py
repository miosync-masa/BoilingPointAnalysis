import numpy as np
from scipy.interpolate import UnivariateSpline
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import stats

@dataclass
class PeakResult:
    T_peak: float
    alpha_peak: float
    sharpness: float  # -d2alpha/dT2 at T_peak
    ci95_T: tuple
    ci95_alpha: tuple
    effect_size: float  # |Δalpha| / SEM

def block_bootstrap_mean(x, block=50, n_boot=1000, rng=None):
    """1D series -> block bootstrap mean; returns (mean, sem, samples)"""
    rng = np.random.default_rng(rng)
    N = len(x)
    n_blocks = max(1, N // block)
    idx_blocks = [slice(i*block, (i+1)*block) for i in range(n_blocks)]
    boots = []
    for _ in range(n_boot):
        take = rng.integers(0, n_blocks, size=n_blocks)
        sample = np.concatenate([x[idx_blocks[i]] for i in take])
        boots.append(sample.mean())
    boots = np.asarray(boots)
    return x.mean(), boots.std(ddof=1), boots

def replicate_layer_bootstrap(values, n_boot=5000, rng=None):
    """replicate means -> bootstrap CI"""
    rng = np.random.default_rng(rng)
    boots = []
    values = np.asarray(values)
    k = len(values)
    for _ in range(n_boot):
        take = rng.integers(0, k, size=k)
        boots.append(values[take].mean())
    boots = np.asarray(boots)
    mean = values.mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, (lo, hi), boots.std(ddof=1)

def find_alpha_peak(T, alpha, smooth=0.0):
    """s=0で補間、必要なら微小sでノイズ抑制"""
    spl = UnivariateSpline(T, alpha, s=smooth, k=3)
    d1 = spl.derivative(1)
    d2 = spl.derivative(2)
    
    # 零点候補（導関数が0）：局所極大を探索
    roots = d1.roots()
    
    # 温度範囲内の極大のみ
    cand = []
    for r in roots:
        if T.min() <= r <= T.max():
            if d2(r) < 0:  # 極大条件
                cand.append((r, spl(r), -d2(r)))
    
    if not cand:
        # フォールバック：観測点最大
        idx = np.argmax(alpha)
        return PeakResult(T[idx], alpha[idx], np.nan, (np.nan, np.nan), (np.nan, np.nan), np.nan)
    
    # 鋭さ最大を採用
    r, a, sharp = max(cand, key=lambda x: x[2])
    return PeakResult(r, a, sharp, (np.nan, np.nan), (np.nan, np.nan), np.nan)

def alpha_peak_with_ci(T, alpha_reps, sem_reps=None, smooth=0.0, n_boot=5000, rng=None):
    """レプリカ集合（list of arrays）からピーク温度・値・CIを推定"""
    rng = np.random.default_rng(rng)
    T = np.asarray(T)
    
    # レプリカごとにスプライン → ピーク温度収集
    peaks_T, peaks_alpha = [], []
    for a in alpha_reps:
        pr = find_alpha_peak(T, np.asarray(a), smooth=smooth)
        peaks_T.append(pr.T_peak)
        peaks_alpha.append(pr.alpha_peak)
    
    mean_T, ciT, _ = replicate_layer_bootstrap(peaks_T, n_boot=n_boot, rng=rng)
    mean_a, cia, _ = replicate_layer_bootstrap(peaks_alpha, n_boot=n_boot, rng=rng)
    
    # 合成曲線で鋭さ評価
    alpha_mean = np.mean(np.vstack(alpha_reps), axis=0)
    pr_mean = find_alpha_peak(T, alpha_mean, smooth=smooth)
    pr_mean.ci95_T = ciT
    pr_mean.ci95_alpha = cia
    
    # 効果量（Δalpha/SEM）: 谷と頂の差を使用
    i_max = np.argmax(alpha_mean)
    # 近傍の最小（沸点-2°C 付近に現れやすい）
    i_min = np.argmin(alpha_mean[max(0, i_max-3):i_max]) + max(0, i_max-3) if i_max>0 else 0
    
    if sem_reps is not None:
        sem = np.mean(sem_reps)
    else:
        sem = np.std(np.vstack(alpha_reps), ddof=1, axis=0)[i_max] / np.sqrt(len(alpha_reps))
    
    pr_mean.effect_size = abs(alpha_mean[i_max] - alpha_mean[i_min]) / max(sem, 1e-12)
    return pr_mean

def detect_interface_drop(z_series_reps, T_idx_boil_minus2, pre_window=3, sigma_rule=3.0):
    """
    z(t) の事前定義ドロップ検出。各レプリカの z(T_b−2) 変化量を Z-score 化して合意率を返す。
    z_series_reps: list of 1D arrays aligned in T
    T_idx_boil_minus2: int index for T_b − 2°C
    """
    votes, zs = [], []
    for z in z_series_reps:
        pre = z[max(0, T_idx_boil_minus2-pre_window):T_idx_boil_minus2]  # 沸点前のベース
        mu, sd = np.mean(pre), np.std(pre, ddof=1)
        delta = z[T_idx_boil_minus2] - mu
        Z = (delta) / (sd if sd>0 else 1e-12)
        zs.append(Z)
        votes.append(Z <= -sigma_rule)
    agree = np.mean(votes)  # 一致率
    return agree, np.array(zs)

# ==========================================
# 追加：実際の解析実行用関数
# ==========================================

def calculate_alpha_from_gromacs(KE_kJmol, VirXX, VirYY, VirZZ, N, V):
    """
    GROMACSデータからα係数を計算
    
    Parameters:
    -----------
    KE_kJmol : float
        運動エネルギー (kJ/mol)
    VirXX, VirYY, VirZZ : float
        ビリアルテンソル成分 (kJ/mol)
    N : int
        分子数
    V : float
        体積 (m³)
    
    Returns:
    --------
    alpha : float
        相互作用係数
    """
    NA = 6.02214076e23  # アボガドロ定数
    n_mol = N/NA
    
    # エネルギー変換
    K_sys = KE_kJmol * 1000 * n_mol  # J
    Phi_kJmol = (VirXX + VirYY + VirZZ) / 3
    W_sys = -Phi_kJmol * 1000 * n_mol  # J (符号注意)
    
    # α計算
    alpha = 1 + W_sys / (2 * K_sys) if K_sys > 0 else 0
    
    return alpha

def analyze_model_peak(model_name, temperatures, alpha_data_reps, plot=True):
    """
    モデルごとのα極大解析
    
    Parameters:
    -----------
    model_name : str
        水モデル名 (TIP3P, SPC/E, TIP4P/2005)
    temperatures : array-like
        温度配列 (°C)
    alpha_data_reps : list of lists
        各温度・各レプリカのα値 [temp][rep]
    plot : bool
        グラフ描画するか
    
    Returns:
    --------
    result : PeakResult
        極大解析結果
    """
    T = np.array(temperatures)
    
    # 各レプリカのα値を整理
    alpha_reps = []
    for rep_idx in range(len(alpha_data_reps[0])):
        alpha_rep = [alpha_data_reps[t_idx][rep_idx] for t_idx in range(len(T))]
        alpha_reps.append(alpha_rep)
    
    # 極大解析
    result = alpha_peak_with_ci(T, alpha_reps, smooth=0.0, n_boot=5000)
    
    # 結果表示
    print(f"\n=== {model_name} α極大解析結果 ===")
    print(f"極大温度: {result.T_peak:.1f} °C (95% CI: {result.ci95_T[0]:.1f}-{result.ci95_T[1]:.1f})")
    print(f"極大α値: {result.alpha_peak:.4f} (95% CI: {result.ci95_alpha[0]:.4f}-{result.ci95_alpha[1]:.4f})")
    print(f"鋭さ: {result.sharpness:.6f}")
    print(f"効果量: {result.effect_size:.2f}")
    
    # t検定（極大点 vs 他の点）
    i_peak = np.argmax(np.mean(alpha_data_reps, axis=1))
    for i, t in enumerate(T):
        if i != i_peak:
            t_stat, p_val = stats.ttest_ind(alpha_data_reps[i_peak], alpha_data_reps[i])
            print(f"  {result.T_peak:.0f}°C vs {t:.0f}°C: t={t_stat:.2f}, p={p_val:.6f}")
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左：α vs T プロット
        alpha_mean = np.mean(alpha_data_reps, axis=1)
        alpha_std = np.std(alpha_data_reps, axis=1)
        
        ax1.errorbar(T, alpha_mean, yerr=alpha_std, fmt='o-', capsize=5, label='観測値')
        
        # スプライン補間
        T_fine = np.linspace(T.min(), T.max(), 200)
        spl = UnivariateSpline(T, alpha_mean, s=0.0, k=3)
        ax1.plot(T_fine, spl(T_fine), 'r--', alpha=0.5, label='スプライン補間')
        ax1.axvline(result.T_peak, color='g', linestyle=':', label=f'極大 {result.T_peak:.1f}°C')
        
        ax1.set_xlabel('温度 (°C)')
        ax1.set_ylabel('α')
        ax1.set_title(f'{model_name} - α温度依存性')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右：レプリカ分布
        for i, t in enumerate(T):
            ax2.scatter([t]*len(alpha_data_reps[i]), alpha_data_reps[i], alpha=0.5)
        ax2.set_xlabel('温度 (°C)')
        ax2.set_ylabel('α')
        ax2.set_title(f'{model_name} - レプリカ分布')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_alpha_peak_analysis.png', dpi=300)
        plt.show()
    
    return result

# ==========================================
# 使用例
# ==========================================

if __name__ == "__main__":
    # TIP4P/2005の例
    temperatures_tip4p = [97, 99, 100, 101, 103]
    
    # ダミーデータ（実際はGROMACSから読み込み）
    alpha_data_tip4p = [
        [0.6304, 0.6310, 0.6298, 0.6315, 0.6302],  # 97°C
        [0.6701, 0.6695, 0.6708, 0.6699, 0.6703],  # 99°C (極大)
        [0.6593, 0.6598, 0.6589, 0.6595, 0.6591],  # 100°C
        [0.6611, 0.6615, 0.6608, 0.6612, 0.6610],  # 101°C
        [0.6454, 0.6458, 0.6450, 0.6456, 0.6452],  # 103°C
    ]
    
    result = analyze_model_peak("TIP4P/2005", temperatures_tip4p, alpha_data_tip4p, plot=False)
    
    # 統計的有意性の判定
    if result.effect_size > 5:
        print("\n★ α極大は統計的に極めて有意！")
    elif result.effect_size > 2:
        print("\n★ α極大は統計的に有意！")
    else:
        print("\n△ より多くのデータが必要")
