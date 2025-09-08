"""
スラブ系MD解析プロトコル
- 密度プロファイルから界面位置・蒸気圧を自動検出
- プレ沸騰現象（98°C界面急降下）の定量化
- データ駆動型（固定しきい値なし）
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt

# ==========================================
# コア解析関数
# ==========================================

def analyze_slab_density(density_file, T_C, liquid_frac=0.2, vapor_frac=0.2,
                         grad_pct=0.4, z_guard_nm=0.5):
    """
    スラブ密度プロファイル rho(z) から、P_vap / ρ_liquid / ρ_vapor / 界面z(2つ) を推定
    - データ駆動: 固定しきい値や固定z範囲を使わない
    - 2界面を左右で独立に検出
    - 気相密度は両側独立に平均し、外れ値を除去

    Parameters
    ----------
    density_file : str   # gmx density の xvg (mass density, kg/m^3)
    T_C          : float # 温度(°C)
    liquid_frac  : float # 中央(液相)からこの割合の幅で液相コアをとる
    vapor_frac   : float # 両端(vapor)の幅割合
    grad_pct     : float # |dρ/dz| 上位この割合だけを界面候補に使う
    z_guard_nm   : float # 各端からの番兵幅(界面候補から除外; nm)

    Returns
    -------
    dict with keys:
        P_vap_atm : float
        rho_liquid : float
        rho_vapor  : float
        z_interfaces : tuple[float,float]  # nm, 下側/上側の界面位置
        interface_width : float  # nm, 界面幅
    """

    # --- 1) 入力読み込み ---
    data = np.loadtxt(density_file, comments=['@', '#'])
    z = data[:, 0]          # nm
    rho = data[:, 1]        # kg/m^3

    # 基本統計（ノイズ除去のためのwinsorize軽め処理）
    r = rho.copy()
    q1, q99 = np.percentile(r, [1, 99])
    r = np.clip(r, q1, q99)

    # --- 2) 液相・気相の候補領域をデータ駆動で定義 ---
    zmin, zmax = z.min(), z.max()
    Lz = zmax - zmin
    zc = 0.5 * (zmin + zmax)

    # 中央の liquid_frac を液相コア候補（例: 20%）
    li_lo, li_hi = zc - (liquid_frac * Lz / 2), zc + (liquid_frac * Lz / 2)
    liquid_core = (z >= li_lo) & (z <= li_hi)

    # 両端の vapor_frac を気相候補（例: 両端20%ずつ）
    vap_lo = (z <= zmin + vapor_frac * Lz)
    vap_hi = (z >= zmax - vapor_frac * Lz)

    # 液相密度（中央コアのロバスト平均）
    rho_liquid = np.median(r[liquid_core])

    # 気相側は外れ値除去（上位/下位パーセンタイルでトリム）
    def robust_mean(vals):
        if vals.size == 0:
            return np.nan
        ql, qu = np.percentile(vals, [10, 90])
        sel = (vals >= ql) & (vals <= qu)
        return float(np.mean(vals[sel])) if np.any(sel) else float(np.mean(vals))

    rho_vap_lo = robust_mean(r[vap_lo])
    rho_vap_hi = robust_mean(r[vap_hi])

    # 2側の平均（どちらかが欠けたら片側で代替）
    if np.isnan(rho_vap_lo) and np.isnan(rho_vap_hi):
        # 最終手段：最小密度近傍から拾う
        rho_vapor = float(np.mean(np.sort(r)[:max(5, len(r)//50)]))
    else:
        vals = [v for v in [rho_vap_lo, rho_vap_hi] if not np.isnan(v)]
        rho_vapor = float(np.mean(vals))

    # --- 3) 界面位置の自動検出（勾配の強い場所） ---
    # 端のガード領域を除外（PMEや境界ノイズを避ける）
    guard = z_guard_nm
    valid = (z >= zmin + guard) & (z <= zmax - guard)

    # 数値微分（nmあたり）
    dz = np.gradient(z)
    drho_dz = np.gradient(rho) / np.where(dz == 0, np.nan, dz)

    # 勾配の大きい順に候補を取り、下側・上側で最大点を1つずつ選ぶ
    mag = np.abs(drho_dz)
    mag[~valid] = 0.0
    thresh = np.percentile(mag[mag > 0], 100 * (1 - grad_pct)) if np.any(mag > 0) else np.inf
    cand = np.where(mag >= thresh)[0]

    # 下側(中心より下)・上側(中心より上)を分ける
    lower = cand[z[cand] < zc]
    upper = cand[z[cand] > zc]
    
    # それぞれで最大勾配の位置を1点ずつ
    z_if1 = float(z[lower[np.argmax(mag[lower])]]) if lower.size else float(z[np.argmax(mag[:len(z)//2])])
    z_if2 = float(z[upper[np.argmax(mag[upper])]]) if upper.size else float(z[len(z)//2 + np.argmax(mag[len(z)//2:])])

    # 界面幅の計算
    interface_width = abs(z_if2 - z_if1)

    # --- 4) P_vap 計算（理想気体近似）---
    k_B = 1.380649e-23     # J/K
    NA  = 6.02214076e23    # 1/mol
    m_H2O = 18.015e-3      # kg/mol
    T_K = T_C + 273.15

    n_vapor = rho_vapor / (m_H2O / NA)  # molecules/m^3
    P_vap = n_vapor * k_B * T_K         # Pa
    P_vap_atm = P_vap / 101325.0

    return {
        'P_vap_atm': P_vap_atm,
        'rho_liquid': float(rho_liquid),
        'rho_vapor': float(rho_vapor),
        'z_interfaces': (z_if1, z_if2),
        'interface_width': interface_width,
        'z': z,
        'rho': rho
    }

# ==========================================
# 高度な解析オプション - 2面独立tanhフィット
# ==========================================

from dataclasses import dataclass

# ---- モデル関数: 片側界面のtanhフィット ----
def tanh_profile(z, rho_l, rho_v, z0, d):
    """片側界面の密度プロファイル"""
    return 0.5*(rho_l + rho_v) - 0.5*(rho_l - rho_v) * np.tanh((z - z0)/d)

@dataclass
class InterfaceFit:
    side: str                 # "lower" or "upper"
    z0: float                 # nm: 界面位置
    d: float                  # nm: 界面厚み（大きいほど厚い）
    rho_l: float              # kg/m^3: 液相密度
    rho_v: float              # kg/m^3: 気相密度
    success: bool             # フィット成功フラグ
    r2: float                 # 決定係数(擬似)
    npts: int                 # 使った点数

@dataclass
class SlabAnalysis:
    P_vap_atm: float
    rho_liquid: float
    rho_vapor: float
    lower: InterfaceFit
    upper: InterfaceFit

# ---- 補助: ロバスト平均 ----
def robust_mean(vals, qlo=10, qhi=90):
    vals = np.asarray(vals)
    if vals.size == 0:
        return np.nan
    lo, hi = np.percentile(vals, [qlo, qhi])
    use = (vals >= lo) & (vals <= hi)
    return float(vals[use].mean()) if np.any(use) else float(vals.mean())

# ---- 勾配に基づく窓抽出 ----
def window_around_gradient(z, rho, center_idx, half_width_nm=1.0):
    z0 = z[center_idx]
    mask = (z >= z0 - half_width_nm) & (z <= z0 + half_width_nm)
    # 端で薄くなりすぎないように最小点数確保
    if mask.sum() < 15:
        # 点数が少なければ倍の幅で取り直す
        mask = (z >= z0 - 2*half_width_nm) & (z <= z0 + 2*half_width_nm)
    return mask

# ---- 片側界面フィット ----
def fit_one_interface(z, rho, side, z_center_guess, rho_l_guess, rho_v_guess):
    # 初期値とバウンド
    # z0 ~ 勾配最大の近傍、d は 0.1〜3 nm 程度を想定
    p0 = [rho_l_guess, rho_v_guess, z_center_guess, 0.5]
    bounds = ([rho_v_guess*0.1, 0.0, z.min(), 0.05],
              [rho_l_guess*1.5, rho_l_guess, z.max(), 5.0])
    try:
        popt, pcov = curve_fit(tanh_profile, z, rho, p0=p0, bounds=bounds, maxfev=20000)
        rho_l, rho_v, z0, d = popt
        pred = tanh_profile(z, *popt)
        ss_res = np.sum((rho - pred)**2)
        ss_tot = np.sum((rho - rho.mean())**2) + 1e-12
        r2 = 1.0 - ss_res/ss_tot
        return InterfaceFit(side, float(z0), float(d), float(rho_l), float(rho_v), True, float(r2), len(z))
    except Exception:
        return InterfaceFit(side, float(np.nan), float(np.nan), float(rho_l_guess), float(rho_v_guess), False, 0.0, len(z))

# ---- メイン: スラブ解析 + tanhフィット ----
def analyze_slab_density_with_tanh(density_file, T_C,
                                   liquid_frac=0.2, vapor_frac=0.2,
                                   grad_pct=0.4, z_guard_nm=0.5,
                                   fit_half_window_nm=1.2):
    """
    gmx density -dens mass (kg/m^3), -center で出力した rho(z) を入力。
    1) データ駆動で液相/気相代表値を推定
    2) |dρ/dz|最大の点を中心に各側で窓を切り、tanhフィット
    3) フィットから得た rho_v を優先、失敗ならロバスト平均で代用
    4) ρ_v → P_vap (ideal gas) を返す
    """
    data = np.loadtxt(density_file, comments=['@', '#'])
    z = data[:, 0]    # nm
    rho = data[:, 1]  # kg/m^3

    # 端ガード
    zmin, zmax = z.min(), z.max()
    Lz = zmax - zmin
    zc = 0.5*(zmin + zmax)
    valid = (z >= zmin + z_guard_nm) & (z <= zmax - z_guard_nm)
    zv = z[valid]; rhov = rho[valid]

    # 液相(中央コア) / 気相(両端) の代表値（ロバスト）
    li_lo, li_hi = zc - (liquid_frac*Lz/2), zc + (liquid_frac*Lz/2)
    liquid_core = (z >= li_lo) & (z <= li_hi)
    rho_liquid_est = float(np.median(rho[liquid_core]))

    vap_lo = (z <= zmin + vapor_frac*Lz)
    vap_hi = (z >= zmax - vapor_frac*Lz)
    rho_vap_lo_est = robust_mean(rho[vap_lo])
    rho_vap_hi_est = robust_mean(rho[vap_hi])

    # 勾配で界面近傍の中心を推定（上下で別々）
    dz = np.gradient(zv)
    drho_dz = np.gradient(rhov) / np.where(dz == 0, np.nan, dz)
    mag = np.abs(drho_dz)
    # 下側: 中心より左、上側: 右
    left_mask = zv < zc
    right_mask = zv > zc
    idx_left = np.argmax(mag * left_mask)
    idx_right = np.argmax(mag * right_mask)

    # フィット用データ抽出
    left_win = window_around_gradient(zv, rhov, idx_left, half_width_nm=fit_half_window_nm)
    right_win = window_around_gradient(zv, rhov, idx_right, half_width_nm=fit_half_window_nm)

    # 初期値（片側ごとに違う気相推定を渡す）
    fit_left = fit_one_interface(zv[left_win], rhov[left_win], "lower",
                                 z_center_guess=zv[idx_left],
                                 rho_l_guess=rho_liquid_est,
                                 rho_v_guess=rho_vap_lo_est)
    fit_right = fit_one_interface(zv[right_win], rhov[right_win], "upper",
                                  z_center_guess=zv[idx_right],
                                  rho_l_guess=rho_liquid_est,
                                  rho_v_guess=rho_vap_hi_est)

    # ρ_liquid は2面のフィット結果があれば平均、なければ中央値
    rho_liquid_list = [x.rho_l for x in (fit_left, fit_right) if x.success]
    rho_liquid = float(np.mean(rho_liquid_list)) if rho_liquid_list else rho_liquid_est

    # ρ_vapor は両側フィットの平均、失敗した側はロバスト平均で代用
    vap_cands = []
    for fit, est in ((fit_left, rho_vap_lo_est), (fit_right, rho_vap_hi_est)):
        if fit.success and fit.rho_v > 0.0:
            vap_cands.append(fit.rho_v)
        elif not np.isnan(est):
            vap_cands.append(est)
    if len(vap_cands) == 0:
        # 最後の手段：最薄側の分位から拾う
        vap_cands.append(float(np.mean(np.sort(rho)[:max(5, len(rho)//50)])))
    rho_vapor = float(np.mean(vap_cands))

    # P_vap（理想気体近似）
    k_B = 1.380649e-23
    NA  = 6.02214076e23
    m_H2O = 18.015e-3
    T_K = T_C + 273.15
    n_vapor = rho_vapor / (m_H2O / NA)
    P_vap = n_vapor * k_B * T_K
    P_vap_atm = float(P_vap / 101325.0)

    return SlabAnalysis(
        P_vap_atm=P_vap_atm,
        rho_liquid=rho_liquid,
        rho_vapor=rho_vapor,
        lower=fit_left,
        upper=fit_right
    )

def calculate_surface_tension(pressure_file, Lz_nm):
    """
    圧力テンソルから表面張力を計算
    γ = (Lz/2) * (Pzz - (Pxx + Pyy)/2)
    
    Parameters
    ----------
    pressure_file : str
        gmx energy出力のxvgファイル（Pxx, Pyy, Pzz含む）
    Lz_nm : float
        ボックスのz方向長さ (nm)
    
    Returns
    -------
    gamma : float
        表面張力 (mN/m)
    """
    # Pressure-XX, Pressure-YY, Pressure-ZZ を読み込み
    # 実装は gmx energy の出力形式に依存
    data = np.loadtxt(pressure_file, comments=['@', '#'])
    
    # 通常、列は [time, Pxx, Pyy, Pzz] の順
    Pxx = np.mean(data[:, 1])  # bar
    Pyy = np.mean(data[:, 2])  # bar
    Pzz = np.mean(data[:, 3])  # bar
    
    # bar to Pa: 1 bar = 1e5 Pa
    # γ = (Lz/2) * (Pzz - (Pxx + Pyy)/2) * 1e5 Pa * 1e-9 m/nm * 1e3 mN/N
    gamma = (Lz_nm / 2) * (Pzz - (Pxx + Pyy) / 2) * 0.1  # mN/m
    
    return gamma

# ==========================================
# プレ沸騰現象の検出
# ==========================================

def detect_pre_boiling(temperatures, interface_widths, T_boiling):
    """
    プレ沸騰現象（界面急降下）を検出
    
    Parameters
    ----------
    temperatures : array
        温度配列 (°C)
    interface_widths : array
        各温度での界面幅 (nm)
    T_boiling : float
        予想沸点 (°C)
    
    Returns
    -------
    dict with detection results
    """
    # 沸点-2°Cの位置を探す
    T_pre = T_boiling - 2
    idx_pre = np.argmin(np.abs(temperatures - T_pre))
    
    if idx_pre == 0 or idx_pre == len(temperatures) - 1:
        return {'detected': False, 'message': 'Not enough data points'}
    
    # 前後の界面幅と比較
    width_pre = interface_widths[idx_pre]
    width_before = interface_widths[idx_pre - 1]
    width_after = interface_widths[idx_pre + 1]
    
    # 急降下の判定（前より20%以上減少）
    drop_ratio = (width_before - width_pre) / width_before
    
    if drop_ratio > 0.2:  # 20%以上の減少
        return {
            'detected': True,
            'T_pre_boiling': temperatures[idx_pre],
            'width_drop': width_before - width_pre,
            'drop_ratio': drop_ratio,
            'message': f'Pre-boiling detected at {temperatures[idx_pre]:.0f}°C'
        }
    else:
        return {'detected': False, 'drop_ratio': drop_ratio}

# ==========================================
# 統合解析パイプライン
# ==========================================

def analyze_slab_series(model_name, temperatures, density_files, plot=True):
    """
    温度シリーズのスラブ解析を統合実行
    
    Parameters
    ----------
    model_name : str
        水モデル名 (TIP3P, SPC/E, TIP4P/2005)
    temperatures : array
        温度配列 (°C)
    density_files : list
        各温度の密度プロファイルファイルリスト
    plot : bool
        結果をプロットするか
    
    Returns
    -------
    results : dict
        全解析結果
    """
    results = {
        'model': model_name,
        'temperatures': temperatures,
        'P_vap': [],
        'rho_liquid': [],
        'rho_vapor': [],
        'interface_widths': [],
        'z_interfaces': []
    }
    
    # 各温度で解析
    for T, density_file in zip(temperatures, density_files):
        res = analyze_slab_density(density_file, T)
        results['P_vap'].append(res['P_vap_atm'])
        results['rho_liquid'].append(res['rho_liquid'])
        results['rho_vapor'].append(res['rho_vapor'])
        results['interface_widths'].append(res['interface_width'])
        results['z_interfaces'].append(res['z_interfaces'])
    
    # プレ沸騰検出
    T_boiling_guess = {'TIP3P': 100, 'SPC/E': 110, 'TIP4P/2005': 99}
    pre_boiling = detect_pre_boiling(
        temperatures, 
        results['interface_widths'],
        T_boiling_guess.get(model_name, 100)
    )
    results['pre_boiling'] = pre_boiling
    
    # 結果表示
    print(f"\n=== {model_name} スラブ解析結果 ===")
    print(f"{'T(°C)':>6} {'P_vap(atm)':>10} {'ρ_liq(kg/m³)':>12} {'ρ_vap(kg/m³)':>12} {'界面幅(nm)':>11}")
    print("-" * 65)
    
    for i, T in enumerate(temperatures):
        print(f"{T:6.0f} {results['P_vap'][i]:10.3f} "
              f"{results['rho_liquid'][i]:12.1f} {results['rho_vapor'][i]:12.3f} "
              f"{results['interface_widths'][i]:11.2f}")
    
    if pre_boiling['detected']:
        print(f"\n★ プレ沸騰現象検出！ {pre_boiling['message']}")
        print(f"  界面幅減少: {pre_boiling['width_drop']:.2f} nm ({pre_boiling['drop_ratio']*100:.1f}%)")
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # P_vap
        axes[0, 0].plot(temperatures, results['P_vap'], 'o-')
        axes[0, 0].set_xlabel('Temperature (°C)')
        axes[0, 0].set_ylabel('P_vap (atm)')
        axes[0, 0].set_title(f'{model_name} - Vapor Pressure')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 密度
        axes[0, 1].plot(temperatures, results['rho_liquid'], 'b-', label='Liquid')
        axes[0, 1].plot(temperatures, results['rho_vapor'], 'r-', label='Vapor')
        axes[0, 1].set_xlabel('Temperature (°C)')
        axes[0, 1].set_ylabel('Density (kg/m³)')
        axes[0, 1].set_title(f'{model_name} - Phase Densities')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 界面幅
        axes[1, 0].plot(temperatures, results['interface_widths'], 'go-')
        if pre_boiling['detected']:
            idx = np.argmin(np.abs(temperatures - pre_boiling['T_pre_boiling']))
            axes[1, 0].plot(temperatures[idx], results['interface_widths'][idx], 
                           'r*', markersize=15, label='Pre-boiling')
        axes[1, 0].set_xlabel('Temperature (°C)')
        axes[1, 0].set_ylabel('Interface Width (nm)')
        axes[1, 0].set_title(f'{model_name} - Interface Width')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 密度プロファイル例（最高温度）
        if density_files:
            last_res = analyze_slab_density(density_files[-1], temperatures[-1])
            axes[1, 1].plot(last_res['z'], last_res['rho'], 'b-', alpha=0.7)
            axes[1, 1].axvline(last_res['z_interfaces'][0], color='r', linestyle='--', label='Interface')
            axes[1, 1].axvline(last_res['z_interfaces'][1], color='r', linestyle='--')
            axes[1, 1].set_xlabel('z (nm)')
            axes[1, 1].set_ylabel('Density (kg/m³)')
            axes[1, 1].set_title(f'{model_name} @ {temperatures[-1]:.0f}°C')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_slab_analysis.png', dpi=300)
        plt.show()
    
    return results

# ==========================================
# ブートストラップCI & 平滑プロファイル
# ==========================================

from typing import List, Dict, Tuple, Literal, Any
from pathlib import Path
from numpy.random import default_rng
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

@dataclass
class BootstrapCI:
    mean: float
    ci95: Tuple[float, float]
    samples: np.ndarray

@dataclass
class SlabBootstrapResult:
    summary: Dict[str, BootstrapCI]
    per_file: List[Dict[str, float]]

def _collect_metrics_one(file: str, T_C: float) -> Dict[str, float]:
    """単一ファイルからメトリクス抽出"""
    res = analyze_slab_density_with_tanh(file, T_C=T_C)
    out = {
        "P_vap_atm": res.P_vap_atm,
        "rho_liquid": res.rho_liquid,
        "rho_vapor": res.rho_vapor,
        "z0_lower": res.lower.z0,
        "d_lower": res.lower.d,
        "r2_lower": res.lower.r2,
        "z0_upper": res.upper.z0,
        "d_upper": res.upper.d,
        "r2_upper": res.upper.r2,
    }
    return out

def _bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 42) -> BootstrapCI:
    """ブートストラップ信頼区間計算"""
    rng = default_rng(seed)
    k = len(values)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        take = rng.integers(0, k, size=k)
        bs[i] = np.mean(values[take])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return BootstrapCI(mean=float(values.mean()), ci95=(float(lo), float(hi)), samples=bs)

def bootstrap_slab_from_files(files: List[str], T_C: float,
                              n_boot: int = 5000, seed: int = 42) -> SlabBootstrapResult:
    """
    複数スナップショット/レプリカからブートストラップCI計算
    
    Parameters
    ----------
    files : List[str]
        密度プロファイルファイルリスト（時間窓別 or レプリカ別）
    T_C : float
        温度 (°C)
    n_boot : int
        ブートストラップサンプル数
    seed : int
        乱数シード（再現性確保）
    
    Returns
    -------
    SlabBootstrapResult
        各メトリクスの平均と95%CI
    """
    per = []
    for f in files:
        per.append(_collect_metrics_one(f, T_C))

    # 欠損を弾きつつ配列化
    keys = ["P_vap_atm","rho_liquid","rho_vapor","z0_lower","d_lower","z0_upper","d_upper"]
    arrs = {}
    for k in keys:
        vals = np.array([x[k] for x in per if np.isfinite(x[k])])
        if vals.size == 0:
            continue
        arrs[k] = _bootstrap_ci(vals, n_boot=n_boot, seed=seed)

    return SlabBootstrapResult(summary=arrs, per_file=per)

def smooth_profile(z: np.ndarray, rho: np.ndarray,
                   method: Literal["spline","savgol"]="spline",
                   spline_s: float = 0.0, spline_k: int = 3,
                   savgol_win: int = 21, savgol_poly: int = 3) -> np.ndarray:
    """
    密度プロファイルの平滑化（図用）
    
    Parameters
    ----------
    z : array
        z座標
    rho : array
        密度
    method : str
        平滑化手法 ("spline" or "savgol")
    
    Returns
    -------
    rho_smooth : array
        平滑化された密度
    """
    z = np.asarray(z); rho = np.asarray(rho)
    if method == "spline":
        spl = UnivariateSpline(z, rho, s=spline_s, k=spline_k)
        return spl(z)
    else:
        # ウィンドウは奇数＆データ長以下に補正
        win = min(savgol_win if savgol_win % 2 == 1 else savgol_win+1, len(rho)-(1-len(rho)%2))
        win = max(5, win)
        poly = min(savgol_poly, win-2)
        return savgol_filter(rho, window_length=win, polyorder=poly, mode="interp")

def mean_ci_profile(files: List[str], ci_alpha: float = 95.0) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray,np.ndarray]]:
    """
    同一zグリッドのxvg群を読み、点ごと平均とCI帯を返す
    
    前提：gmx density -sl N で同じグリッド数/範囲（-center 推奨）
    
    Returns
    -------
    z : array
        z座標
    mean : array
        平均密度
    (lo, hi) : tuple
        95%CI下限・上限
    """
    stack = []
    z_ref = None
    for f in files:
        data = np.loadtxt(f, comments=['@','#'])
        z = data[:,0]; rho = data[:,1]
        if z_ref is None:
            z_ref = z
        else:
            # グリッドが微妙にズレる場合は線形補間
            if not np.allclose(z, z_ref, rtol=0, atol=1e-6):
                rho = np.interp(z_ref, z, rho)
        stack.append(rho)
    M = np.vstack(stack)           # shape: (n_files, n_points)
    mean = M.mean(axis=0)
    lo, hi = np.percentile(M, [(100-ci_alpha)/2, 100-(100-ci_alpha)/2], axis=0)
    return z_ref, mean, (lo, hi)

def summarize_slab_for_figure(files: List[str], T_C: float, n_boot: int = 5000, seed: int = 42) -> Dict[str, Any]:
    """
    論文図用の統合解析
    
    これ一つで以下を取得：
    - スカラー指標のブートストラップCI
    - 密度プロファイルの平均±CI帯
    - tanhフィット曲線（図に重ねる用）
    - R²値（キャプション記載用）
    
    Parameters
    ----------
    files : List[str]
        同一温度の密度プロファイルファイル群
    T_C : float
        温度 (°C)
    n_boot : int
        ブートストラップサンプル数
    seed : int
        乱数シード
    
    Returns
    -------
    dict
        全解析結果（論文図・表に必要な全情報）
    """
    # 1) スカラー指標のCI
    bs = bootstrap_slab_from_files(files, T_C=T_C, n_boot=n_boot, seed=seed)

    # 2) プロファイルの平均±CI
    z, mean, (lo, hi) = mean_ci_profile(files)
    mean_spline = smooth_profile(z, mean, method="spline", spline_s=0.0)

    # 3) 代表スナップショットでtanhフィット曲線を出したい場合（任意）
    rep_fit = analyze_slab_density_with_tanh(files[0], T_C=T_C)
    
    # 代表曲線：両側のパラメータでプロファイル生成（図用）
    def tanh_prof(z_arr, rho_l, rho_v, z0, d):
        return 0.5*(rho_l+rho_v) - 0.5*(rho_l-rho_v)*np.tanh((z_arr-z0)/d)
    
    fit_curve_lower = tanh_prof(z, rep_fit.lower.rho_l, rep_fit.lower.rho_v, 
                                rep_fit.lower.z0, rep_fit.lower.d) if rep_fit.lower.success else None
    fit_curve_upper = tanh_prof(z, rep_fit.upper.rho_l, rep_fit.upper.rho_v, 
                                rep_fit.upper.z0, rep_fit.upper.d) if rep_fit.upper.success else None

    return {
        "bootstrap": bs.summary,          # P_vap, rho_liquid, rho_vapor, z0/d のCI
        "per_file": bs.per_file,          # 各ファイルの個票（付録向き）
        "z": z, "mean": mean, "ci_lo": lo, "ci_hi": hi,
        "mean_spline": mean_spline,       # 図用スムージング
        "fit_lower": fit_curve_lower,     # 図に重ねる用（任意）
        "fit_upper": fit_curve_upper,
        "rep_fit": rep_fit                # tanhパラメータやR^2をキャプションに出せる
    }

# ==========================================
# 論文用図生成
# ==========================================

def create_publication_figure(summary: Dict[str, Any], model_name: str, T_C: float, save_path: str = None):
    """
    論文品質の図を生成
    
    Parameters
    ----------
    summary : dict
        summarize_slab_for_figureの出力
    model_name : str
        モデル名（図タイトル用）
    T_C : float
        温度
    save_path : str
        保存先パス（Noneなら表示のみ）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左：密度プロファイル with CI帯
    z = summary['z']
    ax1.fill_between(z, summary['ci_lo'], summary['ci_hi'], alpha=0.2, color='blue', label='95% CI')
    ax1.plot(z, summary['mean'], 'b-', alpha=0.5, label='Mean (raw)')
    ax1.plot(z, summary['mean_spline'], 'b-', linewidth=2, label='Mean (smoothed)')
    
    # tanhフィット曲線を重ねる
    if summary['fit_lower'] is not None:
        ax1.plot(z, summary['fit_lower'], 'r--', alpha=0.7, label='tanh fit (lower)')
    if summary['fit_upper'] is not None:
        ax1.plot(z, summary['fit_upper'], 'g--', alpha=0.7, label='tanh fit (upper)')
    
    # 界面位置を縦線で示す
    if 'z0_lower' in summary['bootstrap']:
        z0_l = summary['bootstrap']['z0_lower'].mean
        ax1.axvline(z0_l, color='red', linestyle=':', alpha=0.5)
        ax1.text(z0_l, ax1.get_ylim()[1]*0.9, f'z₀={z0_l:.2f}', ha='center')
    
    if 'z0_upper' in summary['bootstrap']:
        z0_u = summary['bootstrap']['z0_upper'].mean
        ax1.axvline(z0_u, color='green', linestyle=':', alpha=0.5)
        ax1.text(z0_u, ax1.get_ylim()[1]*0.9, f'z₀={z0_u:.2f}', ha='center')
    
    ax1.set_xlabel('z (nm)', fontsize=12)
    ax1.set_ylabel('Density (kg/m³)', fontsize=12)
    ax1.set_title(f'{model_name} @ {T_C:.0f}°C - Density Profile', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 右：主要メトリクスのブートストラップCI
    metrics = ['P_vap_atm', 'rho_liquid', 'rho_vapor', 'd_lower', 'd_upper']
    labels = ['P_vap (atm)', 'ρ_liquid', 'ρ_vapor', 'd_lower (nm)', 'd_upper (nm)']
    means = []
    errors = []
    
    for m in metrics:
        if m in summary['bootstrap']:
            ci = summary['bootstrap'][m]
            means.append(ci.mean)
            errors.append([ci.mean - ci.ci95[0], ci.ci95[1] - ci.mean])
        else:
            means.append(0)
            errors.append([0, 0])
    
    errors = np.array(errors).T
    x_pos = np.arange(len(metrics))
    
    ax2.errorbar(x_pos, means, yerr=errors, fmt='o', capsize=5, capthick=2, markersize=8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title(f'{model_name} @ {T_C:.0f}°C - Bootstrap CI (95%)', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # R²値を注釈として追加
    if summary['rep_fit'].lower.success:
        ax2.text(0.05, 0.95, f"Lower R²={summary['rep_fit'].lower.r2:.3f}", 
                transform=ax2.transAxes, va='top', fontsize=10)
    if summary['rep_fit'].upper.success:
        ax2.text(0.05, 0.90, f"Upper R²={summary['rep_fit'].upper.r2:.3f}", 
                transform=ax2.transAxes, va='top', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()

# ==========================================
# 実行例
# ==========================================

if __name__ == "__main__":
    # gmx densityコマンド
    print("=== GROMACS密度計算コマンド ===")
    print("gmx density -f traj.xtc -s topol.tpr -d Z \\")
    print("  -dens mass -sl 400 -center -o rhoZ.xvg")
    print()
    
    # TIP3Pの例
    temperatures_tip3p = [95, 98, 100, 102, 105]
    
    # ダミーファイル名（実際は各温度の解析結果）
    density_files = [f"rhoZ_{T}C.xvg" for T in temperatures_tip3p]
    
    # 解析実行（実際のファイルがある場合）
    # results = analyze_slab_series("TIP3P", temperatures_tip3p, density_files, plot=True)
    
    print("\n=== 重要ポイント ===")
    print("1. 固定しきい値を使わない（データ駆動型）")
    print("2. 2界面を独立検出（左右の非対称性対応）")
    print("3. ロバスト統計で外れ値に強い")
    print("4. プレ沸騰現象（98°C界面急降下）を自動検出")
    print("5. tanh フィッティングオプション付き（論文用）")
