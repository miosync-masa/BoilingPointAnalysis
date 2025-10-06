"""
=============================================================================
EDRパラメータフィッティング統合版 v3.2 (CPU + CUDA)
Miosync, Inc. / 

【概要】
板材成形における破壊予測のための統一理論（EDR理論）実装
- CPU版：既存の逐次実装（安定・検証用）
- CUDA版：GPU並列実装（100-1000倍高速化）

【アーキテクチャ】
CPU版: 単一経路を逐次処理
CUDA版: 候補パラメータ×経路の外積を並列化
  - Grid:  n_candidates ブロック
  - Block: 64 スレッド（実働 n_paths 本）
  - 時間: 逐次ループ（状態依存のため）
  - Λタイムライン不要（peak, D のみ保持）

【使用方法】
# CPU版（既存）
result = simulate_lambda(schedule, mat, edr)

# CUDA版（高速）
optimizer = CUDAOptimizer(schedules_list, failed_labels, mat)
losses = optimizer.evaluate_candidates(candidates_list)

【著者】
飯泉 真道 (Masamichi Iizumi)
環 (Tamaki) - AI Co-Developer

【日付】
2025-01-06
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple, Optional
from scipy.optimize import minimize, Bounds, differential_evolution
from scipy.signal import savgol_filter
from collections import deque
import time

# CUDA関連（条件付きインポート）
CUDA_AVAILABLE = False
try:
    from numba import cuda
    import math
    CUDA_AVAILABLE = cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"✓ CUDA利用可能: {cuda.get_current_device().name.decode()}")
    else:
        print("⚠️  CUDA無効: CPU modeで実行")
except ImportError:
    print("⚠️  Numba未インストール: CPU modeのみ")

# =============================================================================
# Section 1: データ構造（共通）
# =============================================================================

@dataclass
class MaterialParams:
    """材料パラメータ"""
    rho: float = 7800.0      # 密度 [kg/m3]
    cp: float = 500.0        # 比熱 [J/kg/K]
    k: float = 40.0          # 熱伝導率 [W/m/K]
    thickness: float = 0.0008 # 板厚 [m]
    sigma0: float = 600e6    # 初期降伏応力 [Pa]
    n: float = 0.15          # 加工硬化指数
    m: float = 0.02          # 速度感受指数
    r_value: float = 1.0     # ランクフォード値

@dataclass
class EDRParams:
    """EDR理論パラメータ"""
    V0: float = 2e9            # 基準凝集エネルギー [Pa = J/m3]
    av: float = 3e4            # 空孔影響係数
    ad: float = 1e-7           # 転位影響係数
    chi: float = 0.1           # 摩擦発熱の内部分配率
    K_scale: float = 0.2       # K総量スケール
    triax_sens: float = 0.3    # 三軸度感度
    Lambda_crit: float = 1.0   # 臨界Λ
    # 経路別スケール係数
    K_scale_draw: float = 0.15   # 深絞り用
    K_scale_plane: float = 0.25  # 平面ひずみ用
    K_scale_biax: float = 0.20   # 等二軸用
    # FLC V字パラメータ
    beta_A: float = 0.35       # 谷の深さ
    beta_bw: float = 0.28      # 谷の幅

@dataclass
class PressSchedule:
    """FEM or 実験ログの時系列データ"""
    t: np.ndarray                 # 時間 [s]
    eps_maj: np.ndarray           # 主ひずみ
    eps_min: np.ndarray           # 副ひずみ
    triax: np.ndarray             # 三軸度 σm/σeq
    mu: np.ndarray                # 摩擦係数
    pN: np.ndarray                # 接触圧 [Pa]
    vslip: np.ndarray             # すべり速度 [m/s]
    htc: np.ndarray               # 熱伝達係数 [W/m2/K]
    Tdie: np.ndarray              # 金型温度 [K]
    contact: np.ndarray           # 接触率 [0-1]
    T0: float = 293.15            # 板の初期温度 [K]

@dataclass
class ExpBinary:
    """破断/安全のラベル付き実験"""
    schedule: PressSchedule
    failed: int                   # 1:破断, 0:安全
    label: str = ""

@dataclass
class FLCPoint:
    """FLC: 経路比一定での限界点（実測）"""
    path_ratio: float            # β = eps_min/eps_maj
    major_limit: float           # 実測限界主ひずみ
    minor_limit: float           # 実測限界副ひずみ
    rate_major: float = 1.0      # 主ひずみ速度 [1/s]
    duration_max: float = 1.0    # 試験上限時間 [s]
    label: str = ""

# =============================================================================
# Section 2: CPU版実装（既存の全機能）
# =============================================================================

# -----------------------------------------------------------------------------
# 2.1) 物理計算ヘルパ関数
# -----------------------------------------------------------------------------

def triax_from_path(beta: float) -> float:
    """ひずみ経路比βから三軸度ηを計算（平面応力J2塑性）"""
    b = float(np.clip(beta, -0.95, 1.0))
    return (1.0 + b) / (np.sqrt(3.0) * np.sqrt(1.0 + b + b*b))

def beta_multiplier(beta, A=0.35, bw=0.28):
    """β依存ゲイン（V字形状）"""
    b = np.clip(beta, -0.95, 0.95)
    return 1.0 + A * np.exp(-(b / bw)**2)

def cv_eq(T, c0=1e-6, Ev_eV=1.0):
    """平衡空孔濃度"""
    kB_eV = 8.617e-5
    return c0*np.exp(-Ev_eV/(kB_eV*T))

def step_cv(cv, T, rho_d, dt, tau0=1e-3, Q_eV=0.8, k_ann=1e6, k_sink=1e-15):
    """空孔濃度の時間発展"""
    kB_eV = 8.617e-5
    tau = tau0*np.exp(Q_eV/(kB_eV*T))
    dcv = (cv_eq(T)-cv)/tau - k_ann*cv**2 - k_sink*cv*rho_d
    return cv + dcv*dt

def step_rho(rho_d, epdot_eq, T, dt, A=1e14, B=1e-4, Qv_eV=0.8):
    """転位密度の時間発展"""
    kB_eV = 8.617e-5
    Dv = 1e-6*np.exp(-Qv_eV/(kB_eV*T))
    drho = A*max(epdot_eq,0.0) - B*rho_d*Dv
    return max(rho_d + drho*dt, 1e10)

def equiv_strain_rate(epsM_dot, epsm_dot):
    """相当ひずみ速度"""
    return np.sqrt(2.0/3.0)*np.sqrt((epsM_dot-epsm_dot)**2 + epsM_dot**2 + epsm_dot**2)

def mu_effective(mu0, T, pN, vslip):
    """温度・速度・荷重依存の有効摩擦係数"""
    s = (vslip * 1e3) / (pN / 1e6 + 1.0)
    stribeck = 0.7 + 0.3 / (1 + s)
    temp_reduction = 1.0 - 1e-4 * max(T - 293.15, 0)
    return mu0 * stribeck * temp_reduction

def flow_stress(ep_eq, epdot_eq, mat: MaterialParams, T=None, Tref=293.15, alpha=3e-4):
    """温度依存の流動応力"""
    rate_fac = (max(epdot_eq,1e-6)/1.0)**mat.m
    aniso = (2.0 + mat.r_value)/3.0
    temp_fac = 1.0 - alpha*max((0 if T is None else (T-Tref)), 0.0)
    return mat.sigma0 * temp_fac * (1.0 + ep_eq)**mat.n * rate_fac / aniso

def sanity_check(schedule: PressSchedule):
    """入力データの妥当性チェック"""
    assert np.all(schedule.pN < 5e9), "pN too large? Expected [Pa]"
    assert np.all(schedule.pN > 0), "pN must be positive [Pa]"
    assert np.all(schedule.Tdie > 150) and np.all(schedule.Tdie < 1500), "Tdie out of range?"
    assert np.all(schedule.t >= 0), "Time must be non-negative [s]"
    assert np.all(schedule.contact >= 0) and np.all(schedule.contact <= 1), "Contact rate must be in [0,1]"
    assert np.all(schedule.mu >= 0) and np.all(schedule.mu < 1), "Friction coefficient out of range"
    if len(schedule.t) > 1:
        dt = np.diff(schedule.t)
        assert np.all(dt > 0), "Time must be monotonically increasing"

def smooth_signal(x, window_size=11):
    """移動平均によるスムージング"""
    if window_size <= 1 or len(x) <= window_size:
        return x
    kernel = np.ones(window_size) / window_size
    padded = np.pad(x, (window_size//2, window_size//2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(x)]

def get_path_k_scale(beta: float, edr: EDRParams) -> float:
    """ひずみ経路に応じたK_scaleを返す"""
    if abs(beta + 0.5) < 0.1:  # 深絞り
        return edr.K_scale_draw
    elif abs(beta) < 0.1:  # 平面ひずみ
        return edr.K_scale_plane
    elif abs(beta - 0.5) < 0.2:  # 等二軸
        return edr.K_scale_biax
    else:
        return edr.K_scale

# -----------------------------------------------------------------------------
# 2.2) メインシミュレーション関数（CPU版）
# -----------------------------------------------------------------------------

def simulate_lambda(schedule: PressSchedule,
                    mat: MaterialParams,
                    edr: EDRParams,
                    debug: bool = False) -> Dict[str, np.ndarray]:
    """
    CPU版 Λ計算（既存実装）
    
    Args:
        schedule: 時系列データ
        mat: 材料パラメータ
        edr: EDRパラメータ
        debug: デバッグ出力フラグ
    
    Returns:
        Dict containing: t, Lambda, Damage, T, sigma_eq, eps_maj, eps_min
    """
    # 入力チェック
    sanity_check(schedule)
    
    t = schedule.t
    N = len(t)
    
    # 等間隔補間
    dt_mean = np.mean(np.diff(t))
    if not np.allclose(np.diff(t), dt_mean, rtol=1e-2, atol=1e-4):
        t_uniform = np.arange(t[0], t[-1]+dt_mean, dt_mean)
        def interp(x): return np.interp(t_uniform, t, x)
        epsM = interp(schedule.eps_maj)
        epsm = interp(schedule.eps_min)
        tria = interp(schedule.triax)
        mu   = interp(schedule.mu)
        pN   = interp(schedule.pN)
        vs   = interp(schedule.vslip)
        htc  = interp(schedule.htc)
        Tdie = interp(schedule.Tdie)
        ctc  = np.clip(interp(schedule.contact),0,1)
        t = t_uniform
    else:
        epsM, epsm = schedule.eps_maj, schedule.eps_min
        tria = schedule.triax; mu = schedule.mu; pN = schedule.pN
        vs = schedule.vslip; htc = schedule.htc; Tdie = schedule.Tdie
        ctc = np.clip(schedule.contact,0,1)

    dt = np.mean(np.diff(t))
    epsM_dot = np.gradient(epsM, dt)
    epsm_dot = np.gradient(epsm, dt)
    epdot_eq = equiv_strain_rate(epsM_dot, epsm_dot)

    T = np.full_like(t, schedule.T0, dtype=float)
    cv = 1e-7
    rho_d = 1e11
    ep_eq = 0.0

    Lam = np.zeros_like(t[:-1])
    D   = np.zeros_like(t[:-1])
    sigma_eq_log = np.zeros_like(t[:-1])

    rho = mat.rho; cp = mat.cp; h0 = mat.thickness
    
    # 板厚関連
    h_eff = h0
    eps3 = 0.0
    
    # 経路平均β
    beta_avg = np.mean(epsm / (epsM + 1e-10))
    k_scale_path = get_path_k_scale(beta_avg, edr)
    
    # β履歴
    beta_hist = deque(maxlen=5)

    # 時間発展ループ
    for k in range(len(t)-1):
        # 板厚更新
        d_eps3 = - (epsM_dot[k] + epsm_dot[k]) * dt
        eps3 += d_eps3
        h_eff = max(h0 * np.exp(eps3), 0.2*h0)
        
        # 熱収支
        q_fric = mu[k]*pN[k]*vs[k]*ctc[k]
        dTdt = (2.0*htc[k]*(Tdie[k]-T[k]) + 2.0*edr.chi*q_fric) / (rho*cp*h_eff)
        dTdt = np.clip(dTdt, -1000, 1000)
        T[k+1] = T[k] + dTdt*dt
        T[k+1] = np.clip(T[k+1], 200, 2000)

        # 欠陥更新
        rho_d = step_rho(rho_d, epdot_eq[k], T[k], dt)
        cv    = step_cv(cv, T[k], rho_d, dt)

        # K計算（加熱時のみ）
        K_th = rho*cp*max(dTdt, 0.0)
        
        sigma_eq = flow_stress(ep_eq, epdot_eq[k], mat, T=T[k])
        K_pl = 0.9 * sigma_eq * epdot_eq[k]
        
        mu_eff = mu_effective(mu[k], T[k], pN[k], vs[k])
        q_fric_eff = mu_eff * pN[k] * vs[k] * ctc[k]
        K_fr = (2.0*edr.chi*q_fric_eff)/h_eff
        
        # 瞬間β
        num = epsm_dot[k]
        den = epsM_dot[k] if abs(epsM_dot[k]) > 1e-8 else np.sign(epsM_dot[k])*1e-8 + 1e-8
        beta_inst = num / den
        beta_hist.append(beta_inst)
        beta_smooth = float(np.mean(beta_hist))
        
        # K_total
        K_total = k_scale_path * (K_th + K_pl + K_fr)
        K_total *= beta_multiplier(beta_smooth, A=edr.beta_A, bw=edr.beta_bw)
        K_total = max(K_total, 0)

        # V_eff
        T_ratio = min((T[k] - 273.15) / (1500.0 - 273.15), 1.0)
        temp_factor = 1.0 - 0.5 * T_ratio
        V_eff = edr.V0 * temp_factor * (1.0 - edr.av*cv - edr.ad*np.sqrt(max(rho_d,1e10)))
        V_eff = max(V_eff, 0.01*edr.V0)

        # 三軸度
        D_triax = np.exp(-edr.triax_sens*max(tria[k],0.0))

        # Λ計算
        Lam[k] = K_total / max(V_eff*D_triax, 1e7)
        Lam[k] = min(Lam[k], 10.0)
        
        # 損傷積分
        D[k] = (D[k-1] if k>0 else 0.0) + max(Lam[k]-edr.Lambda_crit, 0.0)*dt
        ep_eq += epdot_eq[k]*dt
        sigma_eq_log[k] = sigma_eq

    if debug:
        print(f"T_max: {T.max()-273:.1f}°C, Λ_max: {Lam.max():.3f}, "
              f"σ_max: {sigma_eq_log.max()/1e6:.1f}MPa, D_end: {D[-1]:.4f}")

    return {
        "t": t[:-1], "Lambda": Lam, "Damage": D, "T": T[:-1],
        "sigma_eq": sigma_eq_log, "eps_maj": epsM[:-1], "eps_min": epsm[:-1]
    }

# -----------------------------------------------------------------------------
# 2.3) 損失関数
# -----------------------------------------------------------------------------

def loss_for_binary_improved_v2(exps: List[ExpBinary],
                                mat: MaterialParams,
                                edr: EDRParams,
                                margin: float=0.08,
                                Dcrit: float=0.01,
                                debug: bool=False) -> float:
    """改善版損失関数v2：スムージング＋D積分判定＋安全マージン"""
    loss = 0.0
    correct = 0
    delta = 0.03
    
    for i, e in enumerate(exps):
        res = simulate_lambda(e.schedule, mat, edr, debug=False)
        
        Lam_raw = res["Lambda"]
        Lam_smooth = smooth_signal(Lam_raw, window_size=11)
        
        peak = float(np.max(Lam_smooth))
        D_end = float(res["Damage"][-1])
        
        if e.failed == 1:
            # 破断：ピーク超え「かつ」滞留も必要
            condition_met = (peak > edr.Lambda_crit and D_end > Dcrit)
            if not condition_met:
                peak_penalty = max(0, edr.Lambda_crit - peak)**2
                D_penalty = max(0, Dcrit - D_end)**2
                loss += 10.0 * (peak_penalty + D_penalty)
            else:
                correct += 1
                if peak < edr.Lambda_crit + margin:
                    loss += (edr.Lambda_crit + margin - peak)**2
                if D_end < 2*Dcrit:
                    loss += (2*Dcrit - D_end)**2
        else:
            # 安全
            if peak > edr.Lambda_crit - delta:
                loss += (peak - (edr.Lambda_crit - delta))**2 * 3.0
            if D_end >= 0.5*Dcrit:
                loss += 10.0 * (D_end - 0.5*Dcrit)**2
            else:
                correct += 1
        
        if debug:
            if e.failed == 1:
                status = "✓" if (peak > edr.Lambda_crit and D_end > Dcrit) else "✗"
            else:
                status = "✓" if (peak < edr.Lambda_crit - delta) else "✗"
            print(f"Exp{i}({e.label}): Λ_max={peak:.3f}, D={D_end:.4f}, "
                  f"failed={e.failed}, {status}")
    
    accuracy = correct / len(exps) if exps else 0
    if debug:
        print(f"Accuracy: {accuracy:.2%}")
    
    return loss / max(len(exps), 1)

def loss_for_flc(flc_pts: List[FLCPoint],
                 mat: MaterialParams,
                 edr: EDRParams) -> float:
    """FLC誤差（β重み付け版）"""
    err = 0.0
    for p in flc_pts:
        w = 1.5 if abs(p.path_ratio) < 0.1 else 1.0
        Em, em = predict_FLC_point(
            path_ratio=p.path_ratio,
            major_rate=p.rate_major,
            duration_max=p.duration_max,
            mat=mat, edr=edr
        )
        err += w * ((Em - p.major_limit)**2 + (em - p.minor_limit)**2)
    return err / max(len(flc_pts), 1)

# -----------------------------------------------------------------------------
# 2.4) 段階的フィッティング（CPU版）
# -----------------------------------------------------------------------------

def fit_step1_critical_params_v2(exps: List[ExpBinary],
                                 mat: MaterialParams,
                                 initial_edr: EDRParams,
                                 verbose: bool = True) -> EDRParams:
    """Step1: K_scale系とtriax_sensの最適化"""
    if verbose:
        print("\n=== Step 1: K_scale variants & triax_sens optimization ===")
    
    def objective(x):
        edr = EDRParams(
            V0=initial_edr.V0, av=initial_edr.av, ad=initial_edr.ad, chi=initial_edr.chi,
            K_scale=x[0], triax_sens=x[1], Lambda_crit=initial_edr.Lambda_crit,
            K_scale_draw=x[2], K_scale_plane=x[3], K_scale_biax=x[4],
            beta_A=initial_edr.beta_A, beta_bw=initial_edr.beta_bw
        )
        return loss_for_binary_improved_v2(exps, mat, edr, margin=0.08, Dcrit=0.01)
    
    x0 = [initial_edr.K_scale, 0.3, 0.15, 0.25, 0.20]
    bounds = [
        (0.05, 1.0), (0.1, 0.5), (0.05, 0.3), (0.1, 0.4), (0.05, 0.3)
    ]
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 300})
    
    updated_edr = EDRParams(
        V0=initial_edr.V0, av=initial_edr.av, ad=initial_edr.ad, chi=initial_edr.chi,
        K_scale=res.x[0], triax_sens=res.x[1], Lambda_crit=initial_edr.Lambda_crit,
        K_scale_draw=res.x[2], K_scale_plane=res.x[3], K_scale_biax=res.x[4],
        beta_A=initial_edr.beta_A, beta_bw=initial_edr.beta_bw
    )
    
    if verbose:
        print(f"K_scale: {initial_edr.K_scale:.3f} -> {res.x[0]:.3f}")
        print(f"triax_sens: {initial_edr.triax_sens:.3f} -> {res.x[1]:.3f}")
        print(f"K_scale_draw: {res.x[2]:.3f}, K_scale_plane: {res.x[3]:.3f}, K_scale_biax: {res.x[4]:.3f}")
        print(f"Loss: {res.fun:.4f}")
    
    return updated_edr

def fit_step2_V0(exps: List[ExpBinary],
                 mat: MaterialParams,
                 edr_from_step1: EDRParams,
                 verbose: bool = True) -> EDRParams:
    """Step2: V0を追加最適化"""
    if verbose:
        print("\n=== Step 2: V0 optimization ===")
    
    def objective(x):
        edr = EDRParams(
            V0=x[0], av=edr_from_step1.av, ad=edr_from_step1.ad, chi=edr_from_step1.chi,
            K_scale=edr_from_step1.K_scale, triax_sens=edr_from_step1.triax_sens,
            Lambda_crit=edr_from_step1.Lambda_crit,
            K_scale_draw=edr_from_step1.K_scale_draw,
            K_scale_plane=edr_from_step1.K_scale_plane,
            K_scale_biax=edr_from_step1.K_scale_biax,
            beta_A=edr_from_step1.beta_A, beta_bw=edr_from_step1.beta_bw
        )
        return loss_for_binary_improved_v2(exps, mat, edr, margin=0.08, Dcrit=0.01)
    
    x0 = [edr_from_step1.V0]
    bounds = [(5e8, 5e9)]
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
    
    updated_edr = EDRParams(
        V0=res.x[0], av=edr_from_step1.av, ad=edr_from_step1.ad, chi=edr_from_step1.chi,
        K_scale=edr_from_step1.K_scale, triax_sens=edr_from_step1.triax_sens,
        Lambda_crit=edr_from_step1.Lambda_crit,
        K_scale_draw=edr_from_step1.K_scale_draw,
        K_scale_plane=edr_from_step1.K_scale_plane,
        K_scale_biax=edr_from_step1.K_scale_biax,
        beta_A=edr_from_step1.beta_A, beta_bw=edr_from_step1.beta_bw
    )
    
    if verbose:
        print(f"V0: {edr_from_step1.V0:.2e} -> {res.x[0]:.2e}")
        print(f"Loss: {res.fun:.4f}")
    
    return updated_edr

def fit_step3_fine_tuning_v2(exps: List[ExpBinary],
                             flc_pts: List[FLCPoint],
                             mat: MaterialParams,
                             edr_from_step2: EDRParams,
                             verbose: bool = True) -> Tuple[EDRParams, Dict]:
    """Step3: 全パラメータ微調整"""
    if verbose:
        print("\n=== Step 3: Fine tuning all parameters ===")
    
    theta0 = np.array([
        edr_from_step2.V0, edr_from_step2.av, edr_from_step2.ad, edr_from_step2.chi,
        edr_from_step2.K_scale, edr_from_step2.triax_sens, 1.0,
        edr_from_step2.K_scale_draw, edr_from_step2.K_scale_plane, edr_from_step2.K_scale_biax,
        edr_from_step2.beta_A, edr_from_step2.beta_bw
    ])
    
    bounds = [
        (theta0[0]*0.5, theta0[0]*2.0), (1e4, 1e6), (1e-8, 1e-6), (0.05, 0.3),
        (0.05, 1.0), (0.1, 0.5), (0.95, 1.05),
        (0.05, 0.3), (0.1, 0.4), (0.05, 0.3), (0.2, 0.5), (0.2, 0.35)
    ]
    
    def objective(theta):
        edr = EDRParams(
            V0=theta[0], av=theta[1], ad=theta[2], chi=theta[3],
            K_scale=theta[4], triax_sens=theta[5], Lambda_crit=theta[6],
            K_scale_draw=theta[7], K_scale_plane=theta[8], K_scale_biax=theta[9],
            beta_A=theta[10], beta_bw=theta[11]
        )
        L_binary = loss_for_binary_improved_v2(exps, mat, edr, margin=0.08, Dcrit=0.01)
        L_flc = loss_for_flc(flc_pts, mat, edr) if flc_pts else 0.0
        return L_binary + 0.8 * L_flc
    
    res = differential_evolution(objective, bounds, seed=42, maxiter=150, popsize=20,
                                atol=1e-10, tol=1e-10)
    
    final_edr = EDRParams(
        V0=res.x[0], av=res.x[1], ad=res.x[2], chi=res.x[3],
        K_scale=res.x[4], triax_sens=res.x[5], Lambda_crit=res.x[6],
        K_scale_draw=res.x[7], K_scale_plane=res.x[8], K_scale_biax=res.x[9],
        beta_A=res.x[10], beta_bw=res.x[11]
    )
    
    info = {'success': res.success, 'fval': res.fun, 'nit': res.nit, 'message': res.message}
    
    if verbose:
        print(f"Final loss: {res.fun:.4f}")
        print(f"Iterations: {res.nit}")
        print(f"Lambda_crit: {res.x[6]:.3f}, triax_sens: {res.x[5]:.3f}")
    
    return final_edr, info

def fit_edr_params_staged_v2(binary_exps: List[ExpBinary],
                             flc_pts: List[FLCPoint],
                             mat: MaterialParams,
                             initial_edr: Optional[EDRParams] = None,
                             verbose: bool = True) -> Tuple[EDRParams, Dict]:
    """段階的フィッティングのメイン関数（CPU版）"""
    
    if initial_edr is None:
        initial_edr = EDRParams()
    
    edr_step1 = fit_step1_critical_params_v2(binary_exps, mat, initial_edr, verbose)
    edr_step2 = fit_step2_V0(binary_exps, mat, edr_step1, verbose)
    final_edr, info = fit_step3_fine_tuning_v2(binary_exps, flc_pts, mat, edr_step2, verbose)
    
    if verbose:
        print("\n=== Final Validation ===")
        loss_final = loss_for_binary_improved_v2(binary_exps, mat, final_edr, 
                                                 margin=0.08, Dcrit=0.01, debug=True)
        print(f"Final binary loss: {loss_final:.4f}")
    
    return final_edr, info

# -----------------------------------------------------------------------------
# 2.5) ヘルパー関数
# -----------------------------------------------------------------------------

def predict_fail(res: Dict[str,np.ndarray], margin: float=0.0) -> int:
    Lam_smooth = smooth_signal(res["Lambda"], window_size=11)
    return int(np.max(Lam_smooth) > 1.0 + margin)

def time_at_lambda_cross(res: Dict[str,np.ndarray], crit: float=1.0) -> Optional[int]:
    Lam_smooth = smooth_signal(res["Lambda"], window_size=11)
    idx = np.where(Lam_smooth>crit)[0]
    return int(idx[0]) if len(idx)>0 else None

def predict_FLC_point(path_ratio: float, major_rate: float, duration_max: float,
                     mat: MaterialParams, edr: EDRParams,
                     base_contact: float=1.0, base_mu: float=0.08,
                     base_pN: float=200e6, base_vslip: float=0.02,
                     base_htc: float=8000.0, Tdie: float=293.15,
                     T0: float=293.15) -> Tuple[float,float]:
    """FLC点予測"""
    dt = 1e-3
    N  = int(duration_max/dt)+1
    t  = np.linspace(0, duration_max, N)
    epsM = major_rate*t
    epsm = path_ratio*major_rate*t

    schedule = PressSchedule(
        t=t, eps_maj=epsM, eps_min=epsm,
        triax=np.full(N, triax_from_path(path_ratio)),
        mu=np.full(N, base_mu), pN=np.full(N, base_pN),
        vslip=np.full(N, base_vslip), htc=np.full(N, base_htc),
        Tdie=np.full(N, Tdie), contact=np.full(N, base_contact), T0=T0
    )
    res = simulate_lambda(schedule, mat, edr)
    k = time_at_lambda_cross(res, crit=edr.Lambda_crit)
    if k is None:
        return float(res["eps_maj"][-1]), float(res["eps_min"][-1])
    return float(res["eps_maj"][k]), float(res["eps_min"][k])

# -----------------------------------------------------------------------------
# 2.6) プロット関数
# -----------------------------------------------------------------------------

def plot_flc(experimental: List[FLCPoint], predicted: List[Tuple[float,float]],
             title: str = "FLC: Experimental vs EDR v2"):
    fig = plt.figure(figsize=(8, 6))
    Em_exp = [p.major_limit for p in experimental]
    em_exp = [p.minor_limit for p in experimental]
    Em_pre = [p[0] for p in predicted]
    em_pre = [p[1] for p in predicted]
    plt.plot(em_exp, Em_exp, 'o', markersize=10, label='Experimental FLC')
    plt.plot(em_pre, Em_pre, 's--', markersize=8, label='EDR v2 Predicted')
    plt.xlabel('Minor strain')
    plt.ylabel('Major strain')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_lambda_t(res: Dict[str,np.ndarray], title="Lambda timeline v2"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    Lam_smooth = smooth_signal(res["Lambda"], window_size=11)
    ax1.plot(res["t"], res["Lambda"], 'b-', alpha=0.3, linewidth=1, label='Raw Λ')
    ax1.plot(res["t"], Lam_smooth, 'b-', linewidth=2, label='Smoothed Λ')
    ax1.axhline(1.0, ls='--', color='red', alpha=0.5, label='Λ_crit')
    ax1.fill_between(res["t"], 0, Lam_smooth, where=(Lam_smooth>1.0), color='red', alpha=0.2)
    ax1.set_ylabel('Lambda (Λ)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(res["t"], res["Damage"], 'g-', linewidth=2, label='Damage D')
    ax2.axhline(0.05, ls='--', color='orange', alpha=0.5, label='D_crit')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Damage D')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# Section 3: CUDA版実装（高速化）
# =============================================================================

if CUDA_AVAILABLE:
    
    THREADS_PER_BLOCK = 64
    WARP_SIZE = 32
    
    # -------------------------------------------------------------------------
    # 3.1) Device関数（CUDA）
    # -------------------------------------------------------------------------
    
    @cuda.jit(device=True, inline=True)
    def triax_from_path_device(beta):
        b = max(min(beta, 1.0), -0.95)
        return (1.0 + b) / (math.sqrt(3.0) * math.sqrt(1.0 + b + b*b))
    
    @cuda.jit(device=True, inline=True)
    def beta_multiplier_device(beta, A, bw):
        b = max(min(beta, 0.95), -0.95)
        return 1.0 + A * math.exp(-(b / bw) * (b / bw))
    
    @cuda.jit(device=True, inline=True)
    def cv_eq_device(T, c0=1e-6, Ev_eV=1.0):
        kB_eV = 8.617e-5
        return c0 * math.exp(-Ev_eV / (kB_eV * T))
    
    @cuda.jit(device=True, inline=True)
    def step_cv_device(cv, T, rho_d, dt):
        tau0 = 1e-3; Q_eV = 0.8; k_ann = 1e6; k_sink = 1e-15; kB_eV = 8.617e-5
        tau = tau0 * math.exp(Q_eV / (kB_eV * T))
        dcv = (cv_eq_device(T) - cv) / tau - k_ann * cv * cv - k_sink * cv * rho_d
        return cv + dcv * dt
    
    @cuda.jit(device=True, inline=True)
    def step_rho_device(rho_d, epdot_eq, T, dt):
        A = 1e14; B = 1e-4; Qv_eV = 0.8; kB_eV = 8.617e-5
        Dv = 1e-6 * math.exp(-Qv_eV / (kB_eV * T))
        drho = A * max(epdot_eq, 0.0) - B * rho_d * Dv
        return max(rho_d + drho * dt, 1e10)
    
    @cuda.jit(device=True, inline=True)
    def equiv_strain_rate_device(epsM_dot, epsm_dot):
        sqrt_2_3 = 0.8164965809277260
        return sqrt_2_3 * math.sqrt(
            (epsM_dot - epsm_dot) * (epsM_dot - epsm_dot) + 
            epsM_dot * epsM_dot + epsm_dot * epsm_dot
        )
    
    @cuda.jit(device=True, inline=True)
    def mu_effective_device(mu0, T, pN, vslip):
        s = (vslip * 1e3) / (pN / 1e6 + 1.0)
        stribeck = 0.7 + 0.3 / (1.0 + s)
        temp_reduction = 1.0 - 1e-4 * max(T - 293.15, 0.0)
        return mu0 * stribeck * temp_reduction
    
    @cuda.jit(device=True, inline=True)
    def flow_stress_device(ep_eq, epdot_eq, sigma0, n, m, r_value, T):
        Tref = 293.15; alpha = 3e-4
        rate_fac = math.pow(max(epdot_eq, 1e-6) / 1.0, m)
        aniso = (2.0 + r_value) / 3.0
        temp_fac = 1.0 - alpha * max(T - Tref, 0.0)
        return sigma0 * temp_fac * math.pow(1.0 + ep_eq, n) * rate_fac / aniso
    
    @cuda.jit(device=True, inline=True)
    def get_k_scale_path_device(beta_avg, K_scale, K_scale_draw, K_scale_plane, K_scale_biax):
        if abs(beta_avg + 0.5) < 0.1:
            return K_scale_draw
        elif abs(beta_avg) < 0.1:
            return K_scale_plane
        elif abs(beta_avg - 0.5) < 0.2:
            return K_scale_biax
        else:
            return K_scale
    
    # -------------------------------------------------------------------------
    # 3.2) メインカーネル
    # -------------------------------------------------------------------------
    
    @cuda.jit
    def eval_candidates_kernel(
        # スケジュール: [n_paths, n_time]
        epsM, epsm, triax, mu, pN, vslip, htc, Tdie, contact, T0, dt,
        # 材料パラメータ
        rho, cp, h0, sigma0, n_param, m_param, r_value,
        # 候補パラメータ: [n_candidates]
        V0_arr, av_arr, ad_arr, chi_arr, K_scale_arr, triax_sens_arr, Lambda_crit_arr,
        K_scale_draw_arr, K_scale_plane_arr, K_scale_biax_arr, beta_A_arr, beta_bw_arr,
        # ラベル: [n_paths]
        failed_labels,
        # 出力: [n_candidates]
        loss_out
    ):
        """
        CUDA評価カーネル（B + A 設計）
        
        Grid:  n_candidates ブロック
        Block: THREADS_PER_BLOCK スレッド
        
        各スレッドが (candidate, path) ペアを担当
        時間方向は逐次処理（状態依存）
        Λタイムライン不保持（peak, D のみ）
        """
        
        cand = cuda.blockIdx.x
        path = cuda.threadIdx.x
        
        n_paths = epsM.shape[0]
        n_time = epsM.shape[1]
        
        if path >= n_paths:
            return
        
        # 候補パラメータ読み込み
        V0 = V0_arr[cand]; av = av_arr[cand]; ad = ad_arr[cand]; chi = chi_arr[cand]
        K_scale = K_scale_arr[cand]; triax_sens = triax_sens_arr[cand]
        Lcrit = Lambda_crit_arr[cand]
        K_scale_draw = K_scale_draw_arr[cand]
        K_scale_plane = K_scale_plane_arr[cand]
        K_scale_biax = K_scale_biax_arr[cand]
        beta_A = beta_A_arr[cand]; beta_bw = beta_bw_arr[cand]
        failed = failed_labels[path]
        
        # 経路平均β
        beta_sum = 0.0
        for t in range(n_time):
            em = epsm[path, t]; eM = epsM[path, t]
            denom = eM if abs(eM) > 1e-8 else (1e-8 if eM >= 0.0 else -1e-8)
            beta_sum += em / denom
        beta_avg = beta_sum / n_time
        k_scale_path = get_k_scale_path_device(beta_avg, K_scale, K_scale_draw, 
                                               K_scale_plane, K_scale_biax)
        
        # 状態変数初期化（全てレジスタ）
        T = T0[path]; cv = 1e-7; rho_d = 1e11; ep_eq = 0.0; eps3 = 0.0; h_eff = h0
        peak_Lambda = 0.0; D_cumsum = 0.0
        beta_hist_0 = 0.0; beta_hist_1 = 0.0; beta_hist_2 = 0.0
        beta_hist_3 = 0.0; beta_hist_4 = 0.0; beta_count = 0
        
        # 時間発展ループ（逐次）
        for t in range(n_time - 1):
            eM0 = epsM[path, t]; eM1 = epsM[path, t + 1]
            em0 = epsm[path, t]; em1 = epsm[path, t + 1]
            epsM_dot = (eM1 - eM0) / dt
            epsm_dot = (em1 - em0) / dt
            epdot_eq = equiv_strain_rate_device(epsM_dot, epsm_dot)
            
            # 板厚更新
            d_eps3 = -(epsM_dot + epsm_dot) * dt
            eps3 += d_eps3
            h_eff = max(h0 * math.exp(eps3), 0.2 * h0)
            
            # 熱収支
            q_fric = mu[path, t] * pN[path, t] * vslip[path, t] * contact[path, t]
            dTdt = (2.0 * htc[path, t] * (Tdie[path, t] - T) + 2.0 * chi * q_fric) / (rho * cp * h_eff)
            dTdt = max(min(dTdt, 1000.0), -1000.0)
            T = max(min(T + dTdt * dt, 2000.0), 200.0)
            
            # 欠陥更新
            rho_d = step_rho_device(rho_d, epdot_eq, T, dt)
            cv = step_cv_device(cv, T, rho_d, dt)
            
            # K計算
            K_th = rho * cp * max(dTdt, 0.0)
            sigma_eq = flow_stress_device(ep_eq, epdot_eq, sigma0, n_param, m_param, r_value, T)
            K_pl = 0.9 * sigma_eq * epdot_eq
            mu_eff = mu_effective_device(mu[path, t], T, pN[path, t], vslip[path, t])
            q_fric_eff = mu_eff * pN[path, t] * vslip[path, t] * contact[path, t]
            K_fr = (2.0 * chi * q_fric_eff) / h_eff
            
            # β移動平均
            denom = epsM_dot if abs(epsM_dot) > 1e-8 else (1e-8 if epsM_dot >= 0.0 else -1e-8)
            beta_inst = epsm_dot / denom
            
            if beta_count < 5:
                if beta_count == 0: beta_hist_0 = beta_inst
                elif beta_count == 1: beta_hist_1 = beta_inst
                elif beta_count == 2: beta_hist_2 = beta_inst
                elif beta_count == 3: beta_hist_3 = beta_inst
                elif beta_count == 4: beta_hist_4 = beta_inst
                beta_count += 1
                beta_smooth = (beta_hist_0 + beta_hist_1 + beta_hist_2 + 
                              beta_hist_3 + beta_hist_4) / beta_count
            else:
                beta_hist_0 = beta_hist_1; beta_hist_1 = beta_hist_2
                beta_hist_2 = beta_hist_3; beta_hist_3 = beta_hist_4
                beta_hist_4 = beta_inst
                beta_smooth = (beta_hist_0 + beta_hist_1 + beta_hist_2 + 
                              beta_hist_3 + beta_hist_4) * 0.2
            
            # K_total
            K_total = k_scale_path * (K_th + K_pl + K_fr)
            K_total *= beta_multiplier_device(beta_smooth, beta_A, beta_bw)
            K_total = max(K_total, 0.0)
            
            # V_eff
            T_ratio = min((T - 273.15) / (1500.0 - 273.15), 1.0)
            temp_factor = 1.0 - 0.5 * T_ratio
            V_eff = V0 * temp_factor * (1.0 - av * cv - ad * math.sqrt(max(rho_d, 1e10)))
            V_eff = max(V_eff, 0.01 * V0)
            
            # 三軸度
            D_triax = math.exp(-triax_sens * max(triax[path, t], 0.0))
            
            # Λ計算と集計
            Lam = K_total / max(V_eff * D_triax, 1e7)
            Lam = min(Lam, 10.0)
            
            if Lam > peak_Lambda:
                peak_Lambda = Lam
            if Lam > Lcrit:
                D_cumsum += (Lam - Lcrit) * dt
            
            ep_eq += epdot_eq * dt
        
        # 損失計算
        margin = 0.08; Dcrit = 0.01; delta = 0.03
        path_loss = 0.0
        
        if failed == 1:
            condition_met = (peak_Lambda > Lcrit) and (D_cumsum > Dcrit)
            if not condition_met:
                peak_penalty = max(0.0, Lcrit - peak_Lambda)
                D_penalty = max(0.0, Dcrit - D_cumsum)
                path_loss = 10.0 * (peak_penalty * peak_penalty + D_penalty * D_penalty)
            else:
                if peak_Lambda < Lcrit + margin:
                    path_loss += (Lcrit + margin - peak_Lambda) * (Lcrit + margin - peak_Lambda)
                if D_cumsum < 2.0 * Dcrit:
                    path_loss += (2.0 * Dcrit - D_cumsum) * (2.0 * Dcrit - D_cumsum)
        else:
            if peak_Lambda > Lcrit - delta:
                path_loss += (peak_Lambda - (Lcrit - delta)) * (peak_Lambda - (Lcrit - delta)) * 3.0
            if D_cumsum >= 0.5 * Dcrit:
                path_loss += 10.0 * (D_cumsum - 0.5 * Dcrit) * (D_cumsum - 0.5 * Dcrit)
        
        # ブロック内集約
        shared_losses = cuda.shared.array(THREADS_PER_BLOCK, dtype=float64)
        shared_losses[path] = path_loss
        cuda.syncthreads()
        
        s = THREADS_PER_BLOCK // 2
        while s > 0:
            if path < s and path + s < n_paths:
                shared_losses[path] += shared_losses[path + s]
            cuda.syncthreads()
            s //= 2
        
        if path == 0:
            loss_out[cand] = shared_losses[0] / n_paths
    
    # -------------------------------------------------------------------------
    # 3.3) ホスト側ラッパー
    # -------------------------------------------------------------------------
    
    class CUDAOptimizer:
        """CUDA最適化器"""
        
        def __init__(self, schedules_list: List[Dict], failed_labels: List[int], mat: MaterialParams):
            self.n_paths = len(schedules_list)
            self.mat = mat
            self.failed_labels = np.array(failed_labels, dtype=np.int32)
            
            # データ整形
            max_len = max(len(s['t']) for s in schedules_list)
            self.n_time = max_len
            
            # ホスト配列（SoA）
            self.epsM = np.zeros((self.n_paths, max_len), dtype=np.float64)
            self.epsm = np.zeros((self.n_paths, max_len), dtype=np.float64)
            self.triax = np.zeros((self.n_paths, max_len), dtype=np.float64)
            self.mu = np.zeros((self.n_paths, max_len), dtype=np.float64)
            self.pN = np.zeros((self.n_paths, max_len), dtype=np.float64)
            self.vslip = np.zeros((self.n_paths, max_len), dtype=np.float64)
            self.htc = np.zeros((self.n_paths, max_len), dtype=np.float64)
            self.Tdie = np.zeros((self.n_paths, max_len), dtype=np.float64)
            self.contact = np.zeros((self.n_paths, max_len), dtype=np.float64)
            self.T0 = np.zeros(self.n_paths, dtype=np.float64)
            
            # コピー＋パディング
            for i, sched in enumerate(schedules_list):
                n = len(sched['t'])
                self.epsM[i, :n] = sched['eps_maj']
                self.epsm[i, :n] = sched['eps_min']
                self.triax[i, :n] = sched['triax']
                self.mu[i, :n] = sched['mu']
                self.pN[i, :n] = sched['pN']
                self.vslip[i, :n] = sched['vslip']
                self.htc[i, :n] = sched['htc']
                self.Tdie[i, :n] = sched['Tdie']
                self.contact[i, :n] = sched['contact']
                self.T0[i] = sched.get('T0', 293.15)
                
                if n < max_len:
                    self.epsM[i, n:] = sched['eps_maj'][-1]
                    self.epsm[i, n:] = sched['eps_min'][-1]
                    self.triax[i, n:] = sched['triax'][-1]
                    self.mu[i, n:] = sched['mu'][-1]
                    self.pN[i, n:] = sched['pN'][-1]
                    self.vslip[i, n:] = sched['vslip'][-1]
                    self.htc[i, n:] = sched['htc'][-1]
                    self.Tdie[i, n:] = sched['Tdie'][-1]
                    self.contact[i, n:] = sched['contact'][-1]
            
            self.dt = (schedules_list[0]['t'][-1] - schedules_list[0]['t'][0]) / (len(schedules_list[0]['t']) - 1)
            
            # GPU転送（1回のみ）
            self.d_epsM = cuda.to_device(self.epsM)
            self.d_epsm = cuda.to_device(self.epsm)
            self.d_triax = cuda.to_device(self.triax)
            self.d_mu = cuda.to_device(self.mu)
            self.d_pN = cuda.to_device(self.pN)
            self.d_vslip = cuda.to_device(self.vslip)
            self.d_htc = cuda.to_device(self.htc)
            self.d_Tdie = cuda.to_device(self.Tdie)
            self.d_contact = cuda.to_device(self.contact)
            self.d_T0 = cuda.to_device(self.T0)
            self.d_failed_labels = cuda.to_device(self.failed_labels)
            
            print(f"✓ CUDA初期化完了: {self.n_paths} 経路 × {self.n_time} ステップ")
        
        def evaluate_candidates(self, candidates_list: List[EDRParams]) -> np.ndarray:
            """候補パラメータを一括評価"""
            n_candidates = len(candidates_list)
            
            # 候補配列化
            V0_arr = np.array([c.V0 for c in candidates_list], dtype=np.float64)
            av_arr = np.array([c.av for c in candidates_list], dtype=np.float64)
            ad_arr = np.array([c.ad for c in candidates_list], dtype=np.float64)
            chi_arr = np.array([c.chi for c in candidates_list], dtype=np.float64)
            K_scale_arr = np.array([c.K_scale for c in candidates_list], dtype=np.float64)
            triax_sens_arr = np.array([c.triax_sens for c in candidates_list], dtype=np.float64)
            Lambda_crit_arr = np.array([c.Lambda_crit for c in candidates_list], dtype=np.float64)
            K_scale_draw_arr = np.array([c.K_scale_draw for c in candidates_list], dtype=np.float64)
            K_scale_plane_arr = np.array([c.K_scale_plane for c in candidates_list], dtype=np.float64)
            K_scale_biax_arr = np.array([c.K_scale_biax for c in candidates_list], dtype=np.float64)
            beta_A_arr = np.array([c.beta_A for c in candidates_list], dtype=np.float64)
            beta_bw_arr = np.array([c.beta_bw for c in candidates_list], dtype=np.float64)
            
            # GPU転送
            d_V0 = cuda.to_device(V0_arr)
            d_av = cuda.to_device(av_arr)
            d_ad = cuda.to_device(ad_arr)
            d_chi = cuda.to_device(chi_arr)
            d_K_scale = cuda.to_device(K_scale_arr)
            d_triax_sens = cuda.to_device(triax_sens_arr)
            d_Lambda_crit = cuda.to_device(Lambda_crit_arr)
            d_K_scale_draw = cuda.to_device(K_scale_draw_arr)
            d_K_scale_plane = cuda.to_device(K_scale_plane_arr)
            d_K_scale_biax = cuda.to_device(K_scale_biax_arr)
            d_beta_A = cuda.to_device(beta_A_arr)
            d_beta_bw = cuda.to_device(beta_bw_arr)
            
            loss_out = np.zeros(n_candidates, dtype=np.float64)
            d_loss_out = cuda.to_device(loss_out)
            
            # カーネル起動
            eval_candidates_kernel[n_candidates, THREADS_PER_BLOCK](
                self.d_epsM, self.d_epsm, self.d_triax, self.d_mu, self.d_pN, 
                self.d_vslip, self.d_htc, self.d_Tdie, self.d_contact, self.d_T0, self.dt,
                self.mat.rho, self.mat.cp, self.mat.thickness, self.mat.sigma0, 
                self.mat.n, self.mat.m, self.mat.r_value,
                d_V0, d_av, d_ad, d_chi, d_K_scale, d_triax_sens, d_Lambda_crit,
                d_K_scale_draw, d_K_scale_plane, d_K_scale_biax, d_beta_A, d_beta_bw,
                self.d_failed_labels, d_loss_out
            )
            
            d_loss_out.copy_to_host(loss_out)
            return loss_out

# =============================================================================
# Section 4: 統合インターフェース
# =============================================================================

def convert_schedule_to_dict(schedule_np: PressSchedule) -> Dict:
    """PressSchedule → dict変換"""
    return {
        't': schedule_np.t, 'eps_maj': schedule_np.eps_maj, 'eps_min': schedule_np.eps_min,
        'triax': schedule_np.triax, 'mu': schedule_np.mu, 'pN': schedule_np.pN,
        'vslip': schedule_np.vslip, 'htc': schedule_np.htc, 'Tdie': schedule_np.Tdie,
        'contact': schedule_np.contact, 'T0': schedule_np.T0
    }

def benchmark_cpu_vs_cuda(exps: List[ExpBinary], mat: MaterialParams, edr: EDRParams, n_candidates: int = 100):
    """CPU vs CUDA ベンチマーク"""
    print("\n" + "="*60)
    print("CPU vs CUDA ベンチマーク")
    print("="*60)
    
    # CPU版
    print(f"\n[CPU版] 単一候補×{len(exps)}経路...")
    start = time.time()
    for e in exps:
        _ = simulate_lambda(e.schedule, mat, edr)
    time_cpu = time.time() - start
    print(f"  実行時間: {time_cpu:.4f}秒")
    print(f"  1経路あたり: {time_cpu/len(exps):.4f}秒")
    
    if not CUDA_AVAILABLE:
        print("\n⚠️  CUDA未利用: GPU版スキップ")
        return
    
    # CUDA版
    print(f"\n[CUDA版] {n_candidates}候補×{len(exps)}経路...")
    schedules_list = [convert_schedule_to_dict(e.schedule) for e in exps]
    failed_labels = [e.failed for e in exps]
    
    optimizer = CUDAOptimizer(schedules_list, failed_labels, mat)
    
    # 候補生成
    candidates = []
    for i in range(n_candidates):
        edr_test = EDRParams(
            V0=edr.V0 * (0.8 + 0.4 * i / n_candidates),
            av=edr.av, ad=edr.ad, chi=edr.chi,
            K_scale=edr.K_scale * (0.5 + i / n_candidates),
            triax_sens=edr.triax_sens, Lambda_crit=1.0,
            K_scale_draw=edr.K_scale_draw, K_scale_plane=edr.K_scale_plane,
            K_scale_biax=edr.K_scale_biax, beta_A=edr.beta_A, beta_bw=edr.beta_bw
        )
        candidates.append(edr_test)
    
    start = time.time()
    losses = optimizer.evaluate_candidates(candidates)
    time_cuda = time.time() - start
    
    print(f"  実行時間: {time_cuda:.4f}秒")
    print(f"  1候補あたり: {time_cuda/n_candidates*1000:.2f}ms")
    print(f"  1(候補×経路)あたり: {time_cuda/(n_candidates*len(exps))*1000:.2f}ms")
    
    # 比較
    print(f"\n[比較結果]")
    print(f"  CPU版想定（{n_candidates}候補）: {time_cpu * n_candidates:.1f}秒")
    print(f"  CUDA版実測: {time_cuda:.4f}秒")
    print(f"  高速化率: {time_cpu * n_candidates / time_cuda:.0f}x")
    
    # 4時間最適化の推定
    print(f"\n[4時間最適化の推定]")
    n_iterations_total = 3000  # 3段階×1000評価
    estimated_cuda = time_cuda / n_candidates * n_iterations_total / 60
    print(f"  元のコード: 240分（4時間）")
    print(f"  CUDA版推定: {estimated_cuda:.1f}分")
    print(f"  短縮時間: {240 - estimated_cuda:.1f}分")

# =============================================================================
# Section 5: デモデータ生成
# =============================================================================

def generate_demo_experiments() -> List[ExpBinary]:
    """デモ実験データ生成"""
    def mk_schedule(beta, mu_base, mu_jump=False, high_stress=False):
        dt = 1e-3; T = 0.6
        t = np.arange(0, T+dt, dt)
        
        if high_stress:
            epsM = 0.5 * (t/T)**0.8
        else:
            epsM = 0.35 * (t/T)
        epsm = beta * epsM
        
        mu = np.full_like(t, mu_base)
        if mu_jump:
            j = int(0.25/dt)
            mu[j:] += 0.06
        
        triax_val = triax_from_path(beta)
        
        return PressSchedule(
            t=t, eps_maj=epsM, eps_min=epsm,
            triax=np.full_like(t, triax_val), mu=mu,
            pN=np.full_like(t, 250e6 if high_stress else 200e6),
            vslip=np.full_like(t, 0.03), htc=np.full_like(t, 8000.0),
            Tdie=np.full_like(t, 293.15), contact=np.full_like(t, 1.0), T0=293.15
        )
    
    exps = [
        ExpBinary(mk_schedule(-0.5, 0.08, False, False), failed=0, label="safe_draw"),
        ExpBinary(mk_schedule(-0.5, 0.08, True, True), failed=1, label="draw_fail"),
        ExpBinary(mk_schedule(0.0, 0.08, False, False), failed=0, label="safe_plane"),
        ExpBinary(mk_schedule(0.0, 0.08, True, True), failed=1, label="plane_fail"),
        ExpBinary(mk_schedule(0.5, 0.10, False, False), failed=0, label="safe_biax"),
        ExpBinary(mk_schedule(0.5, 0.10, True, True), failed=1, label="biax_fail"),
    ]
    return exps

def generate_demo_flc() -> List[FLCPoint]:
    """デモFLCデータ生成"""
    return [
        FLCPoint(-0.5, 0.35, -0.175, 0.6, 1.0, "draw"),
        FLCPoint(0.0, 0.28, 0.0, 0.6, 1.0, "plane"),
        FLCPoint(0.5, 0.22, 0.11, 0.6, 1.0, "biax"),
    ]

def evaluate_flc_fit(experimental: List[FLCPoint], predicted: List[Tuple[float, float]]) -> float:
    """FLC適合度評価"""
    errors = []
    for exp, pred in zip(experimental, predicted):
        deM = pred[0] - exp.major_limit
        dem = pred[1] - exp.minor_limit
        err = np.sqrt(deM**2 + dem**2)
        errors.append(err)
        print(f"  β={exp.path_ratio:+.1f}: 誤差={err:.4f} (ΔMaj={deM:+.3f}, ΔMin={dem:+.3f})")
    
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    
    print(f"\nFLC適合度評価:")
    print(f"  平均誤差: {mean_err:.4f}")
    print(f"  最大誤差: {max_err:.4f}")
    print(f"  精度評価: ", end="")
    
    if mean_err < 0.05:
        print("✅ 優秀（<5%）")
    elif mean_err < 0.10:
        print("🟡 良好（<10%）")
    elif mean_err < 0.20:
        print("🟠 要改善（<20%）")
    else:
        print("🔴 不良（>20%）")
    
    return mean_err

# =============================================================================
# Section 6: メイン実行
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print(" EDRパラメータフィッティング統合版 v3.2 (CPU + CUDA)")
    print(" Miosync, Inc. / Nidec Corporation")
    print("="*80)
    
    # 材料パラメータ
    mat = MaterialParams()
    
    # デモデータ生成
    print("\n[デモデータ生成]")
    exps = generate_demo_experiments()
    flc_data = generate_demo_flc()
    print(f"  実験数: {len(exps)}")
    print(f"  FLC点数: {len(flc_data)}")
    
    # ベンチマーク（CPU vs CUDA）
    edr_init = EDRParams()
    benchmark_cpu_vs_cuda(exps, mat, edr_init, n_candidates=100)
    
    # CPU版フィッティング（デモ）
    print("\n" + "="*80)
    print(" CPU版フィッティング実行（デモ）")
    print("="*80)
    
    edr_fit, info = fit_edr_params_staged_v2(exps, flc_data, mat, verbose=True)
    
    print("\n[最終結果]")
    print(f"  V0: {edr_fit.V0:.2e} Pa")
    print(f"  K_scale: {edr_fit.K_scale:.3f}")
    print(f"  triax_sens: {edr_fit.triax_sens:.3f}")
    print(f"  Lambda_crit: {edr_fit.Lambda_crit:.3f}")
    
    # FLC予測
    print("\n[FLC予測]")
    preds = []
    for p in flc_data:
        Em, em = predict_FLC_point(p.path_ratio, p.rate_major, p.duration_max, mat, edr_fit)
        preds.append((Em, em))
        print(f"  β={p.path_ratio:+.1f}: 実測({p.major_limit:.3f}, {p.minor_limit:.3f}) "
              f"→ 予測({Em:.3f}, {em:.3f})")
    
    flc_error = evaluate_flc_fit(flc_data, preds)
    
    # プロット
    print("\n[プロット生成]")
    plot_flc(flc_data, preds, title="FLC: Experimental vs EDR (Unified)")
    
    res = simulate_lambda(exps[3].schedule, mat, edr_fit)
    plot_lambda_t(res, title="Lambda & Damage (plane strain failure) - Unified")
    
    print("\n" + "="*80)
    print(" 実行完了！")
    print(" Nidecプレゼン用統合版準備完了 ✅")
    print("="*80)
