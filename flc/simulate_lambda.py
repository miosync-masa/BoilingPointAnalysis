"""
EDRパラメータフィッティング改善版v3.1
環の提案 + ご主人さまの物理的洞察を統合
- 冷却は回復側として扱う（K_thは加熱時のみ）
- スムージングとD積分による安定判定
- パラメータ境界の最適化
- β依存ゲインでFLCのV字形状を再現
- 温度依存の流動応力・摩擦係数（物理増強）
- 入力データの妥当性チェック
- FLC損失のβ重み付け（平面ひずみ重視）
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple, Optional
from scipy.optimize import minimize, Bounds, differential_evolution
from scipy.signal import savgol_filter

# =========================
# 1) データ構造
# =========================

@dataclass
class MaterialParams:
    rho: float = 7800.0      # kg/m3
    cp: float = 500.0        # J/kg/K
    k: float = 40.0          # W/m/K
    thickness: float = 0.0008 # m
    sigma0: float = 600e6
    n: float = 0.15          # 加工硬化
    m: float = 0.02          # 速度感受
    r_value: float = 1.0     # ランクフォード

@dataclass
class EDRParams:
    V0: float = 2e9            # Pa = J/m3
    av: float = 3e4            # 空孔の影響
    ad: float = 1e-7           # 転位の影響
    chi: float = 0.1           # 摩擦発熱の内部分配
    K_scale: float = 0.2       # K総量のスケール
    triax_sens: float = 0.3    # 三軸度感度（環の提案で下限を下げた）
    Lambda_crit: float = 1.0   # 臨界Λ
    # 経路別スケール係数（環の提案）
    K_scale_draw: float = 0.15   # 深絞り用
    K_scale_plane: float = 0.25  # 平面ひずみ用
    K_scale_biax: float = 0.20   # 等二軸用
    # FLCのV字を作るパラメータ（ご主人さまの新提案）
    beta_A: float = 0.35       # 谷の深さ（0.2～0.5推奨）
    beta_bw: float = 0.28      # 谷の幅（0.2～0.35推奨）

@dataclass
class PressSchedule:
    """FEM or 実験ログを並べた時系列"""
    t: np.ndarray                 # [s] shape=(N,)
    eps_maj: np.ndarray           # 主ひずみ
    eps_min: np.ndarray           # 副ひずみ
    triax: np.ndarray             # σm/σeq
    mu: np.ndarray                # 摩擦係数
    pN: np.ndarray                # 接触圧[Pa]
    vslip: np.ndarray             # すべり速度[m/s]
    htc: np.ndarray               # HTC[W/m2/K]
    Tdie: np.ndarray              # 金型温度[K]
    contact: np.ndarray           # 接触率[0-1]
    T0: float = 293.15            # 板の初期温度[K]

@dataclass
class ExpBinary:
    """破断/安全のラベル付与実験"""
    schedule: PressSchedule
    failed: int                   # 1:破断, 0:安全
    label: str = ""

@dataclass
class FLCPoint:
    """FLC: 経路比一定での限界点（実測）"""
    path_ratio: float            # β
    major_limit: float           # 実測限界主ひずみ
    minor_limit: float           # 実測限界副ひずみ
    rate_major: float = 1.0      # 主ひずみ速度[1/s]
    duration_max: float = 1.0    # 試験上限[s]
    label: str = ""

# =========================
# 1.5) 物理的に正しい三軸度計算
# =========================

def triax_from_path(beta: float) -> float:
    """
    ひずみ経路比βから三軸度ηを計算（平面応力J2塑性）
    η(β) = (1+β)/(√3 * √(1+β+β²))
    """
    b = float(np.clip(beta, -0.95, 1.0))
    return (1.0 + b) / (np.sqrt(3.0) * np.sqrt(1.0 + b + b*b))

# =========================
# 2) 物性ヘルパ
# =========================

def beta_multiplier(beta, A=0.35, bw=0.28):
    """
    beta=eps_min/eps_maj（経路比）を想定
    平面ひずみ(β=0)で 1+A に、±0.5付近で ~1 に戻るV字を作る
    """
    b = np.clip(beta, -0.95, 0.95)
    return 1.0 + A * np.exp(-(b / bw)**2)

def beta_multiplier_asymmetric(beta, A_neg=0.35, A_pos=0.5, bw=0.28):
    """
    非対称β依存ゲイン（左右で異なる強度）
    深絞り側（β<0）と等二軸側（β>0）で異なる増幅
    """
    b = np.clip(beta, -0.95, 0.95)
    if b < 0:
        # 深絞り側（β<0）
        return 1.0 + A_neg * np.exp(-(b/bw)**2)
    else:
        # 等二軸側（β>0）- より強い増幅でFLCの右端急落を再現
        return 1.0 + A_pos * np.exp(-(b/bw)**2)

def cv_eq(T, c0=1e-6, Ev_eV=1.0):
    kB_eV = 8.617e-5
    return c0*np.exp(-Ev_eV/(kB_eV*T))

def step_cv(cv, T, rho_d, dt, tau0=1e-3, Q_eV=0.8, k_ann=1e6, k_sink=1e-15):
    kB_eV = 8.617e-5
    tau = tau0*np.exp(Q_eV/(kB_eV*T))
    dcv = (cv_eq(T)-cv)/tau - k_ann*cv**2 - k_sink*cv*rho_d
    return cv + dcv*dt

def step_rho(rho_d, epdot_eq, T, dt, A=1e14, B=1e-4, Qv_eV=0.8):
    kB_eV = 8.617e-5
    Dv = 1e-6*np.exp(-Qv_eV/(kB_eV*T))
    drho = A*max(epdot_eq,0.0) - B*rho_d*Dv
    return max(rho_d + drho*dt, 1e10)

def equiv_strain_rate(epsM_dot, epsm_dot):
    return np.sqrt(2.0/3.0)*np.sqrt((epsM_dot-epsm_dot)**2 + epsM_dot**2 + epsm_dot**2)

def mu_effective(mu0, T, pN, vslip):
    """温度・速度・荷重依存の有効摩擦係数（Stribeck風）"""
    # 速度・荷重比でストライベック曲線を模擬
    s = (vslip * 1e3) / (pN / 1e6 + 1.0)
    stribeck = 0.7 + 0.3 / (1 + s)
    # 温度上昇で潤滑性向上
    temp_reduction = 1.0 - 1e-4 * max(T - 293.15, 0)
    return mu0 * stribeck * temp_reduction

def flow_stress(ep_eq, epdot_eq, mat: MaterialParams, T=None, Tref=293.15, alpha=3e-4):
    """温度依存を考慮した流動応力計算"""
    rate_fac = (max(epdot_eq,1e-6)/1.0)**mat.m
    aniso = (2.0 + mat.r_value)/3.0
    temp_fac = 1.0 - alpha*max((0 if T is None else (T-Tref)), 0.0)
    return mat.sigma0 * temp_fac * (1.0 + ep_eq)**mat.n * rate_fac / aniso

# =========================
# 2.5) スムージングヘルパ（新規追加）
# =========================

def sanity_check(schedule: PressSchedule):
    """入力データの単位・範囲チェック"""
    assert np.all(schedule.pN < 5e9), "pN too large? Expected [Pa]"
    assert np.all(schedule.pN > 0), "pN must be positive [Pa]"
    assert np.all(schedule.Tdie > 150) and np.all(schedule.Tdie < 1500), "Tdie out of range? Expected [K]"
    assert np.all(schedule.t >= 0), "Time must be non-negative [s]"
    assert np.all(schedule.contact >= 0) and np.all(schedule.contact <= 1), "Contact rate must be in [0,1]"
    assert np.all(schedule.mu >= 0) and np.all(schedule.mu < 1), "Friction coefficient out of realistic range"
    if len(schedule.t) > 1:
        dt = np.diff(schedule.t)
        assert np.all(dt > 0), "Time must be monotonically increasing"

def smooth_signal(x, window_size=11):
    """移動平均によるスムージング（スパイク除去）"""
    if window_size <= 1 or len(x) <= window_size:
        return x
    kernel = np.ones(window_size) / window_size
    # パディングで端を処理
    padded = np.pad(x, (window_size//2, window_size//2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(x)]

# =========================
# 3) Λ計算（改善版v2）
# =========================

def get_path_k_scale(beta: float, edr: EDRParams) -> float:
    """ひずみ経路に応じたK_scaleを返す（環の提案）"""
    if abs(beta + 0.5) < 0.1:  # 深絞り領域
        return edr.K_scale_draw
    elif abs(beta) < 0.1:  # 平面ひずみ領域
        return edr.K_scale_plane
    elif abs(beta - 0.5) < 0.2:  # 等二軸領域
        return edr.K_scale_biax
    else:
        # 中間は線形補間
        return edr.K_scale

def simulate_lambda(schedule: PressSchedule,
                    mat: MaterialParams,
                    edr: EDRParams,
                    debug: bool = False) -> Dict[str, np.ndarray]:
    """改善版v3：冷却は回復側、経路別K_scale、β依存ゲイン、温度依存物性"""
    from collections import deque
    
    # 入力データの妥当性チェック
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
    
    # 板厚関連の初期化
    h_eff = h0
    eps3 = 0.0
    
    # ひずみ経路比を計算（K_scale選択用）
    beta_avg = np.mean(epsm / (epsM + 1e-10))
    k_scale_path = get_path_k_scale(beta_avg, edr)
    
    # β履歴の初期化（β依存ゲイン用）
    beta_hist = deque(maxlen=5)

    for k in range(len(t)-1):
        # 板厚更新
        d_eps3 = - (epsM_dot[k] + epsm_dot[k]) * dt
        eps3 += d_eps3
        h_eff = max(h0 * np.exp(eps3), 0.2*h0)
        
        # 熱収支
        q_fric = mu[k]*pN[k]*vs[k]*ctc[k]  # W/m2
        dTdt = (2.0*htc[k]*(Tdie[k]-T[k]) + 2.0*edr.chi*q_fric) / (rho*cp*h_eff)
        dTdt = np.clip(dTdt, -1000, 1000)
        T[k+1] = T[k] + dTdt*dt
        T[k+1] = np.clip(T[k+1], 200, 2000)

        # 欠陥更新
        rho_d = step_rho(rho_d, epdot_eq[k], T[k], dt)
        cv    = step_cv(cv, T[k], rho_d, dt)

        # K計算（改善版：冷却は回復側）
        K_th = rho*cp*max(dTdt, 0.0)  # 加熱時のみカウント！
        
        # 温度依存の流動応力（新機能）
        sigma_eq = flow_stress(ep_eq, epdot_eq[k], mat, T=T[k])
        K_pl = 0.9 * sigma_eq * epdot_eq[k]
        
        # 温度・速度・荷重依存の摩擦係数（新機能）
        mu_eff = mu_effective(mu[k], T[k], pN[k], vs[k])
        q_fric_eff = mu_eff * pN[k] * vs[k] * ctc[k]
        K_fr = (2.0*edr.chi*q_fric_eff)/h_eff
        
        # 瞬間βの計算（ゼロ割保護）
        num = epsm_dot[k]
        den = epsM_dot[k] if abs(epsM_dot[k]) > 1e-8 else np.sign(epsM_dot[k])*1e-8 + 1e-8
        beta_inst = num / den
        beta_hist.append(beta_inst)
        beta_smooth = float(np.mean(beta_hist))
        
        # K_total計算と経路別・β依存ゲイン適用
        K_total = k_scale_path * (K_th + K_pl + K_fr)
        K_total *= beta_multiplier(beta_smooth, A=edr.beta_A, bw=edr.beta_bw)  # β依存ゲイン！
        K_total = max(K_total, 0)

        # V_eff（温度依存性を強化）
        T_ratio = min((T[k] - 273.15) / (1500.0 - 273.15), 1.0)  # 融点への近さ
        temp_factor = 1.0 - 0.5 * T_ratio  # 温度が上がるとV_effが下がる
        V_eff = edr.V0 * temp_factor * (1.0 - edr.av*cv - edr.ad*np.sqrt(max(rho_d,1e10)))
        V_eff = max(V_eff, 0.01*edr.V0)

        # 三軸度ファクタ（感度を調整）
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

# =========================
# 4) 改善された損失関数v2
# =========================

def loss_for_binary_improved_v2(exps: List[ExpBinary],
                                mat: MaterialParams,
                                edr: EDRParams,
                                margin: float=0.08,
                                Dcrit: float=0.01,  # 0.05から0.01に緩和
                                debug: bool=False) -> float:
    """改善版v2：スムージング＋D積分判定＋安全マージン"""
    loss = 0.0
    correct = 0
    delta = 0.03  # 安全マージン
    
    for i, e in enumerate(exps):
        res = simulate_lambda(e.schedule, mat, edr, debug=False)
        
        # スムージングしてスパイクを除去
        Lam_raw = res["Lambda"]
        Lam_smooth = smooth_signal(Lam_raw, window_size=11)
        
        peak = float(np.max(Lam_smooth))  # スムージング後のピーク
        D_end = float(res["Damage"][-1])  # 累積損傷
        
        if e.failed == 1:
            # 破断：ピーク超え「かつ」滞留も必要
            condition_met = (peak > edr.Lambda_crit and D_end > Dcrit)
            if not condition_met:
                # 両条件を満たさない場合のペナルティ
                peak_penalty = max(0, edr.Lambda_crit - peak)**2
                D_penalty = max(0, Dcrit - D_end)**2
                loss += 10.0 * (peak_penalty + D_penalty)
            else:
                correct += 1
                # マージンを確保
                if peak < edr.Lambda_crit + margin:
                    loss += (edr.Lambda_crit + margin - peak)**2
                if D_end < 2*Dcrit:
                    loss += (2*Dcrit - D_end)**2
        else:
            # 安全：peak < 1-δ を目標に（安全側に余裕を持たせる）
            if peak > edr.Lambda_crit - delta:
                loss += (peak - (edr.Lambda_crit - delta))**2 * 3.0  # 係数を増やして重要視
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
    """FLC誤差（β重み付け版：平面ひずみを重視）"""
    err = 0.0
    for p in flc_pts:
        # 平面ひずみ（β≈0）を重め評価
        w = 1.5 if abs(p.path_ratio) < 0.1 else 1.0
        Em, em = predict_FLC_point(
            path_ratio=p.path_ratio,
            major_rate=p.rate_major,
            duration_max=p.duration_max,
            mat=mat, edr=edr
        )
        err += w * ((Em - p.major_limit)**2 + (em - p.minor_limit)**2)
    return err / max(len(flc_pts), 1)

# =========================
# 5) 段階的フィッティング（改善版v2）
# =========================

def fit_step1_critical_params_v2(exps: List[ExpBinary],
                                 mat: MaterialParams,
                                 initial_edr: EDRParams,
                                 verbose: bool = True) -> EDRParams:
    """Step1: K_scale系とtriax_sensの最適化"""
    if verbose:
        print("\n=== Step 1: K_scale variants & triax_sens optimization ===")
    
    def objective(x):
        edr = EDRParams(
            V0=initial_edr.V0,
            av=initial_edr.av,
            ad=initial_edr.ad,
            chi=initial_edr.chi,
            K_scale=x[0],
            triax_sens=x[1],
            Lambda_crit=initial_edr.Lambda_crit,
            K_scale_draw=x[2],
            K_scale_plane=x[3],
            K_scale_biax=x[4],
            beta_A=initial_edr.beta_A,      # V字の深さを保持
            beta_bw=initial_edr.beta_bw     # V字の幅を保持
        )
        return loss_for_binary_improved_v2(exps, mat, edr, margin=0.08, Dcrit=0.01)
    
    # 初期値と境界（環の提案を反映）
    x0 = [initial_edr.K_scale, 0.3, 0.15, 0.25, 0.20]
    bounds = [
        (0.05, 1.0),   # K_scale
        (0.1, 0.5),    # triax_sens（下限を下げた）
        (0.05, 0.3),   # K_scale_draw
        (0.1, 0.4),    # K_scale_plane
        (0.05, 0.3)    # K_scale_biax
    ]
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                  options={'maxiter': 300})
    
    updated_edr = EDRParams(
        V0=initial_edr.V0,
        av=initial_edr.av,
        ad=initial_edr.ad,
        chi=initial_edr.chi,
        K_scale=res.x[0],
        triax_sens=res.x[1],
        Lambda_crit=initial_edr.Lambda_crit,
        K_scale_draw=res.x[2],
        K_scale_plane=res.x[3],
        K_scale_biax=res.x[4],
        beta_A=initial_edr.beta_A,      # V字の深さを保持
        beta_bw=initial_edr.beta_bw     # V字の幅を保持
    )
    
    if verbose:
        print(f"K_scale: {initial_edr.K_scale:.3f} -> {res.x[0]:.3f}")
        print(f"triax_sens: {initial_edr.triax_sens:.3f} -> {res.x[1]:.3f}")
        print(f"K_scale_draw: {res.x[2]:.3f}")
        print(f"K_scale_plane: {res.x[3]:.3f}")
        print(f"K_scale_biax: {res.x[4]:.3f}")
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
            V0=x[0],
            av=edr_from_step1.av,
            ad=edr_from_step1.ad,
            chi=edr_from_step1.chi,
            K_scale=edr_from_step1.K_scale,
            triax_sens=edr_from_step1.triax_sens,
            Lambda_crit=edr_from_step1.Lambda_crit,
            K_scale_draw=edr_from_step1.K_scale_draw,
            K_scale_plane=edr_from_step1.K_scale_plane,
            K_scale_biax=edr_from_step1.K_scale_biax,
            beta_A=edr_from_step1.beta_A,      # V字の深さを保持
            beta_bw=edr_from_step1.beta_bw     # V字の幅を保持
        )
        return loss_for_binary_improved_v2(exps, mat, edr, margin=0.08, Dcrit=0.01)
    
    x0 = [edr_from_step1.V0]
    bounds = [(5e8, 5e9)]
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                  options={'maxiter': 100})
    
    updated_edr = EDRParams(
        V0=res.x[0],
        av=edr_from_step1.av,
        ad=edr_from_step1.ad,
        chi=edr_from_step1.chi,
        K_scale=edr_from_step1.K_scale,
        triax_sens=edr_from_step1.triax_sens,
        Lambda_crit=edr_from_step1.Lambda_crit,
        K_scale_draw=edr_from_step1.K_scale_draw,
        K_scale_plane=edr_from_step1.K_scale_plane,
        K_scale_biax=edr_from_step1.K_scale_biax,
        beta_A=edr_from_step1.beta_A,      # V字の深さを保持
        beta_bw=edr_from_step1.beta_bw     # V字の幅を保持
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
    """Step3: 全パラメータ微調整（Lambda_critは1.0付近に制約）"""
    if verbose:
        print("\n=== Step 3: Fine tuning all parameters ===")
    
    names = ['V0', 'av', 'ad', 'chi', 'K_scale', 'triax_sens', 'Lambda_crit',
             'K_scale_draw', 'K_scale_plane', 'K_scale_biax', 'beta_A', 'beta_bw']
    
    theta0 = np.array([
        edr_from_step2.V0,
        edr_from_step2.av,
        edr_from_step2.ad,
        edr_from_step2.chi,
        edr_from_step2.K_scale,
        edr_from_step2.triax_sens,
        1.0,  # Lambda_crit
        edr_from_step2.K_scale_draw,
        edr_from_step2.K_scale_plane,
        edr_from_step2.K_scale_biax,
        edr_from_step2.beta_A,      # V字の深さ
        edr_from_step2.beta_bw       # V字の幅
    ])
    
    bounds = [
        (theta0[0]*0.5, theta0[0]*2.0),  # V0
        (1e4, 1e6),                       # av
        (1e-8, 1e-6),                     # ad
        (0.05, 0.3),                      # chi
        (0.05, 1.0),                      # K_scale
        (0.1, 0.5),                       # triax_sens
        (0.95, 1.05),                     # Lambda_crit（1.0付近に制約）
        (0.05, 0.3),                      # K_scale_draw
        (0.1, 0.4),                       # K_scale_plane
        (0.05, 0.3),                      # K_scale_biax
        (0.2, 0.5),                       # beta_A（V字の深さ）
        (0.2, 0.35)                       # beta_bw（V字の幅）
    ]
    
    def objective(theta):
        edr = EDRParams(
            V0=theta[0],
            av=theta[1],
            ad=theta[2],
            chi=theta[3],
            K_scale=theta[4],
            triax_sens=theta[5],
            Lambda_crit=theta[6],
            K_scale_draw=theta[7],
            K_scale_plane=theta[8],
            K_scale_biax=theta[9],
            beta_A=theta[10],      # V字の深さ
            beta_bw=theta[11]      # V字の幅
        )
        L_binary = loss_for_binary_improved_v2(exps, mat, edr, margin=0.08, Dcrit=0.01)
        L_flc = loss_for_flc(flc_pts, mat, edr) if flc_pts else 0.0
        return L_binary + 0.8 * L_flc
    
    res = differential_evolution(objective, bounds, seed=42,
                                maxiter=150, popsize=20,
                                atol=1e-10, tol=1e-10)
    
    final_edr = EDRParams(
        V0=res.x[0],
        av=res.x[1],
        ad=res.x[2],
        chi=res.x[3],
        K_scale=res.x[4],
        triax_sens=res.x[5],
        Lambda_crit=res.x[6],
        K_scale_draw=res.x[7],
        K_scale_plane=res.x[8],
        K_scale_biax=res.x[9],
        beta_A=res.x[10],      # V字の深さ
        beta_bw=res.x[11]      # V字の幅
    )
    
    info = {
        'success': res.success,
        'fval': res.fun,
        'nit': res.nit,
        'message': res.message
    }
    
    if verbose:
        print(f"Final loss: {res.fun:.4f}")
        print(f"Iterations: {res.nit}")
        print(f"Success: {res.success}")
        print(f"Lambda_crit: {res.x[6]:.3f}")
        print(f"triax_sens: {res.x[5]:.3f}")
    
    return final_edr, info

# =========================
# 6) 統合フィッティング関数
# =========================

def fit_edr_params_staged_v2(binary_exps: List[ExpBinary],
                             flc_pts: List[FLCPoint],
                             mat: MaterialParams,
                             initial_edr: Optional[EDRParams] = None,
                             verbose: bool = True) -> Tuple[EDRParams, Dict]:
    """段階的フィッティングのメイン関数（改善版v2）"""
    
    if initial_edr is None:
        initial_edr = EDRParams(
            V0=2e9,
            av=3e4,
            ad=1e-7,
            chi=0.1,
            K_scale=0.2,
            triax_sens=0.3,
            Lambda_crit=1.0,
            K_scale_draw=0.15,
            K_scale_plane=0.25,
            K_scale_biax=0.20,
            beta_A=0.35,      # V字の深さ
            beta_bw=0.28      # V字の幅
        )
    
    # Step 1
    edr_step1 = fit_step1_critical_params_v2(binary_exps, mat, initial_edr, verbose)
    
    # Step 2
    edr_step2 = fit_step2_V0(binary_exps, mat, edr_step1, verbose)
    
    # Step 3
    final_edr, info = fit_step3_fine_tuning_v2(binary_exps, flc_pts, mat, edr_step2, verbose)
    
    # 最終検証
    if verbose:
        print("\n=== Final Validation ===")
        loss_final = loss_for_binary_improved_v2(binary_exps, mat, final_edr, 
                                                 margin=0.08, Dcrit=0.01, debug=True)
        print(f"Final binary loss: {loss_final:.4f}")
    
    return final_edr, info

# =========================
# 7) ヘルパー関数
# =========================

def predict_fail(res: Dict[str,np.ndarray], margin: float=0.0) -> int:
    Lam_smooth = smooth_signal(res["Lambda"], window_size=11)
    return int(np.max(Lam_smooth) > 1.0 + margin)

def time_at_lambda_cross(res: Dict[str,np.ndarray], crit: float=1.0) -> Optional[int]:
    Lam_smooth = smooth_signal(res["Lambda"], window_size=11)
    idx = np.where(Lam_smooth>crit)[0]
    return int(idx[0]) if len(idx)>0 else None

def predict_FLC_point(path_ratio: float,
                     major_rate: float,
                     duration_max: float,
                     mat: MaterialParams,
                     edr: EDRParams,
                     base_contact: float=1.0,
                     base_mu: float=0.08,
                     base_pN: float=200e6,
                     base_vslip: float=0.02,
                     base_htc: float=8000.0,
                     Tdie: float=293.15,
                     T0: float=293.15) -> Tuple[float,float]:
    """FLC点予測"""
    dt = 1e-3
    N  = int(duration_max/dt)+1
    t  = np.linspace(0, duration_max, N)
    epsM = major_rate*t
    epsm = path_ratio*major_rate*t

    schedule = PressSchedule(
        t=t,
        eps_maj=epsM,
        eps_min=epsm,
        triax=np.full(N, triax_from_path(path_ratio)),
        mu=np.full(N, base_mu),
        pN=np.full(N, base_pN),
        vslip=np.full(N, base_vslip),
        htc=np.full(N, base_htc),
        Tdie=np.full(N, Tdie),
        contact=np.full(N, base_contact),
        T0=T0
    )
    res = simulate_lambda(schedule, mat, edr)
    k = time_at_lambda_cross(res, crit=edr.Lambda_crit)
    if k is None:
        return float(res["eps_maj"][-1]), float(res["eps_min"][-1])
    return float(res["eps_maj"][k]), float(res["eps_min"][k])

# =========================
# 8) プロット関数
# =========================

def plot_flc(experimental: List[FLCPoint],
             predicted: List[Tuple[float,float]],
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
    
    # Raw vs Smoothed Lambda
    Lam_smooth = smooth_signal(res["Lambda"], window_size=11)
    ax1.plot(res["t"], res["Lambda"], 'b-', alpha=0.3, linewidth=1, label='Raw Λ')
    ax1.plot(res["t"], Lam_smooth, 'b-', linewidth=2, label='Smoothed Λ')
    ax1.axhline(1.0, ls='--', color='red', alpha=0.5, label='Λ_crit')
    ax1.fill_between(res["t"], 0, Lam_smooth, 
                     where=(Lam_smooth>1.0), color='red', alpha=0.2)
    ax1.set_ylabel('Lambda (Λ)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Damage accumulation
    ax2.plot(res["t"], res["Damage"], 'g-', linewidth=2, label='Damage D')
    ax2.axhline(0.05, ls='--', color='orange', alpha=0.5, label='D_crit')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Damage D')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =========================
# 9) デモデータ生成
# =========================

def generate_demo_experiments() -> List[ExpBinary]:
    """より現実的な合成実験データ"""
    def mk_schedule(beta, mu_base, mu_jump=False, high_stress=False):
        dt = 1e-3
        T = 0.6
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
            triax=np.full_like(t, triax_val),
            mu=mu,
            pN=np.full_like(t, 250e6 if high_stress else 200e6),
            vslip=np.full_like(t, 0.03),
            htc=np.full_like(t, 8000.0),
            Tdie=np.full_like(t, 293.15),
            contact=np.full_like(t, 1.0),
            T0=293.15
        )
    
    exps = [
        ExpBinary(mk_schedule(-0.5, 0.08, False, False), failed=0, label="safe_draw"),
        ExpBinary(mk_schedule(-0.5, 0.08, True, True), failed=1, label="draw_lubrication_fail"),
        ExpBinary(mk_schedule(0.0, 0.08, False, False), failed=0, label="safe_plane"),
        ExpBinary(mk_schedule(0.0, 0.08, True, True), failed=1, label="plane_fail"),
        ExpBinary(mk_schedule(0.5, 0.10, False, False), failed=0, label="safe_biax"),
        ExpBinary(mk_schedule(0.5, 0.10, True, True), failed=1, label="biax_fail"),
    ]
    return exps

def generate_demo_flc() -> List[FLCPoint]:
    return [
        FLCPoint(-0.5, 0.35, -0.175, 0.6, 1.0, "draw"),
        FLCPoint(0.0, 0.28, 0.0, 0.6, 1.0, "plane"),
        FLCPoint(0.5, 0.22, 0.11, 0.6, 1.0, "biax"),
    ]

# =========================
# 10) メイン実行
# =========================

def evaluate_flc_fit(experimental: List[FLCPoint],
                    predicted: List[Tuple[float, float]]) -> float:
    """FLC予測精度の評価"""
    errors = []
    for exp, pred in zip(experimental, predicted):
        deM = pred[0] - exp.major_limit
        dem = pred[1] - exp.minor_limit
        err = np.sqrt(deM**2 + dem**2)
        errors.append(err)
        print(f"  β={exp.path_ratio:+.1f}: 誤差={err:.4f} "
              f"(ΔMaj={deM:+.3f}, ΔMin={dem:+.3f})")
    
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

if __name__ == "__main__":
    print("="*60)
    print("EDRパラメータフィッティング改善版v3.1")
    print("環の提案 + ご主人さまの物理的洞察 + 物理増強")
    print("="*60)
    
    # 材料パラメータ
    mat = MaterialParams()
    
    # デモデータ生成
    exps = generate_demo_experiments()
    flc_data = generate_demo_flc()
    
    print(f"\n実験数: {len(exps)}")
    print(f"FLC点数: {len(flc_data)}")
    
    # 段階的フィッティング実行
    edr_fit, info = fit_edr_params_staged_v2(exps, flc_data, mat, verbose=True)
    
    print("\n" + "="*60)
    print("最終結果")
    print("="*60)
    print(f"EDR Parameters:")
    print(f"  V0: {edr_fit.V0:.2e} Pa")
    print(f"  av: {edr_fit.av:.2e}")
    print(f"  ad: {edr_fit.ad:.2e}")
    print(f"  chi: {edr_fit.chi:.3f}")
    print(f"  K_scale: {edr_fit.K_scale:.3f}")
    print(f"  triax_sens: {edr_fit.triax_sens:.3f}")
    print(f"  Lambda_crit: {edr_fit.Lambda_crit:.3f}")
    print(f"  K_scale_draw: {edr_fit.K_scale_draw:.3f}")
    print(f"  K_scale_plane: {edr_fit.K_scale_plane:.3f}")
    print(f"  K_scale_biax: {edr_fit.K_scale_biax:.3f}")
    print(f"  beta_A: {edr_fit.beta_A:.3f}")  # V字の深さ
    print(f"  beta_bw: {edr_fit.beta_bw:.3f}")  # V字の幅
    
    # FLC予測と比較
    print("\n予測FLC生成中...")
    preds = []
    for p in flc_data:
        Em, em = predict_FLC_point(p.path_ratio, p.rate_major, p.duration_max, mat, edr_fit)
        preds.append((Em, em))
        print(f"  β={p.path_ratio:+.1f}: 実測({p.major_limit:.3f}, {p.minor_limit:.3f}) "
              f"→ 予測({Em:.3f}, {em:.3f})")
    
    # FLC適合度評価
    flc_error = evaluate_flc_fit(flc_data, preds)
    
    # プロット
    plot_flc(flc_data, preds, title="FLC: Experimental vs EDR v2 (Improved)")
    
    # 代表的なΛ履歴を表示
    print("\nΛ履歴プロット中...")
    res = simulate_lambda(exps[3].schedule, mat, edr_fit)
    plot_lambda_t(res, title="Lambda & Damage timeline (plane strain failure) - v2")
    
    print("\n改善版v3.1 完了！")
    print("🎉 物理増強版：温度依存・摩擦モデル・β重み付けFLC実装！")
